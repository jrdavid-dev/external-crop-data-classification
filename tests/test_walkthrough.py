"""The walkthrough reads the latest completed run without hiding shortfalls or pending stages."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from crop_pipeline import walkthrough


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(macro, f1):
    return {"macro_f1": macro, "per_crop": {crop: {"f1": value} for crop, value in zip(walkthrough.CROPS, f1)}}


class WalkthroughTests(unittest.TestCase):
    def build(self, root):
        outputs = root / "pipeline_outputs"
        pre = root / "analysis" / "external_x"
        training = outputs / "training" / "run_a"
        testing = outputs / "testing" / "run_b"
        for folder in (outputs / "preprocessing", training / "models", training / "validation", testing / "reports"):
            folder.mkdir(parents=True, exist_ok=True)
        (pre / "features.json").parent.mkdir(parents=True)
        (pre / "features.json").write_text("{}")
        (training / "models" / "models_stage.json").write_text("{}")
        (testing / "scores.json").write_text("{}")
        for pointer, target in (("preprocessing/current_features.json", pre / "features.json"),
                                ("training/current_models.json", training / "models" / "models_stage.json"),
                                ("testing/current_scores.json", testing / "scores.json")):
            (outputs / pointer).write_text(json.dumps({"format_version": 1, "status": "prepared",
                                                       "path": str(target), "sha256": "x"}))
        write_csv(pre / "usable_samples.csv", [
            {"cohort": "fixed_full", "partition": "train", "crop": "carrot", "usable_parcel_groups": 61,
             "usable_pixels": 1199, "target_parcels": 60, "target_pixels": 1200},
            {"cohort": "fixed_full", "partition": "train", "crop": "potato", "usable_parcel_groups": 70,
             "usable_pixels": 1300, "target_parcels": 60, "target_pixels": 1200}])
        write_csv(pre / "reference_profile.csv", [
            {"source": "francecrops", "crop": "cabbage", "partition": "train", "parcels": 3},
            {"source": "francecrops", "crop": "potato", "partition": "train", "parcels": 2},
            {"source": "cropharvest", "crop": "carrot", "partition": "test", "parcels": 4}])
        manifest = training / "validation" / "fixed_full.json"
        manifest.write_text(json.dumps({"validation": metrics(0.654, (0.7, 0.4, 0.9))}))
        (training / "validation" / "models.json").write_text(json.dumps({"fixed_full": {"path": str(manifest)}}))
        (testing / "reports" / "test_scores.json").write_text(json.dumps({
            "fixed_full": metrics(0.2, (0.1, 0.0, 0.5)), "aligned_full": {"status": "no_usable_test_rows"}}))
        (testing / "reports" / "majority_baseline.json").write_text(json.dumps({
            "fixed_full": {"metrics": {"macro_f1": 0.25}}, "aligned_full": {"metrics": {}}}))
        return pre, training, testing

    def test_reads_runs_named_by_stage_pointers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pre, training, testing = self.build(root)
            self.assertEqual(walkthrough.current_runs(root),
                             {"preprocessing": pre, "training": training, "testing": testing})

    def test_pending_stage_is_reported_not_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build(root)
            (root / "pipeline_outputs/testing/current_scores.json").write_text(
                json.dumps({"format_version": 1, "status": "pending", "reason": "rerun testing 01"}))
            with self.assertRaisesRegex(RuntimeError, "rerun testing 01"):
                walkthrough.current_runs(root)

    def test_a_cell_needs_both_parcel_and_pixel_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            pre, _, _ = self.build(Path(temporary))
            rows = walkthrough.sample_status(pre)
            self.assertEqual([row["Met"] for row in rows], [False, True])
            self.assertEqual(rows[0]["Pixels / target"], "1,199 / 1,200")

    def test_reference_totals_per_source_and_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            pre, _, _ = self.build(Path(temporary))
            rows = walkthrough.reference_table(pre)
            self.assertEqual([(row["Source"], row["Total"]) for row in rows], [("francecrops", 5), ("cropharvest", 4)])

    def test_scores_keep_unavailable_values_and_compare_with_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, training, testing = self.build(Path(temporary))
            validation = walkthrough.validation_scores(training)
            self.assertEqual(validation[0]["Macro-F1"], 0.65)
            self.assertEqual(validation[0]["Crops ≥ 0.50"], "2 / 3")
            test = {row["Recipe"]: row for row in walkthrough.test_scores(testing)}
            self.assertFalse(test["Fixed Jan–Apr · full"]["Above baseline"])
            self.assertIsNone(test["Aligned green-up · full"]["Macro-F1"])
            self.assertIsNone(test["Aligned green-up · full"]["Above baseline"])


class WalkthroughDetailTests(unittest.TestCase):
    def test_funnel_keeps_each_step_and_partition_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            pre = Path(temporary)
            write_csv(pre / "reference_profile.csv", [
                {"source": "a", "crop": "carrot", "partition": "train", "parcels": 30},
                {"source": "b", "crop": "carrot", "partition": "train", "parcels": 10}])
            write_csv(pre / "window_counts.csv", [
                {"window": "fixed", "partition": "train", "crop": "carrot", "parcel_groups": 7},
                {"window": "aligned", "partition": "train", "crop": "carrot", "parcel_groups": 9}])
            write_csv(pre / "usable_samples.csv", [
                {"cohort": cohort, "partition": "train", "crop": "carrot", "usable_parcel_groups": value,
                 "usable_pixels": 1, "target_parcels": 60, "target_pixels": 1200}
                for cohort, value in (("fixed_full", 6), ("aligned_full", 8), ("fixed_matched", 3))])
            row = walkthrough.sample_funnel(pre)[0]
            self.assertEqual((row["Collected"], row["Complete Jan–Apr window"], row["Complete green-up window"],
                              row["Usable fixed (full)"], row["Usable aligned (full)"], row["Usable matched"],
                              row["Target parcels"]), (40, 7, 9, 6, 8, 3, 60))
            self.assertEqual(walkthrough.sample_funnel(pre)[3]["Collected"], 0)

    def test_greenup_onsets_count_only_accepted_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            pre = Path(temporary)
            write_csv(pre / "greenup" / "candidate_greenup.csv", [
                {"status": "candidate", "reason": "supported_rise", "onset_interval_start": "2021-05-11"},
                {"status": "candidate", "reason": "supported_rise", "onset_interval_start": "2021-05-20"},
                {"status": "rejected", "reason": "gap_hides_onset", "onset_interval_start": "2021-02-01"}])
            (pre / "seasons").mkdir()
            (pre / "seasons" / "automatic_timing_policy.json").write_text("{}")
            result = walkthrough.greenup_outcomes(pre)
            self.assertEqual(result["onset_months"][5], 2)
            self.assertEqual(result["onset_months"][2], 0)
            self.assertEqual(result["outcomes"][0], {"Outcome": "candidate", "Reason": "supported rise", "Parcel-years": 2})

    def test_leaf_candidates_and_kept_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            training = Path(temporary)
            (training / "validation").mkdir()
            (training / "validation" / "validation_candidates.json").write_text(json.dumps({"fixed_full": {
                "candidates": [{"min_samples_leaf": 2, "metrics": {"macro_f1": 0.661}},
                               {"min_samples_leaf": 4, "metrics": {"macro_f1": 0.672}}],
                "selected_min_samples_leaf": 4}}))
            self.assertEqual(walkthrough.leaf_selection(training),
                             [{"Recipe": "Fixed Jan–Apr · full", "Leaf 2": 0.66, "Leaf 4": 0.67, "Kept leaf size": 4}])

    def test_aligned_test_rows_take_country_from_their_site_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            testing = Path(temporary)
            (testing / "features_test.json").write_text(json.dumps({"rows": {
                "fixed_full": [{"country": "Senegal", "site_cell": "15,-17", "crop": "cabbage", "partition_group": "a"}],
                "aligned_full": [{"site_cell": "15,-17", "crop": "cabbage", "partition_group": "a"},
                                 {"site_cell": "15,-17", "crop": "cabbage", "partition_group": "a"},
                                 {"site_cell": "0,0", "crop": "potato", "partition_group": "b"}]}}))
            rows = walkthrough.test_geography(testing, "aligned_full")
            self.assertEqual(rows, [{"Country": "Senegal", "Carrot": 0, "Cabbage": 1, "Potato": 0},
                                    {"Country": "unrecorded", "Carrot": 0, "Cabbage": 0, "Potato": 1}])

    def test_forest_importance_is_grouped_by_variable_and_bin(self):
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        columns = [f"bin_{b:02d}_{v}" for b in (1, 2) for v in ("NDVI", "B11")]
        rng = np.random.default_rng(0)
        features = rng.normal(size=(60, len(columns)))
        labels = np.where(features[:, 0] > 0, "carrot", "potato")
        model = RandomForestClassifier(n_estimators=5, max_samples=.5, min_samples_leaf=2, random_state=42).fit(features, labels)
        with tempfile.TemporaryDirectory() as temporary:
            training = Path(temporary)
            (training / "models" / "fixed_full").mkdir(parents=True)
            joblib.dump({"classifier": model, "feature_columns": columns}, training / "models" / "fixed_full" / "model.joblib")
            details = walkthrough.forest_details(training)
        self.assertEqual(details["settings"]["Trees"], 5)
        self.assertEqual(set(details["by_variable"]), {"NDVI", "B11"})
        self.assertEqual(list(details["by_bin"]), [1, 2])
        self.assertAlmostEqual(sum(details["by_variable"].values()), 1.0)
        self.assertGreater(details["by_variable"]["NDVI"], details["by_variable"]["B11"])


if __name__ == "__main__":
    unittest.main()
