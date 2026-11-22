"""Annual harmonic recipe: features, Sentinel-1 cache and the inputs -> test stages."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.annual import features as annual
from experiments.annual import sentinel1
from experiments.annual import workflow as annual_workflow, development_views

CROPS = development_views.CROPS
SOURCES = ("sen4agrinet", "francecrops", "eurocrops")


def curves(crop, rng, n):
    t = (np.arange(annual.BINS) + .5) / annual.BINS
    phase = {"carrot": 0.0, "cabbage": 2.0, "potato": 4.0}[crop]
    season = 0.15 + 0.1 * np.cos(2 * np.pi * t - phase)
    s2 = np.repeat(season[None, :, None], 5, axis=2) * np.linspace(.6, 1.4, 5) + rng.normal(0, .005, (n, 24, 5))
    s1 = np.stack([-12 + 3 * np.cos(2 * np.pi * t - phase), -18 + 2 * np.sin(2 * np.pi * t - phase)], axis=-1)
    return s2, s1[None] + rng.normal(0, .2, (n, 24, 2))


def synthetic(pixels=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, s2, s1 = [], [], []
    for parcel in range(120):
        crop = CROPS[parcel % 3]
        if parcel < 90:
            partition, cohort, source = ("validation" if parcel % 5 == 0 else "train"), None, SOURCES[(parcel // 3) % 3]
        else:
            partition, cohort, source = "test", ("jecam" if parcel < 108 else "denmark_marker_cabbage_2023_v1"), "cropharvest"
            crop = "cabbage" if cohort != "jecam" else crop
        values, radar = curves(crop, rng, pixels)
        for p in range(pixels):
            rows.append({"partition_group": f"g{parcel}", "region_id": f"r{parcel}", "partition": partition,
                         "crop": crop, "source_dataset": source, "site_cell": f"{40 + parcel % 9},{parcel % 6}",
                         "label_status": "reported", "year": 2021, "x": 100. * parcel + p, "y": 5., "source_crs": "EPSG:32631",
                         "test_cohort": cohort})
        s2.append(values)
        s1.append(radar)
    s2, s1 = np.concatenate(s2), np.concatenate(s1)
    valid = np.ones(s2.shape[:2], bool)
    return rows, {"s2_values": s2, "s2_valid": valid, "s1_values": s1}


def inputs_run(root, rows, arrays):
    run = root / "pipeline_outputs" / "annual" / "inputs" / "run_fixture"
    run.mkdir(parents=True)
    annual_workflow.save_inputs(run, rows, arrays)
    receipt = annual_workflow.seal(run, "inputs", quality_handoff="fixture")
    annual_workflow._publish(root, "inputs", receipt)
    return receipt


FAST = {**development_views.SELECTION_FOREST, "n_estimators": 15}


class AnnualFeatureTests(unittest.TestCase):
    def test_harmonic_fit_recovers_mean_and_amplitude(self):
        t = (np.arange(24) + .5) / 24
        series = (3 + 2 * np.cos(2 * np.pi * t - 1.0))[None]
        coefficients, ok = annual.fit_harmonics(series, np.ones((1, 24), bool))
        terms = annual.harmonic_terms(coefficients)
        self.assertTrue(ok[0])
        np.testing.assert_allclose(terms[0, :2], [3, 2], atol=1e-9)
        np.testing.assert_allclose(np.arctan2(terms[0, 3], terms[0, 2]), 1.0, atol=1e-9)

    def test_clear_bin_rule_needs_twelve_bins_and_no_gap_over_three_months(self):
        series = np.ones((3, 24))
        valid = np.ones((3, 24), bool)
        valid[0, :13] = False          # 11 clear bins
        valid[1, 2:9] = False          # a 7-bin gap
        valid[2, 2:8] = False          # a 6-bin gap is allowed
        _, ok = annual.fit_harmonics(series, valid)
        self.assertEqual(ok.tolist(), [False, False, True])

    def test_radar_recipe_columns_and_shared_location_exclusion(self):
        rows, arrays = synthetic(pixels=1)
        X, ok, columns = annual.RECIPES["annual_s2_s1"](arrays["s2_values"], arrays["s2_valid"], arrays["s1_values"])
        self.assertEqual(len(columns), 63 + 14 + 6)
        self.assertTrue(ok.all() and np.isfinite(X).all())
        rows[1] = {**rows[1], "x": rows[0]["x"], "y": rows[0]["y"]}  # another development parcel, same location
        rows[100] = {**rows[100], "x": rows[0]["x"], "y": rows[0]["y"]}  # a test pixel is not dropped
        self.assertEqual(annual.shared_development_locations(rows), [0, 1])


class Sentinel1CacheTests(unittest.TestCase):
    def test_cache_only_reads_and_rejects_missing_or_foreign_entries(self):
        rows = [{"x": float(i), "y": 0., "source_crs": "EPSG:32631", "year": 2021} for i in range(5)]
        with TemporaryDirectory() as folder:
            cache = Path(folder)
            with self.assertRaises(RuntimeError):
                sentinel1.sample(rows, cache, ee=None)
            name, part, _, _ = next(sentinel1.chunks(rows))
            features = [{"properties": {"i": i, "VV_00": -10.0 - i, "VH_23": sentinel1.NODATA}} for i in part]
            (cache / name).write_text(json.dumps({"features": features}))
            values = sentinel1.sample(rows, cache, ee=None)
            self.assertEqual(values[3, 0, 0], -13.0)
            self.assertTrue(np.isnan(values[0, 23, 1]))
            (cache / name).write_text(json.dumps({"features": [{"properties": {"i": 99}}]}))
            with self.assertRaises(ValueError):
                sentinel1.sample(rows, cache, ee=None)

    def test_cache_is_reused_when_row_numbers_shift(self):
        rows = [{"x": float(i), "y": 0., "source_crs": "EPSG:32631", "year": 2021} for i in range(5)]
        with TemporaryDirectory() as folder:
            name, part, _, _ = next(sentinel1.chunks(rows))
            features = [{"properties": {"i": i, "VV_00": -10.0 - i}} for i in part]
            (Path(folder) / name).write_text(json.dumps({"features": features}))
            # Dropping a parcel elsewhere shifts these rows by one; the same points reuse the cache.
            shifted = [{"x": 9., "y": 9., "source_crs": "EPSG:32631", "year": 2020}] + rows
            (Path(folder) / next(sentinel1.chunks(shifted))[0]).write_text(
                json.dumps({"features": [{"properties": {"i": 0}}]}))
            values = sentinel1.sample(shifted, folder, ee=None)
            self.assertEqual(values[4, 0, 0], -13.0)
            self.assertTrue(np.isnan(values[0, 0, 0]))

    def test_missing_requests_are_fetched_once_and_cached(self):
        rows = [{"x": float(i), "y": 0., "source_crs": "EPSG:32631", "year": 2021} for i in range(3)]
        with TemporaryDirectory() as folder, patch.object(sentinel1, "_request") as request:
            request.side_effect = lambda ee, rows, part, year, crs: {
                "features": [{"properties": {"i": i, "VV_05": -8.0}} for i in part]}
            first = sentinel1.sample(rows, folder, ee=object(), progress=None)
            second = sentinel1.sample(rows, folder, ee=None)
            self.assertEqual(request.call_count, 1)
            np.testing.assert_array_equal(first, second)


class AnnualMapTests(unittest.TestCase):
    def test_confidence_rule_and_missing_features(self):
        from experiments.annual import maps as annual_maps

        class Model:
            classes_ = np.array(["carrot", "potato"])  # cabbage absent from this fit

            def predict_proba(self, X):
                return np.array([[.9, .1], [.55, .45], [.62, .38], [.5, .5]])

        labels, probabilities, accepted = annual_maps.classify(Model(), np.zeros((4, 2)),
                                                               np.array([True, True, True, False]))
        self.assertEqual(labels.tolist(), ["carrot", "Unknown crop type", "carrot", "No features"])
        self.assertEqual(probabilities[:, CROPS.index("cabbage")].tolist(), [0, 0, 0, 0])
        self.assertEqual(accepted.tolist(), [True, False, True, False])

    def test_range_flag_needs_more_than_a_fifth_of_predictors_outside(self):
        from experiments.annual import maps as annual_maps

        training = np.tile(np.linspace(0, 1, 101)[:, None], (1, 10))
        inside = np.full(10, .5)
        one_out = inside.copy()
        one_out[0] = 2.0                     # 10% outside
        three_out = inside.copy()
        three_out[:3] = -1.0                 # 30% outside
        share, flagged = annual_maps.outside_training_range(np.stack([inside, one_out, three_out]), training)
        np.testing.assert_allclose(share, [0, .1, .3])
        self.assertEqual(flagged.tolist(), [False, False, True])

    def test_unknown_area_is_refused(self):
        from experiments.annual import maps as annual_maps

        with self.assertRaises(ValueError):
            annual_maps.map_area(".", "baguio")


class SelectionAndRegistryTests(unittest.TestCase):
    def test_selection_records_the_default_forest_and_freezing_another_needs_an_explicit_estimator(self):
        from sklearn.linear_model import LogisticRegression

        rows, arrays = synthetic()
        with TemporaryDirectory() as folder, patch.object(development_views, "SELECTION_FOREST", FAST), \
                patch.object(development_views, "FOREST", FAST):
            root = Path(folder)
            inputs_run(root, rows, arrays)
            selection = annual_workflow.select(root, progress=None, recipes=["annual_s2"])
            record = json.loads((selection.parent / "selection.json").read_text())
            self.assertEqual(record["classifier"], annual_workflow.DEFAULT_CLASSIFIER)
            self.assertNotIn("classifier_candidates", record)
            # A selection naming another classifier cannot be frozen as a forest by accident.
            other = annual_workflow._new_run(root, "selection")
            (other / "selection.json").write_text(json.dumps({**record, "classifier": {"family": "logistic_regression",
                                                                                        "setting": "C=1"}}))
            receipt = annual_workflow.seal(other, "selection", inputs_handoff=json.loads(selection.read_text())["inputs_handoff"])
            with self.assertRaises(ValueError):
                annual_workflow.freeze(root, receipt)
            models = annual_workflow.freeze(root, receipt, estimator=LogisticRegression(max_iter=500))
            info = json.loads((models.parent / "model.json").read_text())
            self.assertEqual((info["classifier"]["family"], info["classifier"]["estimator"]),
                             ("logistic_regression", "LogisticRegression"))

    def test_registered_experiment_recipes_are_usable_but_never_selection_defaults(self):
        rows, arrays = synthetic(pixels=1)
        recipe = {"annual_s2_copy_for_test": annual.RECIPES["annual_s2"]}
        try:
            annual_workflow.register_recipes(recipe)
            X, ok, columns = annual_workflow.recipe_matrix("annual_s2_copy_for_test", arrays)
            self.assertEqual(X.shape[1], len(columns))
            self.assertNotIn("annual_s2_copy_for_test", annual_workflow.DEFAULT_RECIPES)
            with self.assertRaises(ValueError):
                annual_workflow.register_recipes({"annual_s2": annual.RECIPES["annual_s2_s1"]})
        finally:
            annual_workflow.RECIPES.pop("annual_s2_copy_for_test", None)
        self.assertEqual(annual_workflow.DEFAULT_RECIPES, tuple(annual.RECIPES))


class AnnualWorkflowTests(unittest.TestCase):
    def test_stages_select_on_development_and_test_once_in_main_layout(self):
        rows, arrays = synthetic()
        with TemporaryDirectory() as folder, patch.object(development_views, "SELECTION_FOREST", FAST), \
                patch.object(development_views, "FOREST", FAST):
            root = Path(folder)
            inputs_run(root, rows, arrays)
            with patch.object(development_views, "view_probabilities",
                              wraps=development_views.view_probabilities) as views:
                selection = annual_workflow.select(root, progress=None)
            self.assertTrue(all(r["partition"] != "test" for call in views.call_args_list for r in call.args[1]))
            chosen = json.loads((selection.parent / "selection.json").read_text())
            self.assertIn(chosen["selected"], annual_workflow.DEFAULT_RECIPES)
            self.assertEqual(set(chosen["candidates"]), set(annual_workflow.DEFAULT_RECIPES))
            self.assertEqual(chosen["classifier"], {"family": "random_forest", "setting": "original"})
            self.assertFalse(chosen["test_rows_used"])
            models = annual_workflow.freeze(root)
            info = json.loads((models.parent / "model.json").read_text())
            self.assertEqual(info["trained_on"], "locked training split")
            self.assertEqual(info["classifier"]["family"], chosen["classifier"]["family"])
            self.assertGreater(info["validation"]["macro_f1"], .9)
            test = annual_workflow.test(root)
            run = test.parent
            scopes = json.loads((run / "reports" / "test_scopes.json").read_text())
            self.assertEqual(scopes, ["jecam", "denmark_marker_cabbage_2023_v1", "combined"])
            with (run / "predictions" / "annual_test_parcels.csv").open() as stream:
                jecam = list(csv.DictReader(stream))
            self.assertEqual(len(jecam), 18)
            self.assertEqual({r["test_cohort"] for r in jecam}, {"jecam"})
            denmark = json.loads((run / "reports" / "denmark_marker_cabbage_2023_v1" / "test_scores.json").read_text())
            self.assertNotIn("macro_f1", denmark["annual"])
            baseline = json.loads((run / "reports" / "majority_baseline.json").read_text())
            self.assertEqual(baseline["annual"]["training_majority_class"], info["training_majority_class"])
            pointer = json.loads((root / "pipeline_outputs" / "annual" / "current_test.json").read_text())
            self.assertEqual(Path(pointer["path"]), test)

    def test_changed_inputs_and_development_overlap_are_refused(self):
        rows, arrays = synthetic()
        with TemporaryDirectory() as folder, patch.object(development_views, "SELECTION_FOREST", FAST), \
                patch.object(development_views, "FOREST", FAST):
            root = Path(folder)
            receipt = inputs_run(root, rows, arrays)
            annual_workflow.select(root, progress=None)
            models = annual_workflow.freeze(root)
            info_path = models.parent / "model.json"
            info = json.loads(info_path.read_text())
            info["development_groups"].append("g100")  # a JECAM parcel
            info_path.write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                annual_workflow.test(root, models)
            (receipt.parent / "pixels.json").write_text("[]")
            with self.assertRaises(ValueError):
                annual_workflow.select(root, progress=None)



class CollectionSplitTests(unittest.TestCase):
    def rows(self):
        return [{"partition_group": g, "partition": p} for g, p in
                (("a", "development"), ("a", "development"), ("b", "development"), ("t", "test"))]

    def test_pooled_development_takes_the_collection_locks(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "collection_snapshot.json").write_text(json.dumps({"report_root": "inventory"}))
            with patch.object(annual_workflow.adaptive, "assignments",
                              return_value={"a": "train", "b": "validation", "t": "test"}) as locks:
                rows, split = annual_workflow.collection_partitions(folder / "quality.json", self.rows())
            locks.assert_called_once_with("inventory")
            self.assertEqual([r["partition"] for r in rows], ["train", "train", "validation", "test"])
            self.assertEqual((split["train_parcels"], split["validation_parcels"]), (1, 1))
            with patch.object(annual_workflow.adaptive, "assignments", return_value={"a": "train"}), \
                    self.assertRaises(ValueError):
                annual_workflow.collection_partitions(folder / "quality.json", self.rows())

    def test_older_handoffs_are_unchanged(self):
        older = [{"partition_group": "a", "partition": "train"}, {"partition_group": "t", "partition": "test"}]
        self.assertEqual(annual_workflow.collection_partitions(Path("unused"), older), (older, None))


if __name__ == "__main__":
    unittest.main()
