import unittest

import numpy as np

from crop_pipeline.features import experiments
from model_aza.training import transfer, tuning

FAST = {**tuning.BASELINE, "n_estimators": 15}


def rows(source_by_crop=None, offset_by_source=None):
    """Separable crops; each source can add a constant reflectance offset."""
    source_by_crop = source_by_crop or {"carrot": ["a", "b"], "cabbage": ["a"], "potato": ["b", "c"]}
    offset_by_source = offset_by_source or {}
    random = np.random.default_rng(1)
    result = []
    for partition in ("train", "validation"):
        for cell in range(10):
            for level, crop in enumerate(transfer.CROPS):
                source = source_by_crop[crop][cell % len(source_by_crop[crop])]
                group = f"{partition}-{cell}-{crop}"
                for _ in range(2):
                    row = {"partition_group": group, "crop": crop, "partition": partition,
                           "source_dataset": source, "label_status": "reported", "site_cell": f"45,{cell}"}
                    for b in range(1, 9):
                        for v in transfer.VARIABLES:
                            row[f"bin_{b:02d}_{v}"] = (.1 + .05 * level * np.sin(b / 3) + .01 * b
                                                        + offset_by_source.get(source, 0) + random.normal(0, .003))
                    for column in experiments.feature_sets()["shape_only"]:
                        row[column] = level + random.normal(0, .1)
                    result.append(row)
    return result


class FeatureVariantTests(unittest.TestCase):
    def test_variants_have_expected_widths_and_standardization_removes_offsets(self):
        data = rows()
        widths = {variant: transfer.matrix(data, variant).shape[1] for variant in transfer.VARIANTS}
        self.assertEqual(widths, {"interval_72": 72, "indices_32": 32, "standardized_72": 72,
                                  "indices_standardized_32": 32, "shape_48": 48, "indices_32_plus_shape_48": 80})
        shifted = [{**row, **{c: row[c] + .3 for c in transfer.interval_columns()}} for row in data[:4]]
        np.testing.assert_allclose(transfer.matrix(shifted, "standardized_72"),
                                   transfer.matrix(data[:4], "standardized_72"), atol=1e-6)

    def test_band_scaling_recomputes_indices_and_shift_repeats_edges(self):
        values = np.full((1, 8, 9), .2)
        values[0, :, transfer.BANDS.index("B8")] = .4
        scaled = transfer.recompute_indices(values)
        self.assertAlmostEqual(scaled[0, 0, transfer.VARIABLES.index("NDVI")], (.4 - .2) / (.4 + .2))
        self.assertAlmostEqual(scaled[0, 0, transfer.VARIABLES.index("SAVI")], 1.5 * .2 / (.6 + .5))
        ramp = np.arange(8, dtype=float).reshape(1, 8, 1).repeat(9, axis=2)
        self.assertEqual(transfer.shift_bins(ramp, 1)[0, :, 0].tolist(), [0, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(transfer.shift_bins(ramp, -1)[0, :, 0].tolist(), [1, 2, 3, 4, 5, 6, 7, 7])

    def test_augmentation_preserves_total_weight_and_labels(self):
        values = np.random.default_rng(0).uniform(.1, .3, size=(5, 8, 9))
        augmented, labels, weights = transfer.augment(values, list("abcde"), np.ones(5),
                                                      shifts=(-1, 1), band_spread=.1, copies=2)
        self.assertEqual(len(augmented), 25)
        self.assertAlmostEqual(weights.sum(), 5)
        self.assertEqual(Counter_of(labels), {k: 5 for k in "abcde"})


def Counter_of(values):
    from collections import Counter
    return dict(Counter(values))


class TransferDiagnosticsTests(unittest.TestCase):
    def test_confounding_table_counts_parcels_per_source(self):
        table = {r["crop"]: r for r in transfer.confounding_table(rows())}
        self.assertEqual(table["cabbage"]["sources_with_parcels"], 1)
        self.assertEqual(table["carrot"]["a"] + table["carrot"]["b"], 20)

    def test_source_predictability_detects_offsets_and_skips_single_source_crops(self):
        data = rows(offset_by_source={"a": .2})
        results = {r["crop"]: r for r in transfer.source_predictability(data, params=FAST)}
        self.assertEqual(results["cabbage"]["status"], "fewer_than_two_sources_with_enough_parcels")
        self.assertGreater(results["carrot"]["source_balanced_accuracy"], .9)
        standardized = {r["crop"]: r for r in transfer.source_predictability(data, "standardized_72", FAST)}
        self.assertLess(standardized["carrot"]["source_balanced_accuracy"], .9)

    def test_leave_one_source_out_scores_only_shared_crops(self):
        results = {r["held_out_source"]: r for r in transfer.leave_one_source_out(rows(), FAST)}
        self.assertEqual(results["a"]["scored_crops"], "carrot")
        self.assertEqual(results["a"]["unseen_crops"], "cabbage")
        self.assertIn("carrot_recall", results["a"])
        self.assertEqual(results["c"]["scored_crops"], "potato")

    def test_spatial_cv_and_augmented_fit_use_training_parcels(self):
        data = rows()
        train = [r for r in data if r["partition"] == "train"]
        pooled = transfer.spatial_cv(train, FAST, "indices_32", {"shifts": (1,), "band_spread": .05})
        self.assertEqual(sum(v["support"] for v in pooled["per_crop"].values()), 30)
        self.assertEqual(len(pooled["fold_macro_f1"]), 5)
        with self.assertRaises(ValueError):
            transfer.fit(train, FAST, "shape_48", {"shifts": (1,)})


class SourceSignatureOptionTests(unittest.TestCase):
    def test_source_balanced_weights_equalize_crops_then_sources(self):
        data = [r for r in rows() if r["partition"] == "train"]
        weights = transfer.training_weights(data, "crop_source_parcel")
        totals = {}
        for row, weight in zip(data, weights):
            totals[row["crop"], row["source_dataset"]] = totals.get((row["crop"], row["source_dataset"]), 0) + weight
        crop_totals = {crop: sum(v for (c, _), v in totals.items() if c == crop) for crop in transfer.CROPS}
        np.testing.assert_allclose(list(crop_totals.values()), [len(data) / 3] * 3)
        np.testing.assert_allclose(totals["carrot", "a"], totals["carrot", "b"])
        with self.assertRaises(ValueError):
            transfer.training_weights(data, "unknown")

    def test_anchor_offsets_chain_through_shared_crops_and_remove_source_shift(self):
        data = rows(offset_by_source={"b": .2, "c": -.1})
        offsets = transfer.anchor_offsets(data, reference="a")
        self.assertEqual(set(offsets), {"a", "b", "c"})
        np.testing.assert_allclose(offsets["b"], .2, atol=.01)   # linked to a by carrot
        np.testing.assert_allclose(offsets["c"], -.1, atol=.01)  # linked to b by potato
        corrected = transfer.calibrate(transfer.series(data), data, offsets)
        raw = transfer.series(data)
        self.assertLess(abs(corrected[[r["source_dataset"] == "b" for r in data]].mean()
                            - corrected[[r["source_dataset"] == "a" for r in data]].mean()),
                        abs(raw[[r["source_dataset"] == "b" for r in data]].mean()
                            - raw[[r["source_dataset"] == "a" for r in data]].mean()))
        with self.assertRaises(transfer.NotApplicable):
            transfer.calibrate(raw, [{**data[0], "source_dataset": "new"}], offsets)

    def test_calibrated_fit_and_anchor_holdout_use_only_training_offsets(self):
        data = rows(offset_by_source={"b": .2})
        train = [r for r in data if r["partition"] == "train"]
        model = transfer.fit(train, FAST, "interval_72", training={"calibration": {"reference": "a"}})
        self.assertEqual(set(model.source_offsets_), {"a", "b", "c"})
        linked = rows({"carrot": ["a", "b"], "cabbage": ["a"], "potato": ["a", "b"]}, {"b": .2})
        calibrated = {(r["held_out_source"], r["held_out_crop"]): r for r in transfer.crop_transfer_holdout(
            linked, FAST, training={"calibration": {"reference": None}})}
        # Source b keeps potato (shared with a) when its carrot is held out; carrot is learned from a.
        self.assertEqual(calibrated["b", "carrot"]["status"], "scored")
        self.assertEqual(calibrated["b", "carrot"]["crop_trained_from"], "a")
        self.assertGreaterEqual(calibrated["b", "carrot"]["recall"], .8)
        self.assertEqual(calibrated["a", "cabbage"]["status"], "not_scorable")  # cabbage has one source
        uncalibrated = {(r["held_out_source"], r["held_out_crop"]): r
                        for r in transfer.crop_transfer_holdout(linked, FAST)}
        self.assertLess(uncalibrated["b", "carrot"]["recall"], calibrated["b", "carrot"]["recall"])
        # A chain of sources (a-carrot-b-potato-c) breaks when the bridge crop is held out.
        chain = {(r["held_out_source"], r["held_out_crop"]): r for r in transfer.crop_transfer_holdout(
            data, FAST, training={"calibration": {"reference": None}})}
        self.assertIn("No anchor offset", chain["b", "carrot"]["reason"])
        plain = {r["held_out_source"]: r for r in transfer.leave_one_source_out(
            data, FAST, training={"calibration": {"reference": None}})}
        self.assertEqual(plain["b"]["status"], "not_scorable")

    def test_pairwise_forest_needs_connected_pairs_and_returns_three_class_probabilities(self):
        data = rows()
        train = [r for r in data if r["partition"] == "train"]
        model = transfer.fit(train, FAST, training={"method": "pairwise"})
        self.assertEqual({(a, b) for a, b, _, _ in model.models}, {("carrot", "potato"), ("carrot", "cabbage")})
        probabilities = model.predict_proba(transfer.matrix(train[:6], "interval_72"))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertGreater(transfer.locked_validation(
            train, [r for r in data if r["partition"] == "validation"], FAST,
            training={"method": "pairwise"})["macro_f1"], .8)
        single = [{**r, "source_dataset": f"{r['crop']}_only"} for r in train]
        with self.assertRaises(transfer.NotApplicable):
            transfer.fit(single, FAST, training={"method": "pairwise"})

    def test_screening_drops_source_identifying_predictors_inside_fit(self):
        data = rows(offset_by_source={"b": .3})
        train = [r for r in data if r["partition"] == "train"]
        model = transfer.fit(train, FAST, training={"screen_drop": 10})
        self.assertEqual(len(model.keep_), 62)
        self.assertEqual(transfer.evaluation_features(model, train[:2], "interval_72").shape, (2, 62))
        self.assertEqual(len(transfer.variant_columns("indices_32_plus_shape_48")), 80)

    def test_evaluate_configuration_summarizes_and_reports_inapplicable_options(self):
        data = {"rows": {c: [{**r, "source_dataset": {"a": "sen4agrinet", "b": "francecrops", "c": "eurocrops"}[
            r["source_dataset"]]} for r in rows()] for c in tuning.COHORTS}}
        summary, details = transfer.evaluate_configuration(data, "fixed_full", FAST, label="plain")
        self.assertEqual(summary["status"], "scored")
        self.assertTrue(0 <= summary["spatial_macro_f1"] <= 1)
        self.assertEqual(len(details["source"]), 3)
        data["rows"]["fixed_full"] = [{**r, "source_dataset": "sen4agrinet" if r["crop"] == "carrot" else
                                       "francecrops" if r["crop"] == "cabbage" else "eurocrops"}
                                      for r in data["rows"]["fixed_full"]]
        summary, _ = transfer.evaluate_configuration(data, "fixed_full", FAST, training={"method": "pairwise"})
        self.assertEqual(summary["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
