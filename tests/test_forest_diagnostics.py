import unittest
from datetime import date
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.training import error_analysis, exploratory, forest_diagnostics as diagnostics


class ForestDiagnosticsTests(unittest.TestCase):
    def test_weighted_oob_distinguishes_pixel_and_whole_parcel_omissions(self):
        rows = [
            {"partition_group": str(parcel), "crop": ("carrot", "cabbage", "potato")[parcel % 3],
             "label_status": "reported", "bin_01_B3": pixel}
            for parcel in range(6) for pixel in (parcel, parcel + 0.1)
        ]
        columns = ["bin_01_B3"]
        values = exploratory.feature_matrix(rows, columns)
        weights = exploratory.normalized_training_weights(rows, diagnostics.WEIGHTS)
        forest = RandomForestClassifier(n_estimators=20, random_state=42, max_samples=0.5)
        forest.fit(values, [r["crop"] for r in rows], sample_weight=weights)

        summary, trees, parcels = diagnostics.oob_statistics(forest, rows, columns)
        expected = {}
        for parcel in range(6):
            omitted = [tree.predict_proba(values[parcel * 2:parcel * 2 + 2]).mean(axis=0)
                       for tree, draw in zip(forest.estimators_, forest.estimators_samples_, strict=True)
                       if parcel * 2 not in draw and parcel * 2 + 1 not in draw]
            expected[str(parcel)] = len(omitted)
            row = parcels[parcel]
            if omitted:
                self.assertEqual(row["whole_parcel_oob_prediction"],
                                 forest.classes_[np.mean(omitted, axis=0).argmax()])
        self.assertEqual({r["partition_group"]: r["whole_parcel_oob_trees"] for r in parcels}, expected)
        self.assertEqual(summary[0]["scored_units"], len(rows))
        self.assertEqual(summary[2]["scored_units"], sum(count > 0 for count in expected.values()))
        self.assertTrue(all(r["drawn_rows_with_replacement"] == 6 for r in trees))
        for tree in trees:
            self.assertEqual(tree["distinct_training_rows"] + tree["oob_training_rows"], len(rows))

    def test_hard_winner_can_differ_from_probability_winner(self):
        rows = [
            {"partition_group": "p", "region_id": "r", "crop": "carrot", "x": x, "y": 0,
             "bin_01_B3": x}
            for x in (0, 1)
        ]
        trees = [
            SimpleNamespace(classes_=np.arange(3),
                            predict_proba=lambda values, check_input=False: np.tile([0.51, 0.49, 0], (2, 1))),
            SimpleNamespace(classes_=np.arange(3),
                            predict_proba=lambda values, check_input=False: np.tile([0.51, 0.49, 0], (2, 1))),
            SimpleNamespace(classes_=np.arange(3),
                            predict_proba=lambda values, check_input=False: np.tile([0, 1, 0], (2, 1))),
        ]
        forest = SimpleNamespace(classes_=np.array(["cabbage", "carrot", "potato"]), estimators_=trees)
        parcels, pixels = diagnostics.hard_vote_statistics(forest, rows, ["bin_01_B3"])
        self.assertEqual(parcels["p"]["hard_vote_winner"], "cabbage")
        self.assertEqual(parcels["p"]["probability_prediction"], "carrot")
        self.assertAlmostEqual(parcels["p"]["hard_vote_for_probability_prediction"], 1 / 3)
        self.assertEqual([r["hard_vote_winner"] for r in pixels], ["cabbage", "cabbage"])

    def test_grouping_preserves_importance_mass_in_each_view(self):
        forest = SimpleNamespace(feature_importances_=[0.1, 0.3, 0.6])
        _, groups = diagnostics.impurity_importance(forest, ["bin_01_B3", "bin_01_NDVI", "bin_02_B3"])
        by_group = {(r["grouping"], r["group"]): r["importance"] for r in groups}
        self.assertAlmostEqual(by_group["band_or_index", "B3"], 0.7)
        self.assertAlmostEqual(by_group["half_month_bin", "01"], 0.4)
        for grouping in ("band_or_index", "half_month_bin"):
            self.assertAlmostEqual(sum(r["importance"] for r in groups if r["grouping"] == grouping), 1)

    def test_reference_dates_keep_placeholders_and_conflicting_dates_unavailable(self):
        self.assertIsNone(error_analysis.observation_date(
            {"raw_matches": [{"collection_date": "2019-01-01"}]}, 2019))
        self.assertIsNone(error_analysis.observation_date(
            {"raw_matches": [{"collection_date": "2019-03-01"}, {"collection_date": "2019-05-01"}]}, 2019))
        self.assertEqual(error_analysis.observation_date({"raw_matches": [{"MAIN_CROP_DATE": 239}]}, 2023),
                         date(2023, 8, 27))

    def test_archive_mask_summary_distinguishes_zero_from_unavailable(self):
        regions = [{"properties": {"fragment_id": region, "partition": "train",
                                   "source_dataset": source, "crop": "carrot"}}
                   for region, source in (("r1", "francecrops"), ("r2", "eurocrops"))]
        quality = [
            {"region_id": "r1", "pre_mask_count": None, "cloud_shadow_count": None},
            {"region_id": "r2", "pre_mask_count": 6, "cloud_shadow_count": 0},
        ]
        results = {row["source"]: row for row in error_analysis.archive_mask_coverage(regions, quality)}
        self.assertIsNone(results["francecrops"]["observed_cloud_or_shadow_rejections"])
        self.assertEqual(results["francecrops"]["region_bins_without_pre_mask_counts"], 1)
        self.assertEqual(results["eurocrops"]["observed_cloud_or_shadow_rejections"], 0)
        self.assertEqual(results["eurocrops"]["region_bins_without_pre_mask_counts"], 0)
