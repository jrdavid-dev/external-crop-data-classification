"""Test the overfitting-remedy comparison: augmentation, leakage-free fitting and caching."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crop_pipeline.training import overfitting, transfer_demo
from transfer_fixtures import cached_v1_folder


def external_data():
    with tempfile.TemporaryDirectory() as folder:
        return transfer_demo.load_external(cached_v1_folder(folder))


class AugmentationTest(unittest.TestCase):
    def setUp(self):
        self.rows = external_data().head(12).reset_index(drop=True)

    def test_copies_keep_labels_and_parcels_and_recompute_indices(self):
        out = overfitting.augment(self.rows, ("noise",), np.random.default_rng(0), copies=2)
        self.assertEqual(len(out), 36)
        self.assertEqual(out["crop"].tolist(), self.rows["crop"].tolist() * 3)
        self.assertEqual(out["partition_group"].tolist(), self.rows["partition_group"].tolist() * 3)
        copy = out.iloc[12:24]
        self.assertFalse(np.allclose(copy["bin_01_B8"].to_numpy(), self.rows["bin_01_B8"].to_numpy()))
        b4, b8 = copy["bin_03_B4"].to_numpy(), copy["bin_03_B8"].to_numpy()
        np.testing.assert_allclose(copy["bin_03_NDVI"], (b8 - b4) / (b8 + b4))
        np.testing.assert_allclose(copy["bin_03_SAVI"], 1.5 * (b8 - b4) / (b8 + b4 + 0.5))
        pd.testing.assert_frame_equal(out.iloc[:12].reset_index(drop=True), self.rows)

    def test_shift_moves_the_series_by_one_half_month(self):
        rows = self.rows.copy()
        for b in range(1, 9):
            rows[f"bin_{b:02d}_B3"] = float(b)
        out = overfitting.augment(rows, ("shift",), np.random.default_rng(1), copies=1).iloc[len(rows):]
        series = out[[f"bin_{b:02d}_B3" for b in range(1, 9)]].to_numpy()
        earlier = np.array([2, 3, 4, 5, 6, 7, 8, 8], float)
        later = np.array([1, 1, 2, 3, 4, 5, 6, 7], float)
        self.assertTrue(all(np.array_equal(r, earlier) or np.array_equal(r, later) for r in series))

    def test_no_augmentation_returns_rows_unchanged(self):
        self.assertIs(overfitting.augment(self.rows, (), np.random.default_rng(0)), self.rows)


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = external_data()
        cls.development = cls.data[cls.data["partition"] == "development"].reset_index(drop=True)

    def test_every_strategy_builds_and_predicts_known_crops(self):
        train = self.development.iloc[::2]
        for strategy in overfitting.STRATEGIES:
            with self.subTest(strategy.name):
                estimator = overfitting.fit(strategy, train, trees=10)
                predictions = overfitting.predict(estimator, self.development)
                self.assertTrue(set(predictions) <= set(overfitting.CLASSES))

    def test_reduction_is_fitted_on_training_rows_only(self):
        strategy = overfitting.Strategy("RF, PCA", "feature selection", reduce="pca:3")
        train = self.development.iloc[:40]
        estimator = overfitting.fit(strategy, train, trees=10)
        scaler = estimator.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, train[overfitting.interval_columns()].to_numpy().mean(axis=0))

    def test_evaluate_reports_gap_and_region_scores(self):
        folds = overfitting.grouped_folds(self.development, folds=3)
        regions = overfitting.holdout_regions(self.development)
        self.assertTrue(regions)
        row = overfitting.evaluate(overfitting.Strategy("LR", "regularization", "lr", {"C": 0.1}),
                                   self.development, folds, regions)
        self.assertAlmostEqual(row["gap"], row["cv_train_macro_f1"] - row["cv_validation_macro_f1"])
        self.assertEqual({key for key in row if key.startswith("region: ")}, {f"region: {r}" for r in regions})
        for train_index, validation_index in folds:
            self.assertFalse(set(self.development.iloc[train_index]["split_unit"])
                             & set(self.development.iloc[validation_index]["split_unit"]))

    def test_cached_comparison_scores_test_once_for_prototype_and_cv_best(self):
        strategies = [overfitting.PROTOTYPE, overfitting.Strategy("LR", "regularization", "lr", {"C": 1.0})]
        scored = []
        original = overfitting.score_on_test

        def recording(strategy, data, seed=overfitting.SEED, trees=10):
            scored.append(strategy.name)
            return original(strategy, data, seed, trees)

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(overfitting, "STRATEGIES", strategies), \
                patch.object(overfitting, "score_on_test", recording), \
                patch.object(overfitting, "compare", wraps=lambda dev, progress=None: pd.DataFrame(
                    {"family": ["baseline", "regularization"], "cv_validation_macro_f1": [0.4, 0.6]},
                    index=pd.Index([s.name for s in strategies], name="strategy"))):
            comparison, test = overfitting.cached_comparison(folder, self.data)
            self.assertEqual(scored, [overfitting.PROTOTYPE.name, "LR"])
            self.assertEqual(test["role"].tolist(), ["prespecified prototype", "CV-best remedy"])
            self.assertTrue((Path(folder) / "strategy_comparison.csv").is_file())
            again, _ = overfitting.cached_comparison(folder, self.data)
            self.assertEqual(len(scored), 2)          # reused, not rerun
            self.assertEqual(list(again.index), list(comparison.index))

    def test_plot_comparison_renders(self):
        import matplotlib

        matplotlib.use("Agg")
        frame = pd.DataFrame({"family": ["baseline", "regularization"], "cv_train_macro_f1": [0.9, 0.7],
                              "cv_validation_macro_f1": [0.5, 0.55], "cv_validation_sd": [0.1, 0.1],
                              "gap": [0.4, 0.15], "region_balanced_accuracy": [0.3, 0.4]},
                             index=pd.Index([overfitting.PROTOTYPE.name, "LR"], name="strategy"))
        overfitting.plot_comparison(frame)


if __name__ == "__main__":
    unittest.main()
