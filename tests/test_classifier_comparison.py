import unittest
from unittest.mock import patch

import numpy as np

from crop_pipeline.training import classifier_comparison as comparison


def xgboost_available():
    """XGBoost installed and loadable (on macOS it also needs the OpenMP runtime)."""
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


class ClassifierComparisonTests(unittest.TestCase):
    def rows(self):
        return [
            {"partition_group": f"{crop}-{parcel}", "crop": crop,
             "label_status": "reported", "band": (position * 10) + parcel + pixel / 10}
            for position, crop in enumerate(comparison.CROPS)
            for parcel in range(4)
            for pixel in range(2)
        ]

    def test_calibration_leaves_whole_parcels_out(self):
        rows = self.rows()
        seen = []
        for fitting, held in comparison.parcel_calibration_folds(rows):
            fitting_ids = {rows[i]["partition_group"] for i in fitting}
            held_ids = {rows[i]["partition_group"] for i in held}
            self.assertFalse(fitting_ids & held_ids)
            self.assertEqual({rows[i]["crop"] for i in held}, set(comparison.CROPS))
            seen.extend(held)
        self.assertEqual(sorted(seen), list(range(len(rows))))

    def test_weighted_models_offer_compatible_class_probabilities(self):
        rows = self.rows()
        for kind, settings in (("logistic_regression", {"C": 1.0}), ("rbf_svm", {"C": 1.0, "gamma": 0.5})):
            with self.subTest(kind=kind):
                model = comparison.fit_classifier(kind, settings, rows, ["band"])
                probabilities = model.predict_proba(np.array([[0.0], [10.0], [20.0]]))
                self.assertEqual(set(model.classes_), set(comparison.CROPS))
                self.assertEqual(probabilities.shape, (3, 3))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
                self.assertTrue(np.all(probabilities >= 0))


    @unittest.skipUnless(xgboost_available(), "optional XGBoost dependency or its OpenMP runtime is missing")
    def test_xgboost_crop_names_follow_probability_columns(self):
        rows = self.rows()
        model = comparison.fit_classifier(
            "xgboost", {"max_depth": 2, "learning_rate": 0.2, "n_estimators": 30}, rows, ["band"]
        )
        probabilities = model.predict_proba(np.array([[0.0], [10.0], [20.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(tuple(model.classes_[probabilities.argmax(axis=1)]), tuple(comparison.CROPS))


    def test_shape_svm_uses_dimension_adjusted_gamma_on_same_folds(self):
        fit, held = self.rows()[:12], self.rows()[12:]
        columns = comparison.source_holdout.experiments.feature_sets()["interval_and_shape"]
        predictions = [{"crop": crop, "prediction": crop} for crop in comparison.CROPS]
        with (
            patch.object(comparison, "fit_classifier") as fit_classifier,
            patch.object(comparison.source_holdout, "parcel_metrics", return_value=({}, predictions)),
        ):
            results = comparison.candidate_scores("rbf_svm", columns, [(fit, held)])
        self.assertEqual(len(results), 9)
        self.assertEqual({row["settings"]["C"] for row in results}, {0.1, 1.0, 10.0})
        np.testing.assert_allclose(
            sorted({row["settings"]["gamma"] for row in results}), [0.3 / 120, 1 / 120, 3 / 120]
        )
        for call in fit_classifier.call_args_list:
            self.assertIs(call.args[2], fit)
            self.assertEqual(len(call.args[3]), 120)
