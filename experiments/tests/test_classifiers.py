"""Classifier-family comparison (experiment) and recipe registration of the green-up and curve-shape experiments."""

import unittest

import numpy as np

from experiments.annual import features as annual
from experiments.annual import workflow as annual_workflow, development_views
from experiments.classifiers import classifiers

CROPS = development_views.CROPS


def development(parcels=60, pixels=3, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for parcel in range(parcels):
        crop = CROPS[parcel % 3]
        for _ in range(pixels):
            rows.append({"partition_group": f"p{parcel}", "crop": crop,
                         "partition": "validation" if parcel % 5 == 0 else "train",
                         "source_dataset": ("a", "b", "c")[(parcel // 3) % 3], "label_status": "reported",
                         "site_cell": f"{40 + parcel % 10},{parcel % 7}"})
            X.append(np.r_[np.eye(3)[CROPS.index(crop)], rng.normal(size=2)])
    return np.asarray(X), rows


class ClassifierFamilyTests(unittest.TestCase):
    def test_both_gradient_boosting_backends_fit_with_weights_and_two_crops(self):
        X, rows = development()
        y = [r["crop"] for r in rows]
        backends = ["sklearn_hist_gradient_boosting"] + (["xgboost"] if classifiers.xgboost_available() else [])
        for backend in backends:
            model = classifiers.GradientBoosting(max_depth=2, n_estimators=20, backend=backend)
            model.fit(X, y, sample_weight=np.ones(len(y)))
            self.assertEqual((model.backend_, set(model.classes_)), (backend, set(CROPS)))
            self.assertEqual(model.predict_proba(X).shape, (len(y), 3))
            two = [i for i, crop in enumerate(y) if crop != "cabbage"]  # dataset holdout without FranceCrops
            model = classifiers.GradientBoosting(max_depth=2, n_estimators=20, backend=backend)
            model.fit(X[two], [y[i] for i in two], sample_weight=np.ones(len(two)))
            self.assertEqual(model.predict_proba(X).shape, (len(y), 2))

    def test_comparison_tunes_each_family_and_ties_go_to_the_forest(self):
        X, rows = development()
        from sklearn.ensemble import RandomForestClassifier

        small = {"random_forest": {"original": RandomForestClassifier(n_estimators=10, random_state=42)},
                 "logistic_regression": {"C=1": classifiers.WeightedLogistic(C=1.0)}}
        tuning, results = classifiers.compare(X, rows, small, progress=None)
        self.assertEqual(set(tuning), set(small))
        self.assertTrue(all(r["selection_score"] is not None for r in results.values()))
        tied = {"random_forest": {"selection_score": .8}, "logistic_regression": {"selection_score": .8}}
        self.assertEqual(classifiers.best_family(tied), "random_forest")
        tied["logistic_regression"]["selection_score"] = .81
        self.assertEqual(classifiers.best_family(tied), "logistic_regression")

    def test_estimator_lookup_gives_freezing_forests_full_size(self):
        model = classifiers.estimator("random_forest", "original", final=True)
        self.assertEqual(model.n_estimators, development_views.FOREST["n_estimators"])
        self.assertEqual(classifiers.estimator("logistic_regression", "C=0.1").C, 0.1)


class RegistrationTests(unittest.TestCase):
    def test_experiment_recipes_register_but_stay_out_of_selection_defaults(self):
        from experiments.curve_shape import shape
        from experiments.greenup import greenup

        for recipes in (greenup.RECIPES, shape.RECIPES):
            self.assertTrue(set(recipes) <= set(annual_workflow.RECIPES))
            self.assertFalse(set(recipes) & set(annual_workflow.DEFAULT_RECIPES))
        self.assertEqual(annual_workflow.DEFAULT_RECIPES, tuple(annual.RECIPES))


if __name__ == "__main__":
    unittest.main()
