"""Development-only transfer views: no test rows, shared folds, ensemble and summary."""

import unittest

import numpy as np

from crop_pipeline.training import development_views

CROPS = development_views.CROPS
SOURCES = ("sen4agrinet", "francecrops", "eurocrops")


def development(pixels=3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for parcel in range(90):
        crop = CROPS[parcel % 3]
        source = SOURCES[(parcel // 3) % 3]
        partition = "validation" if parcel % 5 == 0 else "train"
        for _ in range(pixels):
            rows.append({"partition_group": f"p{parcel}", "crop": crop, "partition": partition,
                         "source_dataset": source, "label_status": "reported",
                         "site_cell": f"{40 + parcel % 10},{parcel % 7}"})
            signal = np.eye(3)[CROPS.index(crop)]
            X.append(np.r_[signal + rng.normal(0, noise, 3), rng.normal(size=2)])
    return np.asarray(X), rows


PARAMS = {**development_views.SELECTION_FOREST, "n_estimators": 20}


class DevelopmentViewTests(unittest.TestCase):
    def test_separable_crops_score_high_in_every_view(self):
        X, rows = development()
        summary = development_views.summarize(development_views.view_probabilities(X, rows, PARAMS))
        self.assertEqual(summary["locked_validation_macro_f1"], 1.0)
        self.assertEqual(summary["spatial_cv_macro_f1"], 1.0)
        self.assertEqual(summary["dataset_holdout_macro_f1"], 1.0)
        self.assertEqual(summary["selection_score"], 1.0)
        self.assertEqual(set(summary["dataset_holdout"]), set(SOURCES))

    def test_spatial_folds_hold_out_each_training_parcel_once(self):
        X, rows = development()
        views = development_views.view_probabilities(X, rows, PARAMS)
        held = [g for parcels in views["spatial_cv"].values() for g in parcels]
        self.assertEqual(sorted(held), sorted({r["partition_group"] for r in rows if r["partition"] == "train"}))
        self.assertEqual(len(views["spatial_cv"]), 5)

    def test_test_rows_are_refused(self):
        X, rows = development()
        rows[0] = {**rows[0], "partition": "test"}
        with self.assertRaises(ValueError):
            development_views.view_probabilities(X, rows, PARAMS)
        rows[0] = {**rows[1], "test_cohort": "jecam"}
        with self.assertRaises(ValueError):
            development_views.view_probabilities(X, rows, PARAMS)

    def test_ensemble_averages_probabilities_on_identical_folds(self):
        X, rows = development(noise=.4)
        first = development_views.view_probabilities(X, rows, PARAMS, seed=1)
        second = development_views.view_probabilities(X[:, ::-1], rows, PARAMS, seed=2)
        combined = development_views.average(first, second)
        group = next(iter(combined["locked_validation"]["validation"]))
        expected = (first["locked_validation"]["validation"][group][1]
                    + second["locked_validation"]["validation"][group][1]) / 2
        np.testing.assert_allclose(combined["locked_validation"]["validation"][group][1], expected)
        broken = {**second, "spatial_cv": dict(list(second["spatial_cv"].items())[:-1])}
        with self.assertRaises(ValueError):
            development_views.average(first, broken)

    def test_holdout_without_a_crop_in_training_still_scores(self):
        X, rows = development()
        # Only FranceCrops supplies cabbage; holding it out scores its potato with a carrot/potato model.
        rows = [{**r, "source_dataset": "francecrops"} if r["crop"] == "cabbage"
                or (r["crop"] == "potato" and int(r["partition_group"][1:]) % 2) else
                {**r, "source_dataset": "sen4agrinet" if int(r["partition_group"][1:]) % 2 else "eurocrops"}
                for r in rows]
        summary = development_views.summarize(development_views.view_probabilities(X, rows, PARAMS))
        self.assertGreater(summary["dataset_holdout"]["francecrops"]["parcels"], 0)
        self.assertEqual(summary["dataset_holdout"]["francecrops"]["macro_f1_present_crops"], 1.0)

    def test_other_classifiers_and_view_subsets(self):
        from sklearn.linear_model import LogisticRegression

        X, rows = development()
        views = development_views.view_probabilities(X, rows, estimator=LogisticRegression(max_iter=500),
                                                     views=("spatial_cv",))
        self.assertEqual(views["locked_validation"], {})
        self.assertEqual(views["dataset_holdout"], {})
        self.assertEqual(len(views["spatial_cv"]), 5)
        pooled = {g: p for folds in views["spatial_cv"].values() for g, p in folds.items()}
        self.assertEqual(development_views.scores(pooled)["macro_f1"], 1.0)
        with self.assertRaises(ValueError):
            development_views.view_probabilities(X, rows, PARAMS, views=("test",))

    def test_missing_crop_suppresses_three_class_macro_f1(self):
        parcels = {"a": ("carrot", np.array([.9, .05, .05])), "b": ("potato", np.array([.1, .1, .8]))}
        metrics = development_views.scores(parcels)
        self.assertIsNone(metrics["macro_f1"])
        self.assertEqual(metrics["macro_f1_present_crops"], 1.0)


if __name__ == "__main__":
    unittest.main()
