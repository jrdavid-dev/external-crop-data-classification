import unittest

import numpy as np

from crop_pipeline.features import annual
from crop_pipeline.training import development_views
from experiments.classifiers import combined


def season(shift_bins, n=1, amplitude=0.25):
    """Reflectance whose NDVI has one growing season peaking ``shift_bins`` half-months after bin 8."""
    t = np.arange(24)
    nir = 0.12 + amplitude * np.exp(-0.5 * ((t - 8 - shift_bins + 12) % 24 - 12) ** 2 / 4)
    s2 = np.zeros((n, 24, 5))
    s2[..., 0], s2[..., 1], s2[..., 2], s2[..., 3], s2[..., 4] = .06, .05, .1, nir, .15
    return s2, np.ones((n, 24), bool)


class CombinedTests(unittest.TestCase):
    def test_detector_finds_the_season_and_its_timing(self):
        s2, valid = season(0)
        timing = combined.greenup_timing(s2, valid)
        self.assertEqual(timing["detected"][0], 1)
        self.assertGreater(timing["amplitude"][0], .15)
        self.assertGreater(timing["onset_to_peak_days"][0], 0)
        self.assertLess(timing["onset"][0], 8.5 / 24)  # onset before the peak bin

    def test_referenced_phases_do_not_depend_on_the_calendar_month(self):
        first, valid = season(0)
        later, _ = season(6)
        X, _, columns = annual.s2_harmonic(np.concatenate([first, later]), np.concatenate([valid, valid]))
        reference = combined.harmonic_reference(np.concatenate([first, later]), np.concatenate([valid, valid]),
                                                {"onset": np.array([.2, .2 + 6 / 24])})
        moved = combined.rereference(X, columns, reference)
        phases = [j for j, c in enumerate(columns) if "_phase" in c and c.split("_")[0] in ("B8", "NDVI", "NDRE", "SAVI")]
        self.assertGreater(np.abs(X[0, phases] - X[1, phases]).max(), .5)        # calendar phases differ
        np.testing.assert_allclose(moved[0, phases], moved[1, phases], atol=1e-6)  # referenced phases match

    def test_pixels_without_a_detected_cycle_keep_full_coverage(self):
        s2, valid = season(0, amplitude=0.0)  # flat: no cycle
        timing = combined.greenup_timing(s2, valid)
        self.assertEqual(timing["detected"][0], 0)
        reference = combined.harmonic_reference(s2, valid, timing)
        self.assertTrue(np.isfinite(reference).all())
        recipes = combined.greenup_recipes(s2, valid, np.full((1, 24, 2), -15.0), timing)
        for X, ok, columns in recipes.values():
            self.assertEqual(X.shape[1], len(columns))

    def test_classifier_families_fit_with_weights_and_tuning_prefers_the_first_on_ties(self):
        rng = np.random.default_rng(0)
        crops = development_views.CROPS
        rows, X = [], []
        for parcel in range(60):
            crop = crops[parcel % 3]
            for _ in range(3):
                rows.append({"partition_group": f"p{parcel}", "crop": crop, "partition": "train",
                             "source_dataset": "s", "label_status": "reported", "site_cell": f"{parcel % 10},0"})
                X.append(np.r_[np.eye(3)[crops.index(crop)], rng.normal(size=2)])
        X = np.asarray(X)
        weights = np.ones(len(rows))
        for family in combined.families().values():
            estimator = next(iter(family.values()))
            model = estimator.fit(X, [r["crop"] for r in rows], sample_weight=weights)
            self.assertEqual(model.predict_proba(X).shape, (len(rows), 3))
        settings = {"a": combined.WeightedLogistic(C=1.0), "b": combined.WeightedLogistic(C=1.0)}
        best, scores = combined.tune(X, rows, settings)
        self.assertEqual(best, "a")
        self.assertEqual(scores["a"], 1.0)


if __name__ == "__main__":
    unittest.main()
