"""Test the notebook 07 analysis helpers in crop_pipeline.transferability on small synthetic inputs."""

import unittest

import numpy as np
import pandas as pd

from crop_pipeline.transferability import atok as atok_data
from crop_pipeline.transferability import climate, external, maps, models, report
from crop_pipeline.transferability.config import CLASSES, MIN_CLEAR, TARGET_SELECTIVE_ACCURACY


def one_hot(labels, confidence=1.0):
    proba = np.full((len(labels), len(CLASSES)), (1 - confidence) / (len(CLASSES) - 1))
    for i, label in enumerate(labels):
        proba[i, CLASSES.index(label)] = confidence
    return proba


class ModelMetricsTest(unittest.TestCase):
    def test_diagnostics_reports_per_crop_accuracy(self):
        truth = ["carrot", "carrot", "cabbage", "potato"]
        result = models.diagnostics(truth, one_hot(["carrot", "cabbage", "cabbage", "potato"]))
        self.assertAlmostEqual(result["OA"], 0.75)
        self.assertAlmostEqual(result["PA_carrot"], 0.5)
        self.assertAlmostEqual(result["UA_cabbage"], 0.5)
        self.assertAlmostEqual(result["balanced_acc"], (0.5 + 1 + 1) / 3)

    def test_baselines_use_the_majority_and_class_shares(self):
        majority, expected = models.baseline_rows(["carrot"] * 3 + ["potato"])
        self.assertEqual(majority["model"], "Baseline: always carrot")
        self.assertAlmostEqual(majority["OA"], 0.75)
        self.assertAlmostEqual(expected["OA"], 0.75 ** 2 + 0.25 ** 2)

    def test_threshold_is_the_lowest_that_reaches_the_target(self):
        truth = np.array(["carrot"] * 10)
        confident = one_hot(["carrot"] * 5, 0.9)
        unsure = one_hot(["potato"] * 5, 0.4)  # wrong and unsure
        proba = np.vstack([confident, unsure])
        threshold, reached = models.choose_threshold(proba, truth)
        self.assertTrue(reached)
        coverage, accuracy = models.selective(proba, truth, threshold)
        self.assertEqual(coverage, 0.5)
        self.assertGreaterEqual(accuracy, TARGET_SELECTIVE_ACCURACY)

    def test_identity_calibration_keeps_predictions(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        calibrated = models.apply_calibrator(models.fit_calibrator(proba, ["carrot", "potato"], "none"), proba)
        np.testing.assert_allclose(calibrated.sum(1), 1)
        np.testing.assert_array_equal(calibrated.argmax(1), proba.argmax(1))

    def test_calibration_error_is_zero_when_confidence_matches_accuracy(self):
        truth = ["carrot"] * 4
        self.assertAlmostEqual(models.expected_calibration_error(truth, one_hot(truth)), 0.0)

    def test_jeffries_matusita_separates_distant_classes(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (200, 3))
        self.assertLess(models.jeffries_matusita(a, rng.normal(0, 1, (200, 3))), 0.2)
        self.assertGreater(models.jeffries_matusita(a, rng.normal(10, 1, (200, 3))), 1.9)


class CalendarTest(unittest.TestCase):
    def test_calendar_vector_wraps_the_year(self):
        vector = external.calendar_vector(11, 2)  # Nov-Feb
        self.assertEqual(vector.sum(), 8)
        self.assertTrue(vector[[0, 1, 2, 3, 20, 21, 22, 23]].all())

    def test_window_coverage_counts_eight_half_months(self):
        hits = external.coverage_from_starts(pd.to_datetime(["2020-03-01", "2020-03-20"]))
        self.assertEqual(hits[4], 0.5)  # early March: first window only
        self.assertEqual(hits[5], 1.0)
        self.assertEqual(hits[12], 0.5)  # early July: second window only


class ClimateTest(unittest.TestCase):
    def test_southern_profiles_shift_six_months(self):
        profile = pd.Series(np.arange(24, dtype=float), index=climate.TMAX + climate.TMIN)
        aligned = climate.season_aligned(profile, latitude=-20)
        self.assertEqual(aligned["tmax_07"], profile["tmax_01"])
        self.assertIs(climate.season_aligned(profile, latitude=20), profile)


class AtokFeaturesTest(unittest.TestCase):
    def test_window_features_interpolate_cloudy_half_months(self):
        cube = np.full((10, 5, 1, 1), 0.2)
        cube[3, :, 0, 0] = np.nan  # one cloudy half-month inside the window
        grid = atok_data.AtokGrid(cube=cube, composites=10, crs=None, transform=None, height=1, width=1, roi_grid=None,
                                  roi_mask=np.ones((1, 1), bool), polygons=pd.DataFrame(), interiors={}, field_area_ha={},
                                  lon=0.0, lat=0.0)
        features, keep, counts = atok_data.window_features(grid, np.array([0]), np.array([0]), 0)
        self.assertTrue(keep.all())
        self.assertEqual(int(counts[0]), 7)
        self.assertAlmostEqual(features.loc[0, "bin_04_B8"], 0.2)
        self.assertAlmostEqual(features.loc[0, "bin_04_NDVI"], 0.0)

    def test_window_features_drop_windows_below_the_clear_minimum(self):
        cube = np.full((8, 5, 1, 1), np.nan)
        cube[:MIN_CLEAR - 1] = 0.3
        grid = atok_data.AtokGrid(cube=cube, composites=8, crs=None, transform=None, height=1, width=1, roi_grid=None,
                                  roi_mask=np.ones((1, 1), bool), polygons=pd.DataFrame(), interiors={}, field_area_ha={},
                                  lon=0.0, lat=0.0)
        features, keep, _ = atok_data.window_features(grid, np.array([0]), np.array([0]), 0)
        self.assertFalse(keep.any())
        self.assertEqual(len(features), 0)


class ReportTest(unittest.TestCase):
    def test_dataset_specification_fills_the_settings(self):
        spec = report.dataset_specification().set_index("item")["specification"]
        self.assertIn(f">= {MIN_CLEAR} of 8 clear", spec["Clear-data rule"])
        self.assertIn(f"{TARGET_SELECTIVE_ACCURACY:.0%} selective accuracy", spec["Unknown (maps)"])
        self.assertNotIn("{", "".join(spec))

    def test_rejection_causes_partition_the_mapped_pixels(self):
        codes = np.array([[0, 1], [2, np.nan]], dtype="float32")
        low = np.array([[False, True], [True, False]])
        ood = np.array([[False, False], [True, False]])
        raw = {("model", 2024, month): (codes, None, None, low, ood) for month in maps.MAP_WINDOWS}
        causes, _ = maps.rejection_causes(raw, {"model": ("slug", None, 0.6)}, 2024)
        shares = causes[["classified", "uncertain only", "out of distribution only", "both"]]
        np.testing.assert_allclose(shares.sum(axis=1), 1)
        self.assertEqual(causes["mapped_pixels"].iloc[0], 3)


if __name__ == "__main__":
    unittest.main()
