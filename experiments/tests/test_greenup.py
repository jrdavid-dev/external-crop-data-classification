"""Green-up recipes: detector timing on annual series and phases measured from green-up."""

import unittest

import numpy as np

from crop_pipeline.features import annual
from experiments.greenup import greenup


def season(shift_bins, n=1, amplitude=0.25):
    """Reflectance whose NDVI has one season peaking ``shift_bins`` half-months after bin 8."""
    t = np.arange(24)
    nir = 0.12 + amplitude * np.exp(-0.5 * ((t - 8 - shift_bins + 12) % 24 - 12) ** 2 / 4)
    s2 = np.zeros((n, 24, 5))
    s2[..., 0], s2[..., 1], s2[..., 2], s2[..., 3], s2[..., 4] = .06, .05, .1, nir, .15
    return s2, np.ones((n, 24), bool)


class GreenupTests(unittest.TestCase):
    def test_detector_finds_one_season_with_onset_before_peak(self):
        s2, valid = season(0)
        cycles = greenup.timing(s2, valid)
        self.assertEqual(cycles["detected"][0], 1)
        self.assertGreaterEqual(cycles["cycles"][0], 1)
        self.assertGreater(cycles["amplitude"][0], greenup.DETECTOR["minimum_amplitude"])
        self.assertGreater(cycles["onset_to_peak_days"][0], 0)
        self.assertLess(cycles["onset"][0], 8.5 / 24)

    def test_flat_series_has_no_cycle_but_keeps_a_reference(self):
        s2, valid = season(0, amplitude=0.0)
        cycles = greenup.timing(s2, valid)
        self.assertEqual(cycles["detected"][0], 0)
        self.assertTrue(np.isfinite(greenup.reference_time(s2, valid, cycles)).all())
        self.assertEqual(greenup.timing_matrix(cycles).tolist(), [[0, 0, 0, 0, 0, 0]])

    def test_phases_from_greenup_do_not_depend_on_the_calendar_month(self):
        first, valid = season(0)
        later, _ = season(6)
        s2, both = np.concatenate([first, later]), np.concatenate([valid, valid])
        X, _, columns = annual.s2_harmonic(s2, both)
        moved = greenup.rereference(X, columns, np.array([.2, .2 + 6 / 24]))
        phases = [j for j, c in enumerate(columns) if "_phase" in c and c.split("_")[0] in ("B8", "NDVI", "NDRE", "SAVI")]
        self.assertGreater(np.abs(X[0, phases] - X[1, phases]).max(), .5)
        np.testing.assert_allclose(moved[0, phases], moved[1, phases], atol=1e-6)

    def test_detected_onset_moves_with_the_season(self):
        first, valid = season(0)
        later, _ = season(6)
        onsets = greenup.timing(np.concatenate([first, later]), np.concatenate([valid, valid]))["onset"]
        self.assertAlmostEqual((onsets[1] - onsets[0]) * 365, 6 * 365 / 24, delta=16)


if __name__ == "__main__":
    unittest.main()
