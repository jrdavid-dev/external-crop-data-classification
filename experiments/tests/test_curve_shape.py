"""Curve-shape descriptors on the annual series: the study's formulas, gap filling and shared coverage."""

import unittest

import numpy as np

from crop_pipeline.features import annual
from experiments.curve_shape import shape
from crop_pipeline.features.experiments import temporal_shape


def season(n=1):
    t = np.arange(24)
    nir = 0.12 + 0.25 * np.exp(-0.5 * (t - 10) ** 2 / 4)
    s2 = np.zeros((n, 24, 5))
    s2[..., 0], s2[..., 1], s2[..., 2], s2[..., 3], s2[..., 4] = .06, .05, .1, nir, .15
    return s2, np.ones((n, 24), bool)


class ShapeFeatureTests(unittest.TestCase):
    def test_matches_the_study_formulas_on_a_complete_year(self):
        s2, valid = season()
        X, ok, columns = shape.shape_features(s2, valid)
        self.assertTrue(ok[0])
        self.assertEqual(len(columns), 4 * 28)
        ndvi = annual.indices(s2)[0, :, 0]
        expected = temporal_shape(ndvi, shape.bin_positions(), "NDVI")
        for name in ("greenup_slope", "senescence_slope", "peak_interval", "amplitude", "integral"):
            self.assertAlmostEqual(X[0, columns.index(f"annual_shape_NDVI_{name}")], expected[f"shape_NDVI_{name}"])
        self.assertEqual(X[0, columns.index("annual_shape_NDVI_peak_interval")], 11)  # peak at bin index 10

    def test_gaps_are_interpolated_and_coverage_follows_the_annual_rule(self):
        s2, valid = season(3)
        valid[1, 3:7] = False             # four missing half-months: filled, still usable
        valid[2, :13] = False             # eleven clear half-months: fails the annual rule
        X, ok, _ = shape.shape_features(s2, valid)
        _, annual_ok, _ = annual.s2_harmonic(s2, valid)
        self.assertEqual(ok.tolist(), annual_ok.tolist())
        self.assertEqual(ok.tolist(), [True, True, False])
        self.assertTrue(np.isfinite(X[1]).all())
        filled = shape.fill_gaps(np.arange(24, dtype=float)[None], np.r_[[True] * 3, [False] * 4, [True] * 17][None])
        np.testing.assert_allclose(filled[0, 3:7], [3, 4, 5, 6])

    def test_recipes_share_pixels_and_add_up(self):
        s2, valid = season(2)
        s1 = np.full((2, 24, 2), -15.0)
        s1[..., 1] -= 5
        sizes = {name: build(s2, valid, s1)[0].shape[1] for name, build in shape.RECIPES.items()}
        a = annual.RECIPES["annual_s2_s1"](s2, valid, s1)[0].shape[1]
        self.assertEqual(sizes["annual_shape"], 112)
        self.assertEqual(sizes["annual_s2_s1_shape"], a + 112)
        self.assertEqual(sizes["annual_shape_s1"], 112 + a - 63)  # radar terms only


if __name__ == "__main__":
    unittest.main()
