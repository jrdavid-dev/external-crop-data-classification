"""Test valid-observation, missing-interval and usable-coverage statistics on a tiny cube."""

import unittest

import numpy as np

from crop_pipeline.imagery import observation_quality as oq


def tiny_cube():
    """Two years x 24 half-months, 2 bands, 2 x 3 pixels, all valid unless changed."""
    cube = np.ones((48, 2, 2, 3), dtype=np.float32)
    cube[0] = np.nan                    # missing interval: no composite in Jan H1, year 1
    cube[5:9, 0, 0, 0] = np.nan         # pixel (0, 0): 4 cloudy half-months in a row, one band only
    cube[:, :, 1, 2] = np.nan           # pixel (1, 2): never observed (outside the footprint)
    return cube


class ObservationQualityTest(unittest.TestCase):
    def setUp(self):
        self.cube = tiny_cube()
        self.mask = np.ones((2, 3), bool)

    def test_slot_validity_separates_missing_intervals(self):
        slots = oq.slot_validity(self.cube, self.mask, 2019)
        self.assertEqual(len(slots), 48)
        self.assertEqual(slots["missing_interval"].tolist(), [True] + [False] * 47)
        self.assertEqual(slots.loc[1, ["year", "month", "half"]].tolist(), [2019, 1, 2])
        self.assertAlmostEqual(slots.loc[5, "valid_share"], 4 / 6)   # cloudy pixel + never observed
        self.assertAlmostEqual(slots.loc[5, "data_share"], 5 / 6)    # one band still has data

    def test_yearly_quality_excludes_footprint_gap_from_cloud_loss(self):
        quality = oq.yearly_quality(self.cube, self.mask, 2019)
        first = quality.loc[2019]
        self.assertEqual(first["missing_intervals"], 1)
        # 5 observed pixels x 24 slots = 120; invalid: 5 in the missing slot + 4 cloudy
        self.assertAlmostEqual(first["valid_share"], (120 - 9) / 120)
        # cloud loss counts only the 23 slots with a composite: 4 of 115
        self.assertAlmostEqual(first["cloud_related_loss"], 4 / 115)
        self.assertAlmostEqual(first["valid_observations_per_pixel"], (120 - 9) / 5)
        self.assertEqual(first["median_longest_gap_half_months"], 1)
        self.assertEqual(quality.loc[2020, "cloud_related_loss"], 0)
        self.assertEqual(list(oq.yearly_quality(self.cube, self.mask, 2019, years=[2020]).index), [2020])

    def test_never_observed_and_longest_run(self):
        self.assertEqual(np.argwhere(oq.never_observed(self.cube)).tolist(), [[1, 2]])
        valid = np.array([[1, 0], [0, 0], [0, 0], [1, 1], [0, 1]], bool)
        self.assertEqual(oq.longest_invalid_run(valid).tolist(), [2, 3])

    def test_usable_coverage_counts_pixels_with_enough_valid_half_months(self):
        windows = {3: "Mar-Jun", 12: "Dec-Mar"}
        coverage = oq.usable_coverage(self.cube, self.mask, 2019, windows, [2019, 2020], min_valid=6)
        # Mar-Jun 2019 = slots 4-11; pixel (0, 0) has 4 of 8 valid, pixel (1, 2) none
        self.assertAlmostEqual(coverage.loc[2019, "Mar-Jun"], 4 / 6)
        self.assertAlmostEqual(coverage.loc[2019, "Dec-Mar"], 5 / 6)
        self.assertTrue(np.isnan(coverage.loc[2020, "Dec-Mar"]))      # runs past the cube
        strict = oq.usable_coverage(self.cube, self.mask, 2019, {3: "Mar-Jun"}, [2019], min_valid=4)
        self.assertAlmostEqual(strict.loc[2019, "Mar-Jun"], 5 / 6)

    def test_validity_grid_blanks_missing_intervals(self):
        grid = oq.validity_grid(oq.slot_validity(self.cube, self.mask, 2019), years=[2019])
        self.assertEqual(grid.shape, (1, 24))
        self.assertTrue(np.isnan(grid.loc[2019, 0]))
        self.assertAlmostEqual(grid.loc[2019, 1], 5 / 6)


if __name__ == "__main__":
    unittest.main()
