"""Label-free application checks for mapped models."""

import unittest

import numpy as np

from experiments.annual import application_review

GRID = {"crs": "EPSG:32651", "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 100.0), "shape": (10, 10)}


class DiagnosticsTests(unittest.TestCase):
    def test_confidence_crop_mix_and_farmland(self):
        labels = np.full((10, 10), 4, "uint8")  # Unknown
        labels[:, :5] = 3                       # potato on the left half
        labels[0, :5] = 2                       # one row of cabbage
        item = {"name": "m", "period": "2024", "labels": labels, "grid": GRID, "months": tuple(range(1, 13)),
                "window_start": None}
        everywhere = np.ones((10, 10), bool)
        farm = np.zeros((10, 10), bool)
        farm[:, :5] = True
        row = application_review.diagnostics(item, everywhere, everywhere, farm, {}, [], 0.25,
                                             {"carrot": 0.2, "cabbage": 0.4, "potato": 0.4})
        self.assertEqual(row["confident_share_of_cropland"], 0.5)
        self.assertEqual((row["cabbage_share_of_confident_cropland"], row["potato_share_of_confident_cropland"]), (0.1, 0.9))
        # |0-0.2| + |0.1-0.4| + |0.9-0.4| = 1.0, halved
        self.assertEqual(row["crop_mix_distance_to_benguet_2024"], 0.5)
        self.assertEqual(row["farmland_confident_share_minus_outside"], 1.0)
        self.assertEqual(row["outside_training_range_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
