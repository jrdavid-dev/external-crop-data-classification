import unittest

import numpy as np

from experiments.transfer import start_month_maps as maps


class StubModel:
    classes_ = np.asarray(["cabbage", "carrot", "potato"])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        self.shape = X.shape
        return np.repeat(self.probabilities[None], len(X), axis=0)


def cube(clear=True):
    values = np.full((24, 5, 2, 2), np.nan, "float32")
    values[:, :, :, :] = [[[[500.]], [[400.]], [[900.]], [[3000.]], [[1800.]]]]  # B3, B4, B5, B8, B11 (scaled)
    if not clear:
        values[10, :, 0, 0] = np.nan  # June 1-15 missing at pixel (0, 0)
    return values


GRID = {"crs": "EPSG:32651", "transform": (10, 0, 0, 0, -10, 20)}


class StartMonthMapTests(unittest.TestCase):
    def test_labels_and_windows(self):
        self.assertEqual(maps.label(5), "May–Aug")
        self.assertEqual(maps.window(2024, 9, "px")["window_end"], "2025-01-01")

    def test_complete_requires_all_eight_bins_of_the_window(self):
        values = cube(clear=False)
        self.assertFalse(maps.complete(values, 5)[0, 0])  # May–Aug includes June 1-15
        self.assertTrue(maps.complete(values, 1)[0, 0])   # Jan–Apr does not
        self.assertTrue(maps.complete(values, 5)[1, 1])

    def test_classify_uses_72_predictors_and_the_confidence_rule(self):
        pixels = [(0, 0), (1, 1)]
        sure = StubModel([.1, .1, .8])
        labels = maps.classify(sure, cube(), 2024, 5, GRID, pixels)
        self.assertEqual(sure.shape, (2, 72))
        self.assertEqual(set(labels.values()), {"potato"})
        unsure = maps.classify(StubModel([.35, .3, .35]), cube(), 2024, 5, GRID, pixels)
        self.assertEqual(set(unsure.values()), {"Unknown crop type"})

    def test_summary_and_codes(self):
        mask = np.ones((2, 2), bool)
        cropland = np.array([[True, True], [False, False]])
        labels = {(0, 0): "potato", (0, 1): "Unknown crop type", (1, 0): "cabbage"}
        row = maps.summarize(5, 2024, labels, mask, cropland, .5)
        self.assertEqual(row["cropland_confident_pixels"], 1)
        self.assertEqual(row["potato_share_of_confident"], 1.)
        self.assertEqual(row["study_window_not_cloud_free"], 1)
        codes = maps.code_raster(mask, labels)
        self.assertEqual(codes[1, 1], maps.CODES["Window not cloud-free"])
        self.assertEqual(codes[1, 0], maps.CODES["cabbage"])


if __name__ == "__main__":
    unittest.main()
