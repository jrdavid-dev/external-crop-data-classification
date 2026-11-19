"""Exploratory aligned Paoay map reuses training timing rules and keeps every exclusion visible."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyproj import Transformer
import rasterio

from crop_pipeline.paoay import aligned
from test_paoay_references import placemark, polygon, write_kmz

# NDVI per half-month for one 2024 pixel: bare until early March, rise in late March, decline after July.
GREENUP = [.15, .15, .15, .15, .15, .5, .7, .75, .8, .8, .8, .8, .75, .6, .4, .2, .15, .15, .15, .15, .15, .15, .15, .15]


def raw_bands(ndvi):
    """Raw reflectance x 10000 for B3, B4, B5, B8, B11 giving this NDVI (NDRE rises with it)."""
    red = .1
    nir = red * (1 + ndvi) / (1 - ndvi)
    return [500, red * 1e4, 1000, nir * 1e4, 2000]


def cube_for(series_by_pixel, shape=(1, 1)):
    cube = np.full((24, 5, *shape), np.nan, dtype="float32")
    for (row, column), series in series_by_pixel.items():
        for index, value in enumerate(series):
            if value is not None:
                cube[index, :, row, column] = raw_bands(value)
    return cube


class StubForest:
    classes_ = np.array(["cabbage", "carrot", "potato"])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, matrix):
        return np.repeat(self.probabilities[None, :], len(matrix), axis=0)


class AlignedPaoayTests(unittest.TestCase):
    def test_year_rule_needs_all_half_months_then_highest_clear_share(self):
        rows = [{"year": 2023, "half_months_present": 24, "mean_clear_share": .7},
                {"year": 2024, "half_months_present": 24, "mean_clear_share": .85},
                {"year": 2026, "half_months_present": 20, "mean_clear_share": .95}]
        self.assertEqual(aligned.choose_year(rows), 2024)
        with self.assertRaises(ValueError):
            aligned.choose_year([rows[2]])

    def test_training_timing_rules_set_window_from_green_up_month(self):
        cube = cube_for({(0, 0): GREENUP})
        windows, audit = aligned.qualified_windows(aligned.observations(cube, 2024, [(0, 0)]), 2024)
        self.assertEqual([w["window_start"] for w in windows], ["2024-03-01"])
        self.assertEqual(windows[0]["window_end"], "2024-07-01")
        self.assertIn("timing_passed", {row["outcome"] for row in audit})

    def test_flat_or_too_cloudy_pixels_get_no_window(self):
        cloudy = [value if index % 2 else None for index, value in enumerate(GREENUP)]
        cube = cube_for({(0, 0): [.15] * 24, (0, 1): cloudy}, shape=(1, 2))
        windows, audit = aligned.qualified_windows(aligned.observations(cube, 2024, [(0, 0), (0, 1)]), 2024)
        self.assertEqual(windows, [])
        self.assertEqual({row["region_id"] for row in audit}, {"px_0_0", "px_0_1"})

    def test_features_need_eight_clear_bins_and_match_model_columns(self):
        grid = {"crs": "EPSG:32651", "transform": (10, 0, 250000, 0, -10, 1830000)}
        window = {"region_id": "px_0_0", "window_start": "2024-03-01", "window_end": "2024-07-01"}
        features = aligned.window_features(cube_for({(0, 0): GREENUP}), window, 0, 0, grid)
        columns = [f"bin_{n:02d}_{v}" for n in range(1, 9)
                   for v in ("B3", "B4", "B5", "B8", "B11", "NDVI", "NDRE", "NDMI", "SAVI")]
        self.assertTrue(all(np.isfinite(features[c]) for c in columns))
        self.assertAlmostEqual(features["bin_02_NDVI"], .5, places=4)
        gap = list(GREENUP)
        gap[8] = None  # May 1–15 inside the March–June window
        self.assertIsNone(aligned.window_features(cube_for({(0, 0): gap}), window, 0, 0, grid))

    def test_confidence_rule_marks_weak_predictions_unknown(self):
        selected = {(0, 0): ({"window_start": "2024-03-01"}, {"f": 1.0})}
        recipe = {"feature_columns": ["f"]}
        strong = aligned.classify(StubForest([.1, .75, .15]), recipe, selected)[(0, 0)]
        weak = aligned.classify(StubForest([.42, .5, .08]), recipe, selected)[(0, 0)]
        self.assertEqual((strong["label"], strong["raw"]), ("carrot", "carrot"))
        self.assertEqual((weak["label"], weak["raw"]), ("Unknown crop type", "carrot"))

    def test_run_writes_keyed_bundle_with_every_pixel_status_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            imagery = root / "imagery"
            series = {(0, 0): GREENUP, (0, 1): [.15] * 24,
                      (1, 0): GREENUP[:8] + [None] + GREENUP[9:], (1, 1): GREENUP}
            cube = cube_for(series, shape=(2, 2))
            transform = rasterio.Affine(10, 0, 250000, 0, -10, 1830000)
            for index, path in enumerate(aligned._files(imagery, 2024)):
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=5, dtype="float32",
                                   crs="EPSG:32651", transform=transform, nodata=-9999) as raster:
                    raster.write(np.nan_to_num(cube[index], nan=-9999))
                    raster.descriptions = aligned.BANDS
            to_wgs = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True).transform
            west, south = to_wgs(250000.5, 1829980.5)
            east, north = to_wgs(250019.5, 1829999.5)
            kmz = write_kmz(root / "inputs/roi.kmz", [
                placemark("Region of Interest", geometry=polygon(west, south, east, north)),
                placemark("Area 1", "notes", geometry=polygon(west, (south + north) / 2, east, north))])
            model = root / "model"
            model.mkdir()
            for name in ("model.joblib", "recipe.json", "model_manifest.json"):
                (model / name).write_text(name)
            recipe = {"feature_columns": ["bin_01_NDVI", "bin_08_NDVI"]}
            with patch.object(aligned, "load_model", return_value=(StubForest([.1, .8, .1]), recipe)):
                manifest = aligned.run([kmz], "Region of Interest", [], imagery, model / "model_manifest.json",
                                       root / "out", years=[2024])
                again = aligned.run([kmz], "Region of Interest", [], imagery, model / "model_manifest.json",
                                    root / "out", years=[2024])
            self.assertEqual(manifest, again)
            bundle = aligned.load(manifest)
            codes = bundle["codes"].tolist()
            self.assertEqual(codes, [[1, 5], [6, 1]])  # carrot, no window, not cloud-free, carrot
            self.assertEqual(bundle["onset"][0, 0], 3)
            self.assertEqual(bundle["manifest"]["identity"]["year"], 2024)
            self.assertTrue(bundle["manifest"]["deviations_from_training"])
            whole = bundle["area_summary"][0]
            self.assertEqual((whole["area"], whole["pixels"], whole["carrot"]), ("Whole study boundary", "4", "2"))
            self.assertEqual(bundle["area_summary"][1]["pixels"], "2")
            self.assertFalse(list((root / "out").glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
