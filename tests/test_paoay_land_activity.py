import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.paoay import land_activity


def reflectance(ndvi, rng):
    """B3, B4, B5, B8, B11 with the requested NDVI and small noise."""
    nir = 0.4 + rng.normal(0, 0.005, np.shape(ndvi))
    red = nir * (1 - ndvi) / (1 + ndvi)
    return np.stack([np.full_like(nir, 0.05), red, (red + nir) / 2, nir, np.full_like(nir, 0.2)])


class LandActivityTests(unittest.TestCase):
    rules = {**land_activity.RULES, 'first_year': 2019, 'last_year': 2021,
             'minimum_clear_half_months': 40, 'k_values': [2, 3, 4]}

    def test_cycle_rule_counts_two_crops_a_year(self):
        slots = np.arange(72)
        ndvi = 0.5 + 0.25 * np.sin(2 * np.pi * slots / 12)
        cycles = land_activity.cycle_bounds(ndvi, np.ones(72, bool), self.rules)
        self.assertGreaterEqual(len(cycles), 5)
        self.assertTrue(all(start < peak < end for start, peak, end in cycles))
        self.assertEqual(land_activity.cycle_bounds(np.full(72, 0.8), np.ones(72, bool), self.rules), [])

    def test_crop_forest_and_bare_pixels_get_rule_names(self):
        rng = np.random.default_rng(0)
        slots = np.arange(72)
        groups = {'active cropland': lambda: 0.45 + 0.3 * np.sin(2 * np.pi * (slots + rng.integers(12)) / 12),
                  'perennial vegetation': lambda: np.full(72, 0.82) + rng.normal(0, 0.01, 72),
                  'bare / built': lambda: np.full(72, 0.12) + rng.normal(0, 0.01, 72)}
        cube = np.full((72, 5, 6, 10), np.nan, dtype='float32')
        expected = {}
        for row, (name, series) in enumerate([*groups.items()] * 2):
            for col in range(10):
                cube[:, :, row, col] = reflectance(series(), rng).T
                expected[row, col] = name
        cube[::9, :, 0, 0] = np.nan  # occasional cloud gaps stay usable
        mask = np.ones((6, 10), bool)
        mask[5, 9] = False
        codes, summaries, table, centers = land_activity.classify(cube, mask, self.rules)
        names = {code: name for code, name in land_activity.LAND_CLASSES.items()}
        for (row, col), name in expected.items():
            if mask[row, col]:
                self.assertEqual(names[codes[row, col]], name, (row, col))
        self.assertEqual(codes[5, 9], 0)
        self.assertEqual(sum(row['chosen'] for row in table), 1)
        self.assertEqual(summaries.shape, (len(land_activity.SUMMARY_FEATURES), 6, 10))

    def test_pixels_without_enough_clear_observations_are_kept_separate(self):
        rng = np.random.default_rng(1)
        cube = np.stack([reflectance(np.full((4, 4), 0.8), rng) for _ in range(72)]).astype('float32')
        cube[:, :, 0, 1] = np.nan
        cube[:50, :, 0, 0] = np.nan
        codes, *_ = land_activity.classify(cube, np.ones((4, 4), bool), {**self.rules, 'k_values': [2]})
        self.assertEqual(codes[0, 0], 5)
        self.assertEqual(codes[0, 1], 5)

    def test_reader_aligns_to_the_crop_grid_and_masks_nodata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Affine(10, 0, 1000, 0, -10, 5000)
            data = np.arange(5 * 8 * 9, dtype='float32').reshape(5, 8, 9) + 100
            data[:, 3, 4] = -9999
            path = land_activity.composite_path(root, (2019, 1, 1))
            path.parent.mkdir(parents=True)
            with rasterio.open(path, 'w', driver='GTiff', height=8, width=9, count=5, dtype='float32',
                               crs='EPSG:32651', transform=source, nodata=-9999) as raster:
                raster.write(data)
                raster.descriptions = land_activity.BANDS
            grid = {'crs': 'EPSG:32651', 'transform': [10, 0, 1020, 0, -10, 4980], 'width': 4, 'height': 3}
            inventory = land_activity.composite_inventory(root, self.rules)
            cube = land_activity.read_cube(root, inventory, grid, self.rules)
            self.assertEqual(cube.shape, (72, 5, 3, 4))
            np.testing.assert_allclose(cube[0, :, 0, 0], data[:, 2, 2] / 10000)
            self.assertTrue(np.isnan(cube[0, :, 1, 2]).all())
            self.assertTrue(np.isnan(cube[1]).all())
            with self.assertRaisesRegex(ValueError, 'aligned'):
                land_activity.read_cube(root, inventory, {**grid, 'transform': [10, 0, 1025, 0, -10, 4980]}, self.rules)
            with self.assertRaisesRegex(ValueError, 'cover'):
                land_activity.read_cube(root, inventory, {**grid, 'width': 20}, self.rules)

    def test_seasonal_tables_count_clear_months_cycles_and_complete_windows(self):
        rules = {**self.rules, 'first_year': 2019, 'last_year': 2019}
        cube = np.ones((24, 5, 2, 2), dtype='float32')
        cube[8, :, 0, 0] = np.nan          # May H1 cloudy on one active pixel
        cube[20:, :, :, :] = np.nan        # November-December not available yet
        codes = np.array([[1, 1], [3, 0]], dtype='uint8')
        cycles = [(0, 0, 2, 6, 10), (0, 1, 2, 8, 11), (1, 0, 4, 8, 12)]  # last one is not cropland
        monthly, windows = land_activity.seasonal_tables(cube, codes, cycles, rules)
        by_month = {row['month']: row for row in monthly}
        self.assertEqual(by_month['Feb']['cycle_start_share'], 1.0)
        self.assertEqual(by_month['Apr']['cycle_peak_share'], 0.5)
        self.assertEqual(by_month['May']['clear_share_active_cropland'], 0.75)
        self.assertEqual(by_month['Nov']['composites'], 0)
        by_start = {row['start_month']: row for row in windows}
        self.assertEqual(by_start[1]['window'], 'Jan–Apr')
        self.assertEqual(by_start[1]['active_cropland_complete_share'], 1.0)
        self.assertEqual(by_start[3]['active_cropland_complete_share'], 0.5)
        self.assertEqual(by_start[9]['composites_available'], 4)
        self.assertTrue(np.isnan(by_start[9]['active_cropland_complete_share']))
        self.assertNotIn(10, by_start)  # a window running past the last year is not listed
        summary = {row['window']: row for row in land_activity.window_summary(windows)}
        self.assertEqual(summary['Mar–Jun']['best_complete_share'], 0.5)
        self.assertEqual(summary['Sep–Dec']['years_with_all_composites'], 0)
        # A composite file that is cloudy everywhere is present but not clear: the window is 0%, not missing.
        cube[6] = np.nan  # April H1, inside the March-June window
        _, windows = land_activity.seasonal_tables(cube, codes, cycles, rules, available_slots=range(20))
        by_start = {row['start_month']: row for row in windows}
        self.assertEqual(by_start[3]['composites_available'], 8)
        self.assertEqual(by_start[3]['active_cropland_complete_share'], 0.0)

    def test_crop_map_split_keeps_every_category_pixel(self):
        codes = np.array([[1, 1, 3], [4, 5, 0]], dtype='uint8')
        baseline = {'roi': np.array([[1, 1, 1], [1, 1, 0]], bool),
                    'data': (np.array([[3, 2, 3], [0, 3, 0]]), np.array([[1, 1, 1], [2, 1, 0]]))}
        rows = land_activity.crop_map_rows(codes, baseline, 0.01)
        potato = {row['land_activity']: row['pixels'] for row in rows if row['crop_map_category'] == 'Potato'}
        self.assertEqual(potato['active cropland'], 1)
        self.assertEqual(potato['perennial vegetation'], 1)
        self.assertEqual(potato['insufficient clear observations'], 1)
        unknown = [row for row in rows if row['crop_map_category'] == 'Unknown crop type' and row['pixels']]
        self.assertEqual([(row['land_activity'], row['share_of_category']) for row in unknown], [('bare / built', 1.0)])

    def test_naming_rules_and_rule_validation(self):
        self.assertEqual(land_activity.land_name({'ndvi_p90': .3, 'ndvi_p10': .1, 'cycles_per_year': 2, 'mean_cycle_amplitude': .3}), 'bare / built')
        self.assertEqual(land_activity.land_name({'ndvi_p90': .9, 'ndvi_p10': .6, 'cycles_per_year': .2, 'mean_cycle_amplitude': .1}), 'perennial vegetation')
        self.assertEqual(land_activity.land_name({'ndvi_p90': .8, 'ndvi_p10': .3, 'cycles_per_year': 1.5, 'mean_cycle_amplitude': .25}), 'active cropland')
        self.assertEqual(land_activity.land_name({'ndvi_p90': .8, 'ndvi_p10': .3, 'cycles_per_year': .5, 'mean_cycle_amplitude': .25}), 'mixed / low activity')
        with self.assertRaises(ValueError):
            land_activity.validate_rules({**land_activity.RULES, 'extra': 1})


if __name__ == '__main__':
    unittest.main()
