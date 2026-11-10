"""Selective crop scores retain unknown, missing and outside pixels separately."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from crop_pipeline.paoay import crop_baseline


class CropBaselineTests(unittest.TestCase):
    def test_thresholds_class_order_and_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with rasterio.open(folder/'pixel_status.tif', 'w', driver='GTiff',
                               width=8, height=1, count=1, dtype='uint8',
                               crs='EPSG:32651', transform=from_origin(0, 10, 10, 10)) as raster:
                raster.write(np.array([[1, 1, 1, 1, 1, 1, 2, 0]], dtype='uint8'), 1)
            feature = {'folder': folder, 'rows': np.zeros(6, dtype=int), 'columns': np.arange(6)}
            crop = {'top_probability': np.array([.8, .6, .59, .9, .8, .9]),
                    'probability_margin': np.array([.2, .1, .3, .09, 0, .3]),
                    'class_codes': np.array([7, 8, 7, 9, 8, 9]),
                    'manifest': {'class_codes': {'carrot': 7, 'cabbage': 8, 'potato': 9}}}
            data, roi = crop_baseline.arrays(crop, feature)
            np.testing.assert_array_equal(data[0], [[1, 2, 0, 0, 0, 3, 0, 0]])
            np.testing.assert_array_equal(data[1], [[1, 1, 2, 2, 2, 1, 3, 0]])
            boundary = np.array([[True, False, True, True, True, False, True, False]])
            clipped, clipped_roi = crop_baseline.arrays(crop, feature, spatial_mask=boundary)
            np.testing.assert_array_equal(clipped_roi, boundary)
            self.assertTrue((clipped[:, ~boundary] == 0).all())
            self.assertEqual(int((clipped[1] == 3).sum()), 1)
            self.assertEqual(int((clipped[1] == 2).sum()), 3)
            summary = crop_baseline.area_rows(data, roi, .01)
            self.assertEqual([r['pixels'] for r in summary], [1, 1, 1, 3, 1])
            self.assertEqual(sum(r['pixels'] for r in summary), int(roi.sum()))
            self.assertEqual(len(crop_baseline.DEFERRED_METHODS), 2)

    def test_invalid_rules(self):
        for rules in ({}, {**crop_baseline.RULES, 'minimum_margin': float('nan')},
                      {**crop_baseline.RULES, 'minimum_top_probability': 2}):
            with self.assertRaises(ValueError):
                crop_baseline.validate_rules(rules)
