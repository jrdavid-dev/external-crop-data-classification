import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, LineString, Polygon, box, shape

from crop_pipeline.sources import annual
from crop_pipeline.imagery.retrieval import SatelliteRetriever
from test_annual_archive import reference_feature


class AnnualGeometryTests(unittest.TestCase):
    def test_real_collapsed_and_self_intersecting_outlines_are_valid_for_retrieval(self):
        fixtures = json.loads((Path(__file__).parent/'fixtures/invalid_annual_tile_outlines.json').read_text())
        for fragment in fixtures:
            original = shape(fragment['geometry'])
            self.assertFalse(original.is_valid)
            repaired = annual.valid_request_geometry(original)
            self.assertTrue(repaired.is_valid)
            self.assertEqual(repaired.bounds, original.bounds)
            fragment['geometry'] = repaired.__geo_interface__
            retriever = object.__new__(SatelliteRetriever)
            retriever.buffer_metres, retriever.output_resolution = 20, 10
            grid = retriever.make_download_grid(fragment)
            self.assertGreater(grid['width'] * grid['height'], 0)

    def test_request_hull_does_not_fill_pixels_outside_original_parcel(self):
        parcel = box(500000, 5200000, 500030, 5200030)
        # A touching line must not add eligible cells or become an invalid request component.
        clipped = GeometryCollection([parcel, LineString([(500030, 5200030), (500070, 5200070)])])
        feature = reference_feature()
        with tempfile.TemporaryDirectory() as directory:
            region = annual.save_tile(feature, clipped, box(499900, 5199900, 500100, 5200100), Path(directory), 0)
            self.assertTrue(shape(region['geometry']).is_valid)
            grid = region['properties']['sampling_grid']
            expected = rasterize([(clipped.__geo_interface__, 1)], out_shape=(grid['height'],grid['width']),
                                 transform=Affine(*grid['transform']), dtype='uint8')
            with rasterio.open(region['properties']['sampling_mask']) as dataset:
                actual = dataset.read(1)
            np.testing.assert_array_equal(actual, expected)

    def test_valid_parcels_are_unchanged_and_zero_area_outlines_are_skipped(self):
        polygon = box(0, 0, 1, 1)
        self.assertIs(annual.valid_request_geometry(polygon), polygon)
        self.assertIsNone(annual.valid_request_geometry(LineString([(0,0),(1,1)])))
        self.assertTrue(annual.valid_request_geometry(Polygon([(0,0),(1,1),(1,0),(0,1),(0,0)])).is_valid)
