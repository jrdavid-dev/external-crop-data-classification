from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.pilots import earth_engine


class S1GridTileTests(unittest.TestCase):
    def test_row_tiles_cover_the_grid_and_respect_the_limit(self):
        self.assertEqual(earth_engine.row_tiles(10, 5, 48), [(0, 10)])
        tiles = earth_engine.row_tiles(10, 5, 48, limit=48 * 5 * 4 * 3)
        self.assertEqual(tiles, [(0, 3), (3, 3), (6, 3), (9, 1)])

    def test_strips_are_reassembled_in_order(self):
        names = [f"{p}_{i:02d}" for p in earth_engine.POLARIZATIONS for i in range(24)]
        height, width = 7, 4
        requests = []

        def compute(request):
            requests.append(request)
            rows = request["grid"]["dimensions"]["height"]
            first = round((request["grid"]["affineTransform"]["translateY"] - 1000.) / -10)
            out = np.zeros((rows, width), dtype=[(n, "f4") for n in names])
            for k, n in enumerate(names):
                out[n] = (first + np.arange(rows))[:, None] * 100 + k
            if first == 0:
                out["VV_00"][0, 0] = earth_engine.NODATA
            return out

        ee = MagicMock()
        ee.data.computePixels.side_effect = compute
        grid = {"window": [0, 0, width, height], "transform": (10, 0, 500., 0, -10, 1000.), "crs": "EPSG:32651"}
        with TemporaryDirectory() as folder, patch.object(earth_engine, "s1_image", return_value="image"), \
                patch.object(earth_engine, "MAX_REQUEST_BYTES", 48 * width * 4 * 3):
            cube = earth_engine.s1_grid(ee, 2024, grid, Path(folder))
            self.assertEqual(len(requests), 3)
            self.assertEqual(cube.shape, (24, 2, height, width))
            self.assertTrue(np.isnan(cube[0, 0, 0, 0]))
            self.assertEqual(cube[5, 1, 6, 2], 6 * 100 + 24 + 5)  # VH bin 5, row 6
            self.assertTrue(np.array_equal(earth_engine.s1_grid(ee, 2024, grid, Path(folder)), cube, equal_nan=True))
            self.assertEqual(len(requests), 3)  # cached


if __name__ == "__main__":
    unittest.main()
