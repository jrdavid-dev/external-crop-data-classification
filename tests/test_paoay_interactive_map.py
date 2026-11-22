import unittest

from affine import Affine

from crop_pipeline.paoay import interactive_map


class InteractiveMapTests(unittest.TestCase):
    def test_polygon_outline_and_hole_are_kept_on_pixel_grid(self):
        polygon = {"type": "Polygon", "coordinates": [
            [[120, 16], [120.03, 16], [120.03, 16.03], [120, 16.03], [120, 16]],
            [[120.01, 16.01], [120.02, 16.01], [120.02, 16.02], [120.01, 16.01]],
        ]}
        grid = Affine(.01, 0, 120, 0, -.01, 16.03)
        polygons = interactive_map._pixel_polygons(polygon, "EPSG:4326", ~grid)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0]), 2)
        self.assertEqual(polygons[0][0][0], [0, 3])
        self.assertEqual(polygons[0][1][0], [1, 2])


if __name__ == "__main__":
    unittest.main()
