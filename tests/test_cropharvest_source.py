"""Check CropHarvest parcels use the FranceCrops interior-sampling rules."""

import tempfile
import unittest
from collections import Counter

from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry

from crop_pipeline.sources import cropharvest
from crop_pipeline.sources import francecrops as france


def utm_polygon(crs, left, bottom, size):
    projector = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transform_geometry(projector.transform, box(left, bottom, left + size, bottom + size))


class CropHarvestSourceTests(unittest.TestCase):
    def test_southern_parcels_use_southern_utm_zones(self):
        self.assertEqual(cropharvest.utm_crs(47.1, -19.6), "EPSG:32738")
        self.assertEqual(cropharvest.utm_crs(2.3, 48.8), "EPSG:32631")

    def test_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-polygon"):
            cropharvest.parcel_feature("kenya:1", Point(34.3, 0.7), "cabbage", 2019,
                                       "Kenya", "kenya", "r")

    def test_inward_buffer_keeps_crop_and_drops_tiny_fields(self):
        large = cropharvest.parcel_feature(
            "jecam:1", utm_polygon("EPSG:32738", 500000, 7800000, 100), "carrot", 2019,
            "Madagascar", "jecam", "r")
        tiny = cropharvest.parcel_feature(
            "jecam:2", utm_polygon("EPSG:32738", 501000, 7800000, 15), "carrot", 2019,
            "Madagascar", "jecam", "r")
        catalog = france.build_parcel_catalog([large, tiny])
        capacity = {row["parcel_id"]: row["interior_pixels"]
                    for row in cropharvest.interior_capacity([large, tiny], catalog)}
        self.assertEqual(capacity, {"jecam:1": 64, "jecam:2": 0})
        self.assertTrue(all(row["crop"] == "carrot" for row in catalog))

    def test_candidates_are_capped_per_field_and_by_budget(self):
        features = [
            cropharvest.parcel_feature(
                f"src:{n}", utm_polygon("EPSG:32631", 500000 + n * 1000, 5400000, 200),
                "potato", 2019, "France", "src", "r")
            for n in range(5)
        ]
        catalog = france.build_parcel_catalog(features)
        with tempfile.TemporaryDirectory() as folder:
            regions, summary, budgets = cropharvest.sample_year_candidates(
                catalog, folder, budget_caps={"potato": 250})
        self.assertEqual(budgets, {"potato": 250})
        self.assertEqual(summary["candidate_pixels_per_crop"], {"potato": 250})
        per_group = Counter(r["properties"]["partition_group"] for r in regions)
        self.assertTrue(all(count == 1 for count in per_group.values()))
        self.assertTrue(all(r["properties"]["selected_pixels"] <= 100 for r in regions))


if __name__ == "__main__":
    unittest.main()
