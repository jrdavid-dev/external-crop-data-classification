import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from model_aza.improved import field_plan


def grid():
    codes = np.full((40, 40), 4, "uint8")
    codes[:20, :20] = 3
    codes[30:, 30:] = 2
    codes[0, 39] = 1
    cropland = np.ones_like(codes, bool)
    cropland[:, :5] = False
    return codes, cropland, np.full(codes.shape, .7)


class FieldPlanTests(unittest.TestCase):
    def test_points_follow_strata_spacing_and_cropland(self):
        codes, cropland, top = grid()
        areas = {"Area 1": np.zeros_like(cropland)}
        areas["Area 1"][10:15, 10:15] = True
        allocation = {"carrot": 5, "cabbage": 3, "potato": 4, "Unknown crop type": 4}
        points, sizes = field_plan.plan_points(codes, cropland, top, areas, allocation, spacing_px=3, seed=2)
        self.assertEqual(points, field_plan.plan_points(codes, cropland, top, areas, allocation, spacing_px=3, seed=2)[0])
        self.assertEqual(sizes["carrot"], 1)
        stratified = [p for p in points if p["group"] == "stratified"]
        self.assertEqual({s: sum(p["stratum"] == s for p in stratified) for s in field_plan.STRATA},
                         {"carrot": 1, "cabbage": 3, "potato": 4, "Unknown crop type": 4})
        self.assertTrue(all(cropland[p["row"], p["col"]] and field_plan.CODES[codes[p["row"], p["col"]]] == p["stratum"]
                            for p in points if p["group"] != "reference_area"))
        cells = [(p["row"], p["col"]) for p in points if p["group"] != "reference_area"]
        self.assertTrue(all((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 9 for i, a in enumerate(cells) for b in cells[i + 1:]))
        reference = [p for p in points if p["group"] == "reference_area"]
        self.assertEqual({p["reference_area"] for p in reference}, {"Area 1"})
        self.assertTrue(all(areas["Area 1"][p["row"], p["col"]] for p in reference))
        self.assertEqual(len({p["visit_id"] for p in points}), len(points))

    def test_outputs_and_accuracy_weighting(self):
        codes, cropland, top = grid()
        points, sizes = field_plan.plan_points(codes, cropland, top, None, {"potato": 2, "cabbage": 2}, spacing_px=2)
        points = field_plan.add_coordinates(points, (10, 0, 300000, 0, -10, 1830000), "EPSG:32651")
        self.assertTrue(all(16 < p["latitude"] < 17 and 120 < p["longitude"] < 122 for p in points))
        with TemporaryDirectory() as folder:
            field_plan.write_outputs(folder, points, sizes, {"seed": 1})
            with (Path(folder) / "field_sheet.csv").open() as stream:
                sheet = list(csv.DictReader(stream))
            self.assertEqual(list(sheet[0]), field_plan.SHEET_COLUMNS)
            self.assertEqual(len(json.loads((Path(folder) / "visit_points.geojson").read_text())["features"]), len(points))
            self.assertIn("<Placemark>", (Path(folder) / "visit_points.kml").read_text())
            self.assertNotIn("map_label", sheet[0])
            self.assertNotIn("map:", (Path(folder) / "visit_points.kml").read_text())
        plan = [{"visit_id": f"V{i}", "stratum": s} for i, s in enumerate(["potato", "potato", "cabbage", "cabbage", "cabbage"])]
        rows = [{"visit_id": "V0", "group": "stratified", "surface_state": "crop", "observed_crop": "Potato"},
                {"visit_id": "V1", "group": "stratified", "surface_state": "bare", "observed_crop": ""},
                {"visit_id": "V2", "group": "stratified", "surface_state": "crop", "observed_crop": "cabbage"},
                {"visit_id": "V3", "group": "reserve", "surface_state": "crop", "observed_crop": "carrot"},
                {"visit_id": "V4", "group": "stratified", "surface_state": "", "observed_crop": ""}]
        result = field_plan.estimate_accuracy(rows, {"potato": 300, "cabbage": 100, "carrot": 0, "Unknown crop type": 0}, plan)
        self.assertEqual(result["per_class"]["potato"]["user_accuracy"], .5)
        self.assertEqual(result["per_class"]["cabbage"]["points"], 1)
        self.assertAlmostEqual(result["overall_accuracy_of_crop_labels"], .75 * .5 + .25 * 1.)


if __name__ == "__main__":
    unittest.main()
