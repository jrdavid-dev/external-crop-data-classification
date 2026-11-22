import csv
import json
from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from crop_pipeline.paoay import area_review

GRID = {"crs": "EPSG:32651", "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 100.0), "shape": (10, 10)}


def write_tif(path, *bands):
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=len(bands), dtype="uint8",
                       crs=GRID["crs"], transform=rasterio.Affine(*GRID["transform"])) as out:
        for index, band in enumerate(bands, 1):
            out.write(np.asarray(band, "uint8"), index)


def item(labels, months=None, start=None, name="map"):
    return {"name": name, "labels": np.asarray(labels, "uint8"), "grid": GRID, "months": months,
            "window_start": None if start is None else np.asarray(start, "uint8"), "period": "test"}


def write_calendar(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "crop", "months", "scope", "source_excerpt"])
        writer.writeheader()
        writer.writerows(rows)


# Left half of the grid is area "West", right half "East" (pixel centres).
AREAS = [{"name": "West", "geometry": box(0, 0, 50, 100)}, {"name": "East", "geometry": box(50, 0, 100, 100)}]


class ProductLoadingTests(unittest.TestCase):
    def test_saved_maps_share_one_label_code_scheme(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            annual = folder / "annual"; annual.mkdir()
            write_tif(annual / "crop_map.tif", np.full((10, 10), 5))
            (annual / "map_summary.json").write_text(json.dumps({"year": 2024}))
            loaded = area_review.product("annual", annual)
            self.assertEqual(loaded["months"], tuple(range(1, 13)))
            self.assertTrue((loaded["labels"] == 5).all())

            fixed = folder / "fixed"; fixed.mkdir()
            classes, status = np.zeros((10, 10)), np.zeros((10, 10))
            classes[0, 0], status[0, 0] = 3, 1  # potato prediction
            classes[0, 1], status[0, 1] = 2, 2  # score rejected -> Unknown, even with a class
            status[0, 2] = 3                     # missing input
            write_tif(fixed / "crop_baseline.tif", classes, status)
            intervals = [{"start": f"2024-0{m}-01"} for m in (1, 2, 3, 4)]
            handoff = fixed / "crop_baseline_handoff.json"
            handoff.write_text(json.dumps({"period": {"start": "2024-01-01", "end": "2024-05-01",
                                                      "intervals": intervals}}))
            loaded = area_review.product("fixed", handoff)
            self.assertEqual(loaded["labels"][0, :4].tolist(), [3, 4, 5, 0])
            self.assertEqual(loaded["months"], (1, 2, 3, 4))

            bundle = folder / "aligned"; bundle.mkdir()
            codes = np.zeros((10, 10)); codes[0, :3] = [1, 5, 6]
            write_tif(bundle / "aligned_map.tif", codes, np.where(codes == 1, 3, 0))
            (bundle / "manifest.json").write_text(json.dumps({"identity": {"year": 2024}, "model_manifest": "x"}))
            loaded = area_review.product("aligned", bundle)
            self.assertEqual(loaded["labels"][0, :3].tolist(), [1, 5, 5])
            self.assertEqual(area_review.pixel_months(loaded, np.array([0, 0]), np.array([0, 1])),
                             [{3, 4, 5, 6}, set()])

    def test_aligned_bundle_is_matched_by_model_not_by_age(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for key, model in (("a", folder / "old.json"), ("b", folder / "current.json")):
                (folder / key).mkdir()
                (folder / key / "manifest.json").write_text(json.dumps({"model_manifest": str(model.resolve())}))
            self.assertEqual(area_review.aligned_bundle_for(folder, folder / "current.json"), folder / "b")
            self.assertIsNone(area_review.aligned_bundle_for(folder, folder / "other.json"))


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "calendar.csv"

    def tearDown(self):
        self.folder.cleanup()

    def entries(self, rows):
        write_calendar(self.path, rows)
        return area_review.read_calendar(self.path, {"West", "East"})

    def test_window_months_decide_consistency(self):
        entries = self.entries([
            {"name": "West", "crop": "Potato", "months": "3;4;5;6", "scope": "crop_calendar", "source_excerpt": "x"},
            {"name": "East", "crop": "cabbage", "months": "7;8", "scope": "crop_calendar", "source_excerpt": "x"},
            {"name": "East", "crop": "lettuce", "months": "1;2", "scope": "crop_calendar", "source_excerpt": "x"}])
        labels = np.full((10, 10), 3)  # potato everywhere
        masks = area_review.area_masks(AREAS, GRID)
        rows = {r["area"]: r for r in area_review.calendar_comparison(item(labels, months=(1, 2, 3, 4)), masks, entries)}
        self.assertEqual(rows["West"]["status"], "dominant_crop_in_calendar")
        self.assertEqual(rows["West"]["calendar_consistent_percent_of_confident"], 100.0)
        # Cabbage is expected only in July-August, outside a January-April map.
        self.assertEqual(rows["East"]["status"], "no_modelled_crop_expected_in_period")
        self.assertEqual(rows["East"]["other_or_context_crops"], "lettuce")
        whole_year = {r["area"]: r for r in area_review.calendar_comparison(item(labels, months=tuple(range(1, 13))),
                                                                            masks, entries)}
        self.assertEqual(whole_year["East"]["status"], "dominant_crop_not_in_calendar")
        self.assertEqual(whole_year["East"]["calendar_consistent_percent_of_confident"], 0.0)

    def test_per_pixel_windows_and_context_scopes(self):
        entries = self.entries([
            {"name": "West", "crop": "carrot", "months": "11;12;1;2", "scope": "crop_calendar", "source_excerpt": "x"},
            {"name": "East", "crop": "cabbage", "months": "1;2;3;4;5;6;7;8;9;10;11;12", "scope": "seedbed",
             "source_excerpt": "x"}])
        labels, start = np.full((10, 10), 4), np.zeros((10, 10))
        labels[:5, :5], start[:5, :5] = 1, 12   # carrot window Dec-Mar overlaps the calendar
        labels[5:, :5], start[5:, :5] = 1, 5    # carrot window May-Aug does not
        labels[:, 5:], start[:, 5:] = 2, 1      # cabbage on seedbeds never counts as consistent
        masks = area_review.area_masks(AREAS, GRID)
        rows = {r["area"]: r for r in area_review.calendar_comparison(item(labels, start=start), masks, entries)}
        self.assertEqual(rows["West"]["calendar_consistent_percent_of_confident"], 50.0)
        self.assertEqual(rows["East"]["status"], "no_modelled_crop_expected_in_period")
        self.assertEqual(rows["East"]["calendar_consistent_percent_of_confident"], 0.0)
        detail = area_review.calendar_detail(item(labels, start=start), masks, entries)
        self.assertEqual(detail[0]["calendar_months"], "JF........ND")
        self.assertEqual(detail[0]["percent_of_confident"], 100.0)

    def test_invalid_entries_are_rejected(self):
        bad = [{"name": "North", "crop": "potato", "months": "1", "scope": "crop_calendar", "source_excerpt": "x"},
               {"name": "West", "crop": "potato", "months": "13", "scope": "crop_calendar", "source_excerpt": "x"},
               {"name": "West", "crop": "potato", "months": "1", "scope": "guess", "source_excerpt": "x"},
               {"name": "West", "crop": "potato", "months": "1", "scope": "crop_calendar", "source_excerpt": ""}]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.entries([row])
        with self.assertRaises(ValueError):
            self.entries([{"name": "West", "crop": "potato", "months": "1", "scope": "crop_calendar",
                           "source_excerpt": "x"}] * 2)


class FarmlandTests(unittest.TestCase):
    def test_farmland_classes_and_label_shares(self):
        landuse = gpd.GeoDataFrame({"main_elu": ["Production Agriculture"] * 3,
                                    "sub_elu": ["Crops", "General Production Agriculture", "Pasture"]},
                                   geometry=[box(0, 0, 50, 100), box(50, 0, 100, 50), box(50, 50, 100, 100)],
                                   crs=GRID["crs"])
        farm = area_review.farmland(landuse)
        self.assertEqual(farm["sub_elu"].tolist(), ["Crops"])
        mask = area_review.polygon_mask(farm, GRID)
        self.assertEqual(int(mask.sum()), 50)
        labels = np.full((10, 10), 4); labels[:, :5] = 3
        rows = area_review.farmland_composition(item(labels), np.ones((10, 10), bool), mask)
        share = {(r["area"], r["label"]): r["percent_of_mapped"] for r in rows}
        self.assertEqual(share[("ELUM farmland", "potato")], 100.0)
        self.assertEqual(share[("Outside ELUM farmland", "Unknown crop type")], 100.0)
        by_area = area_review.landuse_by_area(landuse, AREAS)
        self.assertEqual([(r["area"], r["elu_subclass"], r["area_percent"]) for r in by_area if r["area"] == "West"],
                         [("West", "Crops", 100.0)])


if __name__ == "__main__":
    unittest.main()
