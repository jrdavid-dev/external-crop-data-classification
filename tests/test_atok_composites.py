"""Test the all-Atok composite export plan, submission ledger and local file handling."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from crop_pipeline.imagery import atok_composites as ac


class Chain:
    """Stands in for any Earth Engine object: records calls and returns itself."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def __getattr__(self, attribute):
        def call(*args, **kwargs):
            self.log.append((f"{self.name}.{attribute}", args, kwargs))
            return self
        return call


class FakeTask:
    def __init__(self, number, kwargs):
        self.id, self.kwargs, self.started = f"TASK{number}", kwargs, False

    def start(self):
        self.started = True


class FakeEE:
    def __init__(self):
        self.log, self.tasks, self.states = [], [], {}
        ee = self

        class Export:
            class image:
                @staticmethod
                def toDrive(**kwargs):
                    task = FakeTask(len(ee.tasks), kwargs)
                    ee.tasks.append(task)
                    return task

        class batch:
            pass

        batch.Export = Export
        self.batch = batch

        class data:
            @staticmethod
            def getTaskStatus(task_id):
                return [{"state": ee.states.get(task_id, "RUNNING")}]

        self.data = data

        class Geometry(Chain):
            def __init__(self, *args):
                Chain.__init__(self, ee.log, "Geometry")
                ee.log.append(("Geometry", args, {}))

            @staticmethod
            def Rectangle(*args):
                ee.log.append(("Rectangle", args, {}))
                return Chain(ee.log, "Rectangle")

        self.Geometry = Geometry

    def ImageCollection(self, name):
        self.log.append(("ImageCollection", (name,), {}))
        return Chain(self.log, name)

    def Projection(self, crs):
        return crs


def write_boundary(folder):
    import geopandas as gpd
    from shapely.geometry import box

    path = Path(folder) / "boundary.geojson"
    gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=[box(258695, 1836003, 259000, 1836500),
                                                     box(259000, 1836003, 259503, 1836500)],
                     crs="EPSG:32651").to_crs(4326).to_file(path)
    return path


class HalfMonthTest(unittest.TestCase):
    def test_half_month_boundaries_and_unfinished_periods(self):
        items = ac.half_months(date(2024, 1, 1), date(2024, 3, 31), today=date(2024, 3, 20))
        self.assertEqual([(y, m, h) for y, m, h, _, _ in items],
                         [(2024, 1, 1), (2024, 1, 2), (2024, 2, 1), (2024, 2, 2), (2024, 3, 1)])
        self.assertEqual(items[1][3:], (date(2024, 1, 16), date(2024, 2, 1)))
        self.assertEqual(items[3][3:], (date(2024, 2, 16), date(2024, 3, 1)))   # leap February
        december = ac.half_months(date(2023, 12, 16), date(2023, 12, 31), today=date(2024, 6, 1))
        self.assertEqual([(y, m, h) for y, m, h, _, _ in december], [(2023, 12, 2)])
        self.assertEqual(december[0][4], date(2024, 1, 1))
        self.assertEqual(ac.file_stem(2024, 3, 2), "Sentinel2_CSPlus_2024_03_H2")


class PlanTest(unittest.TestCase):
    def test_grid_is_on_the_existing_10_m_lattice_and_covers_the_buffer(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = ac.plan_exports(write_boundary(folder), date(2024, 1, 1), date(2024, 1, 31), "f",
                                   today=date(2024, 6, 1), buffer_m=100)
        grid = plan["grid"]
        self.assertEqual((grid["transform"][2] - 258690) % 10, 0)
        self.assertEqual((grid["transform"][5] - 1841210) % 10, 0)
        minx, miny, maxx, maxy = plan["region"].bounds
        x0, y1, x1, y0 = grid["bounds"]
        self.assertTrue(x0 <= minx and y1 <= miny and x1 >= maxx and y0 >= maxy)
        self.assertEqual(grid["width"], (x1 - x0) / 10)
        self.assertEqual([item["stem"] for item in plan["items"]],
                         ["Sentinel2_CSPlus_2024_01_H1", "Sentinel2_CSPlus_2024_01_H2"])

    def test_settings_key_changes_with_buffer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_boundary(folder)
            keys = {ac.plan_exports(path, date(2024, 1, 1), date(2024, 1, 31), "f", buffer_m=b)["settings_key"]
                    for b in (100, 200)}
        self.assertEqual(len(keys), 2)


class SubmitTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.plan = ac.plan_exports(write_boundary(self.folder.name), date(2024, 1, 1), date(2024, 2, 29),
                                    "atok_test", today=date(2024, 6, 1))
        self.ledger = Path(self.folder.name) / "ledger.json"

    def tearDown(self):
        self.folder.cleanup()

    def test_exports_use_the_recipe_and_grid(self):
        ee = FakeEE()
        submitted = ac.submit(ee, self.plan, self.ledger)
        self.assertEqual(len(submitted), 4)
        task = ee.tasks[0]
        self.assertTrue(task.started)
        self.assertEqual(task.kwargs["fileNamePrefix"], "Sentinel2_CSPlus_2024_01_H1")
        self.assertEqual(task.kwargs["folder"], "atok_test")
        self.assertEqual(task.kwargs["crs"], "EPSG:32651")
        self.assertEqual(task.kwargs["crsTransform"], self.plan["grid"]["transform"])
        self.assertEqual(task.kwargs["formatOptions"]["noData"], -9999)
        opened = {args[0] for name, args, _ in ee.log if name == "ImageCollection"}
        self.assertEqual(opened, {"COPERNICUS/S2_SR_HARMONIZED", "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"})
        self.assertIn(("COPERNICUS/S2_SR_HARMONIZED.filterDate", ("2024-01-01", "2024-01-16")),
                      [(name, args) for name, args, _ in ee.log])
        self.assertIn(("COPERNICUS/S2_SR_HARMONIZED.linkCollection", ["cs_cdf"]),
                      [(name, args[1]) for name, args, _ in ee.log if name.endswith("linkCollection")])

    def test_ledger_prevents_resubmission_and_retries_failures(self):
        ee = FakeEE()
        ac.submit(ee, self.plan, self.ledger)
        self.assertEqual(ac.submit(ee, self.plan, self.ledger), [])
        ee.states = {"TASK0": "COMPLETED", "TASK1": "FAILED"}
        counts = ac.refresh(ee, self.ledger)
        self.assertEqual(counts, {"COMPLETED": 1, "FAILED": 1, "RUNNING": 2})
        self.assertEqual(ac.submit(ee, self.plan, self.ledger), ["Sentinel2_CSPlus_2024_01_H2"])

    def test_outstanding_limit(self):
        ee = FakeEE()
        self.assertEqual(len(ac.submit(ee, self.plan, self.ledger, limit=3)), 3)
        self.assertEqual(json.loads(self.ledger.read_text())[0]["state"], "READY")


class LocalFilesTest(unittest.TestCase):
    def write_tif(self, path, grid, bands=5, dtype="float32"):
        import rasterio
        from affine import Affine

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", width=grid["width"], height=grid["height"], count=bands,
                           dtype=dtype, crs="EPSG:32651", transform=Affine(*grid["transform"])) as target:
            target.write(np.ones((bands, grid["height"], grid["width"]), dtype))

    def test_check_arrange_and_present(self):
        grid = {"transform": [10, 0, 258690, 0, -10, 1841210], "width": 4, "height": 3}
        with tempfile.TemporaryDirectory() as folder:
            flat, root = Path(folder) / "drive", Path(folder) / "root"
            self.write_tif(flat / "Sentinel2_CSPlus_2024_01_H1.tif", grid)
            self.write_tif(flat / "Sentinel2_CSPlus_2024_01_H2.tif", grid, bands=4)
            (flat / "notes.txt").write_text("x")
            copied = ac.arrange_by_year(flat, root)
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in copied),
                             ["2024/Sentinel2_CSPlus_2024_01_H1.tif", "2024/Sentinel2_CSPlus_2024_01_H2.tif"])
            self.assertEqual(ac.arrange_by_year(flat, root), [])           # unchanged files skipped
            self.assertEqual(ac.check_file(root / "2024/Sentinel2_CSPlus_2024_01_H2.tif", grid), ["4 bands"])
            self.assertEqual(ac.present_files(root, grid), {"Sentinel2_CSPlus_2024_01_H1"})
            shifted = {**grid, "transform": [10, 0, 258700, 0, -10, 1841210]}
            self.assertTrue(ac.check_file(root / "2024/Sentinel2_CSPlus_2024_01_H1.tif", shifted))


if __name__ == "__main__":
    unittest.main()
