"""Check batch-export results are written exactly like the live Earth Engine code."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from crop_pipeline.imagery import batch_export
from crop_pipeline.imagery.retrieval import SatelliteRetriever


def acquisition(date, index, matched=True):
    return {"date": date, "image_index": index, "image_id": f"COPERNICUS/S2_SR_HARMONIZED/{index}",
            "cloud_probability_id": f"COPERNICUS/S2_CLOUD_PROBABILITY/{index}" if matched else None}


def plan(width=3, height=2, intervals=()):
    return {
        "fragment_id": "region",
        "fragment": {"properties": {"partition_group": "cropharvest|X|2019|a", "crop": "carrot",
                                    "selected_pixels": 4}},
        "settings": {"grid": {"crs": "EPSG:32631", "transform": [10, 0, 500000, 0, -10, 5400000],
                              "width": width, "height": height}, "version": 4},
        "acquisitions": [acquisition("2019-01-03", "B_T2"), acquisition("2019-01-03", "A_T1"),
                         acquisition("2019-01-08", "C_T1", matched=False),
                         acquisition("2019-01-20", "D_T1")],
        "download_intervals": list(intervals),
    }


class BatchExportTests(unittest.TestCase):
    def test_days_keep_matched_granules_in_index_order(self):
        self.assertEqual(batch_export.day_entries(plan()["acquisitions"]),
                         ["2019-01-03;A_T1,B_T2", "2019-01-20;D_T1"])
        self.assertEqual(batch_export.day_entries(plan()["acquisitions"], "2019-01-16", "2019-02-01"),
                         ["2019-01-20;D_T1"])

    def test_running_and_collected_work_is_not_resubmitted(self):
        ledger = [{"kind": "daily", "items": ["daily|a"], "state": "RUNNING"},
                  {"kind": "daily", "items": ["daily|b"], "state": "FAILED"},
                  {"kind": "daily", "items": ["daily|c"], "state": "SUCCEEDED", "collected": True}]
        self.assertEqual(batch_export.in_flight(ledger), {"daily|a"})

    def test_chunks_respect_the_work_limit(self):
        self.assertEqual(batch_export.chunk([3, 3, 3, 5], lambda n: n, 6), [[3, 3], [3], [5]])

    def test_daily_rows_match_the_live_format(self):
        table = pd.DataFrame([
            {"region_id": "region", "date": "2019-01-20", "NDVI_median": 0.5, "NDVI_count": 4,
             "NDRE_median": 0.25, "NDRE_count": 4, "NDMI_median": 0.1, "NDMI_count": 3},
            {"region_id": "region", "date": "2019-01-03", "NDVI_median": np.nan, "NDVI_count": 0,
             "NDRE_median": np.nan, "NDRE_count": 0, "NDMI_median": np.nan, "NDMI_count": 0},
        ])
        with tempfile.TemporaryDirectory() as folder:
            batch_export.write_daily_checkpoints({"region": plan()}, ["region"], table, folder, 0.8)
            saved = json.loads(next(Path(folder).glob("observed_checkpoints/region_*.json")).read_text())
        self.assertEqual([row["date"] for row in saved["rows"]], ["2019-01-03", "2019-01-20"])
        self.assertEqual(saved["rows"][0]["NDVI"], None)
        self.assertEqual(saved["rows"][1]["valid_pixels"], 3)
        self.assertFalse(saved["rows"][1]["eligible"])  # 3 of 4 pixels < 80%

    def test_missing_dates_are_rejected(self):
        table = pd.DataFrame([{"region_id": "region", "date": "2019-01-20"}])
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "dates"):
            batch_export.write_daily_checkpoints({"region": plan()}, ["region"], table, folder, 0.8)

    def test_composite_pixels_rebuild_the_grid_and_manifest(self):
        interval = {"name": "2019_01_16_end", "start": "2019-01-16", "end": "2019-02-01"}
        region = plan(intervals=[interval])
        rows = [{"region_id": "region", "interval": interval["name"], "x": c + 0.5, "y": r + 0.5,
                 **{band: float(10 * r + c) for band in ["B3", "B4", "B5", "B8", "B11", "B2"]},
                 "clear_count": 1.0}
                for r in range(2) for c in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            retriever = SatelliteRetriever(None, 2019, Path(folder) / "img", Path(folder) / "rep")
            batch_export.write_composites(retriever, {"region": region}, pd.DataFrame(rows))
            manifest = json.loads(next(Path(folder).glob("rep/composite_progress/*.json")).read_text())
            record = manifest["files"][interval["name"]]
            with rasterio.open(record["path"]) as raster:
                values = raster.read()
                self.assertEqual(list(raster.descriptions), retriever.output_bands)
                self.assertEqual(raster.nodata, -9999)
        self.assertEqual(values[0].tolist(), [[0, 1, 2], [10, 11, 12]])
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["source_image_ids"], ["COPERNICUS/S2_SR_HARMONIZED/D_T1"])

    def test_incomplete_composite_grid_is_rejected(self):
        interval = {"name": "2019_01_16_end", "start": "2019-01-16", "end": "2019-02-01"}
        rows = pd.DataFrame([{"region_id": "region", "interval": interval["name"], "x": 0.5, "y": 0.5,
                              "B3": 1.0, "B4": 1.0, "B5": 1.0, "B8": 1.0, "B11": 1.0, "B2": 1.0,
                              "clear_count": 1.0}])
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "fill the grid"):
            retriever = SatelliteRetriever(None, 2019, Path(folder) / "img", Path(folder) / "rep")
            batch_export.write_composites(retriever, {"region": plan(intervals=[interval])}, rows)


if __name__ == "__main__":
    unittest.main()
