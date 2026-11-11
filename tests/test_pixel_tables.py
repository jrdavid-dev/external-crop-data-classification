"""Validate full-year selected-pixel archives and their window-reader adapter."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from crop_pipeline.common import interval_bins
from crop_pipeline.imagery import batch_export, pixel_tables, sampling_mask, quality
from test_daily_sampling_mask import mask_plan


class PixelTableTests(unittest.TestCase):
    def fixture(self, folder):
        plan = mask_plan(folder)
        plan["download_intervals"] = interval_bins(2020)
        rows = []
        selection = sampling_mask.load_selection(plan)
        for i, interval in enumerate(plan["download_intervals"]):
            for identifier in selection["selected_ids"]:
                row, column = divmod(identifier, plan["settings"]["grid"]["width"])
                missing = i == 1
                rows.append({"region_id": plan["fragment_id"], "interval": interval["name"],
                    "x": column + .5, "y": row + .5,
                    **{b: -9999 if missing else float(1000 + i) for b in quality.STUDY_BANDS},
                    "clear_count": 0 if missing else 2})
        retriever = SimpleNamespace(imagery_folder=folder / "imagery")
        return retriever, plan, pd.DataFrame(rows)

    def test_annual_roundtrip_preserves_selection_values_and_missing_periods(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan, frame = self.fixture(Path(temporary))
            pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, frame, [plan["fragment_id"]])
            root = retriever.imagery_folder / "pixel_tables"
            self.assertTrue(pixel_tables.reusable(root, plan, deep=True))
            path, receipt = pixel_tables.paths(root, plan)
            restored = pixel_tables.plan_from_receipt(receipt)
            self.assertEqual(pixel_tables.paths(root, restored), (path, receipt))
            q, p = pixel_tables.audit_region(root, restored)
            self.assertEqual(len(q), 24)
            self.assertEqual(q[0]["valid_fraction"], 1)
            self.assertEqual(q[1]["valid_fraction"], 0)
            self.assertEqual(p[0]["B3"], .1)
            self.assertEqual(sum(r["valid"] for r in p), 23 * plan["fragment"]["properties"]["selected_pixels"])
            self.assertTrue(all(r["B3"] is None for r in p if r["name"] == q[1]["name"]))
            self.assertEqual(pq.read_table(path).schema.field("B3").type.to_pandas_dtype(), np.float32)
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            self.assertFalse(pixel_tables.reusable(root, plan))

    def test_missing_duplicate_and_unselected_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan, frame = self.fixture(Path(temporary))
            for bad in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])]:
                with self.assertRaises(ValueError):
                    pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, bad, [plan["fragment_id"]])
            bad = frame.copy()
            bad.loc[0, "x"] = 1000.5
            with self.assertRaises(ValueError):
                pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, bad, [plan["fragment_id"]])
            self.assertFalse(list((retriever.imagery_folder / "pixel_tables").glob("**/*.parquet")))

    def test_table_window_adapter_matches_raster_auditor(self):
        import rasterio
        from affine import Affine
        from crop_pipeline.imagery.retrieval import RESAMPLING_POLICY
        from crop_pipeline.common import save_json
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            retriever, plan, frame = self.fixture(folder)
            plan["settings"].update(start="2020-01-01", end="2021-01-01",
                                    cloud_probability_threshold=40, resampling=RESAMPLING_POLICY)
            plan["fragment"]["properties"]["fragment_id"] = plan["fragment_id"]
            grid = plan["settings"]["grid"]
            progress = folder / "progress"
            manifest = {"fragment_id": plan["fragment_id"], "settings": plan["settings"], "files": {}}
            for interval in plan["download_intervals"]:
                group = frame[frame["interval"] == interval["name"]]
                values = np.full((7, grid["height"], grid["width"]), -9999, dtype="float32")
                columns = np.floor(group["x"]).astype(int).to_numpy()
                rows = np.floor(group["y"]).astype(int).to_numpy()
                values[:, rows, columns] = group[quality.STUDY_BANDS + ["clear_count"]].to_numpy().T
                path = folder / f"{interval['name']}.tif"
                with rasterio.open(path, "w", driver="GTiff", width=grid["width"], height=grid["height"],
                                   count=7, dtype="float32", crs=grid["crs"], transform=Affine(*grid["transform"]), nodata=-9999) as output:
                    output.write(values)
                    output.descriptions = tuple(quality.STUDY_BANDS + ["clear_count"])
                manifest["files"][interval["name"]] = {"status": "complete", "path": str(path), "name": interval["name"]}
            save_json(progress / "region_request.json", manifest)
            pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, frame, [plan["fragment_id"]])
            tq, tp = pixel_tables.audit_region(retriever.imagery_folder / "pixel_tables", plan)
            rq, rp = quality.quality_check([plan["fragment"]], progress, folder, folder, folder / "audit", year=2020, required_request_version=4)
            self.assertEqual([{k: v for k, v in row.items() if k not in {"image_path", "clear_pixel_dates", "acquisition_dates", "unmatched_granules", *batch_export.OBSERVATION_COUNTS}} for row in tq],
                             [{k: v for k, v in row.items() if k != "image_path"} for row in rq])
            rp = [{k: (None if v == "" else v) for k, v in row.items()} for row in rp]
            self.assertEqual(tp, rp)

    def test_selected_exports_group_tiles_and_do_not_resubmit_running_work(self):
        from unittest.mock import Mock, patch
        from crop_pipeline.imagery import batch_export
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            retriever, plan, frame = self.fixture(folder)
            plans = []
            for index in range(100):
                item = copy.deepcopy(plan)
                item["fragment_id"] = f"tile{index}"
                plans.append(item)
            ledger = folder / "ledger.json"
            with patch.object(batch_export, "ensure_folder"), \
                 patch.object(pixel_tables, "pending", return_value=[(p, p["download_intervals"]) for p in plans]), \
                 patch.object(batch_export, "composite_collection") as build, \
                 patch.object(batch_export, "start_export", side_effect=["one", "two"]) as start:
                result = batch_export.submit_composites(Mock(), retriever, plans, ledger, "projects/test/assets/temp", 2020,
                                                        pixels_per_task=6000, selected_only=True)
                again = batch_export.submit_composites(Mock(), retriever, plans, ledger, "projects/test/assets/temp", 2020,
                                                       pixels_per_task=6000, selected_only=True)
            self.assertEqual(result["tasks_submitted"], 2)
            self.assertEqual(again["tasks_submitted"], 0)
            self.assertEqual(start.call_count, 2)
            entries = batch_export.load_ledger(ledger)
            self.assertEqual([len(e["request_fingerprints"]) for e in entries], [62, 38])
            self.assertTrue(all(e["kind"] == "pixel_table" for e in entries))
            self.assertTrue(all(c.kwargs["selected_only"] for c in build.call_args_list))

    def test_preprocessing_preview_reads_only_chosen_period(self):
        from crop_pipeline.visualization.preprocessing import load_image
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan, frame = self.fixture(Path(temporary))
            pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, frame, [plan["fragment_id"]])
            path, _ = pixel_tables.paths(retriever.imagery_folder / "pixel_tables", plan)
            image = load_image(path, plan["fragment"], plan["download_intervals"][0]["name"])
            self.assertEqual(int(image["valid"].sum()), 4)
            self.assertEqual(int(image["selected"].sum()), 4)
            self.assertTrue(np.isnan(image["values"][:, ~image["selected"]]).all())
            cloudy = load_image(path, plan["fragment"], plan["download_intervals"][1]["name"])
            self.assertEqual(int(cloudy["valid"].sum()), 0)

    def test_csv_transport_preserves_float_values_for_plain_and_zipped_responses(self):
        from io import BytesIO
        from zipfile import ZipFile
        from unittest.mock import Mock, patch
        from crop_pipeline.imagery import batch_export
        raw = b'region_id,interval,x,y,B3,B4,B5,B8,B11,B2,clear_count\ntile,2020_01_01_15,0.5,1.5,1234.1234130859375,1,2,3,4,5,2\n'
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("table.csv", raw)
        for content in (raw, buffer.getvalue()):
            response = Mock(content=content)
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with patch("requests.get", return_value=response):
                table = batch_export.read_pixel_table_csv(Mock(), "asset")
            self.assertEqual(table.iloc[0]["B3"], 1234.1234130859375)
            self.assertEqual(table.iloc[0]["region_id"], "tile")
