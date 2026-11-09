"""Annual TIFFs preserve raw selected values, temporal labels and coordinates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from crop_pipeline.imagery import annual_tif, pixel_tables
from crop_pipeline.windows.seasons import file_sha256
import test_pixel_tables


class AnnualTifTests(unittest.TestCase):
    def test_pack_all_periods_and_rebuild_deleted_annual_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever, plan, frame = test_pixel_tables.PixelTableTests().fixture(root)
            plan["fragment"]["properties"]["fragment_id"] = plan["fragment_id"]
            pixel_tables.write_tables(retriever, {plan["fragment_id"]: plan}, frame, [plan["fragment_id"]])
            tables = retriever.imagery_folder / "pixel_tables"
            path, receipt = pixel_tables.paths(tables, plan)
            source = {"year": 2020, "pixel_table_root": str(tables),
                      "pixel_table_index": {plan["fragment_id"]: {"path": str(receipt), "sha256": file_sha256(receipt)}}}
            output = root / "annual_tifs/sen4agrinet/2020/campaign/round_0001"
            index = annual_tif.export_source(source, output)
            annual = Path(index[plan["fragment_id"]]["path"])
            with rasterio.open(annual) as image:
                self.assertEqual(image.count, 168)
                self.assertEqual(image.descriptions[0], "2020_01_01_15__B3")
                self.assertEqual(image.descriptions[7], "2020_01_16_end__B3")
                self.assertEqual(str(image.crs), plan["settings"]["grid"]["crs"])
                values = image.read()
                for number, interval in enumerate(plan["download_intervals"]):
                    rows = frame[frame["interval"] == interval["name"]]
                    columns = np.floor(rows["x"]).astype(int).to_numpy()
                    ys = np.floor(rows["y"]).astype(int).to_numpy()
                    np.testing.assert_array_equal(values[number*7:(number+1)*7, ys, columns].T,
                                                   rows[annual_tif.BANDS].to_numpy(dtype="float32"))
                self.assertTrue(np.all(values[:, 0, 1] == -9999))
            before = file_sha256(annual)
            annual.unlink()
            annual_tif.export_source(source, output)
            self.assertEqual(file_sha256(annual), before)
            self.assertEqual(len(list(output.glob("*.tif"))), 1)
            source["annual_tif_index"] = index
            table_quality, table_pixels = pixel_tables.audit_region(tables, plan)
            annual_quality, annual_pixels = annual_tif.audit_region(source, plan["fragment"])
            self.assertEqual(annual_pixels, table_pixels)
            self.assertEqual([{k:v for k,v in r.items() if k != "image_path"} for r in annual_quality],
                             [{k:v for k,v in r.items() if k != "image_path"} for r in table_quality])
            from crop_pipeline.visualization.preprocessing import load_image
            preview = load_image(annual, plan["fragment"], "2020_01_16_end")
            self.assertEqual(preview["values"].shape[0], 7)
            self.assertEqual(int(preview["valid"].sum()), 0)
            path.unlink()  # Window analysis does not require the transport cache.
            self.assertEqual(annual_tif.audit_region(source, plan["fragment"])[1], table_pixels)
