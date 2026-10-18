import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.imagery import quality as workflow


def curve(values, source="observed_daily", dates=None):
    dates = dates or [date(2020, 1, 1) + timedelta(days=5*i) for i in range(len(values))]
    return [{"region_id": "region", "partition_group": "ES|2020|1", "date": day.isoformat(),
             "start": (day-timedelta(days=7)).isoformat(), "end": (day+timedelta(days=8)).isoformat(),
             "NDVI": value, "valid_fraction": 1.0, "source_kind": source}
            for day, value in zip(dates, values)]


class DetectorTests(unittest.TestCase):
    def test_observed_supported_growth(self):
        results = workflow.detect_greenup(curve([.1,.1,.1,.15,.3,.5,.7,.7,.6,.4,.2]))
        self.assertTrue(any(row["status"] == "candidate" for row in results))
        self.assertFalse(any(row.get("validated") for row in results))

    def test_long_gap_rejected(self):
        days = [0,5,10,60,65,70,75,80,85,90,95]
        dates = [date(2020,1,1)+timedelta(days=day) for day in days]
        results = workflow.detect_greenup(curve([.1,.1,.1,.3,.4,.5,.7,.7,.6,.4,.2], dates=dates))
        self.assertTrue(any(row["reason"] == "gap_hides_onset" for row in results))

    def test_composites_stay_provisional(self):
        results = workflow.detect_greenup(curve([.1,.1,.1,.15,.3,.5,.7,.7,.6,.4,.2], source="semi_monthly"))
        self.assertTrue(any(row["status"] == "provisional" for row in results))
        self.assertFalse(any(row["status"] == "candidate" for row in results))

    def test_flat_signal_rejected(self):
        results = workflow.detect_greenup(curve([.2]*12))
        self.assertTrue(all(row["status"] == "rejected" for row in results))

    def test_weak_coverage_rejected(self):
        rows = curve([.1,.1,.1,.15,.3,.5,.7,.7,.6,.4,.2])
        for row in rows: row["valid_fraction"] = .5
        results = workflow.detect_greenup(rows)
        self.assertEqual(results[0]["reason"], "insufficient_supported_observations")

    def test_multiple_cycles_flagged(self):
        values = [.1,.1,.1,.3,.5,.7,.7,.5,.2,.1,.1,.1,.3,.5,.7,.7,.5,.2]
        results = workflow.detect_greenup(curve(values))
        self.assertTrue(any(row["status"] == "ambiguous" for row in results))


class QualityTests(unittest.TestCase):
    def test_portable_paths_scaling_counts_and_missing_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); region_id = "region"; request_id = "request"
            fields = base / "fields"; mask_folder = fields / "balanced_pilot" / "sampling_masks"
            mask_folder.mkdir(parents=True)
            grid = Affine(10,0,100,0,-10,200)
            with rasterio.open(mask_folder / "region.tif", "w", driver="GTiff", width=2,height=1,
                               count=1,dtype="uint8",crs="EPSG:32631",transform=grid) as out:
                out.write(np.array([[1,1]],dtype="uint8"),1)
            image_folder = base / "images" / region_id / request_id
            image_folder.mkdir(parents=True)
            interval = workflow.interval_bins()[0]
            image_path = image_folder / (interval["name"]+".tif")
            values = np.full((7,1,2),1000,dtype="float32")
            values[-1,0]=[2,0]; values[:-1,0,1]=-9999
            with rasterio.open(image_path,"w",driver="GTiff",width=2,height=1,count=7,
                               dtype="float32",crs="EPSG:32631",transform=grid,nodata=-9999) as out:
                out.write(values)
                for i,name in enumerate(workflow.STUDY_BANDS+["clear_count"],1): out.set_band_description(i,name)
            progress = base / "progress"; progress.mkdir()
            record = {**interval,"path":"Z:/old/composite.tif","status":"complete"}
            manifest={"fragment_id":region_id,"settings":{"version":3,"cloud_probability_threshold":40,
                "start":"2020-01-01","end":"2021-01-01"},"files":{interval["name"]:record}}
            (progress / "region_request.json").write_text(json.dumps(manifest))
            region={"properties":{"fragment_id":region_id,"crop":"carrot","partition_group":"ES|2020|1",
                "selected_pixels":2,"sampling_grid":{"transform":list(grid)[:6]},"sampling_mask":"Z:/old/mask.tif"}}
            quality,pixels=workflow.quality_check([region],progress,base/"images",fields,base/"output")
            self.assertEqual(len(quality),24)
            self.assertEqual(quality[0]["status"],"low_coverage")
            self.assertEqual(quality[0]["valid_fraction"],.5)
            self.assertAlmostEqual(pixels[0]["B3"],.1)
            self.assertEqual(pixels[0]["clear_count"],2)
            self.assertEqual(pixels[1]["B3"],"")
            self.assertEqual(sum(row["status"]=="missing" for row in quality),23)


if __name__ == "__main__":
    unittest.main()
