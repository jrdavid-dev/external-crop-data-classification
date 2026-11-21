import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from shapely.geometry import box, mapping

from model_aza.training import start_month_sweep as sweep
from crop_pipeline.windows import external


def region(group, crop, partition, source="sen4agrinet", cohort=None):
    properties = {"fragment_id": group, "partition_group": group, "crop": crop, "partition": partition,
                  "source_dataset": source, "source_year": 2021, "selected_pixels": 30}
    if cohort:
        properties["test_cohort"] = cohort
    return {"properties": properties, "geometry": mapping(box(2.1, 45.2, 2.2, 45.3))}


def fake_build(windows, pixels, quality, fraction, shift):
    window, = windows
    rows = [{"source_crs": "EPSG:32631", "x": 10.0 * i, "y": 5.0, "label_status": window["label_status"],
             "window_start": window["window_start"],
             **{c: .1 + i / 1000 for c in sweep.seasons.feature_columns()}} for i in range(30)]
    return rows, [], [], []


class StartMonthSweepTests(unittest.TestCase):
    def test_fixed_windows_follow_start_month_and_reject_late_starts(self):
        window, = sweep.fixed_windows([region("a", "carrot", "train")], 3)
        self.assertEqual((window["window_start"], window["window_end"]), ("2021-03-01", "2021-07-01"))
        pipeline, = external.fixed_windows([region("a", "carrot", "train")])
        same, = sweep.fixed_windows([region("a", "carrot", "train")], external.FIXED_START_MONTH)
        self.assertEqual(same, pipeline)
        with self.assertRaises(ValueError):
            next(sweep.fixed_windows([region("a", "carrot", "train")], 10))

    def test_test_regions_are_dropped_before_windows_are_built(self):
        regions = [region("a", "carrot", "train"), region("t", "carrot", "test", "cropharvest", "jecam")]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "quality.json"
            path.write_text(json.dumps({"regions": regions, "pixel_paths": {"a": "a.parquet", "t": "t.parquet"},
                                        "quality_rows": [{"region_id": "a"}, {"region_id": "t"}]}))
            with patch.object(sweep.preprocessing_workflow, "choose_input", return_value=path):
                _, inputs = sweep.load_quality(folder)
        self.assertEqual([r["properties"]["fragment_id"] for r in inputs["regions"]], ["a"])
        self.assertEqual(list(inputs["quality_by_region"]), ["a"])
        self.assertEqual(set(inputs["pixels"].paths), {"a"})

    def test_month_rows_cap_parcels_exclude_shared_locations_and_add_savi(self):
        inputs = {"regions": [region("a", "carrot", "train"), region("b", "potato", "validation")],
                  "quality_by_region": {}, "pixels": sweep.annual_archive.PixelPartitions({})}
        with patch.object(sweep.seasons, "build_features", side_effect=fake_build):
            rows, exclusions = sweep.month_rows(inputs, 5)
        # Both parcels share all 30 coordinates, so every development pixel is ambiguous.
        self.assertEqual(rows, [])
        self.assertEqual(exclusions["shared_locations"], 30)
        inputs["regions"] = [region("a", "carrot", "train")]
        with patch.object(sweep.seasons, "build_features", side_effect=fake_build):
            rows, _ = sweep.month_rows(inputs, 5)
        self.assertEqual(len(rows), 20)
        row = rows[0]
        self.assertEqual(row["site_cell"], "45,2")
        self.assertEqual(row["label_status"], "reported")
        b8, b4 = row["bin_01_B8"], row["bin_01_B4"]
        self.assertAlmostEqual(row["bin_01_SAVI"], 1.5 * (b8 - b4) / (b8 + b4 + .5))
        self.assertEqual(len(sweep.transfer.matrix(rows, "interval_72")[0]), 72)
        table = {(r["partition"], r["crop"]): r for r in sweep.coverage(rows, 5)}
        self.assertEqual((table["train", "carrot"]["parcels"], table["train", "carrot"]["pixels"]), (1, 20))
        self.assertFalse(sweep.has_all_crops(rows))


if __name__ == "__main__":
    unittest.main()
