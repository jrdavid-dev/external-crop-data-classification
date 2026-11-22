"""Streaming window handoffs: bounded-memory reads must not change exported features."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline.common import iter_json_array, iter_json_object, json_scalars, save_json
from crop_pipeline.features import experiments
from crop_pipeline.windows import external, seasons


def in_memory_export(path):
    """The previous whole-file algorithm, kept as the reference for equivalence."""
    data = json.loads(Path(path).read_text())
    indexed = {}
    for name in ("fixed", "aligned"):
        indexed[name] = {}
        for row in data[name]:
            indexed[name].setdefault(external.pixel_key(row), row)
        indexed[name], _ = external.exclude_shared_development_locations(indexed[name])
    common = indexed["fixed"].keys() & indexed["aligned"].keys()
    return {f"{name}_{cohort}": [experiments.enhance_window(row, experiments.FEATURE_POLICY)
                                 for row in external.capped(rows if cohort == "full"
                                                            else {k: rows[k] for k in common})]
            for cohort in ("full", "matched") for name, rows in indexed.items()}


def window_rows():
    rows = []
    sources = {"train": "sen4agrinet", "validation": "francecrops", "test": "cropharvest"}
    for partition, source in sources.items():
        for crop in ("carrot", "cabbage", "potato"):
            for parcel in range(2):
                group = f"{partition}-{crop}-{parcel}"
                # 30 pixels per parcel exceeds the 20-pixel cap.
                for pixel in range(30):
                    rows.append({"partition_group": group, "field_id": group, "region_id": group,
                                 "season_key": group, "crop": crop, "partition": partition,
                                 "source_dataset": source, "label_status": "reported",
                                 "source_crs": "EPSG:32631", "x": 10.0 * pixel + 5, "y": 1000.0 * len(rows) // 30,
                                 "window_start": "2020-05-01", "window_end": "2020-09-01",
                                 **{column: .2 + pixel / 100 for column in seasons.feature_columns()}})
    return rows


class StreamingJsonTests(unittest.TestCase):
    def test_reader_matches_json_across_chunk_boundaries_and_layouts(self):
        payload = {"fixed": [{"i": i, "s": 'a"},]' * i, "n": [1.5, None, {"x": []}]} for i in range(40)],
                   "quality": "q", "aligned": [], "fixed_start_month": 12345}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            for indent in (None, 2):
                path.write_text(json.dumps(payload, indent=indent))
                for chunk in (1, 5, 4096):
                    items = list(iter_json_object(path, arrays=("fixed", "aligned"), chunk_size=chunk))
                    self.assertEqual([v for k, v in items if k == "fixed"], payload["fixed"])
                    self.assertEqual({k: v for k, v in items if k not in ("fixed", "aligned")},
                                     {"quality": "q", "fixed_start_month": 12345})
                self.assertEqual(list(iter_json_array(path, "fixed")), payload["fixed"])
                self.assertEqual(json_scalars(path, ("fixed", "aligned"))["fixed_start_month"], 12345)

    def test_truncated_file_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            path.write_text('{"fixed": [{"a": 1}, {"a": 2')
            with self.assertRaises(ValueError):
                list(iter_json_object(path, arrays=("fixed",)))


class StreamingExportTests(unittest.TestCase):
    def test_two_pass_export_matches_whole_file_export(self):
        rows = window_rows()
        duplicate = {**rows[0], "bin_01_B3": 9.0}  # Later duplicate pixel keys stay ignored.
        shared = {**rows[200], "partition_group": rows[0]["partition_group"] + "-other",
                  "x": rows[0]["x"], "y": rows[0]["y"]}
        aligned = rows[::2] + [shared]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            save_json(path, {"fixed": rows + [duplicate], "aligned": aligned,
                             "quality": "fixture", "fixed_start_month": 5})
            expected = in_memory_export(path)
            with patch("crop_pipeline.windows.external.json.loads",
                       side_effect=AssertionError("window handoff loaded whole")):
                external.export_features(path)
            development = json.loads(path.with_name("features_development.json").read_text())
            test = json.loads(path.with_name("features_test.json").read_text())
            self.assertEqual(development["fixed_start_month"], 5)
            for cohort, expected_rows in expected.items():
                self.assertEqual(development["rows"][cohort] + test["rows"][cohort],
                                 [r for r in expected_rows if r["partition"] != "test"]
                                 + [r for r in expected_rows if r["partition"] == "test"])
                self.assertTrue(expected_rows)
            self.assertLessEqual(max(sum(r["partition_group"] == g for r in expected["fixed_full"])
                                     for g in {r["partition_group"] for r in rows}), 20)


if __name__ == "__main__":
    unittest.main()
