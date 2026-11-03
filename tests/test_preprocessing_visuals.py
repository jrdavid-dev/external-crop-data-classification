"""Protect honest saved-data examples and consistent image/mask alignment."""

import tempfile
import csv
import json
from unittest.mock import patch
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import rasterio
from rasterio.transform import from_origin

from crop_pipeline.visualization import preprocessing as visuals


class PreprocessingVisualTests(unittest.TestCase):
    def test_example_selection_is_deterministic_and_uses_retained_counts(self):
        windows, rows = [], []
        for crop in visuals.CROPS:
            for group, count in [("small", 1), ("middle", 3), ("large", 5)]:
                windows.append(
                    {"crop": crop, "partition_group": group, "region_id": group}
                )
                rows.extend(
                    {"crop": crop, "partition_group": group} for _ in range(count)
                )
        first = visuals.choose_examples(windows, rows)
        reversed_result = visuals.choose_examples(
            list(reversed(windows)), list(reversed(rows))
        )
        self.assertEqual(first, reversed_result)
        self.assertTrue(
            all(window["region_id"] == "middle" for window in first.values())
        )

    def test_package_reads_existing_saved_artifact_names_without_moving_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis/run"
            seasons = analysis / "seasons"
            (seasons / "estimated_features").mkdir(parents=True)
            (analysis / "timelines").mkdir()
            (root / "source_handoffs").mkdir()
            windows = [
                {
                    "crop": crop,
                    "region_id": crop,
                    "partition_group": crop,
                    "window_start": "2020-04-01",
                    "window_end": "2020-08-01",
                }
                for crop in visuals.CROPS
            ]
            (seasons / "estimated_windows.json").write_text(json.dumps(windows))
            with (seasons / "estimated_features/complete_features.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["crop", "partition_group"])
                writer.writeheader()
                writer.writerows(
                    {"crop": crop, "partition_group": crop} for crop in visuals.CROPS
                )
            observations = [
                {"region_id": crop, "date": "2020-04-02"} for crop in visuals.CROPS
            ]
            (analysis / "timelines/observed_timelines.json").write_text(
                json.dumps(observations)
            )
            for source, crops in [
                ("sen4agrinet", ["carrot", "potato"]),
                ("francecrops", ["cabbage"]),
            ]:
                fields = root / source
                fields.mkdir()
                (fields / "sampled_download_regions.geojson").write_text(
                    json.dumps(
                        {
                            "features": [
                                {"properties": {"fragment_id": crop}} for crop in crops
                            ]
                        }
                    )
                )
                (root / "source_handoffs" / f"{source}_run.json").write_text(
                    json.dumps({"source_name": source, "field_folder": str(fields)})
                )
                quality = analysis / "quality" / source
                quality.mkdir(parents=True)
                with (quality / "composite_quality.csv").open(
                    "w", newline=""
                ) as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["region_id", "name", "image_path"]
                    )
                    writer.writeheader()
                    writer.writerows(
                        {
                            "region_id": crop,
                            "name": interval["name"],
                            "image_path": "saved.tif",
                        }
                        for crop in crops
                        for interval in visuals.season_workflow.window_intervals(
                            windows[0]
                        )
                    )
            with (
                patch.object(visuals, "load_image", return_value={}),
                patch.object(visuals, "load_reference_shapes", return_value=[]),
            ):
                contexts = visuals.load_context(root / "inventory", "run")
            self.assertEqual(set(contexts), set(visuals.CROPS))
            self.assertEqual(contexts["carrot"]["observations"], [observations[0]])
            self.assertEqual(len(contexts["cabbage"]["images"]), 8)

    def test_saved_nodata_is_excluded_and_grid_mismatch_stops_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            transform = from_origin(100, 200, 10, 10)
            raster_settings = {
                "driver": "GTiff",
                "width": 2,
                "height": 2,
                "crs": "EPSG:32631",
                "transform": transform,
            }
            image_path, mask_path = folder / "image.tif", folder / "mask.tif"
            values = np.full((7, 2, 2), 1500, dtype="float32")
            values[6] = 2
            values[:, 0, 0] = -9999
            with rasterio.open(
                image_path,
                "w",
                **raster_settings,
                count=7,
                dtype="float32",
                nodata=-9999,
            ) as image:
                image.write(values)
            with rasterio.open(
                mask_path, "w", **raster_settings, count=1, dtype="uint8"
            ) as mask:
                mask.write(np.ones((1, 2, 2), dtype="uint8"))
            region = {"properties": {"sampling_mask": str(mask_path)}}
            result = visuals.load_image(image_path, region)
            self.assertEqual(int(result["valid"].sum()), 3)
            self.assertFalse(result["valid"][0, 0])
            with rasterio.open(
                mask_path,
                "w",
                **{**raster_settings, "transform": from_origin(110, 200, 10, 10)},
                count=1,
                dtype="uint8",
            ) as mask:
                mask.write(np.ones((1, 2, 2), dtype="uint8"))
            with self.assertRaisesRegex(ValueError, "grids differ"):
                visuals.load_image(image_path, region)
