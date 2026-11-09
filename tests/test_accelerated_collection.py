"""Synthetic rasters and mocked services verify transport without any downloads."""

import copy
import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import requests
from affine import Affine

from crop_pipeline import progress
from crop_pipeline.common import interval_bins
from crop_pipeline.imagery import accelerated, scene_cache
from crop_pipeline.sources import annual, annual_identity
from test_annual_archive import reference_feature


def fixture(root):
    source = annual.prepare_source(
        "francecrops", [reference_feature(width=100)], root, {}, inward_buffer_metres=0
    )
    inputs = source["years"]["2020"]
    region = annual.read_regions(inputs)[0]
    retriever = accelerated.AcceleratedRetriever(
        None, 2020, inputs["imagery_folder"], inputs["report_folder"]
    )
    plan = {
        "fragment_id": region["properties"]["fragment_id"],
        "fragment": region,
        "settings": retriever.request_settings(region),
        "acquisitions": [],
    }
    return retriever, plan


def write_stack(retriever, plan, intervals, destination):
    grid = plan["settings"]["grid"]
    calendar_names = [interval["name"] for interval in interval_bins(2020)]
    values = [
        calendar_names.index(interval["name"]) * 7 + band
        for interval in intervals
        for band in range(1, 8)
    ]
    data = np.array(values, dtype="float32")[:, None, None] * np.ones(
        (len(values), grid["height"], grid["width"]), dtype="float32"
    )
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        count=len(values),
        width=grid["width"],
        height=grid["height"],
        dtype="float32",
        crs=grid["crs"],
        transform=Affine(*grid["transform"]),
    ) as raster:
        raster.write(data)


class GroupedCollectionTests(unittest.TestCase):
    def test_all_sources_use_grouped_transport_and_gate_incomplete_handoffs(self):
        for source_name in annual.SOURCE_NAMES:
            with (
                self.subTest(source=source_name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan = annual.prepare_source(
                    source_name, [reference_feature(width=100)], root, {}
                )
                backend = SimpleNamespace(Initialize=lambda **keywords: None)

                def inventory(retriever, region, refresh=False):
                    return {
                        "fragment_id": region["properties"]["fragment_id"],
                        "fragment": region,
                        "settings": retriever.request_settings(region),
                        "acquisitions": [],
                    }

                def daily(
                    plans, mask_acquisition, earth_engine, folder, minimum_fraction
                ):
                    checkpoint = Path(folder) / "observed_checkpoints/test.json"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_text("{}")

                with (
                    patch.dict("sys.modules", {"ee": backend}),
                    patch.object(
                        accelerated.AcceleratedRetriever, "inventory_region", inventory
                    ),
                    patch.object(
                        annual.timelines,
                        "retrieve_observed_timelines",
                        side_effect=daily,
                    ),
                ):
                    with patch.object(
                        accelerated, "fetch_group", side_effect=RuntimeError("offline")
                    ):
                        pending = annual.collect_source(
                            plan, root / "inventory", "test", enabled=True
                        )
                    self.assertEqual(pending["status"], "pending_retry")
                    self.assertFalse(
                        annual.registry_path(root / "inventory", source_name).exists()
                    )
                    with patch.object(
                        accelerated, "fetch_group", side_effect=write_stack
                    ) as fetch:
                        complete = annual.collect_source(
                            plan, root / "inventory", "test", enabled=True
                        )
                    self.assertEqual(complete["status"], "completed")
                    self.assertEqual(fetch.call_count, plan["estimate"]["tiles"] * 4)
                    with patch.object(
                        accelerated,
                        "fetch_group",
                        side_effect=AssertionError("must reuse"),
                    ):
                        reused = annual.collect_source(
                            plan, root / "inventory", "test", enabled=True
                        )
                    self.assertEqual(
                        reused["reused_images"], plan["estimate"]["composites"]
                    )
                    self.assertEqual(reused["downloaded_images"], 0)

    def test_group_expression_retains_interval_filters_and_explicit_band_order(self):
        class FakeImage:
            def __init__(self, bands):
                self.bands = bands

            def rename(self, names):
                return FakeImage(names)

            def addBands(self, image):
                return FakeImage(self.bands + image.bands)

        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))
            intervals = interval_bins(2020)[:2]
            plan["acquisitions"] = [
                {"date": intervals[0]["start"], "image_index": "first"},
                {"date": intervals[1]["start"], "image_index": "second"},
            ]
            with patch.object(
                retriever,
                "make_composite",
                return_value=FakeImage(retriever.output_bands),
            ) as composite:
                image = accelerated.make_group_image(retriever, plan, intervals)
            expected = [
                f"interval_{index:02d}_{band}"
                for index in range(2)
                for band in retriever.output_bands
            ]
            self.assertEqual(image.bands, expected)
            self.assertEqual(
                composite.call_args_list[0].args[2][0]["image_index"], "first"
            )
            self.assertEqual(
                composite.call_args_list[1].args[2][0]["image_index"], "second"
            )

    def test_grouped_files_match_band_order_and_retry_only_damaged_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))
            with patch.object(
                accelerated, "fetch_group", side_effect=write_stack
            ) as fetch:
                results = retriever.download_region(plan)
                self.assertEqual(fetch.call_count, 4)
                self.assertTrue(
                    all(result["status"] == "downloaded" for result in results)
                )
                folder, path, manifest = accelerated.download_checkpoint(
                    retriever, plan
                )
                for index, interval in enumerate(interval_bins(2020)):
                    with rasterio.open(folder / f"{interval['name']}.tif") as raster:
                        self.assertEqual(
                            raster.descriptions, tuple(retriever.output_bands)
                        )
                        np.testing.assert_array_equal(
                            raster.read()[:, 0, 0],
                            np.arange(index * 7 + 1, index * 7 + 8),
                        )
                        self.assertEqual(raster.nodata, -9999)
                before = path.read_bytes()
                results = retriever.download_region(plan)
                self.assertTrue(all(result["status"] == "reused" for result in results))
                self.assertEqual(fetch.call_count, 4)
                self.assertEqual(path.read_bytes(), before)
                damaged = folder / f"{results[3]['name']}.tif"
                damaged.write_bytes(b"damaged")
                results = retriever.download_region(plan)
                self.assertEqual(fetch.call_count, 5)
                self.assertEqual(len(fetch.call_args.args[2]), 1)
                self.assertEqual(
                    sum(result["status"] == "reused" for result in results), 23
                )

    def test_failed_group_does_not_stop_other_groups_or_replace_valid_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))
            first_name = interval_bins(2020)[0]["name"]

            def fail_first(retriever, plan, intervals, destination):
                if intervals[0]["name"] == first_name:
                    raise RuntimeError("simulated 503")
                write_stack(retriever, plan, intervals, destination)

            with patch.object(accelerated, "fetch_group", side_effect=fail_first):
                results = retriever.download_region(plan)
            self.assertEqual(sum(result["status"] == "failed" for result in results), 6)
            self.assertEqual(
                sum(result["status"] == "downloaded" for result in results), 18
            )
            with patch.object(
                accelerated, "fetch_group", side_effect=write_stack
            ) as fetch:
                recovered = retriever.download_region(plan)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(
                sum(result["status"] == "reused" for result in recovered), 18
            )

    def test_corrupt_group_grid_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))

            def wrong_grid(retriever, plan, intervals, destination):
                altered = copy.deepcopy(plan)
                altered["settings"]["grid"]["transform"][2] += 10
                write_stack(retriever, altered, intervals, destination)

            with patch.object(accelerated, "fetch_group", side_effect=wrong_grid):
                results = retriever.download_region(plan)
            self.assertTrue(all(result["status"] == "failed" for result in results))
            self.assertFalse(list(retriever.imagery_folder.glob("*/*/*.tif")))

    def test_group_size_respects_memory_budget_and_workers_are_bounded(self):
        groups = accelerated.request_groups(
            interval_bins(2020), {"width": 500, "height": 500}, 24
        )
        self.assertTrue(
            all(len(group) * 500 * 500 * 28 <= 24 * 1024**2 for group in groups)
        )
        with self.assertRaises(ValueError):
            accelerated.request_groups([], {"width": 10000, "height": 1}, 6)
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))
            lock = threading.Lock()
            active = 0
            maximum = 0

            def concurrent_fetch(retriever, plan, intervals, destination):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.02)
                write_stack(retriever, plan, intervals, destination)
                with lock:
                    active -= 1

            with patch.object(accelerated, "fetch_group", side_effect=concurrent_fetch):
                retriever.download_region(plan)
            self.assertEqual(maximum, 2)

    def test_http_retry_is_bounded_and_only_for_transient_errors(self):
        response = requests.Response()
        response.status_code = 503
        error = requests.HTTPError(response=response)
        image = type(
            "Image", (), {"getDownloadURL": lambda self, parameters: "unused"}
        )()
        plan = {
            "settings": {
                "grid": {
                    "crs": "EPSG:32631",
                    "transform": [10, 0, 0, 0, -10, 0],
                    "width": 2,
                    "height": 2,
                }
            }
        }
        with (
            patch.object(accelerated, "make_group_image", return_value=image),
            patch.object(accelerated.requests, "get", side_effect=error) as get,
            patch.object(accelerated.time, "sleep") as sleep,
        ):
            with self.assertRaises(requests.HTTPError):
                accelerated.fetch_group(None, plan, [], Path("unused"))
            self.assertEqual(get.call_count, 2)
            sleep.assert_called_once_with(2)
        response.status_code = 400
        self.assertFalse(accelerated.transient_error(error))

    def test_scene_metadata_is_shared_but_filtered_to_each_exact_parcel(self):
        with tempfile.TemporaryDirectory() as temporary:
            retriever, plan = fixture(Path(temporary))
            first = reference_feature("first", left=501000, width=100)
            second = reference_feature("second", left=502000, width=100)
            metadata = {
                "cloud_features": [],
                "features": [
                    {
                        "geometry": first["geometry"],
                        "properties": {"image_index": "first", "timestamp_ms": 1},
                    },
                    {
                        "geometry": second["geometry"],
                        "properties": {"image_index": "second", "timestamp_ms": 2},
                    },
                ],
            }
            with patch.object(
                scene_cache, "fetch_metadata", return_value=metadata
            ) as fetch:
                first_plan = retriever.inventory_region(first)
                second_plan = retriever.inventory_region(second)
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(first_plan["acquisitions"][0]["image_index"], "first")
                self.assertIsNone(first_plan["acquisitions"][0]["cloud_probability_id"])
                self.assertEqual(
                    second_plan["acquisitions"][0]["image_index"], "second"
                )
                retriever.inventory_region(first)
                self.assertEqual(fetch.call_count, 1)
                retriever.inventory_region(first, refresh=True)
                self.assertEqual(fetch.call_count, 2)

    def test_preparation_cache_skips_geometry_work_and_detects_damaged_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = [reference_feature()]
            first = annual.prepare_source("francecrops", features, root, {})
            with patch.object(
                annual, "projected_cohorts", side_effect=AssertionError("must reuse")
            ):
                second = annual.prepare_source("francecrops", features, root, {})
            self.assertEqual(first["revision"], second["revision"])
            mask = next(
                Path(first["run_folder"]).glob(
                    "fields/*/balanced_pilot/sampling_masks/*.tif"
                )
            )
            mask.write_bytes(b"damaged")
            with patch.object(
                annual, "projected_cohorts", wraps=annual.projected_cohorts
            ) as prepare:
                third = annual.prepare_source("francecrops", features, root, {})
            prepare.assert_called_once()
            self.assertEqual(third["revision"], first["revision"])
            with rasterio.open(mask) as raster:
                self.assertGreater(raster.read(1).sum(), 0)

    def test_notebook_transport_options_do_not_change_processing_identity(self):
        notebook_path = Path("notebooks/03_cropharvest_preparation.ipynb")
        notebook = json.loads(notebook_path.read_text())
        before = progress.processing_notebook_fingerprint(notebook_path)
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            source = source.replace("download_workers = 2\n", "").replace(
                "intervals_per_request = 6\n", ""
            )
            source = source.replace(
                "    download_workers=download_workers, intervals_per_request=intervals_per_request,\n",
                "",
            )
            cell["source"] = source.splitlines(keepends=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "before.ipynb"
            path.write_text(json.dumps(notebook))
            self.assertEqual(before, progress.processing_notebook_fingerprint(path))
        # Geometry repair changes outlines, so the previous planner is not equivalent.
        self.assertNotEqual(
            annual_identity.planning_fingerprint(Path(annual.__file__)),
            next(iter(annual_identity.LEGACY_PLANNERS.values())),
        )
