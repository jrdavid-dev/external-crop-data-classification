"""Local GeoTIFF fixtures verify continuation, reuse, gates and archive identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.sources import annual, annual_identity, collection_pass
from crop_pipeline.imagery.retrieval import SatelliteRetriever
from test_annual_archive import reference_feature


class LocalBackend:
    def __init__(self):
        self.failed_once = False
        self.fetches = []
        self.fail_image = False
        self.fail_inventory = None
        self.fail_daily = None
        self.first_region = None

    def retriever(
        self, earth_engine, year, imagery_folder, report_folder, cloud_threshold
    ):
        backend = self

        class LocalRetriever(SatelliteRetriever):
            def inventory_region(self, region, refresh=False):
                identifier = region["properties"]["fragment_id"]
                if identifier == backend.fail_inventory:
                    raise RuntimeError("simulated inventory outage")
                return {
                    "fragment_id": identifier,
                    "fragment": region,
                    "settings": self.request_settings(region),
                    "acquisitions": [],
                }

            def fetch_composite(self, plan, interval, granules, destination):
                backend.fetches.append((plan["fragment_id"], interval["name"]))
                if backend.fail_image and not backend.failed_once:
                    backend.failed_once = True
                    raise RuntimeError("HTTP 503: simulated temporary outage")
                grid = plan["settings"]["grid"]
                with rasterio.open(
                    destination,
                    "w",
                    driver="GTiff",
                    width=grid["width"],
                    height=grid["height"],
                    count=7,
                    dtype="float32",
                    crs=grid["crs"],
                    transform=Affine(*grid["transform"]),
                ) as raster:
                    raster.write(
                        np.ones((7, grid["height"], grid["width"]), dtype="float32")
                    )

        return LocalRetriever(
            earth_engine, year, imagery_folder, report_folder, cloud_threshold
        )

    def daily(
        self, plans, mask_acquisition, earth_engine, destination, minimum_fraction
    ):
        identifier = plans[0]["fragment_id"]
        if identifier == self.fail_daily:
            raise RuntimeError("simulated daily-summary failure")
        path = Path(destination) / "observed_checkpoints/fixture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return []


def source_plan(root, source, multiple_years=False):
    first = reference_feature("first", width=100)
    second = reference_feature("second", left=501000, width=100, crop="cabbage")
    if multiple_years:
        second["properties"].update(year=2021, partition_group="test|FR|2021|second")
    return annual.prepare_source(
        source, [first, second], root / "archive", {}, inward_buffer_metres=0
    )


def collect(plan, root, backend):
    earth_engine = SimpleNamespace(Initialize=lambda **arguments: None)
    with (
        patch.dict("sys.modules", {"ee": earth_engine}),
        patch.object(annual, "SatelliteRetriever", side_effect=backend.retriever),
        patch.object(
            annual.timelines, "retrieve_observed_timelines", side_effect=backend.daily
        ),
    ):
        return annual.collect_source(
            plan, root / "inventory", "fixture-project", enabled=True
        )


class CollectionRetryTests(unittest.TestCase):
    def test_all_sources_attempt_later_tiles_and_retry_only_one_failed_image(self):
        for source in annual.SOURCE_NAMES:
            with (
                self.subTest(source=source),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan = source_plan(root, source)
                backend = LocalBackend()
                backend.fail_image = True
                first = collect(plan, root, backend)
                expected = plan["estimate"]["composites"]
                self.assertEqual(len(backend.fetches), expected)
                self.assertEqual(first["status"], "pending_retry")
                self.assertEqual(first["pending_images"], 1)
                self.assertEqual(first["completed_tiles"], first["planned_tiles"] - 1)
                self.assertFalse(
                    annual.registry_path(root / "inventory", source).exists()
                )
                queue = json.loads(Path(first["retry_queue"]).read_text())
                self.assertIn("503", queue[0]["pending_images"][0]["error"])
                second = collect(plan, root, backend)
                self.assertEqual(second["status"], "completed")
                self.assertEqual(second["downloaded_images"], 1)
                self.assertEqual(second["reused_images"], expected - 1)
                self.assertEqual(len(backend.fetches), expected + 1)
                self.assertTrue(
                    annual.registry_path(root / "inventory", source).exists()
                )
                self.assertTrue(Path(first["retry_queue"]).is_file())

    def test_daily_failure_still_downloads_images_and_blocks_the_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = source_plan(root, "francecrops")
            backend = LocalBackend()
            first_year = next(iter(plan["years"].values()))
            backend.fail_daily = annual.read_regions(first_year)[0]["properties"][
                "fragment_id"
            ]
            first = collect(plan, root, backend)
            self.assertEqual(first["status"], "pending_retry")
            self.assertEqual(first["pending_images"], 0)
            self.assertEqual(len(backend.fetches), plan["estimate"]["composites"])
            backend.fail_daily = None
            second = collect(plan, root, backend)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["downloaded_images"], 0)

    def test_inventory_failure_does_not_stop_other_years(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = source_plan(root, "cropharvest", multiple_years=True)
            backend = LocalBackend()
            backend.fail_inventory = annual.read_regions(plan["years"]["2020"])[0][
                "properties"
            ]["fragment_id"]
            result = collect(plan, root, backend)
            self.assertEqual(result["status"], "pending_retry")
            self.assertEqual(result["pending_images"], 24)
            self.assertTrue(any("2021" in interval for _, interval in backend.fetches))

    def test_damaged_file_is_retrieved_again_and_existing_registry_is_preserved_on_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = source_plan(root, "sen4agrinet")
            backend = LocalBackend()
            collect(plan, root, backend)
            registry = annual.registry_path(root / "inventory", "sen4agrinet")
            saved_registry = registry.read_bytes()
            image = next(Path(plan["run_folder"]).glob("imagery/*/*/*/*.tif"))
            image.write_bytes(b"damaged")
            backend.fail_image = True
            result = collect(plan, root, backend)
            self.assertEqual(result["status"], "pending_retry")
            self.assertEqual(registry.read_bytes(), saved_registry)
            with self.assertRaisesRegex(RuntimeError, "still pending"):
                annual.resolve_collection(root / "inventory")
            recovered = collect(plan, root, backend)
            self.assertEqual(recovered["downloaded_images"], 1)
            self.assertEqual(recovered["status"], "completed")

    def test_old_equivalent_archive_is_reused_without_rewriting_its_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = {
                "source": "francecrops",
                "footprints_sha256": "same",
                "planning_sha256": annual_identity.planning_fingerprint(Path(annual.__file__)),
                "request_version": 4,
                "helper_sha256": next(iter(annual_identity.LEGACY_PLANNERS)),
                "timelines_sha256": "old",
            }
            folder = root / "runs/full_year_all_parcels/previous"
            folder.mkdir(parents=True)
            path = folder / "run_configuration.json"
            path.write_text(json.dumps(configuration))
            original = path.read_bytes()
            updated = {
                **configuration,
                "helper_sha256": "new-retry-code",
                "timelines_sha256": "corrected-daily-code",
                "planning_sha256": annual_identity.planning_fingerprint(
                    Path(annual.__file__)
                ),
            }
            reused, revision = annual_identity.initialize_archive(
                root, annual.COLLECTION_NAME, updated
            )
            self.assertEqual(reused, folder)
            self.assertEqual(revision, "previous")
            self.assertEqual(path.read_bytes(), original)
            changed = {**updated, "request_version": 5}
            another, _ = annual_identity.initialize_archive(
                root, annual.COLLECTION_NAME, changed
            )
            self.assertNotEqual(another, folder)

    def test_remaining_space_estimate_reduces_after_completed_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = source_plan(root, "francecrops")
            before = collection_pass.estimated_required_space(plan)
            collect(plan, root, LocalBackend())
            after = collection_pass.estimated_required_space(plan)
            self.assertLess(after, before)
            self.assertEqual(after, 64 * 1024**2)
