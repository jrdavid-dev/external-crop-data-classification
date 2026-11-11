"""Keep supplementary collection safe, reproducible and explicitly offline."""

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from crop_pipeline.imagery import batch_export
from crop_pipeline.sources import cropharvest, pilot_sources

ROOT = Path(__file__).resolve().parents[1]


def example_plan():
    return {
        "fragment_id": "parcel",
        "settings": {"version": 4, "cloud_threshold": 40},
        "acquisitions": [
            {
                "date": "2020-04-02",
                "image_index": "one",
                "cloud_probability_id": "cloud",
            }
        ],
    }


class BatchSafetyTests(unittest.TestCase):
    def test_changed_settings_submit_separate_work_and_cannot_collect_old_rows(self):
        original = example_plan()
        changed = copy.deepcopy(original)
        changed["settings"]["cloud_threshold"] = 20
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ledger = folder / "tasks.json"
            ee = MagicMock()
            with (
                patch.object(batch_export, "daily_collection", return_value="table"),
                patch.object(
                    batch_export, "start_export", side_effect=["old", "new"]
                ) as start,
            ):
                batch_export.submit_daily(
                    ee,
                    None,
                    [original],
                    folder,
                    ledger,
                    "projects/test/assets/tmp",
                    2020,
                )
                batch_export.submit_daily(
                    ee,
                    None,
                    [changed],
                    folder,
                    ledger,
                    "projects/test/assets/tmp",
                    2020,
                )
                repeated = batch_export.submit_daily(
                    ee,
                    None,
                    [changed],
                    folder,
                    ledger,
                    "projects/test/assets/tmp",
                    2020,
                )
            self.assertEqual(start.call_count, 2)
            self.assertEqual(repeated["tasks_submitted"], 0)
            entries = batch_export.load_ledger(ledger)
            self.assertNotEqual(entries[0]["asset_id"], entries[1]["asset_id"])
            with self.assertRaisesRegex(ValueError, "request changed"):
                batch_export.validate_task_requests(entries[0], {"parcel": changed})
            batch_export.validate_task_requests(entries[1], {"parcel": changed})

    def test_changed_acquisition_matches_change_the_request_identity(self):
        original = example_plan()
        changed = copy.deepcopy(original)
        changed["acquisitions"][0]["cloud_probability_id"] = "different_cloud_match"
        self.assertNotEqual(
            batch_export.request_fingerprint(original),
            batch_export.request_fingerprint(changed),
        )
        with self.assertRaisesRegex(ValueError, "Legacy"):
            batch_export.validate_task_requests(
                {"items": ["daily|parcel"]}, {"parcel": original}
            )

    def test_exports_do_not_overwrite_and_permanent_names_preserve_content_versions(
        self,
    ):
        ee = MagicMock()
        ee.batch.Export.table.toAsset.return_value.id = "task_id"
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "tasks.json"
            first = batch_export.submit_permanent(
                ee, "rows", "projects/test/assets/features", ledger, "first", "features"
            )
            repeated = batch_export.submit_permanent(
                ee, "rows", "projects/test/assets/features", ledger, "first", "features"
            )
            changed = batch_export.submit_permanent(
                ee,
                "new_rows",
                "projects/test/assets/features",
                ledger,
                "second",
                "features",
            )
        self.assertFalse(repeated["submitted"])
        self.assertNotEqual(first["asset_id"], changed["asset_id"])
        self.assertEqual(ee.batch.Export.table.toAsset.call_count, 2)
        for call in ee.batch.Export.table.toAsset.call_args_list:
            self.assertFalse(call.kwargs["overwrite"])

    def test_batch_wait_can_continue_sequentially_or_stop_without_sleeping(self):
        active = [{"collected": False, "state": "RUNNING"}]
        done = [{"collected": True, "state": "SUCCEEDED"}]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "tasks.json"
            with (
                patch.object(batch_export, "collect", side_effect=[active, done]),
                patch.object(batch_export.time, "sleep") as sleep,
            ):
                result = batch_export.wait_and_collect(
                    None, "project", ledger, {"daily": None}, {}, poll_seconds=0.01
                )
                self.assertEqual(result, done)
                sleep.assert_called_once()
            with (
                patch.object(batch_export, "collect", return_value=active),
                patch.object(batch_export.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    batch_export.wait_and_collect(
                        None, "project", ledger, {"daily": None}, {}, wait=False
                    )
                sleep.assert_not_called()

    def test_successful_retry_clears_only_its_earlier_failed_item(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "tasks.json"
            batch_export.save_ledger(
                ledger,
                [
                    {"kind": "daily", "items": ["same"], "state": "FAILED"},
                    {
                        "kind": "daily",
                        "items": ["same"],
                        "state": "COMPLETED",
                        "collected": True,
                    },
                ],
            )
            with patch.object(batch_export, "collect", return_value=[]):
                self.assertEqual(
                    batch_export.wait_and_collect(
                        None, "project", ledger, {"daily": None}, {}
                    ),
                    [],
                )


class CropHarvestRunTests(unittest.TestCase):
    def test_years_require_supported_observations_or_explicit_overrides(self):
        self.assertEqual(
            cropharvest.resolve_observation_year("2020-06-15", "2021-01-01")["year"],
            2020,
        )
        self.assertEqual(
            cropharvest.resolve_observation_year("2019-01-01", "2021-01-01")["status"],
            "pending",
        )
        self.assertEqual(
            cropharvest.resolve_observation_year("2018-07-01", "2021-01-01")["status"],
            "pending",
        )
        supported = cropharvest.resolve_observation_year("2020-01-01", "2021-01-01")
        self.assertEqual(supported["year"], 2020)
        self.assertEqual(supported["reason"], "season_year_supported_by_export_period")
        decision = cropharvest.resolve_observation_year(
            "2020-01-01", "2021-01-01", 2020
        )
        self.assertEqual(decision["reason"], "explicit_year_override")
        self.assertEqual(decision["year"], 2020)

    def test_changed_configuration_has_separate_outputs_and_same_run_reuses_folder(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            first, revision = cropharvest.initialize_run(
                temporary, "pilot", {"cloud": 40}
            )
            old_result = first / "result.csv"
            old_result.write_text("preserve me")
            same, _ = cropharvest.initialize_run(temporary, "pilot", {"cloud": 40})
            changed, _ = cropharvest.initialize_run(temporary, "pilot", {"cloud": 20})
            self.assertEqual(first, same)
            self.assertNotEqual(first, changed)
            self.assertEqual(old_result.read_text(), "preserve me")

    def test_runtime_switches_do_not_break_reuse_but_processing_changes_version_run(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.ipynb"

            def write_source(code):
                path.write_text(
                    json.dumps(
                        {
                            "cells": [
                                {
                                    "cell_type": "code",
                                    "source": code.splitlines(keepends=True),
                                }
                            ]
                        }
                    )
                )

            write_source(
                "run_satellite_collection = False\ncloud_probability_threshold = 40\n"
            )
            original = cropharvest.notebook_source_fingerprint(path)
            write_source(
                "run_satellite_collection = True\ncloud_probability_threshold = 40\n"
            )
            self.assertEqual(original, cropharvest.notebook_source_fingerprint(path))
            write_source(
                "run_satellite_collection = True\ncloud_probability_threshold = 20\n"
            )
            self.assertNotEqual(original, cropharvest.notebook_source_fingerprint(path))

    def test_candidate_sampling_defaults_to_shared_cap_and_real_capacity(self):
        catalog = [
            {
                "crop": "carrot",
                "partition_group": f"parcel_{index // 100}",
                "source_crs": "EPSG:32631",
                "x": index * 10,
                "y": 100,
                "source_fragments": ["fragment"],
            }
            for index in range(300)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(cropharvest.bilinear.CANDIDATE_BUDGETS, {"carrot": 150}):
                _, summary, budgets = cropharvest.sample_year_candidates(
                    catalog, temporary
                )
        self.assertEqual(budgets["carrot"], 150)
        self.assertEqual(summary["candidate_pixels_per_crop"]["carrot"], 150)

    def test_invalid_bundle_does_not_replace_a_previously_selected_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "source_handoffs"
            folder.mkdir()
            pointer = folder / "active_supplement_pilot.json"
            pointer.write_text("original selection")
            bundle = root / "invalid.json"
            bundle.write_text('{"kind":"wrong"}')
            with self.assertRaisesRegex(ValueError, "does not match"):
                pilot_sources.activate_supplement(root / "inventory", "pilot", bundle)
            self.assertEqual(pointer.read_text(), "original selection")

    def test_completion_requires_every_planned_interval_without_pixel_minimum(self):
        plan = {
            "fragment_id": "parcel",
            "fragment": {"properties": {"crop": "cabbage", "selected_pixels": 12}},
            "download_intervals": [{"name": "first"}, {"name": "second"}],
        }
        summary = {
            "results": [{"region_id": "parcel", "name": "first", "status": "reused"}]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "completion.json"
            with self.assertRaisesRegex(ValueError, "all planned intervals"):
                cropharvest.download_completion([plan], summary, path)
            summary["results"].append(
                {"region_id": "parcel", "name": "second", "status": "downloaded"}
            )
            report = cropharvest.download_completion([plan], summary, path)
            self.assertEqual(report["downloaded_sampled_pixels"], {"cabbage": 12})

    def test_bundle_selection_routes_later_stages_without_overwriting_original_run(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff = root / "cropharvest.json"
            handoff.write_text('{"source": "cropharvest"}')
            bundle = cropharvest.publish_handoff_bundle(
                root / "inventory", "pilot", root / "run", [handoff]
            )
            original_paths, original_name = pilot_sources.resolve_collection(
                root / "inventory", "pilot"
            )
            self.assertEqual(original_name, "pilot")
            pilot_sources.activate_supplement(root / "inventory", "pilot", bundle)
            paths, name = pilot_sources.resolve_collection(root / "inventory", "pilot")
            self.assertEqual(paths[:2], original_paths)
            self.assertEqual(paths[2].resolve(), handoff.resolve())
            self.assertNotEqual(name, original_name)
            handoff.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "handoff changed"):
                pilot_sources.resolve_collection(root / "inventory", "pilot")

    def test_full_source_notebook_runs_offline_with_local_references(self):
        notebook_path = ROOT / "notebooks/preprocessing/04_cropharvest_test_preparation.ipynb"
        notebook = json.loads(notebook_path.read_text())
        projector = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        features = []
        for index, crop in enumerate(("carrot", "cabbage", "potato")):
            geometry = transform(
                projector.transform,
                box(500000 + index * 1000, 5400000, 500100 + index * 1000, 5400100),
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": {
                        "dataset": "jecam",
                        "index": index,
                        "label": crop,
                        "is_crop": 1,
                        "is_test": False,
                        "collection_date": "2020-06-15",
                        "export_end_date": "2021-01-01",
                        "planting_date": None,
                        "harvest_date": None,
                        "lat": geometry.centroid.y,
                        "lon": geometry.centroid.x,
                        "classification_label": (
                            "root_tuber" if crop == "potato" else "vegetables_melons"
                        ),
                    },
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notebooks/preprocessing").mkdir(parents=True)
            (root / "notebooks/preprocessing/04_cropharvest_test_preparation.ipynb").write_bytes(
                notebook_path.read_bytes()
            )
            batch_path = root / "src/crop_pipeline/imagery/batch_export.py"
            batch_path.parent.mkdir(parents=True)
            batch_path.write_bytes(
                (ROOT / "src/crop_pipeline/imagery/batch_export.py").read_bytes()
            )
            labels = root / "data/external/cropharvest/labels.geojson"
            labels_payload = json.dumps({"type": "FeatureCollection", "features": features}).encode()
            labels.parent.mkdir(parents=True)
            labels.write_bytes(labels_payload)
            forbidden = AssertionError(
                "Offline source notebook attempted a network operation"
            )
            namespace = {}
            with (
                chdir(root),
                patch("crop_pipeline.sources.test_cohort.completed_sources", return_value=[]),
                redirect_stdout(io.StringIO()),
                patch("requests.sessions.Session.request", side_effect=forbidden),
                patch.object(cropharvest, "download_labels_payload", side_effect=forbidden),
                patch("subprocess.check_call", side_effect=forbidden),
                patch.object(batch_export, "submit_daily", side_effect=forbidden),
                patch.object(batch_export, "submit_composites", side_effect=forbidden),
                patch.object(batch_export, "submit_permanent", side_effect=forbidden),
            ):
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        source = source.replace('reference_mode = "reuse"', 'reference_mode = "offline"')
                        exec(source, namespace)
            self.assertEqual(namespace["labels_download"]["status"], "reused")
            self.assertEqual(labels.read_bytes(), labels_payload)
            self.assertEqual(namespace["reference_mode"], "offline")
            self.assertFalse(namespace["authenticate_now"])
            self.assertTrue(labels.is_file())
            self.assertTrue((root / "settings.local.json").is_file())
            self.assertFalse(namespace["run_satellite_collection"])
            self.assertEqual(len(namespace["annual_plan"]["years"]), 1)
            self.assertEqual(set(namespace["reviewed_labels"].crop), {"carrot", "cabbage", "potato"})
            self.assertTrue(namespace["reviewed_labels"].reliable.all())
            self.assertEqual(len(namespace["reference_features"]), 3)
            self.assertEqual(namespace["inward_buffer_metres"], 10)
            self.assertEqual(namespace["collection_result"]["status"], "disabled")
            self.assertTrue(
                (
                    namespace["run_folder"] / "labels/observation_year_audit.csv"
                ).is_file()
            )
            plt.close("all")
