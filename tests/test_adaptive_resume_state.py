"""Offline inspection preserves saved operational state and pending-work barriers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive


class AdaptiveResumeStateTests(unittest.TestCase):
    def test_disabled_collection_preserves_existing_state_bytes(self):
        for status in ("pending_retry", "completed", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                state_path = folder / "state.json"
                save_json(state_path, {"status": status, "registry": "saved.json", "task": "original"})
                original = state_path.read_bytes()
                context = {"folder": str(folder), "report_folder": str(folder / "inventory")}
                with patch.object(adaptive.annual, "collect_source", return_value={"status": "disabled"}) as collect:
                    result = adaptive.collect_batch({"source": "sen4agrinet"}, context, "", enabled=False)
                self.assertEqual(result["status"], "disabled")
                self.assertFalse(collect.call_args.kwargs["enabled"])
                self.assertEqual(state_path.read_bytes(), original)

    def test_offline_inspection_does_not_unblock_window_assessment(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "inventory"
            folder = adaptive.root(report) / "rounds/0001/sen4agrinet_overnight_100"
            state_path = folder / "state.json"
            save_json(state_path, {"status": "pending_retry", "retry_queue": "existing.json"})
            context = {"folder": str(folder), "report_folder": str(folder / "inventory")}
            with patch.object(adaptive.annual, "collect_source", return_value={"status": "disabled"}):
                adaptive.collect_batch({"source": "sen4agrinet"}, context, "", enabled=False)
            with self.assertRaisesRegex(RuntimeError, "Finish the current candidate batches"):
                adaptive.resolve_collection(report)
            self.assertEqual(json.loads(state_path.read_text())["status"], "pending_retry")

    def test_new_disabled_preparation_still_records_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            context = {"folder": str(folder), "report_folder": str(folder / "inventory")}
            with patch.object(adaptive.annual, "collect_source", return_value={"status": "disabled"}):
                adaptive.collect_batch({"source": "sen4agrinet"}, context, "", enabled=False)
            self.assertEqual(json.loads((folder / "state.json").read_text()), {"status": "disabled"})

    def test_disabled_empty_plan_preserves_existing_operational_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            state_path = folder / "state.json"
            save_json(state_path, {"status": "pending_retry"})
            original = state_path.read_bytes()
            context = {"folder": str(folder), "selection_status": "no_candidates_for_deficits"}
            result = adaptive.collect_batch(None, context, "", enabled=False)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(state_path.read_bytes(), original)

    def test_first_real_batch_authenticates_after_empty_selections(self):
        batches = [(None, {"folder": "/empty"}),
                   ({"source": "sen4agrinet"}, {"folder": "/active"}),
                   ({"source": "sen4agrinet"}, {"folder": "/extension"})]
        with patch.object(adaptive, "collect_batch", side_effect=[
            {"status": "skipped"}, {"status": "pending_retry"}, {"status": "pending_retry"},
        ]) as collect:
            result = adaptive.collect_batches(batches, "project", enabled=True, authenticate=True)
        self.assertEqual([call.kwargs["authenticate"] for call in collect.call_args_list],
                         [False, True, False])
        self.assertEqual(result["status"], "pending_retry")

    def test_local_publication_failure_keeps_assessment_blocked_and_retries_safely(self):
        from crop_pipeline.imagery import annual_tif
        for fail in (True, False):
            with self.subTest(failure=fail), tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "inventory"
                campaign = adaptive.root(report)
                folder = campaign / "rounds/0001/sen4agrinet_overnight_100"
                state_path = folder / "state.json"
                manifest_path = Path(temporary) / "manifest.json"
                registry_path = Path(temporary) / "registry.json"
                save_json(manifest_path, {"year": "2020"})
                save_json(registry_path, {"handoffs": [{"path": str(manifest_path)}]})
                original_registry = registry_path.read_bytes()
                result = {"status": "completed", "registry": str(registry_path)}
                context = {"folder": str(folder), "report_folder": str(folder / "inventory"),
                           "annual_tif_root": str(Path(temporary) / "annual_tifs"),
                           "source": "sen4agrinet", "campaign": str(campaign), "round": 1}

                def export(manifest, output):
                    self.assertEqual(json.loads(state_path.read_text())["status"], "pending_retry")
                    self.assertFalse((folder / "completed.json").exists())
                    if fail:
                        raise OSError("disk full during annual TIFF publication")
                    return {"saved": True}

                with (
                    patch.object(adaptive.annual, "collect_source", return_value=result),
                    patch.object(annual_tif, "export_source", side_effect=export),
                ):
                    if fail:
                        with self.assertRaisesRegex(OSError, "disk full"):
                            adaptive.collect_batch({"source": "sen4agrinet"}, context, "project", enabled=True)
                    else:
                        returned = adaptive.collect_batch({"source": "sen4agrinet"}, context, "project", enabled=True)
                        self.assertEqual(returned, result)
                state = json.loads(state_path.read_text())
                if fail:
                    self.assertEqual(state["status"], "pending_retry")
                    self.assertEqual(state["reason"], "annual_tiff_publication_failed")
                    self.assertIn("disk full", state["error"])
                    self.assertEqual(state["registry"], str(registry_path))
                    self.assertEqual(registry_path.read_bytes(), original_registry)
                    with self.assertRaisesRegex(RuntimeError, "Finish the current candidate batches"):
                        adaptive.resolve_collection(report)
                else:
                    self.assertEqual(state, result)
                    self.assertTrue((folder / "completed.json").exists())
                    paths, _ = adaptive.resolve_collection(report)
                    self.assertEqual(paths, [manifest_path])
