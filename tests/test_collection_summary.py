"""Collection reports preserve operational state and distinguish reserved roles."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline.collection_summary import inspect_collection, write_summary


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, source="sen4agrinet", partition="train", status="pending_retry"):
    batch = root / f"sen4agrinet_work/adaptive_training/policy/rounds/0001/{source}"
    fields = root / f"fields/{source}/2020"
    props = {"partition_group": source + "-field", "parcel_id": "one", "crop": "potato",
             "country": "FR", "label_status": "reported"}
    save(fields / "source_footprints.geojson", {"features": [{"properties": props}, {"properties": props}]})
    plan = root / f"plans/{source}.json"
    save(plan, {"years": {"2020": {"field_folder": str(fields)}}})
    save(batch / "context.json", {"source": source, "plan_path": str(plan)})
    save(batch / "selection.json", {"source": source, "groups": [source + "-field"],
                                    "assignments": {source + "-field": partition}})
    save(batch / "state.json", {"status": status, "task_id": "do-not-touch"})
    return batch


class CollectionSummaryTests(unittest.TestCase):
    def test_deduplicates_fragments_preserves_state_and_separates_reserved_roles(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, partition="test")
            fixture(root, source="cropharvest", partition="test", status="completed")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                first = write_summary(root)
                second = write_summary(root)
            self.assertEqual(first, second)
            report = json.loads(first.read_text())
            self.assertEqual({r["role"] for r in report["profile"]}, {"reserved_development", "external_test"})
            self.assertEqual([r["selected_parcel_groups"] for r in report["profile"]], [1, 1])
            self.assertEqual(report["issues"], [])
            self.assertTrue(any(b["status"] == "pending_retry" for b in report["batches"]))
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            self.assertTrue((first.parent / "report.md").exists())

    def test_missing_metadata_is_reported_and_never_marks_batch_complete(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            batch = fixture(root)
            (root / "plans/sen4agrinet.json").unlink()
            report = inspect_collection(root)
            self.assertEqual(report["profile"], [])
            self.assertEqual(report["batches"][0]["status"], "pending_retry")
            self.assertEqual(report["batches"][0]["missing_groups"], 1)
            self.assertEqual({r["reason"] for r in report["issues"]},
                             {"missing_metadata", "selected_groups_without_reference_metadata"})

    def test_changed_states_create_new_reports_and_tampering_is_not_overwritten(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            batch = fixture(root)
            first = write_summary(root)
            save(batch / "state.json", {"status": "completed"})
            second = write_summary(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            (second.parent / "report.md").write_text("user edited")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_summary(root)
            self.assertEqual((second.parent / "report.md").read_text(), "user edited")

    def test_settings_credentials_are_not_copied(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / "settings.local.json", {"token": "private-test-value"})
            report = inspect_collection(root)
            self.assertNotIn("private-test-value", json.dumps(report))
            self.assertNotIn(str(root / "settings.local.json"), report["inputs"])


class CollectionNotebookTests(unittest.TestCase):
    def test_archived_notebook_integrity_and_active_paths(self):
        root = Path(__file__).resolve().parents[1]
        archive = root / "notebooks/archive/collection_before_reorganization"
        for record in json.loads((archive / "inventory.json").read_text()):
            saved = root / record["archive"]
            self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), record["sha256"])
            self.assertFalse((root / record["original"]).exists())
            self.assertTrue((root / record["active"]).is_file())

    def test_summary_notebook_runs_offline_from_nested_directory(self):
        root = Path(__file__).resolve().parents[1]
        book = json.loads((root / "notebooks/data_collection/06_dataset_summary.ipynb").read_text())
        with TemporaryDirectory() as folder:
            from contextlib import chdir, redirect_stdout
            import io
            target = Path(folder)
            (target / "pyproject.toml").touch()
            nested = target / "notebooks/data_collection"
            nested.mkdir(parents=True)
            fixture(target)
            before = (target / "sen4agrinet_work/adaptive_training/policy/rounds/0001/sen4agrinet/state.json").read_bytes()
            namespace = {}
            with chdir(nested), redirect_stdout(io.StringIO()), patch(
                "requests.sessions.Session.request", side_effect=AssertionError("No network")
            ):
                for cell in book["cells"]:
                    if cell["cell_type"] == "code":
                        exec("".join(cell["source"]), namespace)
            self.assertTrue(namespace["collection_manifest"].is_file())
            self.assertEqual(before, (target / "sen4agrinet_work/adaptive_training/policy/rounds/0001/sen4agrinet/state.json").read_bytes())
