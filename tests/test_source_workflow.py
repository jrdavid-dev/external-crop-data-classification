"""Common presentation reports saved work without modifying collection state."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from crop_pipeline import source_workflow as workflow


class SourceWorkflowTests(unittest.TestCase):
    def output(self, batches, result, **keywords):
        stream = io.StringIO()
        with redirect_stdout(stream):
            workflow.print_summary("sen4agrinet", batches, result, "reuse", 0, **keywords)
        return stream.getvalue()

    def test_reference_modes_are_explicit_and_reject_unknown_values(self):
        for mode in ("reuse", "offline"):
            self.assertEqual(workflow.reference_options(mode),
                             {"download_if_missing": False, "cached_only": True})
        self.assertEqual(workflow.reference_options("download_missing"),
                         {"download_if_missing": True, "cached_only": False})
        with self.assertRaisesRegex(ValueError, "reference_mode"):
            workflow.reference_options("automatic")

    def test_summary_combines_batches_deduplicates_parcels_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "round"; folder.mkdir()
            selection = {"groups": ["field_2020", "field_2021"]}
            (folder / "selection.json").write_text(json.dumps(selection))
            years = {}
            for year in (2020, 2021):
                field = root / str(year); field.mkdir()
                footprint = {"features": [{"properties": {
                    "partition_group": f"field_{year}", "parcel_id": "physical_field",
                    "country": "FR", "crop": "potato"}}]}
                (field / "source_footprints.geojson").write_text(json.dumps(footprint))
                years[str(year)] = {"field_folder": str(field)}
            context = {"source": "sen4agrinet", "folder": str(folder), "candidate_parcels": 2,
                       "campaign": str(root / "campaign"), "annual_tif_root": str(root / "tifs")}
            batches = [({"years": years}, context)]
            result = {"status": "pending_retry", "batches": [
                {"batch": "old", "result": {"status": "completed", "registry": "saved.json"}},
                {"batch": "extension", "result": {"status": "pending_retry"}}],
                "progress_by_year": {
                    "old/2020": {"tiles": {"complete": 3, "total": 3}, "jobs": {"queued": 0}},
                    "extension/2021": {"tiles": {"complete": 2, "total": 5}, "jobs": {"queued": 2, "running": 1}}}}
            original = deepcopy((batches, result))
            saved = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            text = self.output(batches, result)
            self.assertIn("Selected parcels: potato 1", text)
            self.assertIn("Verified local tiles: 5/8", text)
            self.assertIn("2 queued, 1 running", text)
            self.assertIn("old: completed; extension: pending_retry", text)
            self.assertIn("Rerun this notebook", text)
            self.assertIn("saved.json", text)
            self.assertEqual((batches, result), original)
            self.assertEqual({p: p.read_bytes() for p in root.rglob("*") if p.is_file()}, saved)

    def test_failed_jobs_override_completed_top_level_status(self):
        text = self.output([], {"status": "completed", "progress_by_year": {
            "2020": {"tiles": {"complete": 0, "total": 2}, "jobs": {"failed": 1}}}})
        self.assertIn("Inspect the reported errors", text)
        self.assertNotIn("Next: Run notebook 04", text)

    def test_unknown_counts_are_not_reported_as_completed_or_zero_jobs(self):
        text = self.output([], {"status": "disabled"})
        self.assertIn("Verified local tiles: unavailable", text)
        self.assertIn("Cloud jobs: unavailable", text)
        self.assertIn("Preparation only", text)

    def test_offline_summary_shows_preserved_pending_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            state = folder / "state.json"
            state.write_text(json.dumps({"status": "pending_retry"}))
            before = state.read_bytes()
            batches = [(None, {"folder": str(folder)})]
            stream = io.StringIO()
            with redirect_stdout(stream):
                workflow.print_summary("sen4agrinet", batches, {"status": "disabled"}, "offline", 0)
            self.assertIn("Saved operational state (unchanged)", stream.getvalue())
            self.assertIn("pending_retry", stream.getvalue())
            self.assertIn("Leave offline mode", stream.getvalue())
            self.assertEqual(state.read_bytes(), before)

    def test_reference_exhaustion_has_an_explicit_bounded_expansion_action(self):
        batches = [(None, {"folder": "unused", "selection_status": "no_candidates_for_deficits"})]
        text = self.output(batches, {"status": "completed", "batches": [
            {"batch": "empty", "result": {"status": "skipped"}}]}, reference_note="350 patches checked")
        self.assertIn("350 patches checked", text)
        self.assertIn("No unselected reference candidates remain", text)
        self.assertIn("reference_target_parcels_per_crop", text)
        self.assertIn("reference_scan_max_patches", text)
        self.assertIn("Run notebook 04", text)


if __name__ == "__main__":
    unittest.main()
