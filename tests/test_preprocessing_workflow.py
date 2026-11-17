"""Fresh-kernel preprocessing handoffs, sample accounting and collection isolation."""

from contextlib import chdir, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline import preprocessing_workflow as workflow
from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import seasons


def reference_rows():
    rows = []
    for partition, source in (("train", "sen4agrinet"), ("validation", "francecrops"), ("test", "cropharvest")):
        for crop in workflow.CROPS:
            group = f"{partition}-{crop}"
            rows.append({"partition_group": group, "field_id": group, "region_id": group,
                         "season_key": group, "crop": crop, "partition": partition,
                         "source_dataset": source, "label_status": "reported",
                         "source_crs": "EPSG:32631", "x": len(rows) * 100 + 5, "y": 5,
                         "window_start": "2020-01-01", "window_end": "2020-05-01",
                         **{column: .2 for column in seasons.feature_columns()}})
    return rows


def fake_audit(report, output):
    run = Path(output) / "external_fixture"
    run.mkdir(parents=True, exist_ok=False)
    regions = [{"properties": {"fragment_id": r["region_id"], "partition_group": r["partition_group"],
                                "crop": r["crop"], "partition": r["partition"], "selected_pixels": 1}}
               for r in reference_rows()]
    save_json(run / "quality.json", {"regions": regions, "quality_rows": [], "pixel_paths": {}})
    return run / "quality.json"


def fake_select(quality, report):
    rows = reference_rows()
    save_json(quality.parent / "windows.json", {"fixed": rows, "aligned": rows[:-1], "quality": str(quality)})
    return quality.parent / "windows.json"


class PreprocessingWorkflowTests(unittest.TestCase):
    def test_export_excludes_shared_development_locations_before_matching(self):
        rows = reference_rows()
        train, validation, test = rows[0], rows[3], rows[6]
        overlap = {**validation, "x": train["x"], "source_dataset": "sen4agrinet"}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "windows.json"
            save_json(path, {"fixed": [train, validation, test],
                             "aligned": [train, validation, overlap, test], "quality": "fixture"})
            workflow.external.export_features(path)
            development = json.loads(path.with_name("features_development.json").read_text())["rows"]
            self.assertEqual({r["partition"] for r in development["fixed_full"]}, {"train", "validation"})
            for cohort in ("aligned_full", "fixed_matched", "aligned_matched"):
                self.assertEqual([(r["partition_group"], r["partition"]) for r in development[cohort]],
                                 [(validation["partition_group"], "validation")])
            test_rows = json.loads(path.with_name("features_test.json").read_text())["rows"]
            self.assertTrue(all([r["partition_group"] for r in cohort] == [test["partition_group"]]
                                for cohort in test_rows.values()))
            exclusions = json.loads(path.with_name("development_overlap_exclusions.json").read_text())
            self.assertEqual(exclusions["aligned"]["shared_locations"], 1)
            self.assertEqual({r["partition"] for r in exclusions["aligned"]["excluded"]}, {"train", "validation"})

    def fixture(self, root):
        (root / "pyproject.toml").touch()
        state = root / "sen4agrinet_work/adaptive_training/campaign/rounds/0001/source/state.json"
        save_json(state, {"status": "completed", "task_id": "preserve"})
        return state

    def test_end_to_end_counts_reuse_and_unchanged_collection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            original = state.read_bytes()
            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(
                workflow.external, "select", side_effect=fake_select
            ), patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                quality = workflow.audit(root)
                windows = workflow.select(root)
                handoff = workflow.export_features(root)
                before = {p: p.read_bytes() for p in handoff.parent.rglob("*") if p.is_file()}
                self.assertEqual(workflow.select(root), windows)
                self.assertEqual(workflow.export_features(root), handoff)
            self.assertEqual(state.read_bytes(), original)
            self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))
            rows = seasons.read_table(handoff.parent / "usable_samples.csv")
            self.assertEqual(len(rows), 36)
            fixed_test = next(r for r in rows if r["cohort"] == "fixed_full" and r["partition"] == "test" and r["crop"] == "potato")
            aligned_test = next(r for r in rows if r["cohort"] == "aligned_full" and r["partition"] == "test" and r["crop"] == "potato")
            self.assertEqual(int(fixed_test["usable_pixels"]), 1)
            self.assertEqual(int(aligned_test["usable_pixels"]), 0)
            self.assertEqual(int(aligned_test["pixel_shortfall"]), 400)
            development = json.loads((handoff.parent / "features_development.json").read_text())
            test = json.loads((handoff.parent / "features_test.json").read_text())
            self.assertTrue(all(r["partition"] != "test" for rows in development["rows"].values() for r in rows))
            self.assertTrue(all(r["partition"] == "test" for rows in test["rows"].values() for r in rows))
            self.assertEqual(resolve_stage_pointer(workflow.pointer(root, "features")), handoff)
            self.assertTrue((root / "pipeline_outputs/preprocessing/external_fixture/features/usable_samples.csv").samefile(handoff.parent / "usable_samples.csv"))

    def test_pending_collection_blocks_real_audit_without_mutating_jobs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            report = workflow.paths(root)[1]
            state = adaptive.root(report) / "rounds/0001/source/state.json"
            save_json(state, {"status": "pending_retry", "task_id": "running"})
            before = state.read_bytes()
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                with self.assertRaisesRegex(RuntimeError, "Finish the current candidate batches"):
                    workflow.audit(root)
            self.assertEqual(state.read_bytes(), before)
            self.assertFalse(workflow.paths(root)[2].exists())
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "quality"))

    def test_new_collection_activity_invalidates_snapshot_and_downstream_pointers(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit):
                workflow.audit(root)
            save_json(state, {"status": "pending_retry"})
            with self.assertRaisesRegex(RuntimeError, "Collection changed"):
                workflow.select(root)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "windows"))
            self.assertFalse((workflow.paths(root)[2] / "external_fixture/windows.json").exists())

    def test_partial_outputs_are_preserved_instead_of_overwritten(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit):
                quality = workflow.audit(root)
            incomplete = quality.parent / "windows.json"
            incomplete.write_text("partial output")
            with self.assertRaisesRegex(FileExistsError, "Partial window"):
                workflow.select(root)
            self.assertEqual(incomplete.read_text(), "partial output")

    def test_changed_features_cannot_be_reused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=fake_select):
                workflow.audit(root)
                workflow.select(root)
                handoff = workflow.export_features(root)
            (handoff.parent / "features_test.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "output changed"):
                workflow.export_features(root)

    def test_final_counts_apply_existing_twenty_pixel_cap(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)

            def many_pixels(quality, report):
                rows = [{**row, "x": row["x"] + offset} for row in reference_rows() for offset in range(25)]
                save_json(quality.parent / "windows.json", {"fixed": rows, "aligned": rows, "quality": str(quality)})
                return quality.parent / "windows.json"

            with patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=many_pixels):
                workflow.audit(root)
                workflow.select(root)
                handoff = workflow.export_features(root)
            rows = seasons.read_table(handoff.parent / "usable_samples.csv")
            self.assertTrue(all(int(r["usable_pixels"]) == 20 for r in rows))
            self.assertTrue(all(int(r["usable_parcel_groups"]) == 1 for r in rows))

    def test_collection_change_during_audit_cannot_publish_success(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)

            def racing_audit(report, output):
                handoff = fake_audit(report, output)
                save_json(state, {"status": "pending_retry"})
                return handoff

            with patch.object(workflow.external, "audit", side_effect=racing_audit):
                with self.assertRaisesRegex(RuntimeError, "changed during auditing"):
                    workflow.audit(root)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                resolve_stage_pointer(workflow.pointer(root, "quality"))

    def test_new_notebooks_run_in_fresh_namespaces_without_explicit_paths(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            nested = root / "notebooks/preprocessing"
            nested.mkdir(parents=True)
            with chdir(nested), redirect_stdout(io.StringIO()), patch.object(workflow.external, "audit", side_effect=fake_audit), patch.object(workflow.external, "select", side_effect=fake_select), patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
                for name in ("01_imagery_quality", "02_window_selection", "03_feature_engineering"):
                    book = json.loads((repository / "notebooks/preprocessing" / f"{name}.ipynb").read_text())
                    namespace = {}
                    for cell in book["cells"]:
                        if cell["cell_type"] == "code":
                            exec("".join(cell["source"]), namespace)
            self.assertTrue(resolve_stage_pointer(workflow.pointer(root, "features")).is_file())

    def test_original_notebook_outputs_are_archived_exactly(self):
        repository = Path(__file__).resolve().parents[1]
        archive = repository / "notebooks/archive/preprocessing_before_reorganization"
        for entry in json.loads((archive / "inventory.json").read_text()):
            self.assertEqual(hashlib.sha256((repository / entry["archive"]).read_bytes()).hexdigest(), entry["sha256"])
            self.assertFalse((repository / entry["original"]).exists())
            self.assertTrue((repository / entry["active"]).is_file())
