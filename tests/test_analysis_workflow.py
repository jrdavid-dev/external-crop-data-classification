"""Pilot budgets do not waive complete windows, parcel locks or artifact integrity."""

import json
import tempfile
import unittest
from pathlib import Path

from crop_pipeline import analysis_workflow as workflow
from crop_pipeline.common import save_json
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.sources import adaptive
from crop_pipeline.windows import seasons
from test_exploratory_training import example_rows


def fixture(folder):
    folder = Path(folder)
    rows = [{**row, "window_status": "satellite_estimated", "timing_review_status": "automated_pass",
             "automatic_timing_status": "passed", "automatic_policy_fingerprint": "saved-policy", "missing_bins": ""}
            for row in example_rows()]
    source = folder / "complete_features.csv"
    seasons.write_records(source, rows)
    assignments = {}
    for row in rows:
        number = int(row["partition_group"].split("|")[-1])
        assignments[row["partition_group"]] = "validation" if number == 3 else "test" if number == 4 else "train"
    split = folder / "locked_split.json"
    save_json(split, {"kind": "exploratory_existing_source_parcel_evaluation", "group_assignments": assignments})
    coverage = folder / "coverage.json"
    save_json(coverage, {"imagery_request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY,
                         "targets_met": False, "minimum_valid_fraction": 0.8})
    return source, split, coverage, rows, assignments


class AnalysisWorkflowTests(unittest.TestCase):
    def export(self, folder, mode="pilot"):
        source, split, coverage, rows, assignments = fixture(folder)
        path = workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage, analysis_mode=mode)
        return path, (source, split, coverage, rows, assignments)

    def test_pilot_exports_unmet_budgets_with_original_locked_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            path, (source, split, coverage, original, assignments) = self.export(folder)
            rows, configuration, saved_coverage, manifest = workflow.load_window_handoff(path)
            self.assertEqual(len(rows), len(original))
            self.assertFalse(saved_coverage["targets_met"])
            self.assertFalse(configuration["frozen"])
            self.assertEqual(configuration["analysis_mode"], "pilot")
            self.assertEqual({row["partition_group"]: row["partition"] for row in rows}, assignments)
            self.assertEqual({row["window_status"] for row in rows}, {"satellite_estimated"})
            self.assertTrue(all(row["season_key"] == row["season_id"] for row in rows))
            self.assertEqual(manifest["row_count"], len(original))
            before = {p: p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()}
            self.assertEqual(workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage), path)
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_strict_requires_budgets_but_pilot_still_reports_deficits(self):
        with tempfile.TemporaryDirectory() as folder:
            source, split, coverage, _, _ = fixture(folder)
            with self.assertRaisesRegex(ValueError, "Strict analysis"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage, analysis_mode="strict")
            report = Path(folder)/"reports/inventory"
            feedback = adaptive.root(report)/"feedback.json"
            save_json(feedback, {"targets_met": False, "counts": {"carrot": {"train": {"pixels": 12}}}})
            readiness = workflow.require_collection_readiness(report, "pilot")
            self.assertFalse(readiness["targets_met"])
            self.assertEqual(readiness["counts"]["carrot"]["train"]["pixels"], 12)
            with self.assertRaisesRegex(RuntimeError, "targets are not met"):
                workflow.require_collection_readiness(report, "strict")
            save_json(feedback, {"targets_met": True})
            save_json(adaptive.root(report)/"locked_split.json", {})
            self.assertTrue(workflow.require_collection_readiness(report, "strict")["targets_met"])
            saved = json.loads(coverage.read_text()); saved["targets_met"] = True; save_json(coverage, saved)
            path = workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage, analysis_mode="strict")
            self.assertEqual(workflow.load_window_handoff(path)[1]["analysis_mode"], "strict")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pilot or strict"):
            workflow.validate_mode("ignore_checks")

    def test_pilot_refuses_quality_timing_calendar_and_split_problems(self):
        changes = [
            ({"bin_01_B3": "nan"}, "nonfinite"),
            ({"missing_bins": "2020_04_01_15"}, "Incomplete windows"),
            ({"reconstructed": True}, "Reconstructed windows"),
            ({"reconstructed": "true"}, "Reconstructed windows"),
            ({"reconstructed": 1}, "Reconstructed windows"),
            ({"timing_review_status": "pending"}, "Unqualified timing"),
            ({"window_end": "2020-07-01"}, "eight consecutive"),
            ({"window_start": "2020-04-02"}, "eight consecutive|four calendar months"),
            ({"partition": "test"}, "locked parcel"),
        ]
        for values, reason in changes:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as folder:
                source, split, coverage, rows, _ = fixture(folder)
                rows[0].update(values); seasons.write_records(source, rows)
                with self.assertRaisesRegex(ValueError, reason):
                    workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)

    def test_coverage_threshold_is_recorded_and_cannot_be_lowered_or_misreported(self):
        invalid = [None, True, "nan", .79, 1.01]
        for minimum in invalid:
            with self.subTest(minimum=minimum), tempfile.TemporaryDirectory() as folder:
                source, split, coverage, _, _ = fixture(folder)
                values = json.loads(coverage.read_text())
                values["minimum_valid_fraction"] = minimum
                save_json(coverage, values)
                with self.assertRaisesRegex(ValueError, "minimum valid fraction"):
                    workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
        with tempfile.TemporaryDirectory() as folder:
            source, split, coverage, _, _ = fixture(folder)
            values = json.loads(coverage.read_text())
            values["window_recipe"] = {"minimum_valid_fraction": .9}
            save_json(coverage, values)
            with self.assertRaisesRegex(ValueError, "fractions differ"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
            values["minimum_valid_fraction"] = .9
            save_json(coverage, values)
            path = workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
            self.assertEqual(workflow.load_window_handoff(path)[2]["minimum_valid_fraction"], .9)

    def test_missing_locked_group_or_class_cannot_create_new_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            source, split, coverage, rows, assignments = fixture(folder)
            del assignments[rows[0]["partition_group"]]
            save_json(split, {"kind": "exploratory_existing_source_parcel_evaluation", "group_assignments": assignments})
            with self.assertRaisesRegex(ValueError, "absent from the locked split"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
            _, split, coverage, rows, _ = fixture(folder)
            seasons.write_records(source, [row for row in rows if row["crop"] != "carrot"])
            with self.assertRaisesRegex(ValueError, "all three crops"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)

    def test_too_few_parcels_and_missing_partition_class_are_not_waived(self):
        with tempfile.TemporaryDirectory() as folder:
            source, split, coverage, rows, assignments = fixture(folder)
            seasons.write_records(source, [row for row in rows if row["partition_group"] != "carrot|0"])
            with self.assertRaisesRegex(ValueError, "five source parcel groups"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
            seasons.write_records(source, rows)
            assignments["carrot|4"] = "train"
            save_json(split, {"kind": "exploratory_existing_source_parcel_evaluation", "group_assignments": assignments})
            with self.assertRaisesRegex(ValueError, "carrot"):
                workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)

    def test_changed_inputs_get_new_bundle_and_old_artifacts_remain_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            path, (source, split, coverage, rows, _) = self.export(folder)
            saved = {p: p.read_bytes() for p in path.parent.iterdir()}
            rows[0]["bin_01_B3"] += .01
            seasons.write_records(source, rows)
            changed = workflow.export_window_handoff([source], Path(folder)/"windows", split, coverage)
            self.assertNotEqual(changed, path)
            self.assertEqual({p: p.read_bytes() for p in saved}, saved)
            (path.parent/"window_features.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "file changed"):
                workflow.load_window_handoff(path)

    def test_stage_pointers_block_stale_outputs_and_verify_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _ = self.export(folder)
            pointer = Path(folder)/"window_stage.json"
            workflow.write_stage_pointer(pointer, manifest)
            self.assertEqual(workflow.resolve_stage_pointer(pointer), manifest.resolve())
            workflow.write_stage_pointer(pointer, status="pending", reason="No complete carrot windows")
            with self.assertRaisesRegex(RuntimeError, "No complete carrot windows"):
                workflow.resolve_stage_pointer(pointer)
            workflow.write_stage_pointer(pointer, manifest)
            manifest.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Stage manifest changed"):
                workflow.resolve_stage_pointer(pointer)


if __name__ == "__main__":
    unittest.main()
