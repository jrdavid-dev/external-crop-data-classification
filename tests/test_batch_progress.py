"""Real workload accounting: no overlap, no stale failures, no cloud-only completion."""
import unittest

from crop_pipeline.common import interval_bins
from crop_pipeline.imagery import batch_progress, batch_export


def plans(count):
    return [{"fragment_id": str(i), "settings": {}, "acquisitions": [],
             "download_intervals": interval_bins(2022)} for i in range(count)]


class BatchProgressTests(unittest.TestCase):
    def test_100_tiles_have_consistent_period_and_tile_totals(self):
        workload = plans(100)
        missing = [(plan, plan["download_intervals"]) for plan in workload[85:]]
        result = batch_progress.snapshot(workload, missing, workload[50:], [])
        self.assertEqual(result["half_month_periods"], {"total": 2400, "verified_local": 2040, "pending": 360})
        tiles = result["tiles"]
        self.assertEqual(tiles["complete"], 50)
        self.assertEqual(tiles["pixel_data_complete"], 85)
        self.assertEqual(tiles["daily_summaries_complete"], 50)
        self.assertEqual(tiles["percent_complete"], 50.0)
        self.assertEqual(tiles["complete"] + tiles["pixel_data_only"] + tiles["daily_summaries_only"] + tiles["neither_complete"], 100)
        self.assertEqual(result["jobs"]["work_items_not_submitted"], 410)

    def test_partial_raster_periods_and_daily_only_tiles_are_counted_separately(self):
        workload = plans(2)
        missing = [(workload[0], workload[0]["download_intervals"]),
                   (workload[1], workload[1]["download_intervals"][:12])]
        result = batch_progress.snapshot(workload, missing, [], [], table_mode=False)
        self.assertEqual(result["half_month_periods"], {"total": 48, "verified_local": 12, "pending": 36})
        self.assertEqual(result["tiles"]["complete"], 0)
        self.assertEqual(result["tiles"]["daily_summaries_only"], 2)

    def test_successful_cloud_jobs_are_not_counted_as_local_completion(self):
        workload = plans(1)
        items = [batch_export.work_item("pixel_table", workload[0], interval) for interval in workload[0]["download_intervals"]]
        ledger = [{"items": items, "state": "SUCCEEDED", "collected": False}]
        result = batch_progress.snapshot(workload, [(workload[0], workload[0]["download_intervals"])], [], ledger)
        self.assertEqual(result["tiles"]["complete"], 0)
        self.assertEqual(result["jobs"]["ready_to_download"], 1)
        self.assertEqual(result["half_month_periods"]["verified_local"], 0)

    def test_latest_attempt_supersedes_old_failures_without_double_counting(self):
        workload = plans(2)
        items = [batch_export.work_item("daily", plan) for plan in workload]
        ledger = [{"items": items, "state": "FAILED"},
                  {"items": items, "state": "RUNNING"},
                  {"items": items[:1], "state": "PENDING"}]
        result = batch_progress.snapshot(workload, [], workload, ledger)
        self.assertEqual(result["jobs"]["failed"], 0)
        self.assertEqual(result["jobs"]["running"], 1)
        self.assertEqual(result["jobs"]["queued"], 1)
        self.assertEqual(result["historical_attempts"]["failed_or_cancelled"], 1)

    def test_missing_results_from_collected_job_need_submission_again(self):
        workload = plans(1)
        item = batch_export.work_item("daily", workload[0])
        result = batch_progress.snapshot(workload, [], workload,
            [{"items": [item], "state": "SUCCEEDED", "collected": True}])
        self.assertEqual(result["jobs"]["work_items_not_submitted"], 1)
        self.assertEqual(result["jobs"]["ready_to_download"], 0)

    def test_unknown_tasks_are_not_assumed_queued(self):
        workload = plans(1)
        item = batch_export.work_item("daily", workload[0])
        result = batch_progress.snapshot(workload, [], workload, [{"items": [item], "state": "UNKNOWN"}])
        self.assertEqual(result["jobs"]["unknown"], 1)
        self.assertEqual(result["jobs"]["queued"], 0)

    def test_status_command_only_reads_cloud_state(self):
        import importlib.util
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        from crop_pipeline.common import save_json
        from test_daily_sampling_mask import mask_plan
        module_path = Path(__file__).resolve().parents[1] / "tools/benchmark_pixel_tables.py"
        spec = importlib.util.spec_from_file_location("benchmark_status_test", module_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = mask_plan(root)
            plan["fragment"]["properties"]["fragment_id"] = "region"
            inputs = {key: str(root / key) for key in
                      ("field_folder", "report_folder", "imagery_folder", "timeline_folder")}
            save_json(Path(inputs["field_folder"]) / "sampled_download_regions.geojson",
                      {"features": [plan["fragment"]]})
            save_json(Path(inputs["report_folder"]) / "acquisition_inventory/region.json", plan)
            ee = Mock()
            ee.data.listOperations.return_value = []
            with patch.object(runner, "DEST", root), \
                 patch.object(runner.batch, "submit_daily", side_effect=AssertionError("no submission")), \
                 patch.object(runner.batch, "submit_composites", side_effect=AssertionError("no submission")):
                result = runner.show_status(ee, inputs, "project")
            self.assertEqual(result["tiles"]["complete"], 0)
            self.assertEqual(result["half_month_periods"]["pending"], 24)
            self.assertEqual(json.loads((root / "progress.json").read_text())["tiles"]["total"], 1)
            ee.data.computeFeatures.assert_not_called()
            ee.data.deleteAsset.assert_not_called()
            ee.data.cancelTask.assert_not_called()
            ee.data.listOperations.assert_called_once()
