"""Background collection must return pending without invoking live retrieval."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from crop_pipeline.sources import annual_batch


class AnnualBatchTests(unittest.TestCase):
    def test_pending_returns_without_direct_collection_and_refreshes_before_submit(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = {"fragment_id": "tile", "settings": {}, "acquisitions": [], "download_intervals": []}
            source = {"source": "francecrops", "revision": "abc", "run_folder": folder}
            inputs = {key: folder for key in ("imagery_folder", "report_folder", "timeline_folder")}
            retriever = Mock()
            retriever.inventory_region.return_value = plan
            with patch.object(annual_batch, "AcceleratedRetriever", return_value=retriever), \
                 patch.object(annual_batch.annual, "read_regions", return_value=[{}]), \
                 patch.object(annual_batch.annual, "collect_year") as direct, \
                 patch.object(annual_batch.batch, "collect") as collect, \
                 patch.object(annual_batch.batch, "missing_daily_checkpoints", return_value=[plan]), \
                 patch.object(annual_batch.batch, "submit_daily") as submit_daily, \
                 patch.object(annual_batch.batch, "submit_composites") as submit_images, \
                 patch.object(annual_batch.batch, "pending_composites", return_value=[(plan, [{"name": "2020_01_01_15"}])]):
                order = Mock()
                order.attach_mock(collect, "collect")
                order.attach_mock(submit_daily, "daily")
                order.attach_mock(submit_images, "images")
                result = annual_batch.collect_batch_year(Mock(), source, 2020, inputs, project="test")
            direct.assert_not_called()
            self.assertEqual(result["completed_tiles"], 0)
            self.assertEqual(result["pending_images"], 1)
            self.assertEqual([c[0] for c in order.mock_calls], ["collect", "daily", "images"])
            self.assertEqual(submit_daily.call_args.args[5], "projects/test/assets/crop_pipeline_batch/francecrops_abc_2020")

    def test_complete_batch_uses_existing_handoff_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            source = {"source": "francecrops", "revision": "abc", "run_folder": folder}
            inputs = {key: folder for key in ("imagery_folder", "report_folder", "timeline_folder")}
            with patch.object(annual_batch, "AcceleratedRetriever"), \
                 patch.object(annual_batch.annual, "read_regions", return_value=[]), \
                 patch.object(annual_batch.annual, "collect_year", return_value={"handoff": "verified"}) as direct, \
                 patch.object(annual_batch.batch, "collect"), \
                 patch.object(annual_batch.batch, "submit_composites"), \
                 patch.object(annual_batch.batch, "pending_composites", return_value=[]), \
                 patch.object(annual_batch.batch, "remove_temp_folder"):
                result = annual_batch.collect_batch_year(Mock(), source, 2020, inputs, project="test")
            self.assertEqual(result["handoff"], "verified")
            direct.assert_called_once()

    def test_slice_inventories_only_selected_tiles_and_stays_incomplete(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            regions = [{"properties": {"fragment_id": str(i)}} for i in range(7)]
            source = {"source": "francecrops", "revision": "abc", "run_folder": folder}
            inputs = {key: folder for key in ("imagery_folder", "report_folder", "timeline_folder")}
            retriever = Mock()
            retriever.inventory_region.side_effect = lambda r: {"fragment_id": r["properties"]["fragment_id"]}
            with patch.object(annual_batch, "AcceleratedRetriever", return_value=retriever), \
                 patch.object(annual_batch.annual, "read_regions", return_value=regions), \
                 patch.object(annual_batch.annual, "collect_year") as full, \
                 patch.object(annual_batch.batch, "collect"), \
                 patch.object(annual_batch.batch, "missing_daily_checkpoints", return_value=[]), \
                 patch.object(annual_batch.batch, "submit_composites"), \
                 patch.object(annual_batch.batch, "pending_composites", return_value=[]), \
                 patch.object(annual_batch.batch, "remove_temp_folder"):
                result = annual_batch.collect_batch_year(Mock(), source, 2020, inputs,
                                                         project="test", tile_start=5, tile_count=5)
            self.assertEqual([c.args[0] for c in retriever.inventory_region.call_args_list], regions[5:])
            self.assertEqual(result["selected_batch"]["status"], "completed")
            self.assertEqual(result["selected_batch"]["tile_end"], 7)
            self.assertEqual(result["planned_tiles"], 7)
            self.assertTrue(result["retry_queue"])
            full.assert_not_called()
            self.assertEqual(json.loads((Path(folder)/"batch/2020_coverage.json").read_text()), ["5", "6"])

    def test_active_overlapping_range_is_rejected_before_inventory(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder)/"batch/2020.json"
            ledger.parent.mkdir()
            ledger.write_text(json.dumps([{"state": "RUNNING", "request_fingerprints": {"tile": "hash"}}]))
            inputs = {key: folder for key in ("imagery_folder", "report_folder", "timeline_folder")}
            with patch.object(annual_batch, "AcceleratedRetriever") as retriever, \
                 patch.object(annual_batch.annual, "read_regions", return_value=[{"properties": {"fragment_id": "tile"}}]):
                with self.assertRaisesRegex(ValueError, "Overlapping"):
                    annual_batch.collect_batch_year(Mock(), {"run_folder": folder}, 2020, inputs,
                                                   project="test", tile_count=5000)
                retriever.return_value.inventory_region.assert_not_called()

    def test_last_completed_slice_triggers_full_source_verification(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            coverage = Path(folder)/"batch/2020_coverage.json"
            coverage.parent.mkdir()
            coverage.write_text('["0"]')
            regions = [{"properties": {"fragment_id": str(i)}} for i in range(2)]
            source = {"source": "francecrops", "revision": "abc", "run_folder": folder}
            inputs = {key: folder for key in ("imagery_folder", "report_folder", "timeline_folder")}
            retriever = Mock()
            retriever.inventory_region.return_value = {"fragment_id": "1"}
            with patch.object(annual_batch, "AcceleratedRetriever", return_value=retriever), \
                 patch.object(annual_batch.annual, "read_regions", return_value=regions), \
                 patch.object(annual_batch.annual, "collect_year", return_value={"handoff": "verified"}) as full, \
                 patch.object(annual_batch.batch, "collect"), \
                 patch.object(annual_batch.batch, "missing_daily_checkpoints", return_value=[]), \
                 patch.object(annual_batch.batch, "submit_composites"), \
                 patch.object(annual_batch.batch, "pending_composites", return_value=[]), \
                 patch.object(annual_batch.batch, "remove_temp_folder"):
                result = annual_batch.collect_batch_year(Mock(), source, 2020, inputs,
                                                         project="test", tile_start=1, tile_count=1)
            self.assertEqual(result["handoff"], "verified")
            full.assert_called_once()
            self.assertEqual(json.loads(coverage.read_text()), ["0", "1"])
