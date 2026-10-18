"""Keep larger collection budgets honest and imagery behind timing selection."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from crop_pipeline.imagery import bilinear as collection
from crop_pipeline.imagery.retrieval import SatelliteRetriever


class BilinearCollectionTests(unittest.TestCase):
    def test_capacity_preserves_field_cap_and_requested_budget(self):
        catalog = [
            {
                "crop": "carrot",
                "partition_group": "field",
                "source_crs": "EPSG:32631",
                "x": index * 10,
                "y": 100,
                "source_fragments": ["fragment"],
            }
            for index in range(150)
        ]
        report = collection.capacity_report(catalog, {"carrot": 11000})[0]
        self.assertEqual(report["available_unique_pixels"], 150)
        self.assertEqual(report["available_capped_pixels"], 100)
        self.assertEqual(report["shortfall"], 10900)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "More reference fields required"):
                collection.prepare_candidates(catalog, {"carrot": 11000}, folder)
            self.assertTrue((Path(folder) / "candidate_capacity.json").is_file())

    def test_only_complete_qualified_windows_are_scheduled(self):
        plan = {
            "fragment_id": "field",
            "settings": {"start": "2020-01-01", "end": "2021-01-01"},
        }
        selected, problem = collection.plan_window_intervals(plan, [])
        self.assertIsNone(selected)
        self.assertEqual(problem["reason"], "no_timing_qualified_window")
        selected, problem = collection.plan_window_intervals(
            plan, [{"window_start": "2020-04-01", "window_end": "2020-08-01"}]
        )
        self.assertIsNone(problem)
        self.assertEqual(len(selected["download_intervals"]), 8)
        self.assertEqual(selected["download_intervals"][-1]["end"], "2020-08-01")
        selected, problem = collection.plan_window_intervals(
            plan, [{"window_start": "2020-11-01", "window_end": "2021-03-01"}]
        )
        self.assertIsNone(selected)
        self.assertEqual(problem["reason"], "adjacent_year_imagery_required")

    def test_cross_year_cycle_does_not_discard_complete_in_year_cycle(self):
        plan = {
            "fragment_id": "field",
            "settings": {"start": "2020-01-01", "end": "2021-01-01"},
        }
        windows = [
            {"window_start": "2020-11-01", "window_end": "2021-03-01"},
            {"window_start": "2020-04-01", "window_end": "2020-08-01"},
            {"window_start": "2019-12-01", "window_end": "2020-04-01"},
        ]
        selected, problem = collection.plan_window_intervals(plan, windows)
        self.assertEqual(len(selected["download_intervals"]), 8)
        self.assertEqual(selected["download_intervals"][0]["start"], "2020-04-01")
        self.assertEqual(selected["download_intervals"][-1]["end"], "2020-08-01")
        self.assertEqual(problem["retained_window_count"], 1)
        self.assertEqual(len(problem["excluded_windows"]), 2)
        self.assertEqual(len(windows), 3)

    def test_cycle_ending_at_request_boundary_is_complete_and_retained(self):
        plan = {
            "fragment_id": "field",
            "settings": {"start": "2020-01-01", "end": "2021-01-01"},
        }
        selected, problem = collection.plan_window_intervals(
            plan, [{"window_start": "2020-09-01", "window_end": "2021-01-01"}]
        )
        self.assertIsNone(problem)
        self.assertEqual(len(selected["download_intervals"]), 8)
        self.assertEqual(selected["download_intervals"][-1]["end"], "2021-01-01")

    def test_retriever_downloads_only_selected_calendar_bins(self):
        grid = {"width": 1, "height": 1}
        plan = {
            "fragment_id": "field",
            "settings": {"grid": grid},
            "fragment": {"properties": {"sampling_grid": grid}},
        }
        selected, _ = collection.plan_window_intervals(
            {
                **plan,
                "settings": {"grid": grid, "start": "2020-01-01", "end": "2021-01-01"},
            },
            [{"window_start": "2020-04-01", "window_end": "2020-08-01"}],
        )
        with tempfile.TemporaryDirectory() as folder:
            retriever = SatelliteRetriever(None, 2020, folder, folder)
            with patch.object(
                retriever, "download_interval", return_value={"status": "downloaded"}
            ) as download:
                retriever.download_region(selected)
                self.assertEqual(download.call_count, 8)

    def test_timing_gate_accepts_supported_growth_and_rejects_flat_series(self):
        values = [0.1, 0.1, 0.1, 0.15, 0.3, 0.5, 0.7, 0.7, 0.6, 0.4, 0.2, 0.1, 0.1]
        observations = [
            {
                "region_id": "field",
                "date": (date(2020, 5, 1) + timedelta(days=5 * index)).isoformat(),
                "source_kind": "observed_daily",
                "valid_fraction": 1.0,
                "NDVI": value,
                "NDRE": value * 0.5,
                "NDMI": value * 0.2,
            }
            for index, value in enumerate(values)
        ]
        plan = {
            "fragment_id": "field",
            "settings": {"version": 4, "start": "2020-01-01", "end": "2021-01-01"},
            "fragment": {
                "properties": {
                    "fragment_id": "field",
                    "crop": "carrot",
                    "partition_group": "field",
                    "selected_pixels": 10,
                }
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            selected = collection.select_timing_plans([plan], observations, folder)
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(selected[0]["download_intervals"]), 8)
            flat = [
                {**row, "NDVI": 0.1, "NDRE": 0.05, "NDMI": 0.02} for row in observations
            ]
            self.assertEqual(collection.select_timing_plans([plan], flat, folder), [])
            self.assertTrue((Path(folder) / "timing_gate.json").is_file())

    def test_source_notebooks_are_ready_for_collection_and_gate_downloads(self):
        for name in [
            "notebooks/01_sen4agrinet_preparation.ipynb",
            "notebooks/02_francecrops_preparation.ipynb",
        ]:
            notebook = json.loads((Path(__file__).parents[1] / name).read_text())
            source = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertIn("retrieve_daily_summaries = True", source)
            self.assertIn("download_satellite_imagery = True", source)
            self.assertIn("authenticate_now = True", source)
            self.assertLess(
                source.index("select_timing_plans("),
                source.index("download_composites(selected_plans)"),
            )
            self.assertLess(
                source.index("retrieve_observed_timelines("),
                source.index("select_timing_plans("),
            )
        self.assertEqual(
            collection.CANDIDATE_BUDGETS,
            {"carrot": 3400, "cabbage": 7000, "potato": 10000},
        )

    def test_download_budget_counts_regions_once_and_requires_above_3000(self):
        plan = {
            "fragment_id": "field",
            "fragment": {"properties": {"crop": "carrot", "selected_pixels": 3000}},
            "download_intervals": [{"name": str(index)} for index in range(24)],
        }
        self.assertEqual(
            collection.download_pixel_counts([plan], ["carrot"]), {"carrot": 3000}
        )
        with self.assertRaisesRegex(ValueError, "timing-qualified"):
            collection.require_download_pixel_budget([plan], ["carrot"])
        plan["fragment"]["properties"]["selected_pixels"] = 3001
        self.assertEqual(
            collection.require_download_pixel_budget([plan], ["carrot"]),
            {"carrot": 3001},
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            collection.download_pixel_counts([plan, plan], ["carrot"])
        with self.assertRaisesRegex(ValueError, "timing-qualified"):
            collection.require_download_pixel_budget([plan], ["carrot", "potato"])

    def test_failed_intervals_cannot_count_as_completed_pixel_downloads(self):
        plan = {
            "fragment_id": "field",
            "fragment": {"properties": {"crop": "carrot", "selected_pixels": 3001}},
            "download_intervals": [{"name": "first"}, {"name": "second"}],
        }
        summary = {
            "results": [
                {"region_id": "field", "name": "first", "status": "downloaded"},
                {"region_id": "field", "name": "second", "status": "failed"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "completion.json"
            with self.assertRaisesRegex(ValueError, "Completed image downloads"):
                collection.verify_download_completion(
                    [plan], summary, ["carrot"], report_path
                )
            self.assertEqual(
                json.loads(report_path.read_text())["downloaded_sampled_pixels"],
                {"carrot": 0},
            )
            summary["results"][1]["status"] = "reused"
            report = collection.verify_download_completion(
                [plan], summary, ["carrot"], report_path
            )
            self.assertEqual(report["downloaded_sampled_pixels"], {"carrot": 3001})
