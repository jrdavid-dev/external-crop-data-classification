"""Window orchestration preserves quality rules and remains entirely local."""

import json
import tempfile
import unittest
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch

from crop_pipeline.sources import adaptive
from crop_pipeline.windows.workflow import WindowAnalysis


class WindowAnalysisTests(unittest.TestCase):
    def make_analysis(self, folder):
        root = Path(folder)
        analysis = WindowAnalysis([], [], root / 'analysis', root / 'inventory', adaptive.COLLECTION_NAME)
        analysis.timeline_folder = root / 'analysis/timelines'
        analysis.composite_rows = []
        analysis.example_regions = set()
        analysis.regions, analysis.pixel_rows, analysis.quality_rows = [], [], []
        return analysis

    def test_empty_observation_assessment_saves_losses_without_retrieval(self):
        with tempfile.TemporaryDirectory() as folder, patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
            analysis = self.make_analysis(folder)
            analysis.select()
            self.assertEqual(analysis.feature_rows, [])
            self.assertEqual(analysis.estimated_feature_rows, [])
            self.assertEqual(analysis.configuration['minimum_valid_fraction'], 0.8)
            self.assertEqual(analysis.configuration['cycle_selection_policy'], 'earliest_complete')
            feedback = json.loads((adaptive.root(analysis.report_folder)/'feedback.json').read_text())
            self.assertFalse(feedback['targets_met'])
            coverage = analysis.coverage()
            self.assertEqual(coverage['complete_pixel_windows'], 0)
            self.assertEqual(len(analysis.usable_selection_summary), 9)

    def test_real_selection_keeps_qualified_estimates_distinct_from_reference_truth(self):
        from crop_pipeline.imagery import quality
        from crop_pipeline.windows import seasons
        from test_season_workflow import reviewed_fixture, feature_fixture
        _, _, region, configuration = reviewed_fixture()
        values = [.1, .1, .1, .15, .3, .5, .7, .7, .6, .4, .2]
        observations = [{"region_id": "region", "date": (date(2020, 10, 1) + timedelta(days=5 * index)).isoformat(),
                         "source_kind": "observed_daily", "valid_fraction": 1.0,
                         "NDVI": value, "NDRE": value * .5, "NDMI": value * .2}
                        for index, value in enumerate(values)]
        _, quality_rows, pixel_rows = feature_fixture()
        configuration.update(frozen=False, development_evidence="", detector_parameters=dict(quality.DEFAULT_DETECTOR))
        with tempfile.TemporaryDirectory() as folder:
            analysis = self.make_analysis(folder)
            analysis.source_run_name = "bilinear_expanded"
            analysis.regions = [region]
            analysis.quality_rows, analysis.pixel_rows = quality_rows, pixel_rows
            analysis.example_regions = {"region"}
            seasons.save_json(analysis.timeline_folder / "observed_timelines.json", observations)
            seasons.save_json(analysis.analysis_folder / "seasons/review/frozen_configuration.json", configuration)
            analysis.select()
            self.assertFalse(analysis.accepted_seasons)
            self.assertFalse(analysis.feature_rows)
            self.assertEqual(len(analysis.estimated_feature_rows), 2)
            saved = seasons.read_table(analysis.season_folder / "estimated_windows.csv")
            self.assertEqual(saved[0]["timing_review_status"], "automated_pass")
            self.assertEqual(saved[0]["label_relevance"], "unverified")

    def test_window_threshold_must_match_audit_and_not_relax_quality(self):
        for threshold in (0.7, 0.9):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as folder:
                analysis = self.make_analysis(folder)
                # Prepare a real empty review template using the default policy.
                analysis.select()
                config_path = analysis.configuration_path
                config = dict(analysis.configuration)
                config['minimum_valid_fraction'] = threshold
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, 'coverage threshold differs'):
                    analysis.select()


if __name__ == '__main__':
    unittest.main()
