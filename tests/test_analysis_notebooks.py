"""Fresh-kernel 04→05→06 handoffs work locally and preserve saved inputs."""

from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

from crop_pipeline import analysis_workflow
from crop_pipeline.common import save_json
from crop_pipeline.imagery import quality
from crop_pipeline.sources import adaptive
from crop_pipeline.training import exploratory, earth_engine
from crop_pipeline.windows import seasons, timing_quality
from test_analysis_workflow import fixture

REPOSITORY = Path(__file__).resolve().parents[1]


class FixtureAnalysis:
    """Only stand in for reading real imagery; exercise all handoff code unchanged."""
    def __init__(self, sources, manifests, folder, report, source, minimum, experiments):
        self.analysis_folder = Path(folder)
        self.report_folder = Path(report)
        self.season_folder = self.analysis_folder / 'seasons'
        self.feature_folder = self.season_folder / 'features'
        self.estimated_feature_folder = self.season_folder / 'estimated_features'
        self.regions = list(range(15))
        self.quality_rows = [{'status': 'quality_pass'}]
        self.configuration = {'automatic_timing_policy': dict(timing_quality.DEFAULT_TIMING_POLICY)}
        self.detector_settings = dict(quality.DEFAULT_DETECTOR)

    def audit(self):
        pass

    def select(self):
        self.feature_rows = []
        self.estimated_feature_rows = seasons.read_table(self.report_folder.parent / 'input/complete_features.csv')
        self.usable_selection_summary = [{'crop': crop, 'pixels': sum(row['crop'] == crop for row in self.estimated_feature_rows)} for crop in exploratory.CLASSES]
        seasons.write_records(self.feature_folder / 'complete_features.csv', [])
        seasons.write_records(self.estimated_feature_folder / 'complete_features.csv', self.estimated_feature_rows)
        save_json(self.analysis_folder / 'source_inputs.json', [])

    def coverage(self):
        return json.loads((self.report_folder.parent / 'input/coverage.json').read_text())


def saved_experiments(rows, configuration, output, reuse_completed=False):
    """Predict deliberately poorly without fitting anything; low scores must complete."""
    output = Path(output)
    parts = {name: [row for row in rows if row['partition'] == name] for name in ('train', 'validation', 'test')}
    exploratory.assert_partition_separation(parts)
    validation = {design: exploratory.evaluate_predictions(parts['validation'], ['carrot'] * len(parts['validation'])) for design in ('interval_only', 'interval_and_shape', 'shape_only')}
    selections = earth_engine.select_feature_designs({'external_parcel_holdout': validation})
    models = {}
    from crop_pipeline.features.experiments import feature_sets
    for design, columns in feature_sets().items():
        path = output / design / 'model.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic-model-for-copy-only-' + design.encode())
        models[design] = {'sha256': seasons.file_sha256(path), 'feature_columns': columns}
    metrics = {'purpose': configuration['purpose'], 'split_summary': exploratory.split_summary(rows),
               'validation': validation, 'test': {design: exploratory.evaluate_predictions(parts['test'], ['carrot'] * len(parts['test'])) for design in validation},
               'feature_selection': selections, 'model_artifacts': models,
               'shared_test_sample_fingerprint': seasons.record_fingerprint([row['sample_id'] for row in parts['test']])}
    save_json(output / 'feature_selection.json', selections)
    save_json(output / 'experiment_metrics.json', metrics)
    return metrics, selections


def execute(name, folder, overrides=None):
    notebook = json.loads((REPOSITORY / 'notebooks' / name).read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    namespace = {}
    with chdir(folder):
        exec(''.join(cells[0]['source']), namespace)
        namespace.update(overrides or {})
        for cell in cells[1:]:
            exec(''.join(cell['source']), namespace)
    return namespace


class AnalysisNotebookTests(unittest.TestCase):
    def prepare(self, folder):
        folder = Path(folder)
        report = folder / 'reports/inventory'
        _, split, _, _, _ = fixture(report.parent / 'input')
        campaign = adaptive.root(report)
        campaign.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split, campaign / 'locked_split.json')
        save_json(campaign / 'feedback.json', {'targets_met': False, 'counts': {}, 'next_round': 2})
        save_json(folder / 'settings.local.json', {'report_root': str(report)})
        return report, campaign / 'locked_split.json'

    def test_fresh_kernels_use_exact_features_preserve_split_and_package_low_score(self):
        from crop_pipeline.features import experiments
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report, split = self.prepare(folder)
            before = split.read_bytes()
            with (
                patch('crop_pipeline.sources.pilot_sources.resolve_collection', return_value=([], 'fixture')),
                patch('crop_pipeline.sources.pilot_sources.load_sources', return_value=[]),
                patch('crop_pipeline.windows.workflow.WindowAnalysis', FixtureAnalysis),
                patch('requests.sessions.Session.request', side_effect=AssertionError('No network')),
                patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('No actual fit')),
                redirect_stdout(io.StringIO()),
            ):
                windows = execute('04_four_month_windows.ipynb', folder, {'show_examples': False})
                self.assertIsNotNone(windows['window_handoff'])
                features = execute('05_feature_engineering.ipynb', folder)
                expected, _, _, _ = experiments.load_pilot_feature_handoff(features['feature_handoff'])
                with (
                    patch.object(experiments, 'enhance_rows', side_effect=AssertionError('No feature recomputation')),
                    patch.object(exploratory, 'run_experiments', side_effect=saved_experiments) as train,
                ):
                    trained = execute('06_model_training.ipynb', folder)
            self.assertEqual(train.call_args.args[0], expected)
            self.assertEqual(trained['training_coverage']['training_status'], 'completed')
            self.assertLess(trained['comparison_metrics']['test']['shape_only']['accuracy'], .5)
            self.assertTrue(trained['selected_model_manifest'].is_file())
            self.assertEqual(len(list((trained['training_output_folder'] / 'visuals').glob('*.png'))), 3)
            recipe = json.loads((trained['selected_model_manifest'].parent / 'recipe.json').read_text())
            self.assertEqual(recipe['window_recipe']['minimum_valid_fraction'], 0.8)
            self.assertEqual(recipe['window_recipe']['missing_values'], 'exclude_no_imputation')
            self.assertEqual(split.read_bytes(), before)
            self.assertFalse(json.loads((features['feature_handoff'].parent / 'configuration.json').read_text())['frozen'])

    def test_inadequate_crop_coverage_invalidates_old_handoff_without_moving_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report, split = self.prepare(folder)
            before = split.read_bytes()
            inputs = report.parent / 'input/complete_features.csv'
            rows = [row for row in seasons.read_table(inputs) if row['crop'] != 'carrot']
            seasons.write_records(inputs, rows)
            workflow = report.parent / 'analysis/fixture/seasons/workflow/pilot'
            old = folder / 'old_handoff.json'
            old.write_text('{}')
            analysis_workflow.write_stage_pointer(workflow / 'window_stage.json', old)
            with (
                patch('crop_pipeline.sources.pilot_sources.resolve_collection', return_value=([], 'fixture')),
                patch('crop_pipeline.sources.pilot_sources.load_sources', return_value=[]),
                patch('crop_pipeline.windows.workflow.WindowAnalysis', FixtureAnalysis),
                redirect_stdout(io.StringIO()),
            ):
                result = execute('04_four_month_windows.ipynb', folder, {'show_examples': False})
                self.assertIsNone(result['window_handoff'])
                with self.assertRaisesRegex(RuntimeError, 'pending'):
                    execute('05_feature_engineering.ipynb', folder)
            self.assertEqual(split.read_bytes(), before)
            self.assertEqual(old.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
