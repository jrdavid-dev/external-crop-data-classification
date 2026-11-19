"""Default handoffs across the supported notebook chain, in isolated kernels.

Collection transfer uses tiny real TIFF fixtures. Imagery audit/window algorithms
are mocked here (covered separately); feature export, fit, freeze, score and
report run locally. Small forests keep this an integration test, not an experiment.
"""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline import preprocessing_workflow, share
from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.training import notebook_workflow, source_holdout
from test_jecam_transfer import completed_batch
from test_preprocessing_workflow import fake_audit, reference_rows

REPOSITORY = Path(__file__).resolve().parents[1]


def select_fixture(quality, report):
    from crop_pipeline.common import save_json
    rows = [{**r, 'window_start': '2020-05-01', 'window_end': '2020-09-01'} for r in reference_rows()]
    save_json(quality.parent/'windows.json', {'fixed': rows, 'aligned': rows,
                                            'quality': str(quality), 'fixed_start_month': 5})
    return quality.parent/'windows.json'


def execute(name, root):
    namespace = {}
    book = json.loads((REPOSITORY/'notebooks'/name).read_text())
    with chdir(root), redirect_stdout(io.StringIO()):
        for i, cell in enumerate(book['cells']):
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), f'{name}:cell{i}', 'exec'), namespace)
    return namespace


class CompleteNotebookWorkflowTests(unittest.TestCase):
    def test_import_to_report_default_handoffs_and_snapshot_rejection(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp); sender = base/'sender'; root=base/'receiver'
            report=sender/'data/pipeline_work/inventory'
            for source in ('sen4agrinet', 'francecrops', 'cropharvest'):
                completed_batch(report, source, {source: (1,1,2,2)})
            package=share.export(sender, base/'package')
            share.import_package(root, package)
            (root/'pyproject.toml').touch()
            campaign=root/'data/pipeline_work/adaptive_training'
            before={p: p.read_bytes() for p in campaign.rglob('*') if p.is_file()}
            with patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
                execute('data_collection/08_dataset_summary.ipynb', root)
                with patch.object(preprocessing_workflow.external, 'audit', side_effect=fake_audit), \
                     patch.object(preprocessing_workflow.external, 'select', side_effect=select_fixture):
                    for name in ('01_imagery_quality', '02_window_selection', '03_feature_engineering'):
                        execute(f'preprocessing/{name}.ipynb', root)
                with patch.object(source_holdout, 'RandomForestClassifier', side_effect=lambda **kw:
                                  RandomForestClassifier(**{**kw, 'n_estimators': 3, 'n_jobs': 1})):
                    trained=execute('training/01_train_and_validate.ipynb', root)['training_handoff']
                # Freeze, scoring and reporting must never fit again.
                with patch.object(RandomForestClassifier, 'fit', side_effect=AssertionError('No refit')):
                    frozen=execute('training/02_freeze_models.ipynb', root)['models_handoff']
                    scores=execute('testing/01_external_test_evaluation.ipynb', root)['scores_handoff']
                    diagnostics=execute('testing/02_results_and_diagnostics.ipynb', root)['report_handoff']
                    self.assertEqual(notebook_workflow.freeze(root, trained), frozen)
                    self.assertEqual(notebook_workflow.report(root, scores), diagnostics)
            self.assertTrue((diagnostics.parent/'per_crop.csv').is_file())
            self.assertTrue((diagnostics.parent/'confusion_matrices.csv').is_file())
            self.assertEqual(resolve_stage_pointer(root/'pipeline_outputs/testing/current_report.json'), diagnostics)
            self.assertTrue(all(p.read_bytes()==b for p,b in before.items()))
            self.assertEqual(set(before), {p for p in campaign.rglob('*') if p.is_file()})
            # An unrelated test snapshot must fail rather than score old models.
            feature_pointer=resolve_stage_pointer(preprocessing_workflow.pointer(root, 'features'))
            test=json.loads((feature_pointer.parent/'features_test.json').read_text())
            test['window_sha256']='different-snapshot'
            wrong=root/'wrong_test.json'; wrong.write_text(json.dumps(test))
            with self.assertRaisesRegex(ValueError, 'different experiment snapshot'):
                notebook_workflow.evaluate(root, wrong, frozen)
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                resolve_stage_pointer(root/'pipeline_outputs/testing/current_scores.json')
