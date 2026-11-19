"""Separate, immutable local training and external-test notebook outputs.

Collection state is never read or written. Default inputs come from completed
preprocessing; explicit paths allow an earlier snapshot while collection grows.
"""

import csv
import json
from pathlib import Path
import shutil
from uuid import uuid4

from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.common import save_json
from crop_pipeline import preprocessing_workflow
from crop_pipeline.training import earth_engine, source_holdout
from crop_pipeline.windows import seasons


def _read(path):
    return json.loads(Path(path).read_text())


def _pointer(root, area, stage):
    return Path(root).resolve() / 'pipeline_outputs' / area / f'current_{stage}.json'


def _input(root, explicit, area, stage):
    if not explicit:
        return resolve_stage_pointer(_pointer(root, area, stage))
    path = Path(explicit).expanduser()
    return (Path(root) / path).resolve()


def _features(root, explicit, test=False):
    if explicit:
        return _input(root, explicit, 'preprocessing', 'features')
    handoff = preprocessing_workflow.verify(
        resolve_stage_pointer(_pointer(root, 'preprocessing', 'features')), 'features')
    return handoff.parent / ('features_test.json' if test else 'features_development.json')


def _new_run(root, area):
    parent = Path(root).resolve() / 'pipeline_outputs' / area
    # Never follow a redirected output root into existing archives.
    for path in (parent.parent, parent):
        if path.is_symlink():
            raise ValueError(f'Output directory must not be a symlink: {path}')
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / ('run_' + uuid4().hex[:12])
    run.mkdir(exist_ok=False)
    return run


def _seal(folder, stage, **metadata):
    receipt = folder / f'{stage}.json'
    save_json(receipt, {'kind': 'external_notebook_stage', 'stage': stage, **metadata,
                       'files': {str(p.relative_to(folder)): seasons.file_sha256(p)
                                 for p in sorted(folder.rglob('*')) if p.is_file() and p != receipt}})
    return receipt


def _verify(receipt, stage):
    receipt = Path(receipt).resolve()
    data = _read(receipt)
    if data.get('kind') != 'external_notebook_stage' or data.get('stage') != stage or not data.get('files'):
        raise ValueError(f'Expected completed {stage} handoff')
    for name, digest in data['files'].items():
        path = receipt.parent / name
        if not path.is_file() or seasons.file_sha256(path) != digest:
            raise ValueError(f'Saved {stage} artifact changed: {name}')
    return receipt, data


def _pending(root, area, stage):
    write_stage_pointer(_pointer(root, area, stage), status='pending', reason=f'{stage} attempt has not completed')


def _publish(root, area, stage, receipt):
    write_stage_pointer(_pointer(root, area, stage), receipt)
    return receipt


def _check_development(data):
    source_holdout.require_matching(data['rows'])
    groups = {}
    for rows in data['rows'].values():
        source_holdout.checked_rows(rows, False)
        for row in rows:
            identity = (row['partition'], row['crop'])
            group = row['partition_group']
            if group in groups and groups[group] != identity:
                raise ValueError('Development parcels cross partitions or crop labels across cohorts')
            groups[group] = identity


def train(repository, development_path=None):
    """Fit/tune development models only; each deliberate call creates a new run."""
    _pending(repository, 'training', 'training')
    _pending(repository, 'training', 'models')
    source = _features(repository, development_path)
    data = _read(source)
    _check_development(data)
    run = _new_run(repository, 'training')
    # Copy the loaded JSON, so a later preprocessing run cannot alter this input.
    snapshot = run / 'features_development.json'
    save_json(snapshot, data)
    index = source_holdout.freeze(snapshot, destination=run / 'validation')
    receipt = _seal(run, 'training', development_source=str(source),
                    candidate_index=str(index), implementation_sha256=seasons.file_sha256(source_holdout.__file__))
    return _publish(repository, 'training', 'training', receipt)


def freeze(repository, training_handoff=None):
    """Publish selected fitted models; no fitting and no test features are read."""
    _pending(repository, 'training', 'models')
    handoff, data = _verify(_input(repository, training_handoff, 'training', 'training'), 'training')
    run = handoff.parent
    destination = run / 'models'
    # Repeat execution reuses only a completely verified package.
    receipt = destination / 'models_stage.json'
    if destination.exists():
        _verify(receipt, 'models_stage')
        return _publish(repository, 'training', 'models', receipt)
    index = _read(data['candidate_index'])
    destination.mkdir()
    shutil.copyfile(run / 'validation/development.json', destination / 'development.json')
    published = {}
    for name, entry in index.items():
        original = Path(entry['path'])
        folder = destination / name
        shutil.copytree(original.parent, folder)
        manifest = _read(folder / 'model_manifest.json')
        manifest['feature_handoff'] = str(destination / 'development.json')
        save_json(folder / 'model_manifest.json', manifest)
        target = folder / 'model_manifest.json'
        published[name] = {'path': str(target), 'sha256': seasons.file_sha256(target)}
    save_json(destination / 'models.json', published)
    receipt = _seal(destination, 'models_stage', training_handoff=str(handoff))
    return _publish(repository, 'training', 'models', receipt)


def evaluate(repository, test_path=None, models_handoff=None):
    """Score the frozen model set once per output run; never alter training."""
    _pending(repository, 'testing', 'scores')
    _pending(repository, 'testing', 'report')
    handoff, _ = _verify(_input(repository, models_handoff, 'training', 'models'), 'models_stage')
    source = _features(repository, test_path, test=True)
    data = _read(source)
    source_holdout.require_matching(data['rows'])
    for rows in data['rows'].values():
        source_holdout.checked_rows(rows, True)
    run = _new_run(repository, 'testing')
    snapshot = run / 'features_test.json'
    save_json(snapshot, data)
    reports, predictions = run / 'reports', run / 'predictions'
    reports.mkdir()
    predictions.mkdir()
    model_index = handoff.parent / 'models.json'
    cohorts = set(data.get("test_cohorts", ()))
    groups = {}
    for rows in data['rows'].values():
        for row in rows:
            cohort = row.get("test_cohort", "jecam")
            cohorts.add(cohort)
            if groups.setdefault(row["partition_group"], cohort) != cohort:
                raise ValueError("Test parcel appears in multiple test cohorts")
    supplements = cohorts - {"jecam"}
    if any(not name.isascii() or not name.replace("_", "").isalnum() or name == "combined"
           for name in supplements):
        raise ValueError("Invalid supplemental test cohort name")
    scopes = ["jecam", *sorted(supplements)]
    if supplements:
        scopes.append("combined")
    for scope in scopes:
        report_folder = reports if scope == "jecam" else reports / scope
        prediction_folder = predictions if scope == "jecam" else predictions / scope
        report_folder.mkdir(exist_ok=True)
        prediction_folder.mkdir(exist_ok=True)
        _score_scope(snapshot, model_index, report_folder, prediction_folder, scope)
    save_json(reports / "test_scopes.json", scopes)
    receipt = _seal(run, 'scores', models_handoff=str(handoff),
                    models_handoff_sha256=seasons.file_sha256(handoff), test_source=str(source))
    return _publish(repository, 'testing', 'scores', receipt)


def _score_scope(snapshot, model_index, reports, predictions, scope):
    scores = source_holdout.score(snapshot, model_index, destination=reports, predictions_directory=predictions, scope=scope)
    baseline = {}
    for name, entry in _read(model_index).items():
        manifest = _read(entry['path'])
        path = predictions / f'{name}_test_parcels.csv'
        if not path.exists():
            continue
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        crop = manifest['training_majority_class']
        metrics = earth_engine.score_predictions(rows, [crop] * len(rows), source_holdout.CROPS)
        if 'macro_f1' not in scores[name]:
            metrics.pop('macro_f1', None)
            metrics.pop('balanced_accuracy', None)
        baseline[name] = {'training_majority_class': crop, 'metrics': metrics}
    save_json(reports / 'majority_baseline.json', baseline)


def report(repository, scores_handoff=None):
    """Summarize saved scores without selecting or refitting any model."""
    _pending(repository, 'testing', 'report')
    handoff, _ = _verify(_input(repository, scores_handoff, 'testing', 'scores'), 'scores')
    reports = handoff.parent / 'reports'
    receipt = reports / 'diagnostics/report.json'
    if receipt.exists():
        _verify(receipt, 'report')
        return _publish(repository, 'testing', 'report', receipt)
    scope_file = reports / "test_scopes.json"
    scopes = _read(scope_file) if scope_file.exists() else ["jecam"]
    per_crop, confusion = [], []
    lines = ['# External test results', '',
             'Frozen development models evaluated on the primary JECAM and available supplemental test cohorts. '
             'Predictions average pixel probabilities within each parcel. '
             'These results do not measure Paoay accuracy.', '',
             'Matched cohorts hold locations constant; full cohorts can have different coverage. '
             'The baseline predicts the most common training parcel crop. '
             'Missing crops suppress the three-class macro-F1. No test-based model selection.', '',
             '| Test population | Window cohort | Coverage status | Macro-F1 | Training-majority baseline F1 | Meets F1 criteria |',
             '| --- | --- | --- | --- | --- | --- |']
    for scope in scopes:
        folder = reports if scope == "jecam" else reports / scope
        scores = _read(folder / "test_scores.json")
        baselines = _read(folder / "majority_baseline.json")
        for name, metrics in scores.items():
            baseline = baselines.get(name, {}).get('metrics', {})
            macro, baseline_macro = metrics.get('macro_f1'), baseline.get('macro_f1')
            outcome = 'unavailable'
            if macro is not None and baseline_macro is not None:
                passes = (macro >= .5 and macro > baseline_macro
                          and all(item['f1'] >= .5 for item in metrics['per_crop'].values()))
                outcome = 'yes' if passes else 'no'
            lines.append(f"| {scope} | {name} | {metrics['status']} | {metrics.get('macro_f1', 'unavailable')} | "
                         f"{baseline.get('macro_f1', 'unavailable')} | {outcome} |")
            for crop, values in metrics.get('per_crop', {}).items():
                per_crop.append({'test_population': scope, 'cohort': name, 'crop': crop, **values, **metrics['coverage'][crop]})
            for i, actual in enumerate(metrics.get('classes', [])):
                for j, predicted in enumerate(metrics['classes']):
                    confusion.append({'test_population': scope, 'cohort': name, 'actual': actual, 'predicted': predicted,
                                      'parcels': metrics['confusion_matrix'][i][j]})
    # Diagnostics live separately from sealed scoring files.
    diagnostics = reports / 'diagnostics'
    diagnostics.mkdir(exist_ok=False)
    lines += ['', 'The F1 criteria require macro-F1 of at least 0.50, a score above the baseline, '
               'and F1 of at least 0.50 for every crop.']
    seasons.write_records(diagnostics / 'per_crop.csv', per_crop)
    seasons.write_records(diagnostics / 'confusion_matrices.csv', confusion)
    (diagnostics / 'results.md').write_text('\n'.join(lines) + '\n')
    receipt = _seal(diagnostics, 'report', scores_handoff=str(handoff))
    return _publish(repository, 'testing', 'report', receipt)
