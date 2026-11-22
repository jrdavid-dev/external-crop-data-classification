"""Exploratory Random Forest hyperparameter tuning on development parcels only.

Tuning reads the development feature export (training and validation
partitions). It never reads test features, never changes stage pointers and
never replaces frozen models. Each search writes a new folder under
``pipeline_outputs/tuning/``.
"""

from itertools import product
import json
from pathlib import Path
import time
from uuid import uuid4

from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.common import save_json
from crop_pipeline.training import exploratory, geographic_validation, notebook_workflow, source_holdout
from crop_pipeline.windows import seasons

COHORTS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")
CONFIDENCE_WEIGHTS = {"confirmed": 1., "reported": .5}
# Settings used by Training 01; min_samples_leaf is selected there from (2, 4).
BASELINE = {"n_estimators": 1000, "max_features": "sqrt", "min_samples_leaf": 2,
            "max_samples": .5, "max_depth": None}
DEFAULT_SPACE = {
    "max_features": ["sqrt", "log2", .2, .4],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_samples": [.3, .5, .8, None],
    "max_depth": [None, 12],
}
PARAMETERS = tuple(BASELINE)


def columns():
    return source_holdout.experiments.feature_sets()[source_holdout.DESIGN]


def load_development(repository, development_path=None):
    """Return the checked development export and its path; test exports are rejected."""
    path = notebook_workflow._features(repository, development_path)
    data = json.loads(Path(path).read_text())
    notebook_workflow._check_development(data)
    return Path(path), data


def cohort_split(data, cohort):
    rows = source_holdout.checked_rows(data["rows"][cohort], False)
    train = [row for row in rows if row["partition"] == "train"]
    validation = [row for row in rows if row["partition"] == "validation"]
    if {r["partition_group"] for r in train} & {r["partition_group"] for r in validation}:
        raise ValueError("Development parcels cross the training/validation split")
    return train, validation


def parameter_grid(space=None, n_estimators=500):
    space = DEFAULT_SPACE if space is None else space
    unknown = set(space) - set(PARAMETERS)
    if unknown:
        raise ValueError(f"Unsupported tuning parameters: {sorted(unknown)}")
    names = sorted(space)
    grid = [{"n_estimators": n_estimators, **dict(zip(names, values))}
            for values in product(*(space[name] for name in names))]
    return [{**BASELINE, **params} for params in grid]


def fit(train, params, seed=42):
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **{k: params[k] for k in PARAMETERS})
    weights = exploratory.normalized_training_weights(train, CONFIDENCE_WEIGHTS)
    model.fit(exploratory.feature_matrix(train, columns()), [row["crop"] for row in train],
              sample_weight=weights)
    return model


def score(model, rows):
    metrics, _ = source_holdout.parcel_metrics(model, rows, columns())
    return metrics


def _record(cohort, params, metrics, seconds, **extra):
    record = {"cohort": cohort, **{k: params[k] for k in PARAMETERS}, **extra,
              "macro_f1": metrics.get("macro_f1"),
              "balanced_accuracy": metrics.get("balanced_accuracy"), "fit_seconds": round(seconds, 2)}
    for crop, values in metrics["per_crop"].items():
        record[f"{crop}_f1"] = values["f1"]
    return record


def validation_search(data, grid, cohorts=COHORTS, seed=42, progress=print, every=16):
    """Fit each setting on training parcels; score validation parcels (macro-F1)."""
    records = []
    for cohort in cohorts:
        train, validation = cohort_split(data, cohort)
        for number, params in enumerate(grid, 1):
            started = time.perf_counter()
            metrics = score(fit(train, params, seed), validation)
            records.append(_record(cohort, params, metrics, time.perf_counter() - started, seed=seed))
            if progress and (number % every == 0 or number == len(grid)):
                progress(f"{cohort} {number}/{len(grid)} macro-F1 {records[-1]['macro_f1']:.3f}")
    return records


def top_settings(records, cohort, count=5):
    ranked = sorted((r for r in records if r["cohort"] == cohort and r["macro_f1"] is not None),
                    key=lambda r: (-r["macro_f1"], -r["balanced_accuracy"], r["fit_seconds"]))
    return [{k: r[k] for k in PARAMETERS} for r in ranked[:count]]


def spatial_cross_validation(data, cohort, params, seed=42):
    """Five spatial folds inside the training split; locked validation parcels are unused."""
    train, _ = cohort_split(data, cohort)
    predictions, folds = [], []
    for fit_ids, held_ids in geographic_validation.spatial_folds(train):
        fold_train = [r for r in train if r["partition_group"] in fit_ids]
        held = [r for r in train if r["partition_group"] in held_ids]
        if {r["crop"] for r in fold_train} != set(source_holdout.CROPS):
            raise ValueError(f"{cohort}: a spatial training fold lacks a crop")
        metrics, fold_predictions = source_holdout.parcel_metrics(fit(fold_train, params, seed), held, columns())
        predictions.extend(fold_predictions)
        folds.append(metrics.get("macro_f1"))
    pooled = source_holdout.earth_engine.score_predictions(
        predictions, [r["prediction"] for r in predictions], source_holdout.CROPS)
    return pooled, folds


def seed_stability(data, cohort, params, seeds=(1, 2, 3, 4, 5)):
    train, validation = cohort_split(data, cohort)
    records = []
    for seed in seeds:
        started = time.perf_counter()
        metrics = score(fit(train, params, seed), validation)
        records.append(_record(cohort, params, metrics, time.perf_counter() - started, seed=seed))
    return records


def new_run(repository, stage, development_path, area="tuning", **metadata):
    parent = Path(repository).resolve() / "pipeline_outputs" / area
    for path in (parent.parent, parent):
        if path.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {path}")
    run = parent / f"{stage}_{uuid4().hex[:12]}"
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "inputs.json", {"stage": stage, "development_features": str(development_path),
                                    "development_sha256": seasons.file_sha256(development_path),
                                    **metadata})
    return run


def save_records(run, name, records):
    seasons.write_records(run / f"{name}.csv", records)
    save_json(run / f"{name}.json", records)
    return run / f"{name}.csv"


def latest_run(repository, stage, area="tuning"):
    runs = sorted((Path(repository).resolve() / "pipeline_outputs" / area).glob(f"{stage}_*"),
                  key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No saved {stage} tuning run; run the earlier tuning notebook first")
    return runs[-1]


def load_records(run, name):
    return json.loads((Path(run) / f"{name}.json").read_text())
