"""Annual harmonic recipe: inputs -> development-only selection -> frozen model -> one external test.

A third model family beside the fixed and aligned window models, which stay unchanged
as the baseline. It reads the verified Preprocessing 01 quality handoff, keeps its locked
partitions and test campaigns, and uses the same Random Forest settings, training weights
and parcel probability averaging. Selection uses only development views
(``development_views``); test parcels are read once, by ``test``.

Every stage writes a new ``pipeline_outputs/annual/<stage>/run_<id>/`` folder with a sealed
receipt and publishes ``current_<stage>.json``. Testing writes predictions and scores in the
main testing layout so the external report and the paired comparison can read them.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from crop_pipeline import preprocessing_workflow
from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.common import save_json
from crop_pipeline.features import annual
from crop_pipeline.imagery import sentinel1
from crop_pipeline.training import development_views, earth_engine
from crop_pipeline.windows import seasons

AREA = "annual"
CROPS = development_views.CROPS
MODEL_NAME = "annual"
TIE_BREAK = "fewer predictors"


# Stage bookkeeping, as in notebook_workflow

def _root(repository):
    return Path(repository).resolve()


def _pointer(repository, stage):
    return _root(repository) / "pipeline_outputs" / AREA / f"current_{stage}.json"


def _new_run(repository, stage):
    parent = _root(repository) / "pipeline_outputs" / AREA / stage
    for path in (parent.parent.parent, parent.parent, parent):
        if path.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {path}")
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"run_{uuid4().hex[:12]}"
    run.mkdir(exist_ok=False)
    return run


def seal(folder, stage, **metadata):
    receipt = Path(folder) / f"{stage}_receipt.json"
    save_json(receipt, {"kind": "annual_pipeline_stage", "stage": stage, **metadata,
                        "files": {str(p.relative_to(folder)): seasons.file_sha256(p)
                                  for p in sorted(Path(folder).rglob("*")) if p.is_file() and p != receipt}})
    return receipt


def verify(receipt, stage):
    receipt = Path(receipt).resolve()
    record = json.loads(receipt.read_text())
    if record.get("kind") != "annual_pipeline_stage" or record.get("stage") != stage or not record.get("files"):
        raise ValueError(f"Expected a completed annual {stage} receipt")
    for name, digest in record["files"].items():
        path = receipt.parent / name
        if not path.is_file() or seasons.file_sha256(path) != digest:
            raise ValueError(f"Saved annual {stage} file changed: {name}")
    return receipt, record


def _start(repository, stage):
    write_stage_pointer(_pointer(repository, stage), status="pending", reason=f"{stage} attempt has not completed")


def _publish(repository, stage, receipt):
    write_stage_pointer(_pointer(repository, stage), receipt)
    return receipt


def _input(repository, explicit, stage):
    path = Path(explicit) if explicit else resolve_stage_pointer(_pointer(repository, stage))
    return verify(path if path.is_absolute() else _root(repository) / path, stage)


def save_inputs(folder, rows, arrays):
    (Path(folder) / "pixels.json").write_text(json.dumps(rows))
    np.savez_compressed(Path(folder) / "arrays.npz", **arrays)


def load_inputs(folder):
    with np.load(Path(folder) / "arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    return json.loads((Path(folder) / "pixels.json").read_text()), arrays


# Recipe registry: the pipeline's own recipes, plus any an experiment registers (``register_recipes``),
# so frozen experiment models can be tested and mapped with the same stages.
RECIPES = dict(annual.RECIPES)
# Recipes the selection stage compares by default: the reported annual recipes only.
DEFAULT_RECIPES = tuple(annual.RECIPES)
DEFAULT_CLASSIFIER = {"family": "random_forest", "setting": "original"}


def register_recipes(recipes):
    """Make experiment recipes usable by ``freeze``, ``test`` and the maps; never selection defaults."""
    clashes = {name for name in recipes if name in RECIPES and RECIPES[name] is not recipes[name]}
    if clashes:
        raise ValueError(f"Recipe names already registered: {sorted(clashes)}")
    RECIPES.update(recipes)


def recipe_matrix(name, arrays):
    X, ok, columns = RECIPES[name](arrays["s2_values"], arrays["s2_valid"], arrays["s1_values"])
    return X, ok, columns


# Pure steps

def development_index(rows, built):
    """Development pixels usable by every built recipe."""
    common = np.logical_and.reduce([ok for _, ok, _ in built.values()])
    return np.asarray([i for i, row in enumerate(rows) if common[i] and row["partition"] in ("train", "validation")])


def development_candidates(rows, arrays, recipes=None, params=None, progress=print, built=None):
    """Score each recipe with the default forest on development pixels usable by every recipe."""
    built = built or {name: recipe_matrix(name, arrays) for name in (recipes or RECIPES)}
    index = development_index(rows, built)
    development = [rows[i] for i in index]
    results = {}
    for name, (X, _, columns) in built.items():
        views = development_views.view_probabilities(X[index], development,
                                                     params or development_views.SELECTION_FOREST)
        results[name] = {"predictors": len(columns), "pixels": len(index),
                         "parcels": len({r["partition_group"] for r in development}),
                         **development_views.summarize(views)}
        if progress:
            progress(f"{name}: selection score {results[name]['selection_score']:.3f}")
    return results


def choose(results):
    """Highest selection score (rounded to 1e-6); ties go to fewer predictors."""
    return max(results, key=lambda n: (round(results[n]["selection_score"], 6), -results[n]["predictors"]))


def majority_crop(rows):
    return Counter({r["partition_group"]: r["crop"] for r in rows}.values()).most_common(1)[0][0]


def parcel_table(model, X, rows):
    """Parcel predictions with mean class probabilities and usable pixel counts."""
    parcels = development_views.parcel_probabilities(model, X, rows)
    pixels = Counter(r["partition_group"] for r in rows)
    first = {}
    for row in rows:
        first.setdefault(row["partition_group"], row)
    table = []
    for group, (crop, probabilities) in sorted(parcels.items()):
        row = first[group]
        table.append({"partition_group": group, "crop": crop, "site_cell": row["site_cell"],
                      "prediction": CROPS[int(np.argmax(probabilities))],
                      "test_cohort": row.get("test_cohort") or "", "usable_pixels": pixels[group],
                      **{f"prob_{c}": float(p) for c, p in zip(CROPS, probabilities)}})
    return table


def coverage(table):
    parcels = Counter(r["crop"] for r in table)
    pixels = Counter()
    for r in table:
        pixels[r["crop"]] += int(r["usable_pixels"])
    return {crop: {"parcels": parcels[crop], "pixels": pixels[crop],
                   "target_met": parcels[crop] >= 20 and pixels[crop] >= 400} for crop in CROPS}


def score_table(table):
    """Main-pipeline test score fields; three-class macro-F1 is absent when a crop is missing."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        metrics = earth_engine.score_predictions(table, [r["prediction"] for r in table], CROPS)
    counts = coverage(table)
    if any(not counts[crop]["parcels"] for crop in CROPS):
        metrics.pop("macro_f1")
        metrics.pop("balanced_accuracy")
    metrics["coverage"] = counts
    metrics["status"] = ("incomplete_class_coverage" if any(not c["parcels"] for c in counts.values())
                         else "targets_met" if all(c["target_met"] for c in counts.values())
                         else "pilot_below_targets")
    return metrics


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Stages

def build_inputs(repository, quality_handoff=None, ee=None, pixels_per_parcel=annual.PIXELS_PER_PARCEL,
                 progress=print):
    """Annual Sentinel-2 series and Sentinel-1 at the same pixels, for every parcel in the quality handoff."""
    _start(repository, "inputs")
    root = _root(repository)
    quality_path = preprocessing_workflow.choose_input(root, quality_handoff, "quality")
    quality = json.loads(quality_path.read_text())
    rows, s2_values, s2_valid = annual.load_series(quality["regions"], quality["pixel_paths"], pixels_per_parcel)
    shared = set(annual.shared_development_locations(rows))
    keep = np.asarray([i for i in range(len(rows)) if i not in shared], dtype=int)
    excluded = Counter((rows[i]["partition"], rows[i]["source_dataset"], rows[i]["crop"]) for i in shared)
    rows = [rows[i] for i in keep]
    s2_values, s2_valid = s2_values[keep], s2_valid[keep]
    if ee is None:
        settings = json.loads((root / "settings.local.json").read_text())
        ee = sentinel1.initialize(settings["earth_engine_project"])
    s1_values = sentinel1.sample(rows, root / "pipeline_outputs" / AREA / "cache" / "sentinel1", ee, progress=progress)
    run = _new_run(repository, "inputs")
    save_inputs(run, rows, {"s2_values": s2_values, "s2_valid": s2_valid, "s1_values": s1_values})
    receipt = seal(run, "inputs", quality_handoff=str(quality_path), quality_sha256=seasons.file_sha256(quality_path),
                   recipe_version=annual.RECIPE_VERSION, pixels_per_parcel=pixels_per_parcel, pixels=len(rows),
                   parcels=len({r["partition_group"] for r in rows}),
                   shared_development_locations_excluded=[
                       {"partition": p, "source": s, "crop": c, "pixel_rows": n}
                       for (p, s, c), n in sorted(excluded.items())])
    return _publish(repository, "inputs", receipt)


def select(repository, inputs_handoff=None, progress=print, recipes=None):
    """Choose the recipe on development views only; the classifier is the default Random Forest.

    Every recipe in ``DEFAULT_RECIPES`` (or ``recipes``) is scored with the default forest on the
    development pixels usable by every recipe; the highest selection score (mean of spatial CV and
    dataset-holdout macro-F1) wins, ties to fewer predictors. Test pixels are not read.
    """
    _start(repository, "selection")
    inputs, _ = _input(repository, inputs_handoff, "inputs")
    rows, arrays = load_inputs(inputs.parent)
    built = {name: recipe_matrix(name, arrays) for name in (recipes or DEFAULT_RECIPES)}
    results = development_candidates(rows, arrays, progress=progress, built=built)
    recipe = choose(results)
    run = _new_run(repository, "selection")
    save_json(run / "selection.json", {
        "selected": recipe, "classifier": DEFAULT_CLASSIFIER, "selection_score": results[recipe]["selection_score"],
        "rule": "highest mean of spatial CV and dataset-holdout macro-F1 with the default Random Forest; ties to "
                f"{TIE_BREAK}", "forest": development_views.SELECTION_FOREST, "candidates": results,
        "test_rows_used": False})
    receipt = seal(run, "selection", inputs_handoff=str(inputs))
    return _publish(repository, "selection", receipt)


def freeze(repository, selection_handoff=None, estimator=None):
    """Fit the selected recipe on the locked training split; report locked validation.

    The model is the default Random Forest (``development_views.FOREST``). An experiment may pass
    another ``estimator`` (any classifier taking ``sample_weight``); its record names it.
    """
    _start(repository, "models")
    selection, record = _input(repository, selection_handoff, "selection")
    chosen = json.loads((selection.parent / "selection.json").read_text())
    choice = chosen["selected"]
    classifier = chosen.get("classifier") or DEFAULT_CLASSIFIER
    if estimator is None and classifier != DEFAULT_CLASSIFIER:
        raise ValueError(f"Selection records {classifier}; pass that estimator explicitly to freeze it")
    inputs, _ = verify(record["inputs_handoff"], "inputs")
    rows, arrays = load_inputs(inputs.parent)
    X, ok, columns = recipe_matrix(choice, arrays)
    train = np.asarray([i for i, r in enumerate(rows) if ok[i] and r["partition"] == "train"], dtype=int)
    validation = np.asarray([i for i, r in enumerate(rows) if ok[i] and r["partition"] == "validation"], dtype=int)
    train_rows = [rows[i] for i in train]
    if estimator is None:
        model = development_views.fit(X[train], train_rows, development_views.FOREST)
    else:
        model = development_views.fit(X[train], train_rows, estimator=estimator)
    validation_table = parcel_table(model, X[validation], [rows[i] for i in validation])
    development = [rows[i] for i in np.r_[train, validation]]
    run = _new_run(repository, "models")
    joblib.dump({"classifier": model, "recipe": choice, "feature_columns": columns}, run / "model.joblib")
    write_table(run / "validation_parcels.csv", validation_table)
    save_json(run / "model.json", {
        "model": MODEL_NAME, "recipe": choice, "recipe_version": annual.RECIPE_VERSION,
        "classifier": {**classifier, "estimator": type(model).__name__,
                       "backend": getattr(model, "backend_", None), "params": model.get_params()},
        "label_weights": development_views.LABEL_WEIGHTS,
        "feature_columns": columns, "trained_on": "locked training split",
        "validation": score_table(validation_table), "training_majority_class": majority_crop(train_rows),
        "development_groups": sorted({r["partition_group"] for r in development}),
        "development_split": {r["partition_group"]: r["partition"] for r in development},
        "clear_bin_rule": {"minimum_clear_half_months": annual.MIN_CLEAR_BINS,
                           "maximum_gap_half_months": annual.MAX_GAP_BINS}})
    receipt = seal(run, "models", selection_handoff=str(selection), inputs_handoff=str(inputs))
    return _publish(repository, "models", receipt)


def load_model(models_receipt):
    receipt, record = verify(models_receipt, "models")
    package = joblib.load(receipt.parent / "model.joblib")
    return package, json.loads((receipt.parent / "model.json").read_text()), record


def test(repository, models_handoff=None):
    """Score the frozen model once on JECAM, each supplement and the combined pool."""
    _start(repository, "test")
    models, _ = _input(repository, models_handoff, "models")
    package, info, record = load_model(models)
    inputs, _ = verify(record["inputs_handoff"], "inputs")
    rows, arrays = load_inputs(inputs.parent)
    X, ok, columns = recipe_matrix(package["recipe"], arrays)
    if columns != package["feature_columns"]:
        raise ValueError("Frozen feature columns differ from the inputs")
    index = np.asarray([i for i, r in enumerate(rows) if ok[i] and r["partition"] == "test"], dtype=int)
    test_rows = [rows[i] for i in index]
    if set(info["development_groups"]) & {r["partition_group"] for r in test_rows}:
        raise ValueError("Test parcel overlaps a development parcel")
    campaigns = {}
    for row in test_rows:
        if campaigns.setdefault(row["partition_group"], row["test_cohort"]) != row["test_cohort"]:
            raise ValueError("A test parcel appears in more than one campaign")
    table = parcel_table(package["classifier"], X[index], test_rows)
    supplements = sorted({r["test_cohort"] for r in table} - {"jecam"})
    scopes = ["jecam", *supplements, *(["combined"] if supplements else [])]
    run = _new_run(repository, "test")
    majority = info["training_majority_class"]
    for scope in scopes:
        chosen = [r for r in table if scope == "combined" or r["test_cohort"] == scope]
        folder = "" if scope == "jecam" else scope
        write_table(run / "predictions" / folder / f"{MODEL_NAME}_test_parcels.csv", chosen)
        reports = run / "reports" / folder
        reports.mkdir(parents=True, exist_ok=True)
        save_json(reports / "test_scores.json", {MODEL_NAME: score_table(chosen)})
        baseline = score_table([{**r, "prediction": majority} for r in chosen])
        baseline.pop("coverage"), baseline.pop("status")
        save_json(reports / "majority_baseline.json",
                  {MODEL_NAME: {"training_majority_class": majority, "metrics": baseline}})
    save_json(run / "reports" / "test_scopes.json", scopes)
    receipt = seal(run, "test", models_handoff=str(models))
    return _publish(repository, "test", receipt)
