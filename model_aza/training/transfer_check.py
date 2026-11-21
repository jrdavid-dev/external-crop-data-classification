"""Exploratory check: current frozen models against the combined transfer configuration.

The combined configuration (settings, predictors and training augmentation) was
chosen in transfer notebooks 01-04 on development data only. This module refits
it on the locked training split, then scores both on locked validation, the
reserved JECAM test parcels and Paoay. Nothing here selects on test or Paoay
results, writes stage pointers or replaces frozen models.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from crop_pipeline.paoay import aligned, predictions
from crop_pipeline.training import earth_engine, notebook_workflow, source_holdout
from model_aza.training import transfer, tuning

COHORTS = transfer.FULL_COHORTS


def current_models(repository):
    """The verified frozen packages from the current Training 02 run."""
    root = Path(repository).resolve()
    handoff, _ = notebook_workflow._verify(
        notebook_workflow._input(root, None, "training", "models"), "models_stage")
    index = json.loads((handoff.parent / "models.json").read_text())
    models = {}
    for cohort in COHORTS:
        manifest = Path(index[cohort]["path"])
        classifier, recipe = predictions.load_classifier(manifest)
        if recipe["feature_columns"] != transfer.interval_columns():
            raise ValueError(f"{cohort}: frozen model does not use the 72 interval predictors")
        models[cohort] = {"model": classifier, "variant": "interval_72", "manifest": str(manifest)}
    return handoff, models


def combined_settings(choice):
    """Per cohort: refit settings from the notebook 04 ``selected.json``."""
    return {cohort: {"params": {**choice[cohort]["params"], "n_estimators": 1000},
                     "variant": choice[cohort]["series_variant"],
                     "augmentation": choice[cohort]["augmentation_settings"]} for cohort in COHORTS}


def fit_combined(development, cohort, setting, seed=42):
    train, _ = tuning.cohort_split(development, cohort)
    return transfer.fit(train, setting["params"], setting["variant"], setting["augmentation"], seed)


def test_rows(test, cohort, models, scope="jecam"):
    """Reserved test parcels for one cohort, checked against development parcel groups."""
    rows = source_holdout.checked_rows(test["rows"][cohort], True)
    rows = [row for row in rows if row.get("test_cohort", "jecam") == scope]
    development_groups = set()
    for entry in models:
        manifest = json.loads(Path(entry).read_text())
        development_groups.update(manifest["development_groups"])
    if development_groups & {row["partition_group"] for row in rows}:
        raise ValueError("Test parcel overlaps a development group")
    return rows


def majority_class(train):
    return Counter({row["partition_group"]: row["crop"] for row in train}.values()).most_common(1)[0][0]


def score_rows(model, rows, variant):
    parcels = transfer.parcel_predictions(model, rows, variant)
    return transfer.scores(parcels), parcels


def paired_bootstrap(first, second, repeats=2000, seed=0):
    """Macro-F1 difference (second minus first) over resampled parcels; both lists share parcels."""
    if [r["partition_group"] for r in first] != [r["partition_group"] for r in second]:
        raise ValueError("Parcel lists must match for a paired comparison")
    rng = np.random.default_rng(seed)
    labels = np.asarray([r["crop"] for r in first])
    a = np.asarray([r["prediction"] for r in first])
    b = np.asarray([r["prediction"] for r in second])

    def macro(truth, guess):
        return float(np.mean([_f1(truth, guess, crop) for crop in transfer.CROPS]))

    differences = []
    for _ in range(repeats):
        index = rng.integers(0, len(labels), len(labels))
        differences.append(macro(labels[index], b[index]) - macro(labels[index], a[index]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": macro(labels, b) - macro(labels, a), "ci_low": float(low), "ci_high": float(high),
            "share_of_resamples_improved": float(np.mean(np.asarray(differences) > 0)), "parcels": len(labels)}


def _f1(truth, guess, crop):
    tp = np.sum((truth == crop) & (guess == crop))
    fp = np.sum((truth != crop) & (guess == crop))
    fn = np.sum((truth == crop) & (guess != crop))
    return 0. if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def metric_row(stage, cohort, configuration, metrics, baseline=None):
    row = {"stage": stage, "cohort": cohort, "configuration": configuration,
           "parcels": sum(v["support"] for v in metrics["per_crop"].values()),
           "macro_f1": metrics.get("macro_f1"), "balanced_accuracy": metrics.get("balanced_accuracy"),
           **{f"{crop}_f1": v["f1"] for crop, v in metrics["per_crop"].items()},
           **{f"{crop}_recall": v["recall"] for crop, v in metrics["per_crop"].items()}}
    if baseline is not None:
        row["majority_baseline_macro_f1"] = baseline
    return row


def baseline_macro_f1(rows, crop):
    parcels = {row["partition_group"]: row["crop"] for row in rows}
    predicted = [{"crop": label, "prediction": crop} for label in parcels.values()]
    return earth_engine.score_predictions(predicted, [crop] * len(predicted), transfer.CROPS)["macro_f1"]


# Paoay: one set of per-pixel green-up windows, classified by each aligned model.

def paoay_windows(kmz_paths, roi_name, context_names, imagery_root, years=range(2019, 2026)):
    """Reuse the exploratory aligned recipe (Paoay notebook 07) to build per-pixel windows once."""
    sample = next(iter(sorted(Path(imagery_root).glob("*/Sentinel2_CSPlus_*.tif"))))
    import rasterio
    with rasterio.open(sample) as source:
        crs = source.crs.to_string()
    roi, areas, sources = aligned.study_area(kmz_paths, roi_name, context_names, crs)
    grid, mask = aligned.grid_for(imagery_root, roi)
    completeness = aligned.year_completeness(imagery_root, list(years), grid, mask)
    year = aligned.choose_year(completeness)
    cube, _ = aligned.read_year(imagery_root, year, grid)
    rows = aligned.observations(cube, year, list(zip(*np.nonzero(mask))))
    windows, audit = aligned.qualified_windows(rows, year)
    selected, incomplete = aligned.earliest_complete(cube, windows, grid)
    return {"grid": grid, "mask": mask, "areas": areas, "sources": sources, "year": year,
            "completeness": completeness, "selected": selected, "incomplete": incomplete,
            "window_outcomes": dict(Counter(row["outcome"] for row in audit))}


def paoay_classify(model, variant, windows):
    columns = transfer.variant_columns(variant)
    results = aligned.classify(model, {"feature_columns": columns}, windows["selected"])
    codes, onset, top = aligned.code_raster(windows["mask"], results, windows["incomplete"])
    return results, codes, top


def paoay_status(codes, mask):
    names = {code: name for name, code in aligned.CODES.items()}
    counts = Counter(codes[mask].tolist())
    total = int(mask.sum())
    return {names[code]: counts.get(code, 0) / total for code in sorted(names)}


def paoay_agreement(first, second):
    """Cross-tab of final labels for pixels predicted by both models."""
    keys = sorted(set(first) & set(second))
    table = Counter((first[k]["label"], second[k]["label"]) for k in keys)
    same = sum(count for (a, b), count in table.items() if a == b)
    return {"pixels": len(keys), "same_label_share": same / len(keys) if keys else None,
            "table": [{"current": a, "combined": b, "pixels": n} for (a, b), n in sorted(table.items())]}
