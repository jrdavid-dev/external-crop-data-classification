"""Development-only transfer views shared by every candidate recipe.

Each candidate is scored on three views that never read test parcels:

- ``locked_validation``: fit on the locked training split, score the locked validation split.
- ``spatial_cv``: five site-cell folds inside the training split (``geographic_validation``).
- ``dataset_holdout``: fit on the other development sources, score the crops they share
  with the held-out source.

Fits use the main pipeline's Random Forest settings and training weights (crops equal,
then parcels equal, label confidence within a parcel). Pixel probabilities are averaged
per parcel before scoring. The selection score is the mean of spatial CV macro-F1 and
dataset-holdout macro-F1, the two views that measure transfer to new places.
"""

from collections import defaultdict
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.training import earth_engine, exploratory, geographic_validation

CROPS = exploratory.CLASSES
LABEL_WEIGHTS = {"confirmed": 1.0, "reported": 0.5}
FOREST = {"n_estimators": 1000, "max_features": "sqrt", "min_samples_leaf": 2, "max_samples": 0.5}
SELECTION_FOREST = {**FOREST, "n_estimators": 300}
VIEWS = ("locked_validation", "spatial_cv", "dataset_holdout")


def require_development(rows):
    if any(row["partition"] not in ("train", "validation") or row.get("test_cohort") for row in rows):
        raise ValueError("Development views accept train and validation rows only")


def fit(X, rows, params=SELECTION_FOREST, seed=42, estimator=None):
    """Fit with the pipeline weights; ``estimator`` (any classifier taking ``sample_weight``) replaces the forest."""
    from sklearn.base import clone

    weights = exploratory.normalized_training_weights(rows, LABEL_WEIGHTS)
    model = clone(estimator) if estimator is not None else RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    return model.fit(X, [row["crop"] for row in rows], sample_weight=weights)


def parcel_probabilities(model, X, rows):
    """``{parcel: (reference crop, mean pixel probabilities in CROPS order)}``.

    A crop the model was not fitted on (for example cabbage when FranceCrops is held out)
    gets probability zero.
    """
    fitted = list(model.classes_)
    probabilities = np.zeros((len(rows), len(CROPS)))
    probabilities[:, [CROPS.index(crop) for crop in fitted]] = model.predict_proba(X)
    grouped, labels = defaultdict(list), {}
    for row, values in zip(rows, probabilities):
        group = row["partition_group"]
        if labels.setdefault(group, row["crop"]) != row["crop"]:
            raise ValueError(f"Conflicting labels for {group}")
        grouped[group].append(values)
    return {group: (labels[group], np.mean(values, axis=0)) for group, values in grouped.items()}


def scores(parcels):
    """Parcel metrics; ``macro_f1_present_crops`` averages F1 over reference crops present."""
    rows = [{"partition_group": g, "crop": crop} for g, (crop, _) in sorted(parcels.items())]
    predictions = [CROPS[int(np.argmax(p))] for _, (_, p) in sorted(parcels.items())]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        metrics = earth_engine.score_predictions(rows, predictions, CROPS)
    present = {row["crop"] for row in rows}
    metrics["macro_f1_present_crops"] = float(np.mean([metrics["per_crop"][c]["f1"] for c in CROPS if c in present]))
    if present != set(CROPS):
        metrics["macro_f1"] = None
    return metrics


def view_probabilities(X, rows, params=SELECTION_FOREST, seed=42, estimator=None, views=VIEWS):
    """Held-out parcel probabilities per view and fold, for one predictor matrix.

    ``views`` limits the work, for example to ``("spatial_cv",)`` when tuning settings.
    """
    require_development(rows)
    X = np.asarray(X, dtype=float)
    index = lambda keep: np.asarray([i for i, row in enumerate(rows) if keep(row)], dtype=int)  # noqa: E731

    def held_out(fit_index, held_index):
        model = fit(X[fit_index], [rows[i] for i in fit_index], params, seed, estimator)
        return parcel_probabilities(model, X[held_index], [rows[i] for i in held_index])

    unknown = set(views) - set(VIEWS)
    if unknown:
        raise ValueError(f"Unknown development views: {sorted(unknown)}")
    train = index(lambda r: r["partition"] == "train")
    result = {"locked_validation": {}, "spatial_cv": {}, "dataset_holdout": {}}
    if "locked_validation" in views:
        result["locked_validation"]["validation"] = held_out(train, index(lambda r: r["partition"] == "validation"))
    train_rows = [rows[i] for i in train]
    folds = geographic_validation.spatial_folds(train_rows) if "spatial_cv" in views else ()
    for fold, (fit_ids, held_ids) in enumerate(folds):
        fit_index = np.asarray([i for i in train if rows[i]["partition_group"] in fit_ids], dtype=int)
        if {rows[i]["crop"] for i in fit_index} != set(CROPS):
            raise ValueError("A spatial training fold lacks a crop")
        held_index = np.asarray([i for i in train if rows[i]["partition_group"] in held_ids], dtype=int)
        result["spatial_cv"][f"fold_{fold + 1}"] = held_out(fit_index, held_index)
    for source in sorted({row["source_dataset"] for row in rows}) if "dataset_holdout" in views else ():
        fit_index = index(lambda r: r["source_dataset"] != source)
        trained = {rows[i]["crop"] for i in fit_index}
        shared = trained & {r["crop"] for r in rows if r["source_dataset"] == source}
        if len(trained) < 2 or not shared:
            continue
        held_index = index(lambda r: r["source_dataset"] == source and r["crop"] in shared)
        result["dataset_holdout"][source] = held_out(fit_index, held_index)
    return result


def average(*views):
    """Average parcel probabilities of several candidates over the same views, folds and parcels."""
    first = views[0]
    if any(v.keys() != first.keys() or any(v[k].keys() != first[k].keys() for k in first) for v in views):
        raise ValueError("Ensembled candidates must share views and folds")
    combined = {}
    for view, folds in first.items():
        combined[view] = {}
        for fold, parcels in folds.items():
            if any(v[view][fold].keys() != parcels.keys() for v in views):
                raise ValueError("Ensembled candidates must score the same parcels")
            combined[view][fold] = {g: (crop, np.mean([v[view][fold][g][1] for v in views], axis=0))
                                    for g, (crop, _) in parcels.items()}
    return combined


def summarize(views):
    """Scores per view and the selection score (mean of spatial CV and dataset holdout)."""
    validation = scores(views["locked_validation"]["validation"])
    pooled = {}
    for folds in views["spatial_cv"].values():
        pooled.update(folds)
    spatial = scores(pooled)
    holdout = {source: scores(parcels) for source, parcels in views["dataset_holdout"].items()}
    holdout_mean = float(np.mean([m["macro_f1_present_crops"] for m in holdout.values()])) if holdout else None
    selection = (None if spatial["macro_f1"] is None or holdout_mean is None
                 else (spatial["macro_f1"] + holdout_mean) / 2)
    return {"locked_validation_macro_f1": validation["macro_f1"], "spatial_cv_macro_f1": spatial["macro_f1"],
            "dataset_holdout_macro_f1": holdout_mean, "selection_score": selection,
            "dataset_holdout": {s: {"parcels": sum(m["per_crop"][c]["support"] for c in CROPS),
                                    "macro_f1_present_crops": m["macro_f1_present_crops"]}
                                for s, m in holdout.items()},
            "spatial_cv_per_crop_f1": {c: spatial["per_crop"][c]["f1"] for c in CROPS},
            "locked_validation_per_crop_f1": {c: validation["per_crop"][c]["f1"] for c in CROPS}}
