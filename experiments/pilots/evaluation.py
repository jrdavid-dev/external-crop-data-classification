"""One evaluation protocol for every pilot: random split, spatial CV, dataset holdout, JECAM.

Rows are pixel metadata dictionaries; ``X`` is their predictor matrix. Pixel
probabilities are averaged per parcel before scoring, as in the main pipeline.
Training weights follow the pipeline (crops equal, then parcels, confidence).

``copies`` optionally holds augmented predictor matrices aligned with ``X``
(for example from seasonally shifted series). They are added for training rows
only, inside every view; held-out rows are always scored on ``X`` itself. Each
original row keeps its total weight across its copies.
"""

from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.training import geographic_validation
from experiments.transfer import transfer

CROPS = transfer.CROPS
PARAMS = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2, "max_samples": .5, "max_depth": None}


def fit(X, rows, params=PARAMS, seed=42, weights=None):
    """Random Forest with pipeline weights; rows given zero weight are left out entirely."""
    weights = transfer.training_weights(rows) if weights is None else np.asarray(weights, dtype=float)
    keep = weights > 0
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    model.fit(X[keep], [r["crop"] for r, k in zip(rows, keep) if k], sample_weight=weights[keep])
    return model


def fit_rows(X, rows, index, params=PARAMS, seed=42, weights=None, copies=None):
    """Fit on rows ``index`` of ``X``, adding the same rows of each augmented copy."""
    selected = [rows[i] for i in index]
    base = transfer.training_weights(selected) if weights is None else np.asarray(weights, dtype=float)
    blocks = [X[index]] + [copy[index] for copy in (copies or [])]
    repeat = len(blocks)
    return fit(np.vstack(blocks), selected * repeat, params, seed, np.tile(base / repeat, repeat))


def parcel_predictions(model, X, rows):
    grouped, labels = defaultdict(list), {}
    for row, probabilities in zip(rows, model.predict_proba(X)):
        labels[row["partition_group"]] = row["crop"]
        grouped[row["partition_group"]].append(probabilities)
    return [{"partition_group": g, "crop": labels[g], "prediction": str(model.classes_[np.argmax(np.mean(p, axis=0))])}
            for g, p in sorted(grouped.items())]


def _index(rows, ok, test):
    return np.asarray([i for i, (row, good) in enumerate(zip(rows, ok)) if good and test(row)], dtype=int)


def _score(model, X, rows, index):
    return transfer.scores(parcel_predictions(model, X[index], [rows[i] for i in index]))


def split(X, rows, ok, partition):
    index = _index(rows, ok, lambda r: r["partition"] == partition)
    return X[index], [rows[i] for i in index]


def random_split(X, rows, ok, params=PARAMS, seed=42, copies=None):
    """The locked train/validation split: same sources and regions on both sides."""
    train = _index(rows, ok, lambda r: r["partition"] == "train")
    validation = _index(rows, ok, lambda r: r["partition"] == "validation")
    return _score(fit_rows(X, rows, train, params, seed, copies=copies), X, rows, validation)


def spatial_cv(X, rows, ok, params=PARAMS, seed=42, copies=None):
    """Five site-cell folds inside the training partition."""
    train = _index(rows, ok, lambda r: r["partition"] == "train")
    train_rows = [rows[i] for i in train]
    predictions, folds = [], []
    for fit_ids, held_ids in geographic_validation.spatial_folds(train_rows):
        fit_index = np.asarray([i for i in train if rows[i]["partition_group"] in fit_ids], dtype=int)
        held_index = np.asarray([i for i in train if rows[i]["partition_group"] in held_ids], dtype=int)
        if {rows[i]["crop"] for i in fit_index} != set(CROPS):
            raise ValueError("A spatial training fold lacks a crop")
        model = fit_rows(X, rows, fit_index, params, seed, copies=copies)
        fold = parcel_predictions(model, X[held_index], [rows[i] for i in held_index])
        predictions.extend(fold)
        folds.append(transfer.scores(fold)["macro_f1_present_crops"])
    pooled = transfer.scores(predictions)
    pooled["fold_macro_f1"] = folds
    return pooled


def dataset_holdout(X, rows, ok, params=PARAMS, seed=42, weights_for=None, copies=None):
    """Train on the other development sources; score crops they share with the held-out one."""
    development = _index(rows, ok, lambda r: r["partition"] in ("train", "validation"))
    results = []
    for source in sorted({rows[i]["source_dataset"] for i in development}):
        fit_index = np.asarray([i for i in development if rows[i]["source_dataset"] != source], dtype=int)
        trained = {rows[i]["crop"] for i in fit_index}
        shared = trained & {rows[i]["crop"] for i in development if rows[i]["source_dataset"] == source}
        if len(trained) < 2 or not shared:
            results.append({"held_out_source": source, "status": "not_scorable"})
            continue
        held_index = np.asarray([i for i in development if rows[i]["source_dataset"] == source
                                 and rows[i]["crop"] in shared], dtype=int)
        weights = None if weights_for is None else weights_for([rows[i] for i in fit_index],
                                                               [rows[i] for i in held_index])
        metrics = _score(fit_rows(X, rows, fit_index, params, seed, weights, copies), X, rows, held_index)
        results.append({"held_out_source": source, "status": "scored",
                        "parcels": len({rows[i]["partition_group"] for i in held_index}),
                        "macro_f1_scored_crops": metrics["macro_f1_present_crops"],
                        **{f"{c}_recall": metrics["per_crop"][c]["recall"] for c in CROPS if c in shared}})
    return results


def external_test(X, rows, ok, params=PARAMS, seed=42, weights_for=None, copies=None):
    """Fit on the training partition, score the primary JECAM test parcels."""
    train = _index(rows, ok, lambda r: r["partition"] == "train")
    test = _index(rows, ok, lambda r: r["partition"] == "test")
    weights = None if weights_for is None else weights_for([rows[i] for i in train], [rows[i] for i in test])
    return _score(fit_rows(X, rows, train, params, seed, weights, copies), X, rows, test)


def coverage(rows, ok):
    parcels = defaultdict(set)
    for row, good in zip(rows, ok):
        if good:
            parcels[row["partition"], row["crop"]].add(row["partition_group"])
    return {f"{p}_{c}_parcels": len(parcels[p, c]) for p in ("train", "validation", "test") for c in CROPS}


def evaluate(name, X, rows, ok, params=PARAMS, seed=42, copies=None):
    """All four views for one feature set; returns a summary row and holdout details."""
    random = random_split(X, rows, ok, params, seed, copies)
    spatial = spatial_cv(X, rows, ok, params, seed, copies)
    holdout = dataset_holdout(X, rows, ok, params, seed, copies=copies)
    test = external_test(X, rows, ok, params, seed, copies=copies)
    scored = [r for r in holdout if r["status"] == "scored"]
    summary = {"feature_set": name, "predictors": X.shape[1], "pixels_used": int(ok.sum()),
               "random_split_macro_f1": random["macro_f1"], "spatial_cv_macro_f1": spatial["macro_f1"],
               "dataset_holdout_macro_f1": float(np.mean([r["macro_f1_scored_crops"] for r in scored])) if scored else None,
               "jecam_macro_f1": test.get("macro_f1"),
               **{f"jecam_{c}_f1": test["per_crop"][c]["f1"] for c in CROPS},
               **coverage(rows, ok)}
    return summary, [{"feature_set": name, **r} for r in holdout]


def forward_group_selection(X, rows, ok, groups, params=PARAMS, seed=42, progress=print):
    """Spatial forward feature selection (Meyer et al. 2018) over predictor groups.

    Starts from the best pair of groups by spatial CV macro-F1, then adds the
    group that most improves it, stopping when no group improves the score.
    """
    names = sorted(groups)

    def score(selected):
        columns = [c for name in selected for c in groups[name]]
        return spatial_cv(X[:, columns], rows, ok, params, seed)["macro_f1"]

    history = []
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pair_scores = {pair: score(pair) for pair in pairs}
    selected = list(max(pair_scores, key=pair_scores.get))
    best = pair_scores[tuple(selected)]
    history.append({"step": 1, "selected": ",".join(selected), "spatial_cv_macro_f1": best})
    if progress:
        progress(f"start {selected} {best:.3f}")
    while len(selected) < len(names):
        trials = {name: score(selected + [name]) for name in names if name not in selected}
        name = max(trials, key=trials.get)
        if trials[name] <= best:
            break
        selected.append(name)
        best = trials[name]
        history.append({"step": len(history) + 1, "selected": ",".join(selected), "spatial_cv_macro_f1": best})
        if progress:
            progress(f"add {name} {best:.3f}")
    return selected, history
