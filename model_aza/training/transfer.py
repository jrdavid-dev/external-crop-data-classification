"""Exploratory transfer experiments on development parcels only.

These diagnostics ask whether models learn crops or dataset signatures and try
source-robust alternatives. They never read test features, change stage pointers
or replace frozen models. Outputs go to ``pipeline_outputs/transfer/``.
"""

from collections import Counter, defaultdict
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from crop_pipeline.features import experiments
from crop_pipeline.training import earth_engine, exploratory, geographic_validation, source_holdout
from model_aza.training import tuning

AREA = "transfer"
CROPS = source_holdout.CROPS
FULL_COHORTS = ("fixed_full", "aligned_full")  # Matched cohorts are too small to tune.
BINS = 8
VARIABLES = ("B3", "B4", "B5", "B8", "B11", "NDVI", "NDRE", "NDMI", "SAVI")
BANDS = VARIABLES[:5]
INDICES = VARIABLES[5:]
SAVI_ADJUSTMENT = float(experiments.FEATURE_POLICY["savi_soil_adjustment"])
# Variants derived from the 8 x 9 interval series can be augmented; column variants cannot.
SERIES_VARIANTS = ("interval_72", "indices_32", "standardized_72", "indices_standardized_32")
COLUMN_VARIANTS = ("shape_48", "indices_32_plus_shape_48")
VARIANTS = SERIES_VARIANTS + COLUMN_VARIANTS


def interval_columns():
    return [f"bin_{b:02d}_{v}" for b in range(1, BINS + 1) for v in VARIABLES]


def series(rows):
    """Return interval predictors as an array shaped (rows, bins, variables)."""
    if interval_columns() != list(experiments.feature_sets()["interval_only"]):
        raise ValueError("Interval predictor order differs from the saved model recipe")
    return exploratory.feature_matrix(rows, interval_columns()).reshape(len(rows), BINS, len(VARIABLES))


def _standardize(values):
    centred = values - values.mean(axis=1, keepdims=True)
    return centred / (values.std(axis=1, keepdims=True) + 1e-6)


def from_series(values, variant):
    index_part = values[:, :, len(BANDS):]
    if variant == "interval_72":
        result = values
    elif variant == "indices_32":
        result = index_part
    elif variant == "standardized_72":
        result = _standardize(values)
    elif variant == "indices_standardized_32":
        result = _standardize(index_part)
    else:
        raise ValueError(f"Not a series variant: {variant}")
    return result.reshape(len(values), -1)


def matrix(rows, variant):
    if variant in SERIES_VARIANTS:
        return from_series(series(rows), variant)
    shape = list(experiments.feature_sets()["shape_only"])
    if variant == "shape_48":
        return exploratory.feature_matrix(rows, shape)
    if variant == "indices_32_plus_shape_48":
        return np.hstack([from_series(series(rows), "indices_32"), exploratory.feature_matrix(rows, shape)])
    raise ValueError(f"Unknown feature variant: {variant}")


def recompute_indices(values):
    """Recompute indices from (possibly perturbed) bands, matching the pipeline formulas."""
    b4, b5, b8, b11 = (values[:, :, BANDS.index(name)] for name in ("B4", "B5", "B8", "B11"))
    with np.errstate(divide="ignore", invalid="ignore"):
        indices = [(b8 - b4) / (b8 + b4), (b8 - b5) / (b8 + b5), (b8 - b11) / (b8 + b11),
                   (1 + SAVI_ADJUSTMENT) * (b8 - b4) / (b8 + b4 + SAVI_ADJUSTMENT)]
    result = values.copy()
    for offset, index in enumerate(indices):
        result[:, :, len(BANDS) + offset] = index
    return result


def shift_bins(values, shift):
    """Move the seasonal curve by whole half-month bins, repeating the edge bin."""
    if shift == 0:
        return values.copy()
    order = np.clip(np.arange(BINS) - shift, 0, BINS - 1)
    return values[:, order, :]


def scale_bands(values, rng, spread):
    """Multiply each band by one random gain per row, then recompute indices."""
    gains = rng.uniform(1 - spread, 1 + spread, size=(len(values), 1, len(BANDS)))
    scaled = values.copy()
    scaled[:, :, :len(BANDS)] *= gains
    return recompute_indices(scaled)


def augment(values, labels, weights, *, shifts=(), band_spread=0., copies=1, seed=0):
    """Training-only copies; each original row keeps its total weight across its copies."""
    rng = np.random.default_rng(seed)
    blocks = [values]
    for shift in shifts:
        blocks.append(shift_bins(values, shift))
    if band_spread:
        blocks.extend(scale_bands(values, rng, band_spread) for _ in range(copies))
    stacked = np.concatenate(blocks)
    finite = np.isfinite(stacked).all(axis=(1, 2))
    repeat = len(blocks)
    labels = np.tile(np.asarray(labels), repeat)
    weights = np.tile(np.asarray(weights, dtype=float) / repeat, repeat)
    return stacked[finite], labels[finite], weights[finite]


def forest(params, seed=42):
    return RandomForestClassifier(random_state=seed, n_jobs=-1,
                                  **{k: params[k] for k in tuning.PARAMETERS})


class NotApplicable(ValueError):
    """The requested training option cannot be fitted or applied to these rows."""


WEIGHTINGS = ("crop_parcel", "crop_source_parcel")


def training_weights(rows, weighting="crop_parcel"):
    """``crop_parcel`` is the pipeline rule; ``crop_source_parcel`` also equalizes sources within crop."""
    if weighting == "crop_parcel":
        return exploratory.normalized_training_weights(rows, tuning.CONFIDENCE_WEIGHTS)
    if weighting != "crop_source_parcel":
        raise ValueError(f"Unknown weighting: {weighting}")
    confidence = np.asarray([float(tuning.CONFIDENCE_WEIGHTS[row["label_status"]]) for row in rows])
    parcel_confidence = Counter()
    parcels_by_cell = defaultdict(set)
    sources_by_crop = defaultdict(set)
    for row, value in zip(rows, confidence):
        parcel_confidence[row["partition_group"]] += value
        parcels_by_cell[row["crop"], row["source_dataset"]].add(row["partition_group"])
        sources_by_crop[row["crop"]].add(row["source_dataset"])
    weights = np.asarray([
        value / parcel_confidence[row["partition_group"]]
        / len(parcels_by_cell[row["crop"], row["source_dataset"]]) / len(sources_by_crop[row["crop"]])
        for row, value in zip(rows, confidence)])
    return weights * (len(rows) / weights.sum())


def _parcel_mean_series(values, rows):
    """Mean series per (source, crop), averaging parcel means so large parcels do not dominate."""
    by_parcel = defaultdict(list)
    for index, row in enumerate(rows):
        by_parcel[row["source_dataset"], row["crop"], row["partition_group"]].append(index)
    by_cell = defaultdict(list)
    for (source, crop, _), indices in by_parcel.items():
        by_cell[source, crop].append(values[indices].mean(axis=0))
    return {key: np.mean(means, axis=0) for key, means in by_cell.items()}


def anchor_offsets(rows, reference=None, values=None):
    """Additive per-source offsets estimated from crops that sources share (e.g. potato).

    The reference source gets zero. Each linked source's offset is the mean
    difference of its shared-crop curves from the already calibrated curves.
    Sources that share no crop with a calibrated source get no offset.
    """
    values = series(rows) if values is None else values
    means = _parcel_mean_series(values, rows)
    crops_by_source = defaultdict(set)
    for source, crop in means:
        crops_by_source[source].add(crop)
    if reference is None:
        reference = max(crops_by_source, key=lambda s: (len(crops_by_source[s]),
                                                         sum(r["source_dataset"] == s for r in rows), s))
    if reference not in crops_by_source:
        raise NotApplicable(f"Reference source {reference} is absent")
    offsets = {reference: np.zeros(values.shape[1:])}
    queue = [reference]
    while queue:
        calibrated = queue.pop(0)
        for source in sorted(crops_by_source):
            shared = crops_by_source[source] & crops_by_source[calibrated]
            if source in offsets or not shared:
                continue
            offsets[source] = np.mean([means[source, crop] - (means[calibrated, crop] - offsets[calibrated])
                                       for crop in sorted(shared)], axis=0)
            queue.append(source)
    return offsets


def calibrate(values, rows, offsets):
    missing = {row["source_dataset"] for row in rows} - set(offsets)
    if missing:
        raise NotApplicable(f"No anchor offset for source(s): {sorted(missing)}")
    return values - np.stack([offsets[row["source_dataset"]] for row in rows])


def variant_columns(variant):
    if variant in ("interval_72", "standardized_72"):
        return interval_columns()
    if variant in ("indices_32", "indices_standardized_32"):
        return [f"bin_{b:02d}_{v}" for b in range(1, BINS + 1) for v in INDICES]
    shape = list(experiments.feature_sets()["shape_only"])
    if variant == "shape_48":
        return shape
    if variant == "indices_32_plus_shape_48":
        return variant_columns("indices_32") + shape
    raise ValueError(f"Unknown feature variant: {variant}")


def screening_scores(features, crops, sources, params, seed=42):
    """Per predictor: normalized importance for telling sources apart within crop, minus for crops."""
    crops, sources = np.asarray(crops), np.asarray(sources)
    crop_model = forest(params, seed).fit(features, crops)
    crop_importance = crop_model.feature_importances_ / crop_model.feature_importances_.sum()
    source_importance = []
    for crop in sorted(set(crops)):
        mask = crops == crop
        if len(set(sources[mask])) > 1:
            model = forest(params, seed).fit(features[mask], sources[mask])
            source_importance.append(model.feature_importances_ / model.feature_importances_.sum())
    if not source_importance:
        return None
    return np.mean(source_importance, axis=0) - crop_importance


def _training_features(train, variant, augmentation, seed, training, labels, weights):
    offsets = None
    if training.get("calibration") is not None:
        if variant not in SERIES_VARIANTS:
            raise ValueError("Anchor calibration needs a variant derived from the interval series")
        offsets = anchor_offsets(train, training["calibration"].get("reference"))
    if augmentation or offsets is not None:
        if variant not in SERIES_VARIANTS:
            raise ValueError("Augmentation needs a variant derived from the interval series")
        values = series(train)
        if offsets is not None:
            values = calibrate(values, train, offsets)
        unaugmented = from_series(values, variant)
        if augmentation:
            values, labels, weights = augment(values, labels, weights, seed=seed, **augmentation)
        features = from_series(values, variant)
    else:
        features = unaugmented = matrix(train, variant)
    return features, unaugmented, labels, weights, offsets


def fit(train, params, variant="interval_72", augmentation=None, seed=42, training=None):
    """Fit one classifier. ``training`` options are fitted on ``train`` only:

    ``weighting`` (see WEIGHTINGS), ``calibration`` ({"reference": source or None}),
    ``screen_drop`` (number of source-identifying predictors to drop) and
    ``method`` ("forest" or "pairwise").
    """
    training = training or {}
    if training.get("method", "forest") == "pairwise":
        return PairwiseForest().fit(train, params, variant, augmentation, seed,
                                    {k: v for k, v in training.items() if k != "method"})
    labels = [row["crop"] for row in train]
    weights = training_weights(train, training.get("weighting", "crop_parcel"))
    features, unaugmented, labels, weights, offsets = _training_features(
        train, variant, augmentation, seed, training, labels, weights)
    keep = None
    drop = int(training.get("screen_drop", 0))
    if drop:
        score = screening_scores(unaugmented, [r["crop"] for r in train], [r["source_dataset"] for r in train],
                                 {**params, "n_estimators": min(params["n_estimators"], 300)}, seed)
        if score is not None:
            dropped = set(np.argsort(score)[::-1][:drop].tolist())
            keep = np.asarray([i for i in range(features.shape[1]) if i not in dropped])
    model = forest(params, seed)
    model.fit(features if keep is None else features[:, keep], labels, sample_weight=weights)
    model.source_offsets_ = offsets
    model.keep_ = keep
    return model


class PairwiseForest:
    """Binary forests trained within single sources, combined into crop probabilities.

    Each source holding two or more crops contributes one forest per crop pair.
    Pairwise log-odds are combined by least squares on crop log-strengths
    (Bradley-Terry); the pairs must connect every crop.
    """

    source_offsets_ = None
    keep_ = None

    def fit(self, train, params, variant, augmentation=None, seed=42, training=None):
        training = training or {}
        if training.get("calibration") is not None or training.get("screen_drop"):
            raise ValueError("Pairwise models do not combine with calibration or screening")
        crops_by_source = defaultdict(set)
        for row in train:
            crops_by_source[row["source_dataset"]].add(row["crop"])
        self.classes_ = np.asarray(CROPS)
        self.models = []
        for source in sorted(crops_by_source):
            present = sorted(crops_by_source[source], key=CROPS.index)
            for i, first in enumerate(present):
                for second in present[i + 1:]:
                    rows = [r for r in train if r["source_dataset"] == source and r["crop"] in (first, second)]
                    model = fit(rows, params, variant, augmentation, seed,
                                {"weighting": training.get("weighting", "crop_parcel")})
                    self.models.append((first, second, source, model))
        pairs = {(a, b) for a, b, _, _ in self.models}
        linked, frontier = {CROPS[0]}, [CROPS[0]]
        while frontier:
            crop = frontier.pop()
            for a, b in pairs:
                for x, y in ((a, b), (b, a)):
                    if x == crop and y not in linked:
                        linked.add(y)
                        frontier.append(y)
        if linked != set(CROPS):
            raise NotApplicable("Within-source crop pairs do not connect all crops")
        return self

    def predict_proba(self, features):
        pair_names = sorted({(a, b) for a, b, _, _ in self.models}, key=lambda p: (CROPS.index(p[0]), CROPS.index(p[1])))
        design = np.zeros((len(pair_names), len(CROPS)))
        logits = np.zeros((len(features), len(pair_names)))
        for column, (a, b) in enumerate(pair_names):
            design[column, CROPS.index(a)], design[column, CROPS.index(b)] = 1, -1
            probabilities = [model.predict_proba(features)[:, list(model.classes_).index(a)]
                             for first, second, _, model in self.models if (first, second) == (a, b)]
            p = np.clip(np.mean(probabilities, axis=0), 1e-3, 1 - 1e-3)
            logits[:, column] = np.log(p / (1 - p))
        strengths = logits @ np.linalg.pinv(design).T
        strengths -= strengths.max(axis=1, keepdims=True)
        exp = np.exp(strengths)
        return exp / exp.sum(axis=1, keepdims=True)


def evaluation_features(model, rows, variant):
    """Apply the fitted model's own calibration and predictor screening to new rows."""
    offsets, keep = getattr(model, "source_offsets_", None), getattr(model, "keep_", None)
    if offsets is not None:
        features = from_series(calibrate(series(rows), rows, offsets), variant)
    else:
        features = matrix(rows, variant)
    return features if keep is None else features[:, keep]


def parcel_predictions(model, rows, variant):
    """Average pixel probabilities per parcel, as in the frozen-model scoring."""
    grouped, labels = defaultdict(list), {}
    for row, probabilities in zip(rows, model.predict_proba(evaluation_features(model, rows, variant))):
        labels[row["partition_group"]] = row["crop"]
        grouped[row["partition_group"]].append(probabilities)
    return [{"partition_group": group, "crop": labels[group],
             "prediction": str(model.classes_[np.argmax(np.mean(values, axis=0))])}
            for group, values in sorted(grouped.items())]


def scores(predictions, classes=CROPS):
    with warnings.catch_warnings():
        # Leave-one-source-out folds can hold a single crop; per-crop scores stay defined.
        warnings.filterwarnings("ignore", message="A single label was found")
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        metrics = earth_engine.score_predictions(predictions, [r["prediction"] for r in predictions], list(classes))
    present = {r["crop"] for r in predictions}
    metrics["macro_f1_present_crops"] = float(np.mean(
        [metrics["per_crop"][crop]["f1"] for crop in classes if crop in present])) if present else None
    return metrics


def spatial_cv(train, params, variant="interval_72", augmentation=None, seed=42, training=None):
    """Five site-cell folds inside the locked training split; returns pooled parcel metrics."""
    predictions, folds = [], []
    for fit_ids, held_ids in geographic_validation.spatial_folds(train):
        fold_train = [r for r in train if r["partition_group"] in fit_ids]
        held = [r for r in train if r["partition_group"] in held_ids]
        if {r["crop"] for r in fold_train} != set(CROPS):
            raise ValueError("A spatial training fold lacks a crop")
        fold = parcel_predictions(fit(fold_train, params, variant, augmentation, seed, training), held, variant)
        predictions.extend(fold)
        folds.append(scores(fold)["macro_f1_present_crops"])
    pooled = scores(predictions)
    pooled["fold_macro_f1"] = folds
    return pooled


def locked_validation(train, validation, params, variant="interval_72", augmentation=None, seed=42, training=None):
    return scores(parcel_predictions(fit(train, params, variant, augmentation, seed, training), validation, variant))


def parcel_sources(rows):
    return {row["partition_group"]: (row["crop"], row["source_dataset"]) for row in rows}


def confounding_table(rows):
    """Parcel counts by crop and development source."""
    counts = Counter(parcel_sources(rows).values())
    sources = sorted({source for _, source in counts})
    return [{"crop": crop, **{source: counts[crop, source] for source in sources},
             "sources_with_parcels": sum(counts[crop, source] > 0 for source in sources)} for crop in CROPS]


def source_predictability(rows, variant="interval_72", params=None, seed=42, folds=5, offsets=None):
    """Within each crop with two or more sources, how well do features predict the source?

    Parcel-grouped CV; balanced accuracy near ``1 / sources`` means little dataset signature.
    """
    params = params or {**tuning.BASELINE, "n_estimators": 300}
    results = []
    for crop in CROPS:
        crop_rows = [row for row in rows if row["crop"] == crop]
        groups = [row["partition_group"] for row in crop_rows]
        sources = np.asarray([row["source_dataset"] for row in crop_rows])
        parcel_counts = Counter(parcel_sources(crop_rows).values())
        if len(set(sources)) < 2 or min(parcel_counts.values()) < folds:
            results.append({"crop": crop, "variant": variant, "sources": len(set(sources)),
                            "status": "fewer_than_two_sources_with_enough_parcels"})
            continue
        features = (matrix(crop_rows, variant) if offsets is None
                    else from_series(calibrate(series(crop_rows), crop_rows, offsets), variant))
        predicted = np.empty(len(crop_rows), dtype=object)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit_index, held_index in splitter.split(features, sources, groups):
            model = forest(params, seed)
            model.fit(features[fit_index], sources[fit_index])
            predicted[held_index] = model.predict(features[held_index])
        parcel_truth, parcel_votes = {}, defaultdict(Counter)
        for group, truth, guess in zip(groups, sources, predicted):
            parcel_truth[group] = truth
            parcel_votes[group][guess] += 1
        truth = [parcel_truth[g] for g in sorted(parcel_truth)]
        guess = [parcel_votes[g].most_common(1)[0][0] for g in sorted(parcel_truth)]
        results.append({"crop": crop, "variant": variant, "sources": len(set(sources)), "status": "scored",
                        "parcels": len(truth), "chance_balanced_accuracy": 1 / len(set(sources)),
                        "source_balanced_accuracy": float(balanced_accuracy_score(truth, guess))})
    return results


def leave_one_source_out(rows, params, variant="interval_72", augmentation=None, seed=42, training=None):
    """Train on the other development sources; score crops shared with the held-out source."""
    results = []
    for held_source in sorted({row["source_dataset"] for row in rows}):
        held = [row for row in rows if row["source_dataset"] == held_source]
        train = [row for row in rows if row["source_dataset"] != held_source]
        trained = {row["crop"] for row in train}
        shared = sorted({row["crop"] for row in held} & trained, key=CROPS.index)
        unseen = sorted({row["crop"] for row in held} - trained, key=CROPS.index)
        record = {"held_out_source": held_source, "variant": variant,
                  "trained_crops": ",".join(sorted(trained, key=CROPS.index)),
                  "scored_crops": ",".join(shared), "unseen_crops": ",".join(unseen)}
        if len(trained) < 2 or not shared:
            results.append({**record, "status": "not_scorable"})
            continue
        try:
            model = fit(train, params, variant, augmentation, seed, training)
            predictions = parcel_predictions(model, [r for r in held if r["crop"] in shared], variant)
        except NotApplicable as error:
            results.append({**record, "status": "not_scorable", "reason": str(error)})
            continue
        metrics = scores(predictions)
        record.update({"status": "scored", "parcels": len(predictions),
                       "macro_f1_scored_crops": metrics["macro_f1_present_crops"]})
        for crop in shared:
            record[f"{crop}_recall"] = metrics["per_crop"][crop]["recall"]
        results.append(record)
    return results


def crop_transfer_holdout(rows, params, variant="interval_72", augmentation=None, seed=42, training=None):
    """Hold out one crop of one source; that source's other crops stay in training.

    Scores whether a crop learned from other sources is recognised in this
    source. The source's remaining crops keep it linked for anchor calibration
    (e.g. Sen4AgriNet potato anchors Sen4AgriNet when its carrot is held out).
    Held-out parcels hold one crop, so recall is the score.
    """
    results = []
    crops_by_source = defaultdict(set)
    for row in rows:
        crops_by_source[row["source_dataset"]].add(row["crop"])
    for source in sorted(crops_by_source):
        for crop in sorted(crops_by_source[source], key=CROPS.index):
            others = [s for s in crops_by_source if s != source and crop in crops_by_source[s]]
            record = {"held_out_source": source, "held_out_crop": crop, "variant": variant,
                      "crop_trained_from": ",".join(sorted(others)),
                      "source_crops_kept": ",".join(sorted(crops_by_source[source] - {crop}, key=CROPS.index))}
            if not others or len(crops_by_source[source]) < 2:
                results.append({**record, "status": "not_scorable",
                                "reason": "crop has no other source or source has no other crop"})
                continue
            held = [r for r in rows if r["source_dataset"] == source and r["crop"] == crop]
            train = [r for r in rows if not (r["source_dataset"] == source and r["crop"] == crop)]
            try:
                predictions = parcel_predictions(fit(train, params, variant, augmentation, seed, training), held, variant)
            except NotApplicable as error:
                results.append({**record, "status": "not_scorable", "reason": str(error)})
                continue
            results.append({**record, "status": "scored", "parcels": len(predictions),
                            "recall": float(np.mean([r["prediction"] == crop for r in predictions])),
                            **{f"predicted_{c}": sum(r["prediction"] == c for r in predictions) for c in CROPS}})
    return results


def development_rows(data, cohort):
    """Both locked development partitions; used only by source-level diagnostics."""
    train, validation = tuning.cohort_split(data, cohort)
    return train + validation


def evaluate_configuration(data, cohort, params, variant="interval_72", augmentation=None,
                           training=None, seed=42, label=None):
    """Score one configuration on development data: spatial CV, locked validation,
    leave-one-source-out and per-crop cross-source holdout. Returns (summary, details)."""
    train, validation = tuning.cohort_split(data, cohort)
    rows = train + validation
    summary = {"cohort": cohort, "configuration": label, "variant": variant,
               "augmentation": str(augmentation), "training": str(training or {})}
    try:
        pooled = spatial_cv(train, params, variant, augmentation, seed, training)
        locked = locked_validation(train, validation, params, variant, augmentation, seed, training)
    except NotApplicable as error:
        return {**summary, "status": "not_applicable", "reason": str(error)}, {"source": [], "crop": []}
    holdout = leave_one_source_out(rows, params, variant, augmentation, seed, training)
    crop_holdout = crop_transfer_holdout(rows, params, variant, augmentation, seed, training)
    scored = [r for r in holdout if r["status"] == "scored"]
    crop_scored = [r for r in crop_holdout if r["status"] == "scored"]
    summary.update({
        "status": "scored", "spatial_macro_f1": pooled["macro_f1"], "fold_min": min(pooled["fold_macro_f1"]),
        "fold_max": max(pooled["fold_macro_f1"]), "locked_validation_macro_f1": locked["macro_f1"],
        "source_holdout_macro_f1_mean": float(np.mean([r["macro_f1_scored_crops"] for r in scored])) if scored else None,
        "source_holdouts_scored": len(scored),
        "crop_holdout_recall_mean": float(np.mean([r["recall"] for r in crop_scored])) if crop_scored else None,
        "crop_holdouts_scored": len(crop_scored),
        **{f"spatial_{crop}_f1": v["f1"] for crop, v in pooled["per_crop"].items()},
    })
    tag = {"cohort": cohort, "configuration": label}
    return summary, {"source": [{**tag, **r} for r in holdout], "crop": [{**tag, **r} for r in crop_holdout]}
