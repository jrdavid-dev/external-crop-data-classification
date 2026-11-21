"""Pure summaries behind the walkthrough figures (no file access, no plotting)."""

import numpy as np
import pandas as pd

from model_aza.pilots import data, features

BINS = data.BINS
HALF_MONTHS = [f"{m} {h}" for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
               for h in ("1–15", "16–end")]
SHORT_HALF_MONTHS = [f"{m:02d}{h}" for m in range(1, 13) for h in ("a", "b")]

# Temporal correspondence codes
UNAVAILABLE, UNCERTAIN, IN_SEASON, OFF_SEASON = 0, 1, 2, 3
CORRESPONDENCE_LABELS = {
    IN_SEASON: "Confirmed: clear and inside the detected crop cycle",
    UNCERTAIN: "Uncertain: clear, but no crop cycle visible that year",
    UNAVAILABLE: "Unavailable: no clear observation",
    OFF_SEASON: "Clear, outside the cycle (bare or other cover)",
}


def ndvi(values):
    return data.indices(values)[..., 0]


def parcel_series(rows, values, valid, variable="NDVI"):
    """Median clear-sky series per parcel group: (groups, series [g, 24], clear [g, 24])."""
    if variable in ("NDVI", "NDRE", "NDMI", "SAVI"):
        series = data.indices(values)[..., ("NDVI", "NDRE", "NDMI", "SAVI").index(variable)]
    else:
        series = values[..., data.BANDS.index(variable)]
    series = np.where(valid & np.isfinite(series), series, np.nan)
    members = {}
    for index, row in enumerate(rows):
        members.setdefault(row["partition_group"], []).append(index)
    order = sorted(members)
    out = np.full((len(order), BINS), np.nan)
    for position, group in enumerate(order):
        block = series[members[group]]
        clear_bins = np.isfinite(block).any(axis=0)
        with np.errstate(all="ignore"):
            out[position, clear_bins] = np.nanmedian(block[:, clear_bins], axis=0)
    return order, out, np.isfinite(out)


def growth_season(series, clear, relative=0.5, minimum_amplitude=0.15):
    """The main crop cycle per parcel: (season [n, 24] bool, has_cycle [n] bool).

    The cycle is the contiguous run of half-months around the NDVI peak where clear NDVI stays at or above
    min + ``relative`` x (max - min); cloudy half-months inside the run do not break it. A parcel whose
    clear NDVI never rises by ``minimum_amplitude`` (or has fewer than 3 clear values) shows no cycle.
    """
    season = np.zeros(series.shape, dtype=bool)
    has_cycle = np.zeros(len(series), dtype=bool)
    for i in range(len(series)):
        values = series[i][clear[i]]
        if values.size < 3 or values.max() - values.min() < minimum_amplitude:
            continue
        threshold = values.min() + relative * (values.max() - values.min())
        peak = int(np.nanargmax(np.where(clear[i], series[i], -np.inf)))
        start = end = peak
        while start > 0 and (not clear[i, start - 1] or series[i, start - 1] >= threshold):
            start -= 1
        while end < series.shape[1] - 1 and (not clear[i, end + 1] or series[i, end + 1] >= threshold):
            end += 1
        season[i, start:end + 1] = True
        has_cycle[i] = True
    return season, has_cycle


def temporal_correspondence(series, clear, **season):
    """Per parcel and half-month: confirmed (clear, in the cycle), uncertain (clear, parcel has no cycle),
    off-season (clear, outside the cycle) or unavailable (no clear observation)."""
    cycle, has_cycle = growth_season(series, clear, **season)
    status = np.full(series.shape, UNAVAILABLE)
    status[clear & ~has_cycle[:, None]] = UNCERTAIN
    status[clear & has_cycle[:, None] & cycle] = IN_SEASON
    status[clear & has_cycle[:, None] & ~cycle] = OFF_SEASON
    return status


def valid_share(frame, valid, by):
    """Share of sampled pixels with a clear observation per half-month, for each group in ``by``."""
    shares = pd.concat([frame[list(by)].reset_index(drop=True),
                        pd.DataFrame(valid.astype(float), columns=SHORT_HALF_MONTHS)], axis=1)
    return shares.groupby(list(by))[SHORT_HALF_MONTHS].mean()


def per_crop_scores(truth, prediction, crops):
    """Precision, recall, F1 and support per crop; precision is NaN when a crop is never predicted."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    records = []
    for crop in crops:
        tp = int(np.sum((truth == crop) & (prediction == crop)))
        predicted = int(np.sum(prediction == crop))
        actual = int(np.sum(truth == crop))
        precision = tp / predicted if predicted else np.nan
        recall = tp / actual if actual else np.nan
        f1 = 2 * tp / (predicted + actual) if predicted + actual else np.nan
        records.append({"crop": crop, "precision": precision, "recall": recall, "f1": f1, "support": actual})
    return pd.DataFrame(records)


def confusion(truth, prediction, labels, columns=None):
    """Counts with true labels as rows and predicted labels (``columns``, default ``labels``) as columns."""
    columns = list(columns or labels)
    table = pd.crosstab(pd.Categorical(truth, categories=labels), pd.Categorical(prediction, categories=columns),
                        dropna=False)
    table.index.name, table.columns.name = "reference", "predicted"
    return table


def harmonic_curve(series, clear, omega=1.0, order=2, samples=240):
    """The two-harmonic least-squares fit the improved pipeline uses, evaluated on a fine time axis."""
    coefficients, ok = features.fit_harmonics(series[None, :], clear[None, :], order=order, omega=omega)
    t = np.linspace(0, 1, samples)
    design = np.column_stack([np.ones(samples)] + [f(2 * np.pi * omega * k * t) for k in range(1, order + 1)
                                                   for f in (np.cos, np.sin)])
    with np.errstate(all="ignore"):  # macOS Accelerate can warn spuriously on small finite products
        curve = design @ coefficients[0]
    return t * BINS - .5, curve, bool(ok[0]), coefficients[0]


def feature_family(column):
    """Group a feature column by source variable, e.g. 'NDVI_amp1' -> 'NDVI', 'VHVV_mean' -> 'VH/VV ratio'."""
    variable = column.split("_")[0]
    return {"VHVV": "VH/VV ratio", "CROSSPOL": "Cross-pol ratio"}.get(variable, variable)


def present_crop_f1(truth, prediction, crops):
    """Macro-F1 over the crops present in ``truth`` (a single-crop population reports that crop's F1)."""
    scores = per_crop_scores(truth, prediction, crops)
    present = scores[scores.support > 0]
    return float(np.nan_to_num(present.f1).mean()) if len(present) else np.nan


def population_records(model_name, predictions, totals, crops):
    """One record per test population and the combined pool: F1 over present crops and parcels scored."""
    records = []
    for population, total in totals.items():
        chosen = predictions if population == "combined" else predictions[predictions.test_cohort == population]
        records.append({"model": model_name, "population": population, "scored": len(chosen), "total": int(total),
                        "f1": present_crop_f1(chosen.crop, chosen.prediction, crops) if len(chosen) else np.nan})
    return records


def grouped_permutation_importance(model, X, truth, columns, repeats=5, seed=0):
    """Mean and SD drop in pixel macro-F1 when all columns of one variable family are shuffled together."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    labels = list(model.classes_)
    baseline = f1_score(truth, model.predict(X), labels=labels, average="macro")
    families = {}
    for index, column in enumerate(columns):
        families.setdefault(feature_family(column), []).append(index)
    records = {}
    for family, indices in families.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[:, indices] = X[order][:, indices]
            drops.append(baseline - f1_score(truth, model.predict(shuffled), labels=labels, average="macro"))
        records[family] = {"mean": float(np.mean(drops)), "sd": float(np.std(drops)), "columns": len(indices)}
    return pd.DataFrame(records).T.astype({"mean": float, "sd": float, "columns": int}), float(baseline)


IMPROVED_CODES = {1: "carrot", 2: "cabbage", 3: "potato", 4: "Unknown crop type (low confidence)",
                  5: "No prediction possible"}
MAIN_ALIGNED_CODES = {1: "carrot", 2: "cabbage", 3: "potato", 4: "Unknown crop type (low confidence)",
                      5: "No prediction possible", 6: "No prediction possible"}


def class_names(codes, mapping, cropland=None):
    """Class-name raster from map codes; pixels outside active cropland become 'Not active cropland'."""
    names = np.full(codes.shape, "", dtype=object)
    for code, name in mapping.items():
        names[codes == code] = name
    if cropland is not None:
        names[(codes > 0) & ~cropland] = "Not active cropland"
    return names


def sample_per_group(labels, keys, size, seed="walkthrough"):
    """Indices of up to ``size`` items per label, chosen by the pipeline's stable rank of ``keys`` (reproducible)."""
    from crop_pipeline.sources import sampling

    chosen = {}
    for index, (label, key) in enumerate(zip(labels, keys)):
        chosen.setdefault(label, []).append((sampling.stable_rank(f"{seed}|{key}"), index))
    return np.asarray(sorted(i for members in chosen.values() for _, i in sorted(members)[:size]), dtype=int)


def confident_crop_shares(names, crops):
    """Share of each crop among pixels with a confident crop label (Unknown and no-data excluded)."""
    labelled = np.isin(names, list(crops))
    total = labelled.sum()
    return {crop: float((names == crop).sum() / total) if total else np.nan for crop in crops}, int(total)
