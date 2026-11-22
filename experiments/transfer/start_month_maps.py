"""Exploratory: one fixed-window model per start month, applied to Paoay.

For each start month (January through September, as in the start-month sweep),
a Random Forest with the pipeline's baseline settings and weights is trained on
the development parcels' fixed four-month windows (train + validation, as the
frozen models). It is applied to Paoay pixels whose eight half-months of that
window are all clear in the local Cloud Score+ composites, for the year with
the most such cropland pixels. The pipeline's 0.60 / 0.10 confidence rule turns
unsure pixels into "Unknown crop type".

No test parcels are used. Nothing here changes the frozen models, the main
Paoay outputs or any stage pointer. Imagery differs from training in the same
ways as Paoay notebook 07 (Cloud Score+ composites; each half-month median
counts as one clear observation). The maps are exploratory, with no local labels.
"""

from collections import Counter
from datetime import date

import numpy as np

from crop_pipeline.paoay import aligned
from experiments.transfer import start_month_sweep as sweep, transfer
from crop_pipeline.windows import seasons

YEARS = tuple(range(2019, 2026))
CONFIDENCE = {"minimum_top_probability": .60, "minimum_margin": .10}
CODES = {"carrot": 1, "cabbage": 2, "potato": 3, "Unknown crop type": 4, "Window not cloud-free": 5}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def label(month):
    return f"{MONTH_NAMES[month - 1]}–{MONTH_NAMES[month + 2]}"


def window(year, month, region_id):
    start = date(year, month, 1)
    return {"window_start": start.isoformat(), "window_end": seasons.add_calendar_months(start, 4).isoformat(),
            "region_id": region_id}


def complete(cube, month):
    """(rows, cols) mask: all five bands clear in all eight half-months of the window."""
    first = (month - 1) * 2
    return np.all(np.isfinite(cube[first:first + 8]), axis=(0, 1))


def fit_models(inputs, months=sweep.START_MONTHS, params=None, progress=print):
    """{month: (model, development pixel rows)} for every month whose windows cover all crops."""
    from experiments.tuning import tuning
    params = params or tuning.BASELINE
    models = {}
    for month in months:
        rows, _ = sweep.month_rows(inputs, month)
        if not sweep.has_all_crops(rows):
            if progress:
                progress(f"{label(month)}: a crop has no complete windows; skipped")
            continue
        models[month] = (transfer.fit(rows, params), len(rows))
        if progress:
            progress(f"{label(month)}: trained on {len(rows)} development pixels")
    return models


def classify(model, cube, year, month, grid, pixels):
    """Labels for the given (row, col) pixels; incomplete windows are marked as not cloud-free."""
    rows, keys = [], []
    for r, c in pixels:
        features = aligned.window_features(cube, window(year, month, f"px_{r}_{c}"), int(r), int(c), grid)
        if features is not None:
            rows.append(features)
            keys.append((int(r), int(c)))
    labels = {}
    if rows:
        probabilities = model.predict_proba(transfer.evaluation_features(model, rows, "interval_72"))
        ordered = np.sort(probabilities, axis=1)
        accepted = (ordered[:, -1] >= CONFIDENCE["minimum_top_probability"]) & \
                   (ordered[:, -1] - ordered[:, -2] >= CONFIDENCE["minimum_margin"])
        names = model.classes_[np.argmax(probabilities, axis=1)]
        labels = {key: str(name) if ok else "Unknown crop type" for key, name, ok in zip(keys, names, accepted)}
    return labels


def summarize(month, year, labels, mask, cropland_mask, coverage_share):
    """One summary row: coverage and crop shares of confident cropland pixels."""
    in_cropland = Counter(v for (r, c), v in labels.items() if cropland_mask[r, c])
    confident = {crop: in_cropland.get(crop, 0) for crop in transfer.CROPS}
    total = sum(confident.values())
    return {"start_month": month, "window": label(month), "year": year,
            "cropland_pixels": int(cropland_mask.sum()), "cropland_complete_share": coverage_share,
            "cropland_classified_pixels": sum(in_cropland.values()),
            "cropland_confident_pixels": total,
            **{f"{crop}_share_of_confident": (confident[crop] / total if total else None) for crop in transfer.CROPS},
            **{f"study_{name.replace(' ', '_').lower()}": sum(1 for v in labels.values() if v == name) for name in CODES
               if name != "Window not cloud-free"},
            "study_window_not_cloud_free": int(mask.sum()) - len(labels)}


def code_raster(mask, labels):
    codes = np.zeros(mask.shape, "uint8")
    codes[mask] = CODES["Window not cloud-free"]
    for (r, c), name in labels.items():
        codes[r, c] = CODES[name]
    return codes
