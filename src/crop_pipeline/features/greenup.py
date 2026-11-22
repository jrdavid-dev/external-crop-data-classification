"""Green-up recipes: the external study's detector applied to each pixel's annual NDVI series.

The detector (``imagery.quality.detect_greenup`` with the aligned-map settings) finds the
NDVI dip before growth and the peak after it, and places the onset where NDVI crosses 20% of
that rise. Half-month composites are dated at their bin midpoints in one reference year, since
only timing within the year is used. Two recipes build on the annual Sentinel-2 + Sentinel-1
predictors:

- ``annual_s2_s1_greenup_timing`` adds the strongest detected cycle's rise, baseline and
  onset-to-peak duration, the number of detected cycles and of cycles passing the detector.
- ``annual_s2_s1_greenup_referenced`` also measures every harmonic phase from that onset
  instead of from 1 January, so the description does not depend on the calendar month.

Pixels without a detected cycle keep their annual predictors (timing features zero, phase
referenced to the NDVI harmonic trough), so coverage is the same as the annual recipe.
"""

from datetime import date

import numpy as np

from crop_pipeline.features import annual
from crop_pipeline.imagery import quality

REFERENCE_YEAR = 2021
DETECTOR = {**quality.DEFAULT_DETECTOR, "relative_threshold": 0.2}  # as in paoay.aligned and windows.workflow
ACCEPTED = ("candidate", "provisional", "ambiguous")
TIMING_COLUMNS = ("greenup_cycles", "greenup_accepted_cycles", "greenup_amplitude", "greenup_baseline",
                  "greenup_onset_to_peak_days", "greenup_detected")


def bin_dates(year=REFERENCE_YEAR):
    """(start, end, midpoint) ISO dates for the 24 half-months."""
    rows = []
    for month in range(1, 13):
        end = date(year + (month == 12), month % 12 + 1, 1)
        for start, stop in ((date(year, month, 1), date(year, month, 16)), (date(year, month, 16), end)):
            rows.append((start.isoformat(), stop.isoformat(), (start + (stop - start) // 2).isoformat()))
    return rows


def detector_rows(ndvi, clear):
    dates = bin_dates()
    rows = []
    for i in range(len(ndvi)):
        for b, (start, end, middle) in enumerate(dates):
            ok = bool(clear[i, b]) and np.isfinite(ndvi[i, b])
            rows.append({"region_id": i, "date": middle, "start": start, "end": end, "source_kind": "observed_daily",
                         "valid_fraction": 1.0 if ok else 0.0, "NDVI": float(ndvi[i, b]) if ok else None,
                         "NDRE": None, "NDMI": None})
    return rows


def fraction_of_year(day):
    return ((date.fromisoformat(day) - date(REFERENCE_YEAR, 1, 1)).days % 365) / 365


def timing(s2, s2_valid, chunk=5000):
    """Per pixel: detected cycles, passing cycles and the strongest cycle's rise, baseline, duration and onset."""
    ndvi = annual.indices(s2)[..., 0]
    n = len(ndvi)
    out = {"cycles": np.zeros(n), "accepted": np.zeros(n), "amplitude": np.zeros(n), "baseline": np.full(n, np.nan),
           "onset_to_peak_days": np.zeros(n), "onset": np.full(n, np.nan), "detected": np.zeros(n)}
    best = np.full(n, -np.inf)
    for first in range(0, n, chunk):
        part = np.arange(first, min(n, first + chunk))
        for result in quality.detect_greenup(detector_rows(ndvi[part], s2_valid[part]), DETECTOR):
            if result.get("onset_interval_start") is None or result.get("amplitude") is None:
                continue
            i = part[int(result["region_id"])]
            out["cycles"][i] += 1
            out["accepted"][i] += result["status"] in ACCEPTED
            if result["amplitude"] > best[i]:
                best[i] = result["amplitude"]
                onset = date.fromisoformat(result["onset_interval_start"])
                peak = date.fromisoformat(result["peak_date"]) if result.get("peak_date") else onset
                out["onset"][i] = fraction_of_year(result["onset_interval_start"])
                out["amplitude"][i] = result["amplitude"]
                out["baseline"][i] = result["baseline"]
                out["onset_to_peak_days"][i] = (peak - onset).days % 365
                out["detected"][i] = 1
    return out


def reference_time(s2, s2_valid, cycles):
    """Detected onset as a fraction of the year, else the NDVI first-harmonic trough."""
    cube = np.concatenate([s2, annual.indices(s2)], axis=-1)
    coefficients, _ = annual.fit_harmonics(cube[:, :, len(annual.BANDS)], s2_valid)
    trough = (np.arctan2(coefficients[:, 2], coefficients[:, 1]) / (2 * np.pi) + .5) % 1
    return np.where(np.isfinite(cycles["onset"]), cycles["onset"], np.nan_to_num(trough))


def rereference(X, columns, reference):
    """Rotate each (cos_phaseK, sin_phaseK) pair by -2 pi K reference."""
    X = X.copy()
    for j, column in enumerate(columns):
        if column.endswith(("_cos_phase1", "_cos_phase2")):
            k = int(column[-1])
            s = columns.index(column.replace("_cos_", "_sin_"))
            angle = np.arctan2(X[:, s], X[:, j]) - 2 * np.pi * k * reference
            X[:, j], X[:, s] = np.cos(angle), np.sin(angle)
    return X


def timing_matrix(cycles):
    baseline = np.where(np.isfinite(cycles["baseline"]), cycles["baseline"], 0.0)
    return np.column_stack([cycles["cycles"], cycles["accepted"], cycles["amplitude"], baseline,
                            cycles["onset_to_peak_days"], cycles["detected"]])


def with_timing(s2, s2_valid, s1):
    X, ok, columns = annual.RECIPES["annual_s2_s1"](s2, s2_valid, s1)
    return np.hstack([X, timing_matrix(timing(s2, s2_valid))]), ok, columns + list(TIMING_COLUMNS)


def referenced(s2, s2_valid, s1):
    X, ok, columns = annual.RECIPES["annual_s2_s1"](s2, s2_valid, s1)
    cycles = timing(s2, s2_valid)
    moved = rereference(X, columns, reference_time(s2, s2_valid, cycles))
    names = [f"{c}_from_greenup" if "_phase" in c else c for c in columns]
    return np.hstack([moved, timing_matrix(cycles)]), ok, names + list(TIMING_COLUMNS)


RECIPES = {
    "annual_s2_s1_greenup_timing": with_timing,
    "annual_s2_s1_greenup_referenced": referenced,
}
