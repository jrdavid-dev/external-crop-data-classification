"""The external study's curve-shape descriptors, computed on the annual half-month series.

For each of NDVI, NDRE, NDMI and SAVI, the study's ``features.experiments.temporal_shape`` is
applied to the 24 half-month values of the labelled year: consecutive differences, green-up and
senescence slopes (least squares before and after the first maximum, per day), the one-based peak
interval, amplitude and the trapezoidal integral, on calendar bin midpoints in days. Half-months
without a clear observation are filled by circular linear interpolation between clear ones, and a
pixel is usable under the annual recipe's rule (at least 12 clear half-months, no gap over six), so
every recipe here covers the same pixels as ``annual_s2_s1`` and only the description changes.

Recipes (all fitted with the same Random Forest when compared):

- ``annual_shape`` — curve shape only (Sentinel-2 indices).
- ``annual_shape_s1`` — curve shape plus the annual Sentinel-1 terms, to separate the feature effect
  from the radar effect.
- ``annual_s2_s1_shape`` — the annual Sentinel-2 + Sentinel-1 predictors plus curve shape.
"""

from datetime import date

import numpy as np

from experiments.annual import features as annual
from crop_pipeline.features.experiments import INDEX_NAMES, temporal_shape
from experiments.annual import workflow as annual_workflow

REFERENCE_YEAR = 2021


def bin_positions(year=REFERENCE_YEAR):
    """Half-month midpoints in days from 1 January (the study's time axis)."""
    origin, positions = date(year, 1, 1), []
    for month in range(1, 13):
        end = date(year + (month == 12), month % 12 + 1, 1)
        for start, stop in ((date(year, month, 1), date(year, month, 16)), (date(year, month, 16), end)):
            positions.append((start - origin).days + (stop - start).days / 2)
    return np.asarray(positions)


def fill_gaps(series, clear):
    """Circular linear interpolation of non-clear half-months, per row ([n, 24])."""
    filled = np.where(clear, series, np.nan)
    bins = np.arange(series.shape[1])
    for i in np.nonzero(~np.isfinite(filled).all(axis=1))[0]:
        good = np.isfinite(filled[i])
        if good.sum() < 2:
            continue
        filled[i] = np.interp(bins, np.r_[bins[good] - 24, bins[good], bins[good] + 24], np.tile(filled[i, good], 3))
    return filled


def columns():
    names = []
    for index_name in INDEX_NAMES:
        prefix = f"annual_shape_{index_name}"
        names += [f"{prefix}_difference_{b:02d}_{b + 1:02d}" for b in range(1, 24)]
        names += [f"{prefix}_{name}" for name in ("greenup_slope", "senescence_slope", "peak_interval",
                                                  "amplitude", "integral")]
    return names


def shape_features(s2, s2_valid, s1=None):
    """(X [n, 112], ok [n], columns) from the study's descriptors on the 24-bin annual index series."""
    _, ok, _ = annual.s2_harmonic(s2, s2_valid)
    index_series = annual.indices(s2)
    positions = bin_positions()
    X = np.full((len(s2), 4 * 28), np.nan)
    for k, index_name in enumerate(INDEX_NAMES):
        series = fill_gaps(index_series[..., k], s2_valid & np.isfinite(index_series[..., k]))
        for i in np.nonzero(ok)[0]:
            values = temporal_shape(series[i], positions, index_name)
            prefix = f"shape_{index_name}"
            X[i, 28 * k:28 * (k + 1)] = (
                [values[f"{prefix}_difference_{b:02d}_{b + 1:02d}"] for b in range(1, 24)]
                + [values[f"{prefix}_{name}"] for name in ("greenup_slope", "senescence_slope", "peak_interval",
                                                           "amplitude", "integral")])
    ok = ok & np.isfinite(X).all(axis=1)
    return X, ok, columns()


RECIPES = {
    "annual_shape": shape_features,
    "annual_shape_s1": annual.combine(shape_features, annual.s1_harmonic),
    "annual_s2_s1_shape": annual.combine(annual.s2_harmonic, annual.s1_harmonic, shape_features),
}

# Make these recipes usable by the pipeline's freeze, test and map stages (never selection defaults).
annual_workflow.register_recipes(RECIPES)
