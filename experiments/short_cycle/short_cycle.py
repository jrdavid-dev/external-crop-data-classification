"""Short-cycle radar recipe: describe one crop cycle instead of the whole year.

European parcels grow one crop per summer, so a yearly harmonic curve has one large
seasonal swing. Paoay farms year-round with several short cycles, so its yearly curve
is flat and lands nearest European potato. This recipe instead finds, per pixel, the
four-month window (8 half-months, wrapping across the year end) in which Sentinel-1 VH
rises most from its early-window minimum to its peak, and describes that one cycle in
window order. Calendar position is deliberately left out, so a cycle in March and one in
July are described the same way.

Builders follow ``crop_pipeline.features.annual`` and return ``(X, ok, columns)``.
"""

import numpy as np

from crop_pipeline.features import annual

WINDOW = 8
MIN_RADAR_BINS = 22  # half-months with both polarizations, out of 24


def fill_gaps(series):
    """Fill missing half-months by circular linear interpolation per row ([n, 24])."""
    filled = series.copy()
    bins = np.arange(series.shape[1])
    for i in np.nonzero(~np.isfinite(series).all(axis=1))[0]:
        good = np.isfinite(series[i])
        if good.sum() < 2:
            continue
        x = np.r_[bins[good] - 24, bins[good], bins[good] + 24]
        y = np.tile(series[i, good], 3)
        filled[i] = np.interp(bins, x, y)
    return filled


def cycle_start(vh, tolerance=1e-6):
    """Start bin of the 8-bin circular window with the largest rise from its first-three-bin minimum to its peak.

    Several windows can hold the same cycle; among those within ``tolerance`` of the largest rise,
    the one that puts the peak nearest the window middle wins, so the choice does not depend on
    the calendar month.
    """
    n, bins = vh.shape
    index = (np.arange(bins)[:, None] + np.arange(WINDOW)[None, :]) % bins  # [starts, 8]
    windows = vh[:, index]  # [n, starts, 8]
    rise = windows.max(axis=2) - windows[:, :, :3].min(axis=2)
    best = rise.max(axis=1, keepdims=True)
    peak_position = np.argmax(windows, axis=2)
    # Cost: distance of the peak from the middle; equally centred windows prefer the peak in the
    # later half (position 4 of 0-7), a rule relative to the peak, not to the calendar.
    cost = np.abs(peak_position - (WINDOW - 1) / 2) + 0.01 * (peak_position < WINDOW / 2)
    cost = np.where(rise >= best - tolerance, cost, np.inf)
    return np.argmin(cost, axis=1), best[:, 0]


def s1_cycle(s2, s2_valid, s1):
    """VV, VH and VH-VV for the 8 half-months of the strongest VH cycle, plus its rise and peak position."""
    if s1 is None:
        raise ValueError("This recipe needs Sentinel-1 values")
    ok = np.isfinite(s1).all(axis=2).sum(axis=1) >= MIN_RADAR_BINS
    vv, vh = fill_gaps(s1[..., 0]), fill_gaps(s1[..., 1])
    start, rise = cycle_start(np.nan_to_num(vh, nan=-30.0))
    index = (start[:, None] + np.arange(WINDOW)[None, :]) % vh.shape[1]
    rows = np.arange(len(vh))[:, None]
    window_vv, window_vh = vv[rows, index], vh[rows, index]
    peak = np.argmax(window_vh, axis=1).astype(float)
    X = np.hstack([window_vv, window_vh, window_vh - window_vv, rise[:, None], peak[:, None]])
    columns = ([f"cycle_VV_{b + 1}" for b in range(WINDOW)] + [f"cycle_VH_{b + 1}" for b in range(WINDOW)]
               + [f"cycle_VHVV_{b + 1}" for b in range(WINDOW)] + ["cycle_VH_rise", "cycle_VH_peak_bin"])
    return X, ok & np.isfinite(X).all(axis=1), columns


RECIPES = {
    "s1_cycle": s1_cycle,
    "s1_cycle_plus_s2_harmonic": annual.combine(annual.s2_harmonic, s1_cycle),
    "annual_s2_s1": annual.RECIPES["annual_s2_s1"],
}
