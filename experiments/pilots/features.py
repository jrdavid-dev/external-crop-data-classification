"""Pilot predictors from annual 24-bin series: harmonic regression and the fixed window.

Harmonic regression fits ``c0 + sum_k a_k cos(2 pi k t) + b_k sin(2 pi k t)`` to the
clear half-months of one year (t = bin midpoint as a fraction of the year). It
needs no green-up detection, only enough clear bins spread across the year.
"""

import numpy as np

from experiments.pilots import data

BINS = data.BINS
VARIABLES = (*data.BANDS, "NDVI", "NDRE", "NDMI", "SAVI")
FULL_TERMS = ("mean", "amp1", "cos_phase1", "sin_phase1", "amp2", "cos_phase2", "sin_phase2")
INVARIANT_TERMS = ("mean", "amp1", "amp2", "cos_relphase", "sin_relphase")


def stack(values):
    """[n, 24, 5] reflectance -> [n, 24, 9] bands plus indices."""
    return np.concatenate([values, data.indices(values)], axis=-1)


def longest_gap(valid):
    """Longest run of consecutive invalid bins per row."""
    gaps = np.zeros(len(valid), dtype=int)
    run = np.zeros(len(valid), dtype=int)
    for b in range(valid.shape[1]):
        run = np.where(valid[:, b], 0, run + 1)
        gaps = np.maximum(gaps, run)
    return gaps


def fit_harmonics(series, valid, order=2, min_valid=12, max_gap=6, omega=1.0):
    """Least-squares coefficients [n, 1 + 2*order] per row; ``ok`` marks rows fitted.

    ``omega`` scales the base frequency (Wang et al. use 1.5 within a growing-season
    window); on a full year, omega other than 1 is not periodic at the year end.
    """
    t = (np.arange(BINS) + .5) / BINS
    design = np.column_stack([np.ones(BINS)] + [f(2 * np.pi * omega * k * t) for k in range(1, order + 1)
                                                 for f in (np.cos, np.sin)])
    usable = valid & np.isfinite(series)
    ok = (usable.sum(axis=1) >= min_valid) & (longest_gap(usable) <= max_gap)
    coefficients = np.full((len(series), design.shape[1]), np.nan)
    for i in np.nonzero(ok)[0]:
        rows = usable[i]
        coefficients[i], *_ = np.linalg.lstsq(design[rows], series[i, rows], rcond=None)
    return coefficients, ok


FORMS = {
    "full": FULL_TERMS,
    # Shifting the season by d moves phase1 by d and phase2 by 2d; phase2 - 2 phase1 is unchanged.
    "invariant": INVARIANT_TERMS,
    "amplitude": ("mean", "amp1", "amp2"),
    # phase2 - phase1 changes by d under a shift of d: kept only to test that claim.
    "phase_difference": ("mean", "amp1", "amp2", "cos_phasediff", "sin_phasediff"),
}


def harmonic_terms(coefficients, invariant=False, form=None):
    form = form or ("invariant" if invariant else "full")
    c0 = coefficients[:, 0]
    a1, b1, a2, b2 = (coefficients[:, j] for j in range(1, 5))
    amp1, amp2 = np.hypot(a1, b1), np.hypot(a2, b2)
    phase1, phase2 = np.arctan2(b1, a1), np.arctan2(b2, a2)
    if form == "invariant":
        relative = phase2 - 2 * phase1
        return np.column_stack([c0, amp1, amp2, np.cos(relative), np.sin(relative)])
    if form == "amplitude":
        return np.column_stack([c0, amp1, amp2])
    if form == "phase_difference":
        difference = phase2 - phase1
        return np.column_stack([c0, amp1, amp2, np.cos(difference), np.sin(difference)])
    if form == "full":
        return np.column_stack([c0, amp1, np.cos(phase1), np.sin(phase1), amp2, np.cos(phase2), np.sin(phase2)])
    raise ValueError(f"Unknown harmonic form: {form}")


def harmonic_features(cube, valid, names=VARIABLES, invariant=False, form=None, **fit):
    """cube [n, 24, v] -> (features [n, v * terms], ok [n], column names)."""
    form = form or ("invariant" if invariant else "full")
    blocks, ok = [], np.ones(len(cube), dtype=bool)
    for v in range(cube.shape[2]):
        coefficients, fitted = fit_harmonics(cube[:, :, v], valid, **fit)
        blocks.append(harmonic_terms(coefficients, form=form))
        ok &= fitted
    columns = [f"{name}_{term}" for name in names for term in FORMS[form]]
    return np.hstack(blocks), ok, columns


def evi2(values):
    """Two-band EVI: 2.5 (NIR - red) / (NIR + 2.4 red + 1), from [..., 5] reflectance."""
    red, nir = values[..., data.BANDS.index("B4")], values[..., data.BANDS.index("B8")]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1)


def optical_indices(values):
    """[n, 24, 5] -> [n, 24, 5] EVI2, NDVI, NDRE, NDMI, SAVI (no raw bands)."""
    return np.concatenate([evi2(values)[..., None], data.indices(values)], axis=-1)


OPTICAL_INDEX_NAMES = ("EVI2", "NDVI", "NDRE", "NDMI", "SAVI")
SAR_NAMES = ("VV", "VH", "VHVV", "CROSSPOL")


def sar_variables(values_db):
    """[n, 24, 2] VV, VH dB -> [n, 24, 4]: VV, VH, VH/VV (dB difference), VH/(VV+VH) (linear)."""
    vv, vh = values_db[..., 0], values_db[..., 1]
    cross = 10 ** (vh / 10) / (10 ** (vv / 10) + 10 ** (vh / 10))
    return np.stack([vv, vh, vh - vv, cross], axis=-1)


def temporal_zscore(cube, valid):
    """Per pixel and variable: (x - annual mean) / annual SD over valid bins."""
    masked = np.where(valid[..., None], cube, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(masked, axis=1, keepdims=True)
        sd = np.nanstd(masked, axis=1, keepdims=True)
    return (cube - mean) / np.where(sd > 0, sd, np.nan)


def fixed_interval(cube, valid, start_bin=8):
    """The 72 interval predictors for the eight bins from ``start_bin`` (8 = May 1); all bins must be clear."""
    window = slice(start_bin, start_bin + 8)
    ok = valid[:, window].all(axis=1) & np.isfinite(cube[:, window]).all(axis=(1, 2))
    columns = [f"bin_{b + 1:02d}_{v}" for b in range(8) for v in VARIABLES]
    return cube[:, window].reshape(len(cube), -1), ok, columns


def column_groups(columns):
    """Group predictors by their variable (text before the first term suffix)."""
    groups = {}
    for index, column in enumerate(columns):
        groups.setdefault(column.split("_")[0], []).append(index)
    return groups


def shift_series(cube, valid, shift):
    """Move the annual series by whole half-months (circular, as for an annual cycle)."""
    return np.roll(cube, shift, axis=1), np.roll(valid, shift, axis=1)


def gain_series(values, rng, spread):
    """One random gain per pixel and band on reflectance [n, 24, bands]."""
    return values * rng.uniform(1 - spread, 1 + spread, size=(len(values), 1, values.shape[2]))


def augmented_copies(build, values, valid, shifts=(-1, 1), gain=.1, gain_copies=1, seed=0):
    """Feature matrices from shifted and gain-scaled series; ``build(values, valid)`` returns X."""
    rng = np.random.default_rng(seed)
    copies = [build(*shift_series(values, valid, s)) for s in shifts]
    copies += [build(gain_series(values, rng, gain), valid) for _ in range(gain_copies if gain else 0)]
    return copies


def grid_median(cube, radius=1):
    """3x3 (radius 1) median of clear values on a grid cube [bins, bands, rows, cols]; NaN = not clear."""
    import warnings
    bins, bands, height, width = cube.shape
    padded = np.pad(cube, ((0, 0), (0, 0), (radius, radius), (radius, radius)), constant_values=np.nan)
    shifted = [padded[:, :, radius + dy: radius + dy + height, radius + dx: radius + dx + width]
               for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(shifted), axis=0)


def seasonality_ratio(values, valid, **fit):
    """NDVI first-harmonic amplitude divided by NDVI mean (low for evergreen cover)."""
    ndvi = data.indices(values)[..., 0]
    coefficients, ok = fit_harmonics(ndvi, valid, **fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.hypot(coefficients[:, 1], coefficients[:, 2]) / coefficients[:, 0]
    return ratio, ok
