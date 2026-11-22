"""Annual harmonic predictors: whole-year Sentinel-2 series, optionally with Sentinel-1.

Each pixel keeps its 24 half-month Sentinel-2 composites for the parcel's labelled
year (the annual archive written by Preprocessing 01). A second-order harmonic
regression ``c0 + sum_k a_k cos(2 pi k t) + b_k sin(2 pi k t)`` is fitted to the
clear half-months, so no four-month window or green-up date is needed. A pixel is
usable when at least 12 half-months are clear and no gap exceeds 6 half-months
(three months). Sentinel-1 VV and VH backscatter (dB) on the same half-months adds
harmonic terms for VV and VH and seasonal amplitude terms for VH/VV and VH/(VV+VH).

Pixels are drawn per parcel by the pipeline's stable rank, up to the 20-pixel parcel
cap, and development pixels shared by different parcel groups are dropped, as for the
window recipes.
"""

from collections import defaultdict
from math import floor

import numpy as np
from shapely.geometry import shape

from crop_pipeline.sources import sampling
from crop_pipeline.windows import annual_archive

RECIPE_VERSION = 1
BANDS = ("B3", "B4", "B5", "B8", "B11")
INDICES = ("NDVI", "NDRE", "NDMI", "SAVI")
BINS = 24
PIXELS_PER_PARCEL = 20
MIN_CLEAR_BINS = 12
MAX_GAP_BINS = 6
FULL_TERMS = ("mean", "amp1", "cos_phase1", "sin_phase1", "amp2", "cos_phase2", "sin_phase2")
AMPLITUDE_TERMS = ("mean", "amp1", "amp2")
SEED = "annual"


def bin_index(name):
    """'2022_05_16_end' -> 9 (zero-based half-month of the year)."""
    _, month, start, _ = name.split("_")
    return (int(month) - 1) * 2 + (0 if start == "01" else 1)


def _site_cell(region):
    point = shape(region["geometry"]).representative_point()
    return point.x, point.y


def load_series(regions, pixel_paths, pixels_per_parcel=PIXELS_PER_PARCEL, seed=SEED):
    """Return (pixel rows, reflectance [n, 24, 5], clear flags [n, 24]) for every parcel in ``regions``."""
    archive = annual_archive.PixelPartitions(pixel_paths)
    per_parcel = defaultdict(list)
    for region in regions:
        p = region["properties"]
        lon, lat = _site_cell(region)
        series = defaultdict(lambda: {"values": np.full((BINS, len(BANDS)), np.nan), "valid": np.zeros(BINS, bool)})
        info = {}
        for row in archive.for_regions([p["fragment_id"]]):
            key = (row["source_crs"], float(row["x"]), float(row["y"]))
            b = bin_index(row["name"])
            if row["valid"] and all(row.get(band) is not None for band in BANDS):
                series[key]["values"][b] = [float(row[band]) for band in BANDS]
                series[key]["valid"][b] = True
            info[key] = {"partition_group": p["partition_group"], "region_id": p["fragment_id"],
                         "partition": p["partition"], "crop": p["crop"], "source_dataset": p["source_dataset"],
                         "site_cell": f"{floor(lat)},{floor(lon)}", "label_status": p.get("label_status", "reported"),
                         "year": int(p["source_year"]), "x": key[1], "y": key[2], "source_crs": key[0],
                         "test_cohort": (p.get("test_cohort") or "jecam") if p["partition"] == "test" else None}
        for key, item in series.items():
            per_parcel[p["partition_group"]].append((key, info[key], item))
    rows, values, valid = [], [], []
    for group, items in sorted(per_parcel.items()):
        ranked = sorted(items, key=lambda item: sampling.stable_rank(f"{seed}|{group}|{item[0]}"))
        seen = set()
        for key, info, item in ranked:
            if key in seen:
                continue
            seen.add(key)
            rows.append(info)
            values.append(item["values"])
            valid.append(item["valid"])
            if len(seen) >= pixels_per_parcel:
                break
    return rows, np.asarray(values).reshape(-1, BINS, len(BANDS)), np.asarray(valid, dtype=bool).reshape(-1, BINS)


def shared_development_locations(rows):
    """Indices of development pixels whose ground location belongs to more than one parcel group."""
    groups = defaultdict(set)
    for row in rows:
        if row["partition"] in ("train", "validation"):
            groups[sampling.pixel_location(row)].add(row["partition_group"])
    shared = {location for location, members in groups.items() if len(members) > 1}
    return [i for i, row in enumerate(rows)
            if row["partition"] in ("train", "validation") and sampling.pixel_location(row) in shared]


def indices(values):
    """NDVI, NDRE, NDMI and SAVI from [..., 5] reflectance; NaN where bands are missing."""
    b4, b5, b8, b11 = (values[..., BANDS.index(b)] for b in ("B4", "B5", "B8", "B11"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([(b8 - b4) / (b8 + b4), (b8 - b5) / (b8 + b5), (b8 - b11) / (b8 + b11),
                         1.5 * (b8 - b4) / (b8 + b4 + .5)], axis=-1)


def longest_gap(valid):
    gaps = np.zeros(len(valid), dtype=int)
    run = np.zeros(len(valid), dtype=int)
    for b in range(valid.shape[1]):
        run = np.where(valid[:, b], 0, run + 1)
        gaps = np.maximum(gaps, run)
    return gaps


def fit_harmonics(series, valid, order=2, min_valid=MIN_CLEAR_BINS, max_gap=MAX_GAP_BINS):
    """Least-squares coefficients [n, 1 + 2 order] per row; ``ok`` marks rows that meet the clear-bin rule."""
    t = (np.arange(BINS) + .5) / BINS
    design = np.column_stack([np.ones(BINS)] + [f(2 * np.pi * k * t) for k in range(1, order + 1)
                                                 for f in (np.cos, np.sin)])
    usable = valid & np.isfinite(series)
    ok = (usable.sum(axis=1) >= min_valid) & (longest_gap(usable) <= max_gap)
    coefficients = np.full((len(series), design.shape[1]), np.nan)
    for i in np.nonzero(ok)[0]:
        keep = usable[i]
        coefficients[i], *_ = np.linalg.lstsq(design[keep], series[i, keep], rcond=None)
    return coefficients, ok


def harmonic_terms(coefficients, form="full"):
    c0 = coefficients[:, 0]
    a1, b1, a2, b2 = (coefficients[:, j] for j in range(1, 5))
    amp1, amp2 = np.hypot(a1, b1), np.hypot(a2, b2)
    if form == "amplitude":
        return np.column_stack([c0, amp1, amp2])
    if form == "full":
        phase1, phase2 = np.arctan2(b1, a1), np.arctan2(b2, a2)
        return np.column_stack([c0, amp1, np.cos(phase1), np.sin(phase1), amp2, np.cos(phase2), np.sin(phase2)])
    raise ValueError(f"Unknown harmonic form: {form}")


def harmonic_features(cube, valid, names, form="full"):
    """cube [n, 24, v] -> (features [n, v * terms], ok [n], column names)."""
    blocks, ok = [], np.ones(len(cube), dtype=bool)
    for v in range(cube.shape[2]):
        coefficients, fitted = fit_harmonics(cube[:, :, v], valid)
        blocks.append(harmonic_terms(coefficients, form))
        ok &= fitted
    terms = FULL_TERMS if form == "full" else AMPLITUDE_TERMS
    return np.hstack(blocks), ok, [f"{name}_{term}" for name in names for term in terms]


def sar_variables(values_db):
    """[n, 24, 2] VV, VH dB -> [n, 24, 4]: VV, VH, VH/VV (dB difference), VH/(VV+VH) (linear)."""
    vv, vh = values_db[..., 0], values_db[..., 1]
    cross = 10 ** (vh / 10) / (10 ** (vv / 10) + 10 ** (vh / 10))
    return np.stack([vv, vh, vh - vv, cross], axis=-1)


def s2_harmonic(s2, s2_valid, s1=None):
    cube = np.concatenate([s2, indices(s2)], axis=-1)
    return harmonic_features(cube, s2_valid, (*BANDS, *INDICES))


def s1_harmonic(s2, s2_valid, s1):
    """VV and VH harmonic terms plus seasonal amplitude of VH/VV and VH/(VV+VH)."""
    if s1 is None:
        raise ValueError("This recipe needs Sentinel-1 values")
    sar = sar_variables(s1)
    clear = np.isfinite(sar).all(axis=2)
    raw = harmonic_features(sar[:, :, :2], clear, ("VV", "VH"))
    ratios = harmonic_features(sar[:, :, 2:], clear, ("VHVV", "CROSSPOL"), form="amplitude")
    return np.hstack([raw[0], ratios[0]]), raw[1] & ratios[1], raw[2] + ratios[2]


def combine(*builders):
    def build(s2, s2_valid, s1=None):
        parts = [builder(s2, s2_valid, s1) for builder in builders]
        return (np.hstack([p[0] for p in parts]), np.logical_and.reduce([p[1] for p in parts]),
                [c for p in parts for c in p[2]])
    return build


RECIPES = {
    "annual_s2": s2_harmonic,
    "annual_s2_s1": combine(s2_harmonic, s1_harmonic),
}
