"""Pilot subset: annual 24-bin Sentinel-2 pixel series for sampled parcels.

Development parcels are sampled per (partition, crop, source) inside the locked
train/validation split; all primary JECAM test parcels are kept. Pixels per
parcel are chosen by the pipeline's stable rank, so the subset is reproducible.
The per-pixel ``valid`` flag of the annual archive is used as the clear-sky
test (the main pipeline additionally applies an 80% parcel coverage rule).
"""

from collections import defaultdict
import json
from math import floor
import warnings
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from crop_pipeline import preprocessing_workflow
from crop_pipeline.sources import sampling
from crop_pipeline.windows import annual_archive

BANDS = ("B3", "B4", "B5", "B8", "B11")
BINS = 24
METADATA = ("partition_group", "region_id", "partition", "crop", "source_dataset", "site_cell",
            "label_status", "year", "x", "y", "source_crs", "lon", "lat")


def bin_index(name):
    """'2022_05_16_end' -> 9 (zero-based half-month of the year)."""
    _, month, start, _ = name.split("_")
    return (int(month) - 1) * 2 + (0 if start == "01" else 1)


def load_quality(repository, quality_path=None):
    root = Path(repository).resolve()
    path = preprocessing_workflow.choose_input(root, quality_path or None, "quality")
    return path, json.loads(path.read_text())


def sample_parcels(regions, per_cell, seed="pilot", include_supplements=False):
    """Fragments of sampled parcels: up to ``per_cell[partition]`` parcels per (partition, crop, source).

    Supplemental test campaigns are left out unless ``include_supplements`` is set.
    """
    groups = defaultdict(set)
    for region in regions:
        p = region["properties"]
        if p["partition"] == "test" and p.get("test_cohort", "jecam") != "jecam" and not include_supplements:
            continue
        groups[p["partition"], p["crop"], p["source_dataset"]].add(p["partition_group"])
    chosen = set()
    for (partition, _, _), members in groups.items():
        limit = per_cell.get(partition)
        ranked = sorted(members, key=lambda g: sampling.stable_rank(f"{seed}|{g}"))
        chosen.update(ranked if limit is None else ranked[:limit])
    return [r for r in regions if r["properties"]["partition_group"] in chosen]


def _lonlat(region):
    point = shape(region["geometry"]).representative_point()
    return point.x, point.y


def neighbourhood_median(series, spacing=10., radius=1):
    """Per pixel: median of clear values in the surrounding (2 radius + 1)^2 pixels of the same fragment.

    A bin is clear when at least one neighbour (or the pixel itself) is clear.
    Neighbours outside the fragment are absent, so parcel edges use fewer pixels.
    """
    by_cell = {(key[0], round(key[1] / spacing), round(key[2] / spacing)): key for key in series}
    smoothed = {}
    for key in series:
        crs, cx, cy = key[0], round(key[1] / spacing), round(key[2] / spacing)
        stack = [series[by_cell[(crs, cx + dx, cy + dy)]] for dx in range(-radius, radius + 1)
                 for dy in range(-radius, radius + 1) if (crs, cx + dx, cy + dy) in by_cell]
        values = np.stack([item["values"] for item in stack])
        valid = np.stack([item["valid"] for item in stack])
        masked = np.where(valid[..., None], values, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(masked, axis=0)
        smoothed[key] = {"values": median, "valid": valid.any(axis=0)}
    return smoothed


def load_series(regions, pixel_paths, pixels_per_parcel=10, seed="pilot", neighbourhood=0):
    """Return (metadata rows, values [n, 24, 5] reflectance, valid [n, 24] bool).

    With ``neighbourhood`` > 0, each pixel's series is the median of its clear
    neighbours within that many pixels (3x3 for 1), restricted to its own fragment.
    Pixel selection is unchanged, so rows match an unfiltered load.
    """
    archive = annual_archive.PixelPartitions(pixel_paths)
    per_parcel = defaultdict(list)
    for region in regions:
        p = region["properties"]
        lon, lat = _lonlat(region)
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
                         "lon": lon, "lat": lat,
                         "test_cohort": (p.get("test_cohort") or "jecam") if p["partition"] == "test" else None}
        if neighbourhood:
            series = neighbourhood_median(series, radius=neighbourhood)
        for key, item in series.items():
            per_parcel[p["partition_group"]].append((key, info[key], item))
    meta, values, valid = [], [], []
    for group, items in per_parcel.items():
        ranked = sorted(items, key=lambda item: sampling.stable_rank(f"{seed}|{group}|{item[0]}"))
        seen = set()
        for key, info, item in ranked:
            if key in seen:
                continue
            seen.add(key)
            meta.append(info)
            values.append(item["values"])
            valid.append(item["valid"])
            if len(seen) >= pixels_per_parcel:
                break
    return meta, np.asarray(values), np.asarray(valid)


def indices(values):
    """NDVI, NDRE, NDMI and SAVI with the pipeline formulas; NaN where bands are missing."""
    b4, b5, b8, b11 = (values[..., BANDS.index(b)] for b in ("B4", "B5", "B8", "B11"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([(b8 - b4) / (b8 + b4), (b8 - b5) / (b8 + b5), (b8 - b11) / (b8 + b11),
                         1.5 * (b8 - b4) / (b8 + b4 + .5)], axis=-1)


def save(folder, meta, arrays):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(folder / "arrays.npz", **arrays)


def load(folder):
    folder = Path(folder)
    with np.load(folder / "arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    return json.loads((folder / "meta.json").read_text()), arrays
