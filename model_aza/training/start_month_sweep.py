"""Exploratory fixed-window start-month sweep on development parcels only.

Rebuilds four-month fixed windows for each start month from a verified
Preprocessing 01 quality handoff, using the unchanged window builder, 80%
validity rule, eight complete bins, shared-location exclusion and 20-pixel cap.
Test regions are dropped before any window is built. Only capped development
rows are kept in memory; no windows or feature handoffs are written, and stage
pointers and frozen models are unchanged.
"""

from collections import Counter
from datetime import date
import json
from math import floor
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from crop_pipeline import preprocessing_workflow
from model_aza.training import transfer
from crop_pipeline.windows import annual_archive, external, seasons

START_MONTHS = tuple(range(1, 10))
MINIMUM_VALID_FRACTION = .8
BASE_VARIABLES = ("B3", "B4", "B5", "B8", "B11", "NDVI", "NDRE", "NDMI")
METADATA = ("partition_group", "region_id", "partition", "crop", "source_dataset", "site_cell",
            "label_status", "source_crs", "x", "y")


def load_quality(repository, quality_path=None):
    """Return the verified quality handoff path and its development regions and inputs."""
    root = Path(repository).resolve()
    path = preprocessing_workflow.choose_input(root, quality_path or None, "quality")
    saved = json.loads(path.read_text())
    regions = [r for r in saved["regions"] if r["properties"]["partition"] in {"train", "validation"}]
    if any(r["properties"].get("test_cohort") for r in regions):
        raise ValueError("Test regions must not enter the development sweep")
    ids = {r["properties"]["fragment_id"] for r in regions}
    quality_by_region = {}
    for row in saved["quality_rows"]:
        if row["region_id"] in ids:
            quality_by_region.setdefault(row["region_id"], []).append(row)
    pixels = annual_archive.PixelPartitions({k: v for k, v in saved["pixel_paths"].items() if k in ids})
    return path, {"regions": regions, "quality_by_region": quality_by_region, "pixels": pixels}


def fixed_windows(regions, start_month):
    """The pipeline's fixed windows, moved to another start month of the same year.

    Kept outside ``windows/external.py`` so published preprocessing receipts,
    which hash that file, stay valid.
    """
    if type(start_month) is not int or start_month not in START_MONTHS:
        raise ValueError("Fixed four-month window must start in January through September")
    for window in external.fixed_windows(regions):
        start = date(int(window["source_year"]), start_month, 1)
        yield {**window, "window_start": start.isoformat(),
               "window_end": seasons.add_calendar_months(start, 4).isoformat()}


def _site_cell(region):
    point = shape(region["geometry"]).representative_point()
    return f"{floor(point.y)},{floor(point.x)}"


def compact_rows(inputs, start_month):
    """Yield one compact record per complete pixel window, in pipeline order."""
    columns = [f"bin_{b:02d}_{v}" for b in range(1, 9) for v in BASE_VARIABLES]
    for region, window in zip(inputs["regions"], fixed_windows(inputs["regions"], start_month)):
        identifier = region["properties"]["fragment_id"]
        accepted = seasons.build_features([window], inputs["pixels"].for_regions([identifier]),
                                          inputs["quality_by_region"].get(identifier, []),
                                          MINIMUM_VALID_FRACTION, 0)[0]
        properties = region["properties"]
        site = _site_cell(region) if accepted else None
        for row in accepted:
            record = {"partition_group": properties["partition_group"], "region_id": identifier,
                      "partition": properties["partition"], "crop": properties["crop"],
                      "source_dataset": properties["source_dataset"], "site_cell": site,
                      "label_status": row.get("label_status", "reported"),
                      "source_crs": row["source_crs"], "x": row["x"], "y": row["y"]}
            record["_values"] = np.asarray([float(row[c]) for c in columns], dtype=float)
            yield record


def select(records):
    """Apply the pipeline's first-occurrence de-duplication, shared-location exclusion and cap."""
    indexed = {}
    for record in records:
        indexed.setdefault(external.pixel_key(record), record)
    retained, exclusions = external.exclude_shared_development_locations(indexed)
    return external.capped(retained), exclusions


def to_feature_rows(records):
    """Expand compact records into rows with the 72 ordered interval predictors."""
    rows = []
    adjustment = transfer.SAVI_ADJUSTMENT
    for record in records:
        values = record["_values"].reshape(8, len(BASE_VARIABLES))
        row = {key: record[key] for key in METADATA}
        for b in range(8):
            for v, name in enumerate(BASE_VARIABLES):
                row[f"bin_{b + 1:02d}_{name}"] = float(values[b, v])
            b8, b4 = values[b, BASE_VARIABLES.index("B8")], values[b, BASE_VARIABLES.index("B4")]
            row[f"bin_{b + 1:02d}_SAVI"] = float((1 + adjustment) * (b8 - b4) / (b8 + b4 + adjustment))
        rows.append(row)
    return rows


def month_rows(inputs, start_month):
    selected, exclusions = select(compact_rows(inputs, start_month))
    return to_feature_rows(selected), exclusions


def coverage(rows, start_month):
    parcels = {}
    for row in rows:
        parcels[row["partition_group"]] = (row["partition"], row["crop"])
    pixels = Counter((row["partition"], row["crop"]) for row in rows)
    counts = Counter(parcels.values())
    return [{"start_month": start_month, "partition": partition, "crop": crop,
             "parcels": counts[partition, crop], "pixels": pixels[partition, crop]}
            for partition in ("train", "validation") for crop in transfer.CROPS]


def has_all_crops(rows):
    return all({r["crop"] for r in rows if r["partition"] == split} == set(transfer.CROPS)
               for split in ("train", "validation"))
