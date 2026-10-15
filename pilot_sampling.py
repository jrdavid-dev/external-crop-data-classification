"""Reproducible candidate sampling and field-balanced usable-pixel selection."""

import hashlib
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry, unary_union

from crop_workflow import save_json, write_table

CANDIDATE_PIXELS_PER_CROP = 3000
USABLE_PIXELS_PER_CROP = 1000


def stable_rank(value, seed=42):
    """Avoid Python's process-dependent hash so reruns preserve sample identity."""
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def choose_spread_groups(groups, count, seed=42):
    """Spread candidate fields geographically before drawing within-field pixels."""
    group_names = sorted(groups)
    if not 0 < count <= len(group_names):
        raise ValueError("Requested group count exceeds available groups.")
    centers = np.asarray([group_center(groups[name]) for name in group_names])
    selected = [int(np.random.default_rng(seed).integers(len(group_names)))]
    distances = np.full(len(group_names), np.inf)
    while len(selected) < count:
        squared_distances = ((centers - centers[selected[-1]]) ** 2).sum(axis=1)
        distances = np.minimum(distances, squared_distances)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return [group_names[index] for index in selected]


def group_center(rows):
    coordinate_system = rows[0]["source_crs"]
    if any(row["source_crs"] != coordinate_system for row in rows):
        raise ValueError("A parcel group spans coordinate systems; resolve it before sampling.")
    transformer = Transformer.from_crs(coordinate_system, "EPSG:3857", always_xy=True)
    return transformer.transform(
        np.mean([row["x"] for row in rows]), np.mean([row["y"] for row in rows])
    )


def sample_candidate_pixels(
    catalog, crops, pixels_per_crop=3000, preferred_groups=60, maximum_pixels_per_group=100, seed=42
):
    """Require the full candidate budget; never duplicate scarce pixels to reach it."""
    if pixels_per_crop < 1 or preferred_groups < 1 or maximum_pixels_per_group < 1:
        raise ValueError("Sampling budgets must be positive.")
    selected = []
    for crop_number, crop in enumerate(crops):
        crop_rows = [row for row in catalog if row["crop"] == crop]
        selected.extend(
            sample_crop_candidates(
                crop_rows,
                pixels_per_crop,
                preferred_groups,
                maximum_pixels_per_group,
                seed + crop_number,
            )
        )
    return selected


def sample_crop_candidates(rows, budget, preferred_groups, maximum_per_group, seed):
    locations = [pixel_location(row) for row in rows]
    if len(locations) != len(set(locations)):
        raise ValueError("Candidate pixels must have unique locations within each crop.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["partition_group"]].append(row)
    capacity = sum(min(len(entries), maximum_per_group) for entries in groups.values())
    if capacity < budget:
        raise ValueError(
            f"Insufficient parcel-group capacity for {budget} candidate pixels; collect more fields."
        )
    # Small vegetable parcels cannot all meet an equal quota. Keep their usable
    # capacity and add spread-out groups until the capped total meets the budget.
    ordered_groups = choose_spread_groups(groups, len(groups), seed)
    chosen_groups = []
    selected_capacity = 0
    for group_name in ordered_groups:
        chosen_groups.append(group_name)
        selected_capacity += min(len(groups[group_name]), maximum_per_group)
        if len(chosen_groups) >= min(preferred_groups, len(groups)) and selected_capacity >= budget:
            break
    pool = [
        row
        for group_name in chosen_groups
        for row in sorted(
            groups[group_name], key=lambda row: stable_rank(pixel_location(row), seed)
        )[:maximum_per_group]
    ]
    selected = round_robin_fields(pool, budget, seed)
    return [{**row, "pixel_id": stable_rank(pixel_location(row), 0)[:20]} for row in selected]


def pixel_location(row):
    return (row["source_crs"], float(row["x"]), float(row["y"]))


def selected_region(crop, group_name, rows, seed):
    coordinate_system = rows[0]["source_crs"]
    pixel_shapes = [box(row["x"] - 5, row["y"] - 5, row["x"] + 5, row["y"] + 5) for row in rows]
    projector = Transformer.from_crs(coordinate_system, "EPSG:4326", always_xy=True)
    geographic_geometry = transform_geometry(projector.transform, unary_union(pixel_shapes))
    properties = {
        "fragment_id": stable_rank(f"pilot3000|{crop}|{group_name}", 0)[:20],
        "field_id": None,
        "partition_group": group_name,
        "crop": crop,
        "source_crs": coordinate_system,
        "selected_pixels": len(rows),
        "source_fragments": sorted(
            {fragment for row in rows for fragment in row["source_fragments"]}
        ),
        "sampling_seed": seed,
        "geometry_status": "selected_pixel_footprints",
    }
    return {"type": "Feature", "properties": properties, "geometry": mapping(geographic_geometry)}


def download_grid(region, buffer_metres=20, resolution=10):
    projector = Transformer.from_crs(
        "EPSG:4326", region["properties"]["source_crs"], always_xy=True
    )
    geometry = transform_geometry(projector.transform, shape(region["geometry"]))
    west, south, east, north = geometry.bounds
    left = math.floor((west - buffer_metres) / resolution) * resolution
    right = math.ceil((east + buffer_metres) / resolution) * resolution
    bottom = math.floor((south - buffer_metres) / resolution) * resolution
    top = math.ceil((north + buffer_metres) / resolution) * resolution
    width, height = int(round((right - left) / resolution)), int(round((top - bottom) / resolution))
    if width * height * 7 * 4 >= 30 * 1024**2:
        raise ValueError(
            "Disconnected sample footprint exceeds direct-download limits; review this parcel group."
        )
    return {
        "crs": region["properties"]["source_crs"],
        "transform": [resolution, 0, left, 0, -resolution, top],
        "width": width,
        "height": height,
    }


def save_sampling_mask(region, rows, field_folder):
    grid = download_grid(region)
    transform = Affine(*grid["transform"])
    sample_mask = np.zeros((grid["height"], grid["width"]), dtype="uint8")
    for row in rows:
        column, raster_row = ~transform * (row["x"], row["y"])
        sample_mask[int(math.floor(raster_row)), int(math.floor(column))] = 1
        row["download_region_id"] = region["properties"]["fragment_id"]
    if int(sample_mask.sum()) != len(rows):
        raise ValueError("Selected centers are not unique on the retrieval grid.")
    path = (
        Path(field_folder)
        / "balanced_pilot"
        / "sampling_masks"
        / f"{region['properties']['fragment_id']}.tif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid["width"],
        height=grid["height"],
        count=1,
        dtype="uint8",
        crs=grid["crs"],
        transform=transform,
        compress="deflate",
    ) as raster:
        raster.write(sample_mask, 1)
        raster.set_band_description(1, "selected_training_pixel")
    region["properties"].update(sampling_mask=str(path.resolve()), sampling_grid=grid)


def save_candidate_pilot(selected_pixels, field_folder, seed=42):
    """Write source-specific regions, masks and counts as a portable acquisition input."""
    field_folder = Path(field_folder)
    groups = defaultdict(list)
    for row in selected_pixels:
        groups[(row["crop"], row["partition_group"])].append(row)
    regions = []
    for (crop, group_name), rows in sorted(groups.items()):
        region = selected_region(crop, group_name, rows, seed)
        save_sampling_mask(region, rows, field_folder)
        regions.append(region)
    save_json(
        field_folder / "sampled_download_regions.geojson",
        {"type": "FeatureCollection", "features": regions},
    )
    columns = [
        "pixel_id",
        "crop",
        "partition_group",
        "download_region_id",
        "source_crs",
        "x",
        "y",
        "source_fragments",
    ]
    write_table(field_folder / "balanced_pilot" / "selected_pixels.csv", selected_pixels, columns)
    summary = {
        "candidate_pixels_per_crop": dict(Counter(row["crop"] for row in selected_pixels)),
        "groups_per_crop": dict(Counter(region["properties"]["crop"] for region in regions)),
        "download_regions": len(regions),
        "sampling_seed": seed,
        "usable_target_per_crop": USABLE_PIXELS_PER_CROP,
        "stage": "candidates_before_quality_filtering",
    }
    save_json(field_folder / "balanced_pilot" / "sampling_summary.json", summary)
    return regions, summary


def trim_usable_pixels(
    reference_rows,
    estimated_rows,
    target_per_crop=1000,
    seed=42,
    crops=("carrot", "cabbage", "potato"),
):
    """Cap unique locations across both policies, giving supported windows priority.

    Only external pilot rows should be supplied. Held-out local evaluation data
    are never balanced or selected using their feature quality here.
    """
    if target_per_crop < 1:
        raise ValueError("Usable-pixel target must be positive.")
    candidates = unique_usable_windows(reference_rows, estimated_rows)
    selected = []
    summaries = []
    for crop in crops:
        crop_rows = [row for row in candidates if row["crop"] == crop]
        kept_rows = round_robin_fields(crop_rows, target_per_crop, seed)
        selected.extend(kept_rows)
        summaries.append(
            {
                "crop": crop,
                "available_usable_pixels": len(crop_rows),
                "selected_usable_pixels": len(kept_rows),
                "target_pixels": target_per_crop,
                "shortfall_pixels": max(0, target_per_crop - len(kept_rows)),
                "status": (
                    "target_met"
                    if len(kept_rows) == target_per_crop
                    else "insufficient_usable_pixels"
                ),
            }
        )
    reference = [row for row in selected if row["window_status"] == "reference_supported"]
    estimated = [row for row in selected if row["window_status"] == "satellite_estimated"]
    return reference, estimated, summaries


def unique_usable_windows(reference_rows, estimated_rows):
    """Choose one season per physical pixel so multiple cycles cannot inflate totals."""
    ordered_rows = sorted(reference_rows, key=lambda row: str(row.get("season_key", "")))
    ordered_rows += sorted(estimated_rows, key=lambda row: str(row.get("season_key", "")))
    unique = {}
    for row in ordered_rows:
        location = pixel_location(row)
        if location in unique and unique[location]["crop"] != row["crop"]:
            raise ValueError("Conflicting crop labels at one usable pixel location.")
        unique.setdefault(location, row)
    return list(unique.values())


def round_robin_fields(rows, target, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["partition_group"]].append(row)
    queues = [
        deque(
            sorted(
                groups[name],
                key=lambda row: (
                    row.get("window_status") != "reference_supported",
                    stable_rank(pixel_location(row), seed),
                ),
            )
        )
        for name in sorted(groups, key=lambda name: stable_rank(name, seed))
    ]
    selected = []
    while queues and len(selected) < target:
        queue = queues.pop(0)
        selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    return selected


def trim_external_pilot(
    reference_rows,
    estimated_rows,
    registry=(),
    target_per_crop=1000,
    seed=42,
    crops=("carrot", "cabbage", "potato"),
):
    """Leave local train/test rows untouched; balance only external candidate sources."""
    local_groups = {row["partition_group"] for row in registry if row.get("source") == "local"}
    local_reference = [row for row in reference_rows if row["partition_group"] in local_groups]
    external_reference = [
        row for row in reference_rows if row["partition_group"] not in local_groups
    ]
    if any(row["partition_group"] in local_groups for row in estimated_rows):
        raise ValueError("Estimated pilot windows cannot be used as local labels.")
    selected_reference, selected_estimates, summary = trim_usable_pixels(
        external_reference, estimated_rows, target_per_crop, seed, crops
    )
    return local_reference + selected_reference, selected_estimates, summary
