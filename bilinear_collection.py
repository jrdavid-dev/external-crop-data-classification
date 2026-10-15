"""Prepare a larger, timing-first collection without changing the existing pilot."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

import crop_workflow
import pilot_sampling
import season_workflow
from satellite_workflow import REQUEST_VERSION

# Carrot capacity is 3,471 under the field cap; keep a feasible rounded budget.
CANDIDATE_BUDGETS = {"carrot": 3400, "cabbage": 7000, "potato": 10000}
COLLECTION_NAME = "bilinear_expanded"


def fragment_pixel_records(feature):
    """Recover label-grid centers using the same rasterization as the source notebook."""
    properties = feature["properties"]
    coordinate_system = properties["source_crs"]
    projector = Transformer.from_crs("EPSG:4326", coordinate_system, always_xy=True)
    geometry = transform_geometry(projector.transform, shape(feature["geometry"]))
    west, south, east, north = geometry.bounds
    left = math.floor(west / 10 + 1e-6) * 10
    right = math.ceil(east / 10 - 1e-6) * 10
    bottom = math.floor(south / 10 + 1e-6) * 10
    top = math.ceil(north / 10 - 1e-6) * 10
    width, height = int(round((right - left) / 10)), int(round((top - bottom) / 10))
    if width < 1 or height < 1 or width * height > 5000000:
        raise ValueError("Unexpected label grid; investigate before sampling.")
    mask = rasterize(
        [(mapping(geometry), 1)],
        out_shape=(height, width),
        transform=Affine(10, 0, left, 0, -10, top),
        dtype="uint8",
    )
    rows, columns = np.where(mask == 1)
    group = f"{properties['country']}|{properties['year']}|{properties['parcel_id']}"
    return [
        {
            "source_crs": coordinate_system,
            "x": left + int(column) * 10 + 5,
            "y": top - int(row) * 10 - 5,
            "crop": properties["crop"],
            "partition_group": group,
            "source_fragments": [properties["fragment_id"]],
        }
        for row, column in zip(rows, columns)
    ]


def build_reference_catalog(features):
    """Exclude overlapping labels or field identities rather than inflate capacity."""
    locations = defaultdict(list)
    for feature in features:
        for record in fragment_pixel_records(feature):
            locations[pilot_sampling.pixel_location(record)].append(record)
    catalog = []
    for records in locations.values():
        if (
            len({(record["crop"], record["partition_group"]) for record in records})
            != 1
        ):
            continue
        fragments = sorted(
            {fragment for record in records for fragment in record["source_fragments"]}
        )
        catalog.append({**records[0], "source_fragments": fragments})
    return sorted(
        catalog,
        key=lambda row: (
            row["crop"],
            row["partition_group"],
            pilot_sampling.pixel_location(row),
        ),
    )


def capacity_report(catalog, budgets, maximum_per_group=100):
    """Expose reference shortages before contacting Earth Engine or sampling."""
    groups = defaultdict(Counter)
    for row in catalog:
        groups[row["crop"]][row["partition_group"]] += 1
    return [
        {
            "crop": crop,
            "requested_candidates": budget,
            "available_unique_pixels": sum(groups[crop].values()),
            "available_capped_pixels": sum(
                min(count, maximum_per_group) for count in groups[crop].values()
            ),
            "available_groups": len(groups[crop]),
            "shortfall": max(
                0,
                budget
                - sum(min(count, maximum_per_group) for count in groups[crop].values()),
            ),
        }
        for crop, budget in budgets.items()
    ]


def prepare_candidates(catalog, budgets, field_folder, seed=42):
    """Require genuine capacity; never lower the requested budget or duplicate pixels."""
    if any(budget <= 3000 for budget in budgets.values()):
        raise ValueError("Candidate budgets must exceed 3,000 unique pixels per crop.")
    report = capacity_report(catalog, budgets)
    crop_workflow.save_json(Path(field_folder) / "candidate_capacity.json", report)
    shortages = [row for row in report if row["shortfall"]]
    if shortages:
        raise ValueError(
            f"More reference fields required before collection: {shortages}"
        )
    selected = []
    for crop_number, (crop, budget) in enumerate(budgets.items()):
        selected.extend(
            pilot_sampling.sample_crop_candidates(
                [row for row in catalog if row["crop"] == crop],
                budget,
                60,
                100,
                seed + crop_number,
            )
        )
    return pilot_sampling.save_candidate_pilot(selected, field_folder, seed)


def select_timing_plans(plans, observations, destination):
    """Reuse the window notebook's unchanged detector and automatic timing policy."""
    destination = Path(destination)
    regions = [plan["fragment"] for plan in plans]
    if any(plan["settings"]["version"] != REQUEST_VERSION for plan in plans):
        raise ValueError("Future collection requires version 4 bilinear imagery.")
    settings = dict(crop_workflow.DEFAULT_DETECTOR)
    candidates = crop_workflow.detect_greenup(observations, settings)
    season_workflow.prepare_review_files(
        candidates, regions, settings, destination / "review"
    )
    configuration = json.loads(
        (destination / "review/frozen_configuration.template.json").read_text()
    )
    # Timing-first retrieval must consider all supported cycles; completeness
    # chooses the earliest usable cycle after imagery has been audited.
    configuration["cycle_selection_policy"] = "earliest_complete"
    _, proposals, audit = season_workflow.select_season_windows(
        candidates, [], regions, configuration
    )
    windows, assessments, audit = season_workflow.qualify_estimated_windows(
        proposals, audit, observations, configuration
    )
    crop_workflow.save_json(
        destination / "timing_gate.json",
        {
            "configuration": configuration,
            "proposals": proposals,
            "qualified_windows": windows,
            "assessments": assessments,
            "audit": audit,
        },
    )
    by_region = defaultdict(list)
    for window in windows:
        by_region[window["region_id"]].append(window)
    selected, pending = [], []
    for plan in plans:
        selected_plan, problem = plan_window_intervals(
            plan, by_region[plan["fragment_id"]]
        )
        if problem:
            pending.append(problem)
        if selected_plan:
            selected.append(selected_plan)
    crop_workflow.save_json(
        destination / "download_selection.json",
        {
            "qualified_regions": len(selected),
            "planned_composites": sum(
                len(plan["download_intervals"]) for plan in selected
            ),
            "pending": pending,
            "minimum_valid_fraction": 0.8,
        },
    )
    return selected


def window_fits_request(window, request):
    """Keep complete cycles only; clipping would weaken eight-bin completeness."""
    return (
        window["window_start"] >= request["start"]
        and window["window_end"] <= request["end"]
    )


def plan_window_intervals(plan, windows):
    """Retain in-year cycles independently and audit unsupported cycles separately."""
    if not windows:
        return None, {
            "region_id": plan["fragment_id"],
            "reason": "no_timing_qualified_window",
        }
    request = plan["settings"]
    supported_windows = [
        window for window in windows if window_fits_request(window, request)
    ]
    excluded_windows = [
        window for window in windows if not window_fits_request(window, request)
    ]
    problem = None
    if excluded_windows:
        problem = {
            "region_id": plan["fragment_id"],
            "reason": "adjacent_year_imagery_required",
            "excluded_windows": [
                {
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                }
                for window in excluded_windows
            ],
            "retained_window_count": len(supported_windows),
        }
    if not supported_windows:
        return None, problem
    intervals = {
        interval["name"]: interval
        for window in supported_windows
        for interval in season_workflow.window_intervals(window)
    }
    return {
        **plan,
        "download_intervals": sorted(
            intervals.values(), key=lambda interval: interval["start"]
        ),
    }, problem


def download_pixel_counts(plans, crops):
    """Count each selected sampling footprint once, not its repeated image intervals."""
    counts = Counter({crop: 0 for crop in crops})
    identifiers = set()
    for plan in plans:
        identifier = plan["fragment_id"]
        if identifier in identifiers:
            raise ValueError(
                "Duplicate region plans would inflate the download-pixel count."
            )
        identifiers.add(identifier)
        properties = plan["fragment"]["properties"]
        crop = properties["crop"]
        if crop not in counts:
            raise ValueError(
                "Download plan contains a crop outside the requested source budget."
            )
        counts[crop] += int(properties["selected_pixels"])
    return dict(counts)


def require_download_pixel_budget(plans, crops):
    """Stop before raster transfer if timing filtering leaves 3,000 or fewer per crop."""
    counts = download_pixel_counts(plans, crops)
    shortages = {crop: count for crop, count in counts.items() if count <= 3000}
    if shortages:
        raise ValueError(
            f"More timing-qualified reference pixels are needed before downloading: {shortages}. "
            "Each crop must exceed 3,000 unique sampled pixels. Add reference fields; do not relax timing or quality rules."
        )
    return counts


def verify_download_completion(plans, summary, crops, report_path):
    """Distinguish transferred sampling footprints from complete, usable feature pixels."""
    completed_intervals = {
        (row["region_id"], row["name"])
        for row in summary["results"]
        if row["status"] in {"downloaded", "reused"}
    }
    completed_plans = [
        plan
        for plan in plans
        if plan["download_intervals"]
        and all(
            (plan["fragment_id"], interval["name"]) in completed_intervals
            for interval in plan["download_intervals"]
        )
    ]
    counts = download_pixel_counts(completed_plans, crops)
    report = {
        "downloaded_sampled_pixels": counts,
        "minimum_exclusive_per_crop": 3000,
        "status": (
            "complete"
            if all(count > 3000 for count in counts.values())
            else "insufficient_completed_downloads"
        ),
        "usable_pixels": "not_measured_until_window_quality_checks",
    }
    crop_workflow.save_json(report_path, report)
    if report["status"] != "complete":
        raise ValueError(
            f"Completed image downloads do not yet cover more than 3,000 sampled pixels per crop: {counts}. Rerun failed downloads or add timing-qualified fields."
        )
    return report
