"""CropHarvest polygon labels prepared with the same sampling rules as FranceCrops."""

import math
from collections import Counter

from shapely.geometry import mapping

from crop_pipeline.imagery.quality import save_json
from crop_pipeline.sources import sampling as pilot_sampling

DATASET_NAME = "nasaharvest/cropharvest"
# Same seed offsets as bilinear.prepare_candidates, which follows CANDIDATE_BUDGETS order.
CROP_SEED_ORDER = ("carrot", "cabbage", "potato")
MAXIMUM_PIXELS_PER_GROUP = 100
PREFERRED_GROUPS = 60


def utm_crs(longitude, latitude):
    """UTM zone of the parcel centroid, southern-hemisphere zones included."""
    zone = int(math.floor((longitude + 180) / 6)) + 1
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def parcel_feature(uid, geometry, crop, year, country, source_dataset, revision):
    """One labelled CropHarvest polygon as a FranceCrops-style parcel feature."""
    if (
        geometry is None
        or geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError(f"Invalid or non-polygon footprint: {uid}")
    longitude, latitude = geometry.centroid.coords[0]
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "fragment_id": f"cropharvest_{year}_{uid.replace(':', '_')}",
            "parcel_id": uid,
            "partition_group": f"cropharvest|{country}|{year}|{uid}",
            "crop": crop,
            "country": country,
            "year": int(year),
            "source_crs": utm_crs(longitude, latitude),
            "dataset": DATASET_NAME,
            "dataset_revision": revision,
            "source_dataset": source_dataset,
            "label_status": "reported",
            "label_evidence": f"CropHarvest {source_dataset} label '{crop}' ({revision})",
        },
    }


def interior_capacity(features, catalog, maximum_per_group=MAXIMUM_PIXELS_PER_GROUP):
    """Interior pixels per field after the inward buffer, listing fields left with none."""
    counts = Counter(row["partition_group"] for row in catalog)
    rows = []
    for feature in features:
        properties = feature["properties"]
        pixels = counts.get(properties["partition_group"], 0)
        rows.append(
            {
                "parcel_id": properties["parcel_id"],
                "partition_group": properties["partition_group"],
                "crop": properties["crop"],
                "year": properties["year"],
                "source_dataset": properties["source_dataset"],
                "interior_pixels": pixels,
                "capped_pixels": min(pixels, maximum_per_group),
            }
        )
    return rows


def sample_year_candidates(catalog, field_folder, seed=42, budget_caps=None):
    """Select up to the pipeline candidate budget from one observation year.

    Where CropHarvest cannot supply that budget (or the 3,000-candidate minimum), the
    budget is the full capped capacity instead. Selection, spreading and masks are unchanged.
    """
    selected, budgets = [], {}
    for crop_number, crop in enumerate(CROP_SEED_ORDER):
        rows = [row for row in catalog if row["crop"] == crop]
        if not rows:
            continue
        groups = Counter(row["partition_group"] for row in rows)
        budget = sum(min(count, MAXIMUM_PIXELS_PER_GROUP) for count in groups.values())
        if budget_caps and crop in budget_caps:
            budget = min(budget, budget_caps[crop])
        budgets[crop] = budget
        selected.extend(
            pilot_sampling.sample_crop_candidates(
                rows, budget, PREFERRED_GROUPS, MAXIMUM_PIXELS_PER_GROUP, seed + crop_number
            )
        )
    regions, summary = pilot_sampling.save_candidate_pilot(selected, field_folder, seed)
    return regions, summary, budgets


def download_completion(plans, summary, report_path):
    """Sampled pixels whose qualified half-month composites all downloaded, per crop."""
    completed = {
        (row["region_id"], row["name"])
        for row in summary["results"]
        if row["status"] in {"downloaded", "reused"}
    }
    counts = Counter()
    incomplete = []
    for plan in plans:
        properties = plan["fragment"]["properties"]
        if all((plan["fragment_id"], interval["name"]) in completed
               for interval in plan["download_intervals"]):
            counts[properties["crop"]] += int(properties["selected_pixels"])
        else:
            incomplete.append(plan["fragment_id"])
    report = {
        "downloaded_sampled_pixels": dict(counts),
        "incomplete_regions": incomplete,
        "minimum_exclusive_per_crop_in_pipeline": 3000,
        "usable_pixels": "not_measured_until_window_quality_checks",
    }
    save_json(report_path, report)
    return report
