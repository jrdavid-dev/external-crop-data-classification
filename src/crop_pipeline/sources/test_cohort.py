"""Reserve CropHarvest JECAM in a campaign separate from development."""

import json
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree

from crop_pipeline.sources import adaptive, pilot_sources

REQUIRED_DEVELOPMENT = {"sen4agrinet", "francecrops"}
DEVELOPMENT = REQUIRED_DEVELOPMENT | {"eurocrops"}
TEST = "cropharvest"
TEST_POLICY = {**adaptive.DEFAULT_POLICY, "test_source": TEST}
CABBAGE_SUPPLEMENT_POLICY = {**TEST_POLICY, "cohort": "non_jecam_cabbage_supplement_v1"}
CARROT_POTATO_SUPPLEMENT_POLICY = {**TEST_POLICY, "cohort": "non_jecam_carrot_potato_supplement_v1"}


def source_role(source_name):
    """Annual handoffs append the observation year to the source name."""
    base, separator, year = source_name.rpartition("_")
    return base if separator and len(year) == 4 and year.isdigit() else source_name


def completed_sources(report_folder, policy=None):
    paths, _ = adaptive.resolve_collection(report_folder, policy)
    return pilot_sources.load_sources(paths)


def source_footprints(sources):
    for source in sources:
        folder = Path(source["field_folder"])
        regions = json.loads((folder / "sampled_download_regions.geojson").read_text())["features"]
        groups = {item["properties"]["partition_group"] for item in regions}
        footprints = json.loads((folder / "source_footprints.geojson").read_text())["features"]
        found = set()
        for item in footprints:
            group = item["properties"]["partition_group"]
            if group in groups:
                found.add(group)
                yield item
        if found != groups:
            raise ValueError(f"Missing original source footprints in {folder}")


def exclude_overlap(candidates, development):
    """Reject whole test parcels whose original outlines overlap development."""
    index = STRtree([shape(item["geometry"]) for item in development])
    excluded = set()
    candidates = list(candidates)
    for item in candidates:
        geometry = shape(item["geometry"])
        if any(geometry.intersection(index.geometries[i]).area > 0
               for i in index.query(geometry, predicate="intersects")):
            excluded.add(item["properties"]["partition_group"])
    return [item for item in candidates if item["properties"]["partition_group"] not in excluded], excluded


def prepare(report_folder, archive_root, features, provenance, limit, extra_batches=0, candidate_crops=None):
    candidates = [item for item in features if item["properties"]["source_dataset"] == "jecam"]
    print(f"JECAM candidates: {len(candidates)} | Development overlap checked in notebook 05")
    return adaptive.prepare_round_batches(
        TEST, candidates, archive_root, report_folder,
        {**provenance, "cohort": "jecam", "development_overlap": "pending_notebook_05"},
        policy=TEST_POLICY, parcels_per_crop=limit, extra_batches=extra_batches, candidate_crops=candidate_crops,
    )


def cabbage_supplement_candidates(report_folder, features):
    """Screen non-JECAM cabbage against completed development and JECAM footprints."""
    candidates = [
        item for item in features
        if item["properties"]["crop"] == "cabbage"
        and item["properties"]["source_dataset"] != "jecam"
    ]
    development = completed_sources(report_folder)
    jecam = completed_sources(report_folder, TEST_POLICY)
    return exclude_overlap(candidates, source_footprints(development + jecam))


def prepare_cabbage_supplement(report_folder, archive_root, features, provenance, limit):
    candidates, excluded = cabbage_supplement_candidates(report_folder, features)
    print(f"Supplemental cabbage: {len(candidates)} eligible footprints, {len(excluded)} overlapping parcel groups excluded")
    return adaptive.prepare_round_batches(
        TEST, candidates, archive_root, report_folder,
        {**provenance, "cohort": CABBAGE_SUPPLEMENT_POLICY["cohort"],
         "excluded_overlap_groups": sorted(excluded)},
        policy=CABBAGE_SUPPLEMENT_POLICY, parcels_per_crop=limit, candidate_crops=("cabbage",),
    )


def prepare_carrot_potato_supplement(report_folder, archive_root, features, provenance, limit):
    """Collect separately after screening against development and reserved test parcels."""
    from crop_pipeline.windows.seasons import file_sha256

    crops = ("carrot", "potato")
    features = list(features)
    development = completed_sources(report_folder)
    occupied = list(source_footprints(development))
    for policy in (TEST_POLICY, CABBAGE_SUPPLEMENT_POLICY):
        campaign = adaptive.root(report_folder, policy)
        selections = [json.loads(path.read_text())
                      for path in campaign.glob("rounds/*/*/selection.json")]
        keys = {key for selection in selections for key in selection["physical_keys"]}
        selected = [f for f in features if adaptive.physical_key(TEST, f["properties"]) in keys]
        occupied.extend(selected)
        selected_keys = {adaptive.physical_key(TEST, f["properties"]) for f in selected}
        for path in sorted(campaign.glob("rounds/*/*/completed.json")):
            handoffs = json.loads(path.read_text())["handoffs"]
            handoff_paths = [Path(record["path"]) for record in handoffs]
            if any(file_sha256(p) != record["sha256"] for p, record in zip(handoff_paths, handoffs)):
                raise ValueError("Existing test handoff changed after registration")
            occupied.extend(source_footprints(pilot_sources.load_sources(handoff_paths)))
        if keys - selected_keys:
            raise ValueError("Current references lack saved test selections needed for overlap screening")
    candidates = [f for f in features if f["properties"]["crop"] in crops
                  and f["properties"]["source_dataset"] != "jecam"]
    candidates, excluded = exclude_overlap(candidates, occupied)
    print(f"Supplemental carrot and potato: {len(candidates)} eligible footprints, "
          f"{len(excluded)} overlapping parcel groups excluded")
    return adaptive.prepare_round_batches(
        TEST, candidates, archive_root, report_folder,
        {**provenance, "cohort": CARROT_POTATO_SUPPLEMENT_POLICY["cohort"],
         "excluded_overlap_groups": sorted(excluded)},
        policy=CARROT_POTATO_SUPPLEMENT_POLICY, parcels_per_crop=limit,
        extra_batches=5, candidate_crops=crops,
    )
