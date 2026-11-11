"""Reserve CropHarvest JECAM in a campaign separate from development."""

import json
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree

from crop_pipeline.sources import adaptive, pilot_sources

DEVELOPMENT = {"sen4agrinet", "francecrops"}
TEST = "cropharvest"
TEST_POLICY = {**adaptive.DEFAULT_POLICY, "test_source": TEST}


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


def prepare(report_folder, archive_root, features, provenance, limit):
    sources = [s for s in completed_sources(report_folder) if source_role(s["source_name"]) in DEVELOPMENT]
    candidates = [item for item in features if item["properties"]["source_dataset"] == "jecam"]
    eligible, excluded = exclude_overlap(candidates, source_footprints(sources))
    print(f"JECAM candidates: {len(candidates)} | overlapping groups excluded: {len(excluded)}")
    return [adaptive.prepare_batch(
        TEST, eligible, archive_root, report_folder,
        {**provenance, "cohort": "jecam", "excluded_overlap_groups": sorted(excluded)},
        policy=TEST_POLICY, round_parcels_per_crop=limit,
    )]
