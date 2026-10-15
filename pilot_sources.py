"""Read source-specific acquisition handoffs without downloading any data."""

import json
from pathlib import Path

import crop_workflow
from crop_workflow import save_json
from season_workflow import file_sha256


def load_sources(source_manifest_paths):
    sources = []
    for path in source_manifest_paths:
        path = Path(path)
        if not path.is_file():
            print("Source not prepared:", path)
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        region_path = Path(source["field_folder"]) / "sampled_download_regions.geojson"
        if source.get("regions_sha256") != file_sha256(region_path):
            raise ValueError(
                "Source sampling changed after its handoff; rerun acquisition preparation and export."
            )
        source["manifest_path"] = str(path.resolve())
        sources.append(source)
    if not sources:
        raise FileNotFoundError(
            "Run a source-preparation notebook first; no acquisition handoffs were found."
        )
    return sources


def audit_sources(sources, analysis_folder, minimum_fraction=0.8):
    """Audit each source in its own observation year before combining record tables."""
    regions, quality_rows, pixel_rows, observed_rows = [], [], [], []
    seen_region_ids = set()
    for source in sources:
        source_regions, source_quality, source_pixels, source_observations = audit_source(
            source, analysis_folder, minimum_fraction
        )
        identifiers = [region["properties"]["fragment_id"] for region in source_regions]
        if len(identifiers) != len(set(identifiers)) or seen_region_ids.intersection(identifiers):
            raise ValueError("Source region identifiers collide; preserve source namespaces.")
        seen_region_ids.update(identifiers)
        regions.extend(source_regions)
        quality_rows.extend(source_quality)
        pixel_rows.extend(source_pixels)
        observed_rows.extend(source_observations)
    save_json(Path(analysis_folder) / "timelines" / "observed_timelines.json", observed_rows)
    save_json(Path(analysis_folder) / "source_inputs.json", sources)
    return regions, quality_rows, pixel_rows


def audit_source(source, analysis_folder, minimum_fraction):
    field_folder = Path(source["field_folder"])
    regions = json.loads((field_folder / "sampled_download_regions.geojson").read_text())[
        "features"
    ]
    for region in regions:
        region["properties"].update(
            source_dataset=source["source_name"],
            source_year=source["year"],
            dataset_revision=source["dataset_revision"],
        )
    quality, pixels = crop_workflow.quality_check(
        regions,
        source["progress_folder"],
        source["imagery_folder"],
        field_folder,
        Path(analysis_folder) / "quality" / source["source_name"],
        cloud_threshold=source["cloud_threshold"],
        minimum_fraction=minimum_fraction,
        year=int(source["year"]),
    )
    observed_path = Path(source["timeline_folder"]) / "observed_timelines.json"
    observations = json.loads(observed_path.read_text()) if observed_path.exists() else []
    identifiers = {region["properties"]["fragment_id"] for region in regions}
    observations = [row for row in observations if row["region_id"] in identifiers]
    if not observations:
        print(
            "No observed daily summaries for",
            source["source_name"],
            "— automatic timing will remain pending.",
        )
    return regions, quality, pixels, observations


def save_source_manifest(
    path,
    source_name,
    year,
    field_folder,
    imagery_folder,
    progress_folder,
    timeline_folder,
    revision,
    cloud_threshold=40,
):
    """Persist the acquisition boundary so window preparation needs no notebook globals."""
    payload = {
        "source_name": source_name,
        "year": year,
        "dataset_revision": revision,
        "field_folder": str(Path(field_folder).resolve()),
        "imagery_folder": str(Path(imagery_folder).resolve()),
        "progress_folder": str(Path(progress_folder).resolve()),
        "timeline_folder": str(Path(timeline_folder).resolve()),
        "cloud_threshold": cloud_threshold,
        "candidate_target_per_crop": 3000,
        "usable_target_per_crop": 1000,
    }
    region_path = Path(field_folder) / "sampled_download_regions.geojson"
    payload["regions_sha256"] = file_sha256(region_path)
    save_json(path, payload)
    return payload
