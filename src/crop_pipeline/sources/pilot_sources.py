"""Read source-specific acquisition handoffs without downloading any data."""

import json
from pathlib import Path

from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.imagery.quality import save_json
from crop_pipeline.windows.seasons import file_sha256
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.sources.sampling import USABLE_PIXELS_PER_CROP


def validate_source_contract(source):
    """Allow legacy imagery only when the window run explicitly opts in."""
    if (
        source.get("allow_existing_version_3", False)
        and source.get("request_version") == 3
        and source.get("resampling") == "nearest"
    ):
        return
    if (
        source.get("request_version") != REQUEST_VERSION
        or source.get("resampling") != RESAMPLING_POLICY
    ):
        raise ValueError(
            "Source uses an old imagery contract. Rerun satellite retrieval with bilinear B5/B11 and export its handoff; existing nearest-neighbor files are preserved."
        )


def load_sources(
    source_manifest_paths, allow_existing_version_3=False, require_all=True
):
    """Check required handoffs before reading inputs to avoid partial pilot runs."""
    source_manifest_paths = [Path(path) for path in source_manifest_paths]
    missing_paths = [path for path in source_manifest_paths if not path.is_file()]
    if require_all and missing_paths:
        missing_names = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Required source handoffs missing: {missing_names}. "
            "Finish both source-preparation notebooks before running the window notebook."
        )
    sources = []
    for path in source_manifest_paths:
        path = Path(path)
        if not path.is_file():
            print("Source not prepared:", path)
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        source["allow_existing_version_3"] = allow_existing_version_3
        validate_source_contract(source)
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
        source_regions, source_quality, source_pixels, source_observations = (
            audit_source(source, analysis_folder, minimum_fraction)
        )
        identifiers = [region["properties"]["fragment_id"] for region in source_regions]
        if len(identifiers) != len(set(identifiers)) or seen_region_ids.intersection(
            identifiers
        ):
            raise ValueError(
                "Source region identifiers collide; preserve source namespaces."
            )
        seen_region_ids.update(identifiers)
        regions.extend(source_regions)
        quality_rows.extend(source_quality)
        pixel_rows.extend(source_pixels)
        observed_rows.extend(source_observations)
    save_json(
        Path(analysis_folder) / "timelines" / "observed_timelines.json", observed_rows
    )
    save_json(Path(analysis_folder) / "source_inputs.json", sources)
    return regions, quality_rows, pixel_rows


def audit_source(source, analysis_folder, minimum_fraction):
    validate_source_contract(source)
    field_folder = Path(source["field_folder"])
    regions = json.loads(
        (field_folder / "sampled_download_regions.geojson").read_text()
    )["features"]
    for region in regions:
        region["properties"].update(
            source_dataset=source["source_name"],
            source_year=source["year"],
            dataset_revision=source["dataset_revision"],
            imagery_request_version=source.get("request_version", 3),
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
        required_request_version=source.get("request_version", 3),
    )
    observed_path = Path(source["timeline_folder"]) / "observed_timelines.json"
    observations = (
        json.loads(observed_path.read_text()) if observed_path.exists() else []
    )
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
    # Describe the files actually retrieved, even when this kernel has newer helpers.
    progress_records = [
        json.loads(record_path.read_text(encoding="utf-8"))
        for record_path in sorted(Path(progress_folder).glob("*.json"))
    ]
    contract_pairs = {
        (
            record["settings"]["version"],
            json.dumps(record["settings"]["resampling"], sort_keys=True),
        )
        for record in progress_records
    }
    if len(contract_pairs) > 1:
        raise ValueError(
            "Mixed imagery contracts: select one progress folder before exporting."
        )
    request_version, resampling = REQUEST_VERSION, RESAMPLING_POLICY
    if contract_pairs:
        request_version, encoded_resampling = next(iter(contract_pairs))
        resampling = json.loads(encoded_resampling)
    region_path = Path(field_folder) / "sampled_download_regions.geojson"
    regions = json.loads(region_path.read_text(encoding="utf-8"))["features"]
    region_identifiers = {region["properties"]["fragment_id"] for region in regions}
    downloaded_identifiers = {record["fragment_id"] for record in progress_records}
    if downloaded_identifiers - region_identifiers:
        raise ValueError(
            "Downloaded regions differ from the field folder; use the matching pilot_3000 folder."
        )
    payload = {
        "request_version": request_version,
        "resampling": resampling,
        "source_name": source_name,
        "year": year,
        "dataset_revision": revision,
        "field_folder": str(Path(field_folder).resolve()),
        "imagery_folder": str(Path(imagery_folder).resolve()),
        "progress_folder": str(Path(progress_folder).resolve()),
        "timeline_folder": str(Path(timeline_folder).resolve()),
        "cloud_threshold": cloud_threshold,
        "candidate_target_per_crop": 3000,
        "usable_target_per_crop": USABLE_PIXELS_PER_CROP,
    }
    region_path = Path(field_folder) / "sampled_download_regions.geojson"
    payload["regions_sha256"] = file_sha256(region_path)
    save_json(path, payload)
    return payload
