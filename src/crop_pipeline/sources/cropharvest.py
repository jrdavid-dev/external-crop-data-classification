"""CropHarvest polygon labels prepared with the same sampling rules as FranceCrops."""

import math
import ast
import hashlib
import json
from datetime import date
from pathlib import Path
from collections import Counter

from shapely.geometry import mapping

from crop_pipeline.imagery.quality import save_json
from crop_pipeline.imagery import bilinear
from crop_pipeline.windows.seasons import file_sha256
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
    """Use shared crop targets, reducing to genuine capacity without duplicating pixels."""
    caps = dict(bilinear.CANDIDATE_BUDGETS)
    caps.update(budget_caps or {})
    if any(value <= 0 for value in caps.values()):
        raise ValueError("Candidate caps must be positive.")
    selected, budgets = [], {}
    for crop_number, crop in enumerate(CROP_SEED_ORDER):
        rows = [row for row in catalog if row["crop"] == crop]
        if not rows:
            continue
        groups = Counter(row["partition_group"] for row in rows)
        budget = sum(min(count, MAXIMUM_PIXELS_PER_GROUP) for count in groups.values())
        budget = min(budget, caps[crop])
        budgets[crop] = budget
        selected.extend(
            pilot_sampling.sample_crop_candidates(
                rows,
                budget,
                PREFERRED_GROUPS,
                MAXIMUM_PIXELS_PER_GROUP,
                seed + crop_number,
            )
        )
    regions, summary = pilot_sampling.save_candidate_pilot(selected, field_folder, seed)
    return regions, summary, budgets


def download_completion(plans, summary, report_path):
    """Apply the shared all-planned-interval check without a 3,000-pixel minimum."""
    crops = sorted({plan["fragment"]["properties"]["crop"] for plan in plans})
    if not crops:
        raise ValueError("No timing-qualified CropHarvest plans to complete.")
    return bilinear.verify_download_completion(plans, summary, crops, report_path)


def parse_date(value):
    if value is None or str(value) in ("", "NaT", "nan", "None"):
        return None
    return date.fromisoformat(str(value)[:10])


def resolve_observation_year(collection_date, export_end_date, override_year=None):
    """Use supported observation years without turning placeholder days into crop dates.

    Export-end dates describe coverage, not planting. A January 1 placeholder
    supplies only a season year when its year agrees with the export-period
    start; contradictory years remain pending. No planting date is inferred.
    """
    if override_year is not None:
        year = int(override_year)
        if year < 2017 or year > 2100:
            raise ValueError(
                "Observation-year override is outside the supported range."
            )
        return {"year": year, "status": "resolved", "reason": "explicit_year_override"}
    observation = parse_date(collection_date)
    export_end = parse_date(export_end_date)
    if observation is None or export_end is None:
        return {
            "year": None,
            "status": "pending",
            "reason": "missing_observation_or_export_date",
        }
    # Calendar-year subtraction mirrors the metadata's twelve-month coverage period.
    start_year = export_end.year - 1
    try:
        export_start = export_end.replace(year=start_year)
    except ValueError:
        export_start = export_end.replace(year=start_year, day=28)
    if observation.month == 1 and observation.day == 1:
        if observation.year == export_start.year:
            return {
                "year": observation.year,
                "status": "resolved",
                "reason": "season_year_supported_by_export_period",
            }
        return {
            "year": None,
            "status": "pending",
            "reason": "placeholder_year_conflicts_with_export_period",
        }
    if not export_start <= observation <= export_end:
        return {
            "year": None,
            "status": "pending",
            "reason": "observation_outside_export_period",
        }
    return {
        "year": observation.year,
        "status": "resolved",
        "reason": "dated_observation_year",
    }


def initialize_run(root, collection_name, configuration):
    """Version local outputs by inputs/settings so earlier collections remain intact."""
    encoded = json.dumps(configuration, sort_keys=True, allow_nan=False)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    folder = Path(root) / "runs" / collection_name / fingerprint
    path = folder / "run_configuration.json"
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Run fingerprint collision; do not overwrite earlier outputs.")
    if not path.exists():
        save_json(path, configuration)
    return folder, fingerprint


def notebook_source_fingerprint(path):
    """Ignore runtime switches so inspecting a completed run does not change its identity."""
    runtime_switches = {
        "download_missing_labels",
        "download_missing_reference_tables",
        "run_satellite_collection",
        "authenticate_now",
        "publish_earth_engine_assets",
        "activate_completed_run",
        "wait_for_batch_tasks",
        "batch_poll_seconds",
        "maximum_batch_wait_seconds",
    }
    notebook = json.loads(Path(path).read_text())
    normalized = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [
            statement
            for statement in tree.body
            if not (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in runtime_switches
                    for target in statement.targets
                )
            )
        ]
        normalized.append(ast.dump(tree, include_attributes=False))
    return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()


def publish_handoff_bundle(report_root, collection_name, run_folder, handoff_paths):
    """Register immutable handoffs for an explicit downstream collection selection."""
    if not handoff_paths:
        raise ValueError("No completed CropHarvest handoffs to register.")
    records = [
        {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
        for path in handoff_paths
    ]
    payload = {
        "kind": "cropharvest_source_bundle",
        "collection_name": collection_name,
        "run_folder": str(Path(run_folder).resolve()),
        "handoffs": records,
    }
    revision = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    bundle = (
        Path(report_root).parent
        / "source_handoffs"
        / f"cropharvest_{collection_name}_{revision}.json"
    )
    if bundle.exists() and json.loads(bundle.read_text()) != payload:
        raise ValueError("Supplementary handoff differs from its recorded revision.")
    if not bundle.exists():
        save_json(bundle, payload)
    return bundle


# The official package pins this same Zenodo release; avoid silently moving labels.
LABELS_RECORD = "10251170"
LABELS_URL = (
    f"https://zenodo.org/records/{LABELS_RECORD}/files/labels.geojson?download=1"
)
LABELS_MD5 = "54a5070f103bc3e635afba27c139ac8d"


def validate_labels(path):
    """Reject an HTML error page or incomplete JSON before it becomes a pipeline input."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("CropHarvest labels must be a GeoJSON FeatureCollection.")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("CropHarvest labels contain no features.")
    if any(
        not isinstance(feature, dict)
        or feature.get("type") != "Feature"
        or not isinstance(feature.get("properties"), dict)
        for feature in features
    ):
        raise ValueError("CropHarvest labels contain malformed features.")
    return len(features)


def label_file_summary(path, status):
    path = Path(path)
    return {
        "status": status,
        "path": str(path.resolve()),
        "features": validate_labels(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def ensure_labels(path, download_if_missing=True, expected_md5=LABELS_MD5):
    """Fetch labels only when missing; never replace a user's existing label file."""
    import os
    import tempfile

    path = Path(path)
    if path.exists():
        return label_file_summary(path, "reused")
    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing {path}. Enable download_missing_labels to retrieve the pinned CropHarvest labels."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cropharvest_labels_", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        download_labels_payload(temporary_path)
        validate_label_checksum(temporary_path, expected_md5)
        summary = label_file_summary(temporary_path, "downloaded")
        published = publish_labels(temporary_path, path)
        if not published:
            return label_file_summary(path, "reused")
        summary.update(
            path=str(path.resolve()),
            source_url=LABELS_URL,
            zenodo_record=LABELS_RECORD,
            upstream_checksum=f"md5:{expected_md5}",
        )
        save_json(path.with_suffix(".download.json"), summary)
        return summary
    finally:
        temporary_path.unlink(missing_ok=True)


def download_labels_payload(destination):
    """Stream the label file alone; satellite and feature archives are not requested."""
    import requests

    with requests.get(LABELS_URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        write_labels_response(response, destination)


def write_labels_response(response, destination):
    with Path(destination).open("wb") as stream:
        for block in response.iter_content(chunk_size=1024 * 1024):
            stream.write(block)


def validate_label_checksum(path, expected_md5):
    """The release's published MD5 detects interrupted or substituted downloads."""
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_md5:
        raise ValueError(
            "CropHarvest label download failed the pinned Zenodo MD5 checksum. Rerun to retry; no labels were published."
        )


def publish_labels(temporary_path, destination):
    """Publish atomically without replacing labels created by another kernel meanwhile."""
    import os

    try:
        os.link(temporary_path, destination)
    except FileExistsError:
        return False
    return True
