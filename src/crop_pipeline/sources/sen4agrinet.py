"""Scan a pinned Sen4AgriNet release completely before publishing references.

Only label/parcel masks and grid metadata are read. Missing patches cannot be
silently treated as crop absence. Temporary raw files belong to this new run;
existing NetCDFs and historical field inventories are never overwritten.
"""

import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from affine import Affine
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry, unary_union

from crop_pipeline.common import save_json
from crop_pipeline.sources import remote_masks
from crop_pipeline.windows.seasons import file_sha256

DATASET_NAME = "orion-ai-lab/S4A"
CROP_CODES = {"cabbage": 213, "carrot": 231, "potato": 510}


def repository_file_record(item):
    """Keep every NetCDF below data/, without selecting a year, tile or batch."""
    path = PurePosixPath(item.path)
    if path.suffix.lower() != ".nc" or not hasattr(item, "size"):
        return None
    if len(path.parts) < 4 or path.parts[0] != "data" or not path.parts[1].isdigit():
        raise ValueError(
            f"Unrecognized NetCDF layout; inspect rather than skip: {path}"
        )
    if any(part in {"..", "."} for part in path.parts):
        raise ValueError(f"Unsafe repository path: {path}")
    lfs = getattr(item, "lfs", None)
    checksum = (
        lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    )
    return {
        "filename": str(path),
        "year": int(path.parts[1]),
        "tile": path.parts[2],
        "expected_bytes": int(item.size),
        "sha256": checksum,
        "blob_id": getattr(item, "blob_id", None),
    }


def discover_release(reference_root, enabled=True, revision=None):
    """Pin discovery once so restarting does not switch to a moving upstream release."""
    reference_root = Path(reference_root)
    selection_path = reference_root / "selected_release.json"
    if not enabled:
        return load_selected_release(selection_path)
    if revision is None and selection_path.is_file():
        return load_selected_release(selection_path)
    from huggingface_hub import HfApi

    hub = HfApi()
    pinned_revision = hub.repo_info(
        DATASET_NAME, repo_type="dataset", revision=revision
    ).sha
    manifest_path = (
        reference_root / "releases" / pinned_revision / "repository_manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = discover_manifest(hub, pinned_revision)
        save_json(manifest_path, manifest)
    save_json(
        selection_path,
        {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
    )
    return manifest


def discover_manifest(hub, revision):
    records = [
        record
        for item in hub.list_repo_tree(
            DATASET_NAME,
            repo_type="dataset",
            revision=revision,
            path_in_repo="data",
            recursive=True,
        )
        if (record := repository_file_record(item)) is not None
    ]
    records.sort(key=lambda record: record["filename"])
    identifiers = [record["filename"] for record in records]
    if not records or len(set(identifiers)) != len(records):
        raise ValueError("Dataset-wide NetCDF listing is empty or contains duplicates.")
    return {
        "dataset": DATASET_NAME,
        "revision": revision,
        "scope": "all_netcdf_files_under_data",
        "files": records,
        "years": sorted({record["year"] for record in records}),
        "year_tile_pairs": sorted(
            {f"{record['year']}/{record['tile']}" for record in records}
        ),
        "total_source_bytes": sum(record["expected_bytes"] for record in records),
    }


def load_selected_release(selection_path):
    if not selection_path.is_file():
        raise FileNotFoundError(
            "No complete dataset-wide reference discovery is saved. Enable retrieve_reference_masks for discovery and selective retrieval."
        )
    selection = json.loads(selection_path.read_text())
    if file_sha256(Path(selection["path"])) != selection["sha256"]:
        raise ValueError("Selected Sen4AgriNet repository manifest changed.")
    return json.loads(Path(selection["path"]).read_text())


def run_folder(reference_root, manifest, reference_only=True):
    """Extractor changes create a new inventory rather than altering old references."""
    configuration = {
        "dataset": DATASET_NAME,
        "revision": manifest["revision"],
        "crop_codes": CROP_CODES,
        "helper_sha256": file_sha256(Path(__file__)),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "scope": manifest["scope"],
        "reference_only": reference_only,
        "range_reader_sha256": file_sha256(Path(remote_masks.__file__)),
    }
    revision = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()[:16]
    folder = (
        Path(reference_root)
        / "releases"
        / manifest["revision"]
        / "inventories"
        / revision
    )
    save_json(folder / "configuration.json", configuration)
    return folder


def verified_source(path, record):
    """Size alone cannot establish that a cached file belongs to the pinned release."""
    if not path.is_file() or path.stat().st_size != record["expected_bytes"]:
        return False
    if record.get("sha256"):
        return file_sha256(path) == record["sha256"]
    if record.get("blob_id"):
        return git_blob_checksum(path) == record["blob_id"]
    return False


def git_blob_checksum(path):
    """Git's blob identifier includes its header, unlike an ordinary file SHA-1."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_source(record, manifest, folder, existing_root, download_enabled):
    """Borrow verified existing files read-only; fetch missing files into owned staging."""
    existing_path = Path(existing_root) / record["filename"]
    if verified_source(existing_path, record):
        return existing_path, False
    staged_path = folder / "staging" / record["filename"]
    if verified_source(staged_path, record):
        return staged_path, True
    if not download_enabled:
        raise FileNotFoundError(
            f"Missing verified source for {record['filename']}; no subset fallback is allowed."
        )
    require_staging_space(folder, record["expected_bytes"])
    from huggingface_hub import hf_hub_download

    downloaded_path = Path(
        hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            revision=manifest["revision"],
            filename=record["filename"],
            local_dir=str(folder / "staging"),
            force_download=staged_path.exists(),
        )
    )
    if downloaded_path.resolve() != staged_path.resolve():
        raise ValueError("Download destination escaped the owned staging folder")
    if not verified_source(downloaded_path, record):
        raise ValueError(
            f"Downloaded source failed pinned size/hash validation: {record['filename']}"
        )
    return downloaded_path, True


def require_staging_space(folder, expected_bytes):
    required_bytes = int(expected_bytes * 1.1) + 64 * 1024**2
    if shutil.disk_usage(folder).free < required_bytes:
        raise OSError(
            "Insufficient disk space for the next NetCDF patch; choose a larger sen4agrinet_reference_root."
        )


def collect_variables(group):
    variables = list(group.variables.values())
    for child in group.groups.values():
        variables.extend(collect_variables(child))
    return variables


def find_variable(variables, name):
    matches = [variable for variable in variables if variable.name.lower() == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} mask, found {len(matches)}")
    return matches[0]


def read_mask(variable):
    masked_values = np.ma.asarray(variable[:])
    if masked_values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional {variable.name} mask")
    values = np.asarray(masked_values.data)
    valid = ~np.ma.getmaskarray(masked_values) & np.isfinite(values)
    if np.any(values[valid] != np.floor(values[valid])):
        raise ValueError(f"Noninteger values in {variable.name}")
    return values, valid


def read_grid(variable):
    """Reject uncertain georeferencing instead of moving parcel labels by guesswork."""
    if not hasattr(variable, "crs") or not hasattr(variable, "transform"):
        raise ValueError("Missing source CRS or affine transform")
    source_crs = CRS.from_user_input(str(variable.crs).replace("+init=epsg:", "EPSG:"))
    coefficients = np.asarray(variable.transform).ravel()
    if len(coefficients) != 6 or not source_crs.is_projected:
        raise ValueError("Expected projected CRS and six affine coefficients")
    grid = Affine(*coefficients)
    if grid.b != 0 or grid.d != 0 or grid.a != 10 or grid.e != -10:
        raise ValueError("Expected the existing north-up 10 m label grid")
    group = variable.group()
    horizontal = np.asarray(group.variables["x"][:])
    vertical = np.asarray(group.variables["y"][:])
    expected_horizontal = grid.c + (np.arange(len(horizontal)) + 0.5) * grid.a
    expected_vertical = grid.f + (np.arange(len(vertical)) + 0.5) * grid.e
    if tuple(variable.shape) != (len(vertical), len(horizontal)):
        raise ValueError("Mask shape differs from coordinates")
    if not np.allclose(
        horizontal, expected_horizontal, rtol=0, atol=0.001
    ) or not np.allclose(vertical, expected_vertical, rtol=0, atol=0.001):
        raise ValueError("Pixel centers disagree with the stored grid")
    return source_crs, grid


def extract_patch(path, record):
    """Read masks only; downloaded spectral arrays are not loaded into memory."""
    with Dataset(path, "r") as dataset:
        variables = collect_variables(dataset)
        label_variable, parcel_variable = find_variable(
            variables, "labels"
        ), find_variable(variables, "parcels")
        labels, valid_labels = read_mask(label_variable)
        parcels, valid_parcels = read_mask(parcel_variable)
        source_crs, grid = read_grid(label_variable)
        if (
            read_grid(parcel_variable) != (source_crs, grid)
            or labels.shape != parcels.shape
        ):
            raise ValueError("Label and parcel grids differ")
        metadata = patch_metadata(dataset, record)
        valid = valid_labels & valid_parcels & ~np.isin(parcels, [0, -1])
        return extract_crop_features(
            labels, parcels, valid, source_crs, grid, metadata, record
        )


def patch_metadata(dataset, record):
    year = int(getattr(dataset, "patch_year", record["year"]))
    tile = str(getattr(dataset, "patch_tile", record["tile"]))
    country = str(getattr(dataset, "patch_country_code", "")).strip()
    if year != record["year"] or tile != record["tile"]:
        raise ValueError("Patch metadata contradicts its repository year/tile")
    if not country or country.casefold() == "unknown":
        raise ValueError("Missing country scope for parcel identity")
    return {"year": year, "tile": tile, "country": country}


def extract_crop_features(labels, parcels, valid, source_crs, grid, metadata, record):
    features = []
    for crop, code in CROP_CODES.items():
        identifiers = np.unique(parcels[valid & (labels == code)])
        features.extend(
            parcel_feature(
                labels,
                parcels,
                valid,
                source_crs,
                grid,
                metadata,
                record,
                crop,
                code,
                identifier,
            )
            for identifier in identifiers
        )
    return features


def parcel_feature(
    labels, parcels, valid, source_crs, grid, metadata, record, crop, code, identifier
):
    selected = valid & (labels == code) & (parcels == identifier)
    pieces = [
        shape(geometry)
        for geometry, value in shapes(
            selected.astype("uint8"), mask=selected, transform=grid
        )
        if value == 1
    ]
    geometry = unary_union(pieces)
    projector = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    geographic = transform_geometry(projector.transform, geometry)
    if geographic.is_empty or not geographic.is_valid:
        raise ValueError("Invalid reprojected parcel footprint")
    west, south, east, north = geographic.bounds
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError("Parcel is outside WGS84 coordinate bounds")
    identifier = int(identifier)
    fragment_id = hashlib.sha256(
        f"{record['filename']}|{crop}|{identifier}".encode()
    ).hexdigest()[:20]
    properties = {
        **metadata,
        "fragment_id": fragment_id,
        "parcel_id": identifier,
        "crop": crop,
        "crop_code": code,
        "source_file": record["filename"],
        "source_crs": source_crs.to_string(),
        "crop_pixels": int(selected.sum()),
        "mixed_label_parcel": bool(
            (valid & (parcels == identifier) & (labels != code)).any()
        ),
        "partition_group": f"sen4agrinet|{metadata['country']}|{metadata['year']}|{identifier}",
        "label_status": "reported",
        "source_split": "train",
    }
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(geographic),
    }


def obtain_reference_source(
    record, manifest, folder, existing_root, download_enabled, reference_only
):
    """Range mode has no path that silently invokes full-file download."""
    existing_path = Path(existing_root) / record["filename"]
    staged_path = folder / "staging" / record["filename"]
    for source_path in (existing_path, staged_path):
        if verified_source(source_path, record):
            receipt = {
                "retrieval": "verified_local_netcdf",
                "full_source_checksum_verified": True,
                "network_bytes_this_run": 0,
            }
            return source_path, source_path == staged_path, receipt
    if reference_only:
        source_path, receipt = remote_masks.obtain_reference_masks(
            record,
            manifest,
            folder,
            tuple(CROP_CODES.values()),
            enabled=download_enabled,
        )
        return source_path, False, receipt
    source_path, owned = obtain_source(
        record, manifest, folder, existing_root, download_enabled
    )
    return (
        source_path,
        owned,
        {
            "retrieval": "explicit_full_download",
            "full_source_checksum_verified": True,
            "network_bytes_this_run": record["expected_bytes"],
        },
    )


def extract_reference_patch(path, record):
    """A patch without target labels needs no parcel payload to establish crop absence."""
    with Dataset(path, "r") as dataset:
        variables = collect_variables(dataset)
        label_variable = find_variable(variables, "labels")
        labels, valid = read_mask(label_variable)
        read_grid(label_variable)
        patch_metadata(dataset, record)
        has_targets = bool(np.isin(labels[valid], list(CROP_CODES.values())).any())
    return extract_patch(path, record) if has_targets else []


def checkpoint_path(folder, record):
    identifier = hashlib.sha256(record["filename"].encode()).hexdigest()
    return folder / "checkpoints" / f"{identifier}.json"


def valid_checkpoint(checkpoint, record):
    expected_hash = hashlib.sha256(
        json.dumps(checkpoint.get("features", []), sort_keys=True).encode()
    ).hexdigest()
    return (
        checkpoint.get("source") == record
        and checkpoint.get("status") == "complete"
        and checkpoint.get("features_sha256") == expected_hash
    )


def process_patch(
    record,
    manifest,
    folder,
    existing_root,
    download_enabled,
    retain_new_netcdf,
    reference_only=True,
):
    """A valid checkpoint avoids downloading a large source patch again on restart."""
    path = checkpoint_path(folder, record)
    if path.is_file():
        checkpoint = json.loads(path.read_text())
        if valid_checkpoint(checkpoint, record):
            return {
                **checkpoint,
                "reused_checkpoint": True,
                "network_bytes_this_run": 0,
            }
    source_path, owned_staging, receipt = obtain_reference_source(
        record, manifest, folder, existing_root, download_enabled, reference_only
    )
    features = extract_reference_patch(source_path, record)
    result = {
        "status": "complete",
        "source": record,
        "reference_retrieval": receipt,
        "network_bytes_this_run": receipt.get("network_bytes_this_run", 0),
        "features": features,
        "features_sha256": hashlib.sha256(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest(),
        "crop_fragments": dict(
            Counter(feature["properties"]["crop"] for feature in features)
        ),
    }
    save_json(path, result)
    # Delete only this run's temporary payload, after its reference checkpoint is durable.
    if owned_staging and not retain_new_netcdf:
        source_path.unlink()
    return result


def build_reference_inventory(
    manifest,
    reference_root,
    existing_root,
    download_enabled=True,
    retain_new_netcdf=False,
    reference_only=True,
):
    """Fail closed on any unscanned patch, then publish all years in one completed inventory."""
    folder = run_folder(reference_root, manifest, reference_only=reference_only)
    transfer = {
        "network_bytes": 0,
        "patches_reused_from_checkpoints": 0,
        "patches_processed": 0,
    }
    errors = []
    for number, record in enumerate(manifest["files"], 1):
        try:
            result = process_patch(
                record,
                manifest,
                folder,
                existing_root,
                download_enabled,
                retain_new_netcdf,
                reference_only=reference_only,
            )
            transfer["network_bytes"] += result.get("network_bytes_this_run", 0)
            transfer["patches_reused_from_checkpoints"] += int(
                result.get("reused_checkpoint", False)
            )
            transfer["patches_processed"] += 1
            print(
                f"[{number}/{len(manifest['files'])}] {record['filename']}: {result['crop_fragments']}"
            )
        except Exception as error:
            errors.append(
                {
                    "filename": record["filename"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            save_json(folder / "scan_errors.json", errors)
            raise RuntimeError(
                f"Reference scan incomplete at {record['filename']}; resume Run All. No partial inventory was published."
            ) from error
    save_json(folder / "scan_errors.json", [])
    save_json(folder / "last_scan_transfer.json", transfer)
    return publish_inventory(folder, manifest)


def publish_inventory(folder, manifest):
    """SQLite bounds consolidation memory and keeps ambiguous parcel groups out of training."""
    database_path = folder / "consolidation.sqlite"
    with sqlite3.connect(database_path) as database:
        database.execute("DROP TABLE IF EXISTS fragments")
        database.execute(
            "CREATE TABLE fragments (parcel_group TEXT, crop TEXT, year INTEGER, mixed INTEGER, feature TEXT)"
        )
        for record in manifest["files"]:
            index_checkpoint(database, folder, record)
        database.execute("CREATE INDEX group_index ON fragments(parcel_group)")
        database.execute(
            "CREATE TEMP TABLE excluded AS SELECT parcel_group FROM fragments GROUP BY parcel_group HAVING COUNT(DISTINCT crop)>1 OR MAX(mixed)=1"
        )
        audit = write_year_inventories(database, folder, manifest)
    save_json(folder / "completed_inventory.json", audit)
    return audit


def index_checkpoint(database, folder, record):
    checkpoint = json.loads(checkpoint_path(folder, record).read_text())
    if not valid_checkpoint(checkpoint, record):
        raise ValueError(f"Missing current checkpoint for {record['filename']}")
    database.executemany(
        "INSERT INTO fragments VALUES (?, ?, ?, ?, ?)",
        [
            (
                feature["properties"]["partition_group"],
                feature["properties"]["crop"],
                feature["properties"]["year"],
                int(feature["properties"]["mixed_label_parcel"]),
                json.dumps(feature),
            )
            for feature in checkpoint["features"]
        ],
    )


def write_year_inventories(database, folder, manifest):
    paths = []
    for year in manifest["years"]:
        path = folder / "fields" / str(year) / "download_candidates.geojson"
        cursor = database.execute(
            "SELECT feature FROM fragments WHERE year=? AND parcel_group NOT IN (SELECT parcel_group FROM excluded) ORDER BY parcel_group, crop, feature",
            (year,),
        )
        write_feature_collection(path, cursor)
        paths.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    crops = database.execute(
        "SELECT crop, COUNT(*), COUNT(DISTINCT parcel_group) FROM fragments WHERE parcel_group NOT IN (SELECT parcel_group FROM excluded) GROUP BY crop"
    ).fetchall()
    excluded = database.execute(
        "SELECT parcel_group FROM excluded ORDER BY parcel_group"
    ).fetchall()
    save_json(
        folder / "excluded_parcel_groups.json",
        {
            "reason": "mixed_labels_or_conflicting_target_crops",
            "groups": [row[0] for row in excluded],
        },
    )
    return {
        "status": "complete",
        "dataset": DATASET_NAME,
        "revision": manifest["revision"],
        "scope": manifest["scope"],
        "files_scanned": len(manifest["files"]),
        "years": manifest["years"],
        "year_tile_pairs": manifest["year_tile_pairs"],
        "crop_codes": CROP_CODES,
        "crop_counts": {
            crop: {"fragments": fragments, "parcel_groups": groups}
            for crop, fragments, groups in crops
        },
        "excluded_parcel_groups": len(excluded),
        "inventories": paths,
        "field_root": str((folder / "fields").resolve()),
    }


def write_feature_collection(path, cursor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".geojson.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write('{"type":"FeatureCollection","features":[')
        separator = ""
        for row in cursor:
            stream.write(separator + row[0])
            separator = ","
        stream.write("]}")
    temporary.replace(path)


def load_completed_inventory(folder, manifest):
    """Offline reuse requires a complete, unmodified full-release inventory."""
    path = Path(folder) / "completed_inventory.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Complete dataset-wide inventory is missing; enable rebuild_field_inventory. The old subset will not be substituted."
        )
    completed = json.loads(path.read_text())
    if (
        completed.get("status") != "complete"
        or completed.get("revision") != manifest["revision"]
        or completed.get("files_scanned") != len(manifest["files"])
    ):
        raise ValueError("Completed inventory does not cover the selected full release")
    for entry in completed["inventories"]:
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(
                "Completed reference inventory changed; rebuild it from checkpoints"
            )
    return completed
