"""Transfer completed Notebook 04 batches for local auditing without Earth Engine."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from crop_pipeline.common import save_json
from crop_pipeline.imagery.batch_export import content_key
from crop_pipeline.sources import adaptive, pilot_sources, test_cohort
from crop_pipeline.windows.seasons import file_sha256


def read(path):
    return json.loads(path.read_text())


def campaign_rounds(report_root, source=test_cohort.TEST):
    policy = test_cohort.TEST_POLICY if source == test_cohort.TEST else None
    return adaptive.root(report_root, policy) / "rounds"


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def checked_path(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Invalid JECAM bundle path: {name}")
    return path


def manifest_hashes(record):
    return tuple(sorted(record.get("original_handoff_hashes", [item["sha256"] for item in record["handoffs"]])))


def relocate(manifest, batch_root, regions, staged_fields):
    """Write the sampled regions and resolve their masks and handoff paths."""
    def resolve(name):
        return str(checked_path(batch_root, name).resolve())

    for key in ("field_folder", "timeline_folder", "annual_tif_root"):
        manifest[key] = resolve(manifest[key])
    for item in manifest["annual_tif_index"].values():
        item["path"] = resolve(item["path"])
    for region in regions["features"]:
        properties = region["properties"]
        properties["sampling_mask"] = resolve(properties["sampling_mask"])
    path = staged_fields / "sampled_download_regions.geojson"
    save_json(path, regions)
    manifest["regions_sha256"] = file_sha256(path)


def export_year(source, staging, number):
    if source.get("archive_format") != "selected_pixel_parquet_v1" or not source.get("annual_tif_index"):
        raise ValueError("Finish Notebook 04 through annual TIFF publication before exporting")
    base = Path(f"year_{number:04d}")
    fields = Path(source["field_folder"])
    regions = read(fields / "sampled_download_regions.geojson")
    copy_file(fields / "source_footprints.geojson", staging / base / "fields/source_footprints.geojson")
    index = {}
    for number, region in enumerate(regions["features"]):
        properties = region["properties"]
        identifier = properties["fragment_id"]
        mask = base / "fields/masks" / f"{number:06d}.tif"
        copy_file(Path(properties["sampling_mask"]), staging / mask)
        properties["sampling_mask"] = mask.as_posix()
        item = source["annual_tif_index"][identifier]
        original = Path(item["path"])
        if file_sha256(original) != item["sha256"]:
            raise ValueError(f"Annual TIFF changed after registration: {original}")
        image = base / "tiffs" / f"{number:06d}.tif"
        copy_file(original, staging / image)
        index[identifier] = {**item, "path": image.as_posix()}
        timeline = checked_path(Path(source["timeline_folder"]), f"tiles/{identifier}/observed_timelines.json")
        copy_file(timeline, staging / base / "timelines/tiles" / identifier / "observed_timelines.json")
    region_path = staging / base / "fields/sampled_download_regions.geojson"
    save_json(region_path, regions)
    # TIFF auditing uses the masks and daily summaries, so the transport cache
    # and operational job files stay in the exporting workspace.
    local_keys = {"manifest_path", "imagery_folder", "progress_folder", "pixel_table_root", "pixel_table_index"}
    portable = {key: value for key, value in source.items() if key not in local_keys}
    portable.update(field_folder=(base / "fields").as_posix(),
                    timeline_folder=(base / "timelines").as_posix(),
                    annual_tif_root=(base / "tiffs").as_posix(), annual_tif_index=index,
                    regions_sha256=file_sha256(region_path))
    path = base / "handoff.json"
    save_json(staging / path, portable)
    return path.as_posix()


def export_completed(report_root, destination, source=test_cohort.TEST):
    """Write one portable subfolder per completed local batch."""
    if source not in test_cohort.DEVELOPMENT | {test_cohort.TEST}:
        raise ValueError(f"Unknown collection source: {source}")
    destination = Path(destination).expanduser().resolve()
    exports = []
    for receipt in sorted(campaign_rounds(report_root, source).glob("*/*/completed.json")):
        record = read(receipt)
        if record["source"] != source or "original_handoff_hashes" in record:
            continue
        if read(receipt.with_name("state.json")).get("status") != "completed":
            continue
        paths = [Path(item["path"]) for item in record["handoffs"]]
        hashes = manifest_hashes(record)
        if sorted(file_sha256(path) for path in paths) != list(hashes):
            raise ValueError(f"Completed JECAM handoff changed: {receipt}")
        manifests_by_year = pilot_sources.load_sources(paths)
        batch_id = content_key(hashes)
        target = destination / batch_id
        if target.exists():
            existing = verify_bundle(target, source)
            if tuple(existing["original_handoff_hashes"]) != hashes:
                raise ValueError(f"Export conflicts with an existing bundle: {target}")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=destination) as temp:
                staging = Path(temp) / batch_id
                manifests = [export_year(source, staging, number) for number, source in enumerate(manifests_by_year)]
                selection = read(receipt.with_name("selection.json"))
                files = {path.relative_to(staging).as_posix(): file_sha256(path)
                         for path in sorted(staging.rglob("*")) if path.is_file()}
                save_json(staging / "bundle.json", {
                    "version": 1, "round": record["round"], "source": source,
                    "original_handoff_hashes": list(hashes), "manifests": manifests,
                    "selection": selection, "files": files,
                })
                staging.rename(target)
        exports.append(target)
    if not exports:
        raise FileNotFoundError(f"No completed local {source} batches to export")
    return exports


def verify_bundle(folder, source=test_cohort.TEST):
    bundle = read(checked_path(folder, "bundle.json"))
    if bundle["version"] != 1 or bundle["source"] != source:
        raise ValueError(f"Unsupported JECAM bundle: {folder}")
    for name, digest in bundle["files"].items():
        if file_sha256(checked_path(folder, name)) != digest:
            raise ValueError(f"JECAM bundle checksum mismatch: {name}")
    if not bundle["manifests"] or set(bundle["manifests"]) - bundle["files"].keys():
        raise ValueError(f"JECAM bundle has missing handoffs: {folder}")
    return bundle


def import_completed(report_root, source_folder, source=test_cohort.TEST):
    """Verify bundles, then register them alongside existing local and pending work."""
    if source not in test_cohort.DEVELOPMENT | {test_cohort.TEST}:
        raise ValueError(f"Unknown collection source: {source}")
    source_folder = Path(source_folder).expanduser().resolve()
    folders = [source_folder] if (source_folder / "bundle.json").is_file() else sorted(
        path.parent for path in source_folder.glob("*/bundle.json")
    )
    if not folders:
        raise FileNotFoundError(f"No completed JECAM bundles in {source_folder}")
    rounds = campaign_rounds(report_root, source).resolve()
    known, used, used_groups = {}, set(), set()
    for path in rounds.glob("*/*/selection.json"):
        selection = read(path)
        if selection["source"] != source:
            continue
        used.update(selection["physical_keys"])
        used_groups.update(selection["groups"])
        receipt = path.with_name("completed.json")
        if receipt.is_file():
            known[manifest_hashes(read(receipt))] = path.parent

    allowed_partitions = {"test"} if source == test_cohort.TEST else {"train", "validation", "test"}
    imports, pending = [], []
    for folder in folders:
        bundle = verify_bundle(folder, source)
        hashes = tuple(bundle["original_handoff_hashes"])
        if hashes in known:
            imports.append(known[hashes])
            continue
        selection = bundle["selection"]
        number = bundle["round"]
        if (type(number) is not int or number < 1 or selection["round"] != number
                or selection["source"] != source
                or set(selection["assignments"]) != set(selection["groups"])
                or not set(selection["assignments"].values()) <= allowed_partitions):
            raise ValueError(f"Invalid reserved JECAM selection: {folder}")
        if used.intersection(selection["physical_keys"]) or used_groups.intersection(selection["groups"]):
            raise ValueError(f"JECAM bundle conflicts with an existing parcel selection: {folder}")
        used.update(selection["physical_keys"])
        used_groups.update(selection["groups"])
        target = rounds / f"{number:04d}" / f"{source}_import_{content_key(hashes)}"
        if target.exists():
            raise FileExistsError(f"JECAM import already exists: {target}")
        known[hashes] = target
        pending.append((folder, bundle, target))
        imports.append(target)

    if not pending:
        return imports
    rounds.mkdir(parents=True, exist_ok=True)
    # Stage outside rounds so readers only discover batches after publication.
    with TemporaryDirectory(dir=rounds.parent) as temp:
        staged = []
        for folder, bundle, target in pending:
            staging = Path(temp) / target.name
            for name, digest in bundle["files"].items():
                destination = checked_path(staging, name)
                copy_file(checked_path(folder, name), destination)
                if file_sha256(destination) != digest:
                    raise ValueError(f"JECAM bundle changed during import: {name}")
            verify_files = bundle["files"]
            handoffs = []
            groups = set()
            for name in bundle["manifests"]:
                path = checked_path(staging, name)
                manifest = read(path)
                pilot_sources.validate_source_contract(manifest)
                if (test_cohort.source_role(manifest["source_name"]) != source
                        or manifest.get("archive_format") != "selected_pixel_parquet_v1"):
                    raise ValueError(f"Invalid JECAM handoff: {name}")
                fields = checked_path(staging, manifest["field_folder"])
                region_path = fields / "sampled_download_regions.geojson"
                regions = read(region_path)
                footprints = read(fields / "source_footprints.geojson")
                footprint_groups = {item["properties"]["partition_group"] for item in footprints["features"]}
                for region in regions["features"]:
                    properties = region["properties"]
                    group = properties["partition_group"]
                    identifier = properties["fragment_id"]
                    if group not in footprint_groups:
                        raise ValueError(f"Missing original JECAM footprint: {group}")
                    groups.add(group)
                    image = manifest["annual_tif_index"][identifier]
                    timeline = f"{manifest['timeline_folder']}/tiles/{identifier}/observed_timelines.json"
                    required = {properties["sampling_mask"], image["path"], timeline,
                                region_path.relative_to(staging).as_posix(),
                                (fields / "source_footprints.geojson").relative_to(staging).as_posix()}
                    if required - verify_files.keys() or verify_files[image["path"]] != image["sha256"]:
                        raise ValueError(f"Incomplete JECAM tile: {identifier}")
                relocate(manifest, target, regions, fields)
                save_json(path, manifest)
                handoffs.append({"path": str(target / name), "sha256": file_sha256(path)})
            if not groups or not groups.issubset(bundle["selection"]["groups"]):
                raise ValueError(f"JECAM handoffs differ from the saved selection: {folder}")
            save_json(staging / "selection.json", bundle["selection"])
            save_json(staging / "completed.json", {
                "source": source, "round": bundle["round"], "handoffs": handoffs,
                "original_handoff_hashes": bundle["original_handoff_hashes"],
            })
            staged.append((staging, target))
        for staging, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
    return imports
