"""Skip unchanged parcel preparation only when saved inputs still verify."""

import json
from pathlib import Path

from crop_pipeline.common import save_json
from crop_pipeline.windows.seasons import file_sha256


def preparation_files(plan):
    paths = [Path(plan["run_folder"]) / "collection_plan.json"]
    for inputs in plan["years"].values():
        field_folder = Path(inputs["field_folder"])
        regions_path = field_folder / "sampled_download_regions.geojson"
        paths.extend([regions_path, field_folder / "source_footprints.geojson"])
        regions = json.loads(regions_path.read_text())["features"]
        paths.extend(Path(region["properties"]["sampling_mask"]) for region in regions)
    return paths


def save_receipt(plan):
    """Content checksums catch damaged or edited masks rather than trusting existence."""
    files = {str(path.resolve()): file_sha256(path) for path in preparation_files(plan)}
    save_json(
        Path(plan["run_folder"]) / "preparation_cache.json",
        {"version": 1, "files": files},
    )


def load_prepared(run_folder):
    receipt_path = Path(run_folder) / "preparation_cache.json"
    if not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("version") != 1 or not receipt.get("files"):
            return None
        if any(
            not Path(path).is_file() or file_sha256(Path(path)) != checksum
            for path, checksum in receipt["files"].items()
        ):
            return None
        plan = json.loads((Path(run_folder) / "collection_plan.json").read_text())
        expected = {str(path.resolve()) for path in preparation_files(plan)}
        if expected != set(receipt["files"]):
            return None
        return plan
    except (OSError, ValueError, KeyError):
        return None
