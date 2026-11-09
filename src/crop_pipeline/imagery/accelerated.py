"""Transport optimizations that preserve the existing imagery recipe and files."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import rasterio
import requests
from affine import Affine
from pyproj import CRS

from crop_pipeline.common import interval_bins, save_json
from crop_pipeline.imagery.retrieval import SatelliteRetriever


class AcceleratedRetriever(SatelliteRetriever):
    """Keep scientific processing inherited; change only inventory and transport."""

    download_workers = 2
    intervals_per_request = 6

    def inventory_region(self, region, refresh=False):
        from crop_pipeline.imagery.scene_cache import inventory_region

        return inventory_region(self, region, refresh)

    def download_region(self, plan):
        return download_region(self, plan)


def request_groups(intervals, grid, maximum_intervals):
    """Leave headroom below the direct-download cap for headers and server limits."""
    bytes_per_interval = grid["width"] * grid["height"] * 7 * 4
    capacity = min(maximum_intervals, (24 * 1024**2) // bytes_per_interval)
    if capacity < 1 or max(grid["width"], grid["height"]) >= 10000:
        raise ValueError("Region exceeds direct-download size limits.")
    return [
        intervals[start : start + capacity]
        for start in range(0, len(intervals), capacity)
    ]


def download_region(retriever, plan):
    """Workers return results; only the main thread updates the durable manifest."""
    grid = plan["settings"]["grid"]
    if grid != plan["fragment"]["properties"]["sampling_grid"]:
        raise ValueError(
            "Sampling-mask and retrieval grids differ; regenerate the source."
        )
    intervals = plan.get("download_intervals", interval_bins(retriever.year))
    allowed = {interval["name"]: interval for interval in interval_bins(retriever.year)}
    if any(allowed.get(interval["name"]) != interval for interval in intervals):
        raise ValueError(
            "Selected download intervals must match full calendar half-month bins."
        )
    folder, manifest_path, manifest = download_checkpoint(retriever, plan)
    reused, missing = classify_intervals(retriever, plan, intervals, folder, manifest)
    groups = request_groups(missing, grid, retriever.intervals_per_request)
    results = {result["name"]: result for result in reused}
    with ThreadPoolExecutor(max_workers=retriever.download_workers) as executor:
        futures = [
            executor.submit(download_group, retriever, plan, group, folder)
            for group in groups
        ]
        for future in as_completed(futures):
            checkpoint_group(future.result(), results, manifest, manifest_path)
    return [results[interval["name"]] for interval in intervals]


def checkpoint_group(completed, results, manifest, manifest_path):
    for result, record in completed:
        results[result["name"]] = result
        manifest["files"][result["name"]] = record
    save_json(manifest_path, manifest)


def download_checkpoint(retriever, plan):
    request_id = hashlib.sha256(
        json.dumps(plan["settings"], sort_keys=True).encode()
    ).hexdigest()[:16]
    folder = retriever.imagery_folder / plan["fragment_id"] / request_id
    folder.mkdir(parents=True, exist_ok=True)
    path = (
        retriever.report_folder
        / "composite_progress"
        / f"{plan['fragment_id']}_{request_id}.json"
    )
    manifest = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "fragment_id": plan["fragment_id"],
            "settings": plan["settings"],
            "files": {},
        }
    )
    return folder, path, manifest


def classify_intervals(retriever, plan, intervals, folder, manifest):
    reused, missing = [], []
    for interval in intervals:
        destination = folder / f"{interval['name']}.tif"
        previous = manifest["files"].get(interval["name"])
        if retriever.reusable_file(destination, previous, plan["settings"]["grid"]):
            reused.append(outcome(plan, interval, "reused"))
            continue
        missing.append(interval)
    return reused, missing


def outcome(plan, interval, status):
    return {
        "region_id": plan["fragment_id"],
        "name": interval["name"],
        "status": status,
    }


def interval_record(plan, interval, folder):
    granules = [
        row
        for row in plan["acquisitions"]
        if interval["start"] <= row["date"] < interval["end"]
    ]
    return {
        **interval,
        "path": str((folder / f"{interval['name']}.tif").resolve()),
        "source_image_ids": [row["image_id"] for row in granules],
        "available_dates": sorted({row["date"] for row in granules}),
        "cloud_screened_available_dates": sorted(
            {row["date"] for row in granules if row.get("cloud_probability_id")}
        ),
    }


def download_group(retriever, plan, intervals, folder):
    """A failed group remains pending; smaller groups on a later pass are allowed."""
    try:
        with TemporaryDirectory(prefix="group_", dir=folder) as temporary:
            stack_path = Path(temporary) / "stack.tif"
            fetch_group(retriever, plan, intervals, stack_path)
            return split_group(retriever, plan, intervals, folder, stack_path)
    except Exception as error:
        return [
            failed_interval(plan, interval, folder, error) for interval in intervals
        ]


def failed_interval(plan, interval, folder, error):
    record = interval_record(plan, interval, folder)
    record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return outcome(plan, interval, "failed"), record


def make_group_image(retriever, plan, intervals):
    """Explicit names preserve date and band order without changing the composites."""
    composite_images = [
        make_interval_image(retriever, plan, interval, index)
        for index, interval in enumerate(intervals)
    ]
    combined = composite_images[0]
    for image in composite_images[1:]:
        combined = combined.addBands(image)
    return combined


def make_interval_image(retriever, plan, interval, index):
    granules = [
        row
        for row in plan["acquisitions"]
        if interval["start"] <= row["date"] < interval["end"]
    ]
    image = retriever.make_composite(plan["fragment"], interval, granules)
    return image.rename(
        [f"interval_{index:02d}_{band}" for band in retriever.output_bands]
    )


def fetch_group(retriever, plan, intervals, destination):
    grid = plan["settings"]["grid"]
    image = make_group_image(retriever, plan, intervals)
    parameters = {
        "name": "half_month_group",
        "format": "GEO_TIFF",
        "filePerBand": False,
        "crs": grid["crs"],
        "crs_transform": grid["transform"],
        "dimensions": [grid["width"], grid["height"]],
    }
    # A single bounded retry handles transient overload without hiding persistent failures.
    for attempt in range(2):
        try:
            url = image.getDownloadURL(parameters)
            stream_group(retriever, url, destination)
            return
        except Exception as error:
            if attempt or not transient_error(error):
                raise
            time.sleep(2)


def stream_group(retriever, url, destination):
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        retriever.write_response(response, destination)


def transient_error(error):
    response = getattr(error, "response", None)
    return isinstance(error, (requests.Timeout, requests.ConnectionError)) or getattr(
        response, "status_code", None
    ) in {429, 500, 502, 503, 504}


def validate_stack(raster, grid, interval_count):
    if raster.count != interval_count * 7 or any(
        dtype != "float32" for dtype in raster.dtypes
    ):
        raise ValueError("Grouped response has unexpected bands or data types.")
    if (raster.width, raster.height) != (grid["width"], grid["height"]):
        raise ValueError("Grouped response has unexpected dimensions.")
    if CRS.from_user_input(raster.crs) != CRS.from_user_input(grid["crs"]):
        raise ValueError("Grouped response has unexpected CRS.")
    if not raster.transform.almost_equals(Affine(*grid["transform"])):
        raise ValueError("Grouped response has unexpected transform.")


def split_group(retriever, plan, intervals, folder, stack_path):
    with rasterio.open(stack_path) as raster:
        validate_stack(raster, plan["settings"]["grid"], len(intervals))
        return [
            save_interval(retriever, plan, interval, index, folder, raster)
            for index, interval in enumerate(intervals)
        ]


def save_interval(retriever, plan, interval, index, folder, raster):
    """Validate before atomic replacement; never label a partial file complete."""
    destination = folder / f"{interval['name']}.tif"
    temporary = destination.with_suffix(".tif.part")
    record = interval_record(plan, interval, folder)
    try:
        profile = {
            **raster.profile,
            "count": 7,
            "compress": "deflate",
            "nodata": retriever.nodata_value,
        }
        bands = list(range(index * 7 + 1, index * 7 + 8))
        with rasterio.open(temporary, "w", **profile) as output:
            output.write(raster.read(bands))
        retriever.check_raster(temporary, plan["settings"]["grid"])
        record.update(retriever.finish_raster(temporary))
        temporary.replace(destination)
        return outcome(plan, interval, "downloaded"), record
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return failed_interval(plan, interval, folder, error)
