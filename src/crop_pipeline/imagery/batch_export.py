"""Run the slow Earth Engine steps as batch export tasks instead of live requests.

Daily summaries (timelines.retrieve_observed_timelines) and half-month composites
(SatelliteRetriever.download_interval) are computed server-side with the same masks,
daily mosaics, medians and reducers. Results go to temporary table assets; collecting
writes them into the exact checkpoint, GeoTIFF and manifest files the live code writes,
checks them, and deletes the temporary assets. A local ledger records every task so
finished or running work is never submitted twice.
"""

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.common import save_json
from crop_pipeline.imagery import timelines, sampling_mask

INDEX_BANDS = [("NDVI", "B4"), ("NDRE", "B5"), ("NDMI", "B11")]
MAX_OUTSTANDING_TASKS = 100  # Leave room in the project queue for other work.
FINISHED_STATES = {"SUCCEEDED", "COMPLETED"}
FAILED_STATES = {"FAILED", "CANCELLED", "CANCELLING"}
# Table assets reject rows without a geometry; exported rows carry this placeholder,
# which collecting ignores.
PLACEHOLDER = [0, 0]


# ----------------------------------------------------------------------------- ledger


def load_ledger(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_ledger(path, ledger):
    save_json(path, ledger)


def settings_fingerprint(settings):
    """Same fingerprint the live code uses for checkpoints and request folders."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def request_fingerprint(plan):
    """Bind remote work to settings and acquisition IDs, not just a parcel name."""
    payload = {"settings": plan["settings"], "acquisitions": plan["acquisitions"]}
    if plan.get("fragment", {}).get("properties", {}).get("sampling_mask"):
        payload["daily_contract"] = timelines.summary_contract(
            plan, sampling_mask.load_selection(plan), 0.8)
    return content_key(payload)


def work_item(kind, plan, interval=None):
    parts = [kind, plan["fragment_id"]]
    if interval is not None:
        parts.append(interval["name"])
    parts.append(request_fingerprint(plan))
    return "|".join(parts)


def validate_task_requests(entry, plans_by_id):
    """Never label an earlier export as if it used the current request settings."""
    expected = entry.get("request_fingerprints")
    if not expected:
        raise ValueError(
            "Legacy batch task has no request fingerprints; preserve its ledger and use a new run."
        )
    for region_id, fingerprint in expected.items():
        if (
            region_id not in plans_by_id
            or request_fingerprint(plans_by_id[region_id]) != fingerprint
        ):
            raise ValueError(
                f"Batch request changed for {region_id}; do not collect old results into this run."
            )


def wait_and_collect(
    ee,
    project,
    ledger_path,
    handlers,
    plans_by_id,
    wait=True,
    poll_seconds=30,
    maximum_wait_seconds=43200,
):
    """Allow sequential Run All while preserving a resumable, bounded task wait."""
    if poll_seconds <= 0 or maximum_wait_seconds <= 0:
        raise ValueError("Batch polling and timeout must be positive.")
    deadline = time.monotonic() + maximum_wait_seconds
    while True:
        report = collect(ee, project, ledger_path, handlers, plans_by_id=plans_by_id)
        active = [
            row
            for row in report
            if not row["collected"] and row["state"] not in FAILED_STATES
        ]
        failed_items = {
            item
            for entry in load_ledger(ledger_path)
            if entry["kind"] in handlers and entry.get("state") in FAILED_STATES
            for item in entry["items"]
        }
        completed_items = {
            item
            for entry in load_ledger(ledger_path)
            if entry["kind"] in handlers and entry.get("collected")
            for item in entry["items"]
        }
        unresolved_failures = failed_items - completed_items
        if not active:
            if unresolved_failures:
                raise RuntimeError(
                    "Batch tasks failed; rerun the submit stage to retry. Saved files are preserved."
                )
            return report
        if not wait or time.monotonic() >= deadline:
            raise RuntimeError(
                "Batch work is still running. Resume at its collect stage; no new tasks are needed."
            )
        print(
            f"Waiting for {len(active)} batch tasks; saved progress is retained.",
            flush=True,
        )
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def ensure_folder(ee, folder):
    """Create an asset folder (and its parents) under the project if missing."""
    prefix, _, rest = folder.partition("/assets/")
    path = prefix + "/assets"
    for part in rest.split("/"):
        path = f"{path}/{part}"
        try:
            ee.data.getAsset(path)
        except ee.EEException:
            ee.data.createFolder(path)


def task_state(ee, project, task_id):
    """State and EECU-seconds of one batch task."""
    metadata = ee.data.getOperation(f"projects/{project}/operations/{task_id}")[
        "metadata"
    ]
    return metadata.get("state", "UNKNOWN"), float(
        metadata.get("batchEecuUsageSeconds", 0) or 0
    )


def outstanding_tasks(ledger):
    return sum(not entry.get("collected") and entry.get("state") not in FAILED_STATES
               for entry in ledger)


def in_flight(ledger):
    """Work items covered by tasks that are submitted and not yet collected or failed."""
    covered = set()
    for entry in ledger:
        if (
            entry["kind"] != "permanent"
            and not entry.get("collected")
            and entry.get("state") not in FAILED_STATES
        ):
            covered.update(entry["items"])
    return covered


# ----------------------------------------------------------------------------- server side


def granule_sources(ee, retriever, indices):
    """The region's own granules and their exact cloud-probability matches.

    Image IDs built inside a server-side map are not constants, so granules are looked
    up by system:index in these small collections; the images are the same.
    """
    in_region = ee.Filter.inList("system:index", indices)
    return (
        ee.ImageCollection(retriever.collection_name).filter(in_region),
        ee.ImageCollection(retriever.cloud_probability_collection).filter(in_region),
    )


def masked_mosaic(ee, retriever, sources, indices_csv):
    """One date: masked granules (same order as live code) mosaicked like the live code."""
    granules, probabilities = sources

    def masked(index):
        same_index = ee.Filter.eq("system:index", ee.String(index))
        return retriever.mask_acquisition(
            ee.Image(granules.filter(same_index).first()),
            ee.Image(probabilities.filter(same_index).first()),
        )

    images = ee.String(indices_csv).split(",").map(masked)
    return ee.ImageCollection.fromImages(images).mosaic()


OBSERVATION_COUNTS = ("pre_mask_count", "cloud_shadow_count", "other_rejection_count")


def screening_flags(retriever, image, probability):
    scl = image.select("SCL")
    present = image.select("B3").mask()
    cloudy = scl.remap([3, 8, 9, 10], [1, 1, 1, 1], 0).unmask(0)
    cloudy = cloudy.Or(probability.select("probability").gte(retriever.cloud_probability_threshold).unmask(0))
    return present.rename("present").addBands(cloudy.And(present).rename("cloudy")).unmask(0)


def classify_screened_date(flags, clear):
    """A date with any clear granule survives. Count rejected dates only once."""
    present = flags.select("present").unmask(0, sameFootprint=False).Or(clear)
    rejected = present.And(clear.Not())
    cloud = rejected.And(flags.select("cloudy").unmask(0, sameFootprint=False))
    return present.rename("pre_mask_count").addBands(cloud.rename("cloud_shadow_count")).addBands(
        rejected.And(cloud.Not()).rename("other_rejection_count")
    )


def daily_screening(ee, retriever, sources, indices_csv, clear):
    granules, probabilities = sources

    def flags(index):
        match = ee.Filter.eq("system:index", ee.String(index))
        return screening_flags(retriever, ee.Image(granules.filter(match).first()),
                               ee.Image(probabilities.filter(match).first()))

    daily = ee.ImageCollection.fromImages(ee.String(indices_csv).split(",").map(flags)).max()
    return classify_screened_date(daily, clear)


def daily_index_image(ee, daily):
    """Explicit ratios preserve negative reflectance values and mask zero denominators."""
    indices = []
    for name, other in INDEX_BANDS:
        left, right = daily.select("B8"), daily.select(other)
        denominator = left.add(right)
        indices.append(
            left.subtract(right)
            .divide(denominator)
            .updateMask(denominator.neq(0))
            .rename(name)
        )
    return ee.Image.cat(indices)


def region_indices(acquisitions):
    return sorted(
        {row["image_index"] for row in acquisitions if row.get("cloud_probability_id")}
    )


def day_entries(acquisitions, start=None, end=None):
    """'date;index,index' per date, granules sorted by image_index like the live code."""
    matched = [
        row
        for row in acquisitions
        if row.get("cloud_probability_id")
        and (start is None or start <= row["date"] < end)
    ]
    entries = []
    for observation_date in sorted({row["date"] for row in matched}):
        same_date = sorted(
            (row for row in matched if row["date"] == observation_date),
            key=lambda row: row["image_index"],
        )
        entries.append(
            f"{observation_date};" + ",".join(row["image_index"] for row in same_date)
        )
    return entries


def daily_collection(ee, retriever, plans):
    """One feature per region-date with the median and count of each index."""
    regions = []
    for plan in plans:
        selection = sampling_mask.load_selection(plan)
        grid = selection["grid"]
        regions.append(ee.Feature(sampling_mask.grid_footprint(ee, grid), {
            "region_id": plan["fragment_id"], "crs": grid["crs"],
            "transform": grid["transform"], "width": grid["width"], "height": grid["height"],
            "selected_ids": selection["selected_ids"],
            "days": day_entries(plan["acquisitions"]), "indices": region_indices(plan["acquisitions"]),
        }))

    def region_days(feature):
        feature = ee.Feature(feature)
        selected = sampling_mask.selection_image(ee, {
            "crs": feature.get("crs"), "transform": feature.get("transform"),
            "width": ee.Number(feature.get("width")), "height": ee.Number(feature.get("height")),
        }, feature.get("selected_ids"))
        sources = granule_sources(ee, retriever, feature.get("indices"))
        reducer = ee.Reducer.median().unweighted().combine(ee.Reducer.count(), sharedInputs=True)

        def one_day(entry):
            parts = ee.String(entry).split(";")
            daily = masked_mosaic(ee, retriever, sources, parts.get(1))
            image = timelines.daily_index_image(daily, ee).updateMask(selected)
            statistics = image.reduceRegion(
                reducer=reducer, geometry=feature.geometry(), crs=feature.get("crs"),
                crsTransform=feature.get("transform"), maxPixels=1000000)
            return ee.Feature(ee.Geometry.Point(PLACEHOLDER), statistics).set(
                {"date": parts.get(0), "region_id": feature.get("region_id")})
        return ee.FeatureCollection(ee.List(feature.get("days")).map(one_day))

    return ee.FeatureCollection(regions).map(region_days).flatten()


def composite_collection(ee, retriever, items, selected_only=False):
    """One feature per grid pixel of each (region, interval) half-month composite."""
    bands, outputs, nodata = (
        retriever.spectral_bands,
        retriever.output_bands,
        retriever.nodata_value,
    )
    if selected_only:
        outputs = [*outputs, *OBSERVATION_COUNTS]
    regions = []
    for plan, intervals in items:
        grid = plan["settings"]["grid"]
        left, top = grid["transform"][2], grid["transform"][5]
        right, bottom = left + 10 * grid["width"], top - 10 * grid["height"]
        # Inset 1 m so every pixel centre is inside and no outside centre is.
        rectangle = ee.Geometry.Rectangle(
            [left + 1, bottom + 1, right - 1, top - 1], proj=grid["crs"], geodesic=False
        )
        regions.append(
            ee.Feature(
                rectangle,
                {
                    "region_id": plan["fragment_id"],
                    "crs": grid["crs"],
                    "transform": grid["transform"],
                    "selected_ids": sampling_mask.load_selection(plan)["selected_ids"] if selected_only else [],
                    "width": grid["width"], "height": grid["height"],
                    "intervals": [
                        interval["name"]
                        + "".join(
                            ";" + entry.replace(";", "=")
                            for entry in day_entries(
                                plan["acquisitions"], interval["start"], interval["end"]
                            )
                        )
                        for interval in intervals
                    ],
                    "indices": region_indices(plan["acquisitions"]),
                },
            )
        )

    def region_pixels(feature):
        feature = ee.Feature(feature)
        projection = ee.Projection(feature.get("crs"), feature.get("transform"))
        rectangle = feature.geometry()
        sources = granule_sources(ee, retriever, feature.get("indices"))

        def one_interval(entry):
            parts = ee.String(entry).split(";")
            days = parts.slice(1)
            def build_day(day):
                indices = ee.String(day).split("=").get(1)
                mosaic = masked_mosaic(ee, retriever, sources, indices)
                if selected_only:
                    clear = mosaic.select("B3").mask().unmask(0)
                    mosaic = mosaic.addBands(daily_screening(ee, retriever, sources, indices, clear))
                return mosaic

            dailies = days.map(build_day)
            daily_images = ee.ImageCollection.fromImages(dailies)
            median = daily_images.median().select(bands)
            clear_count = (
                daily_images.select("B3")
                .count()
                .rename("clear_count")
                .unmask(0, sameFootprint=False)
            )
            built = median.unmask(nodata, sameFootprint=False).addBands(clear_count)
            if selected_only:
                counts = daily_images.select(list(OBSERVATION_COUNTS)).sum().unmask(0, sameFootprint=False)
                built = built.addBands(counts)
            empty = ee.Image.constant([nodata] * len(bands) + [0] * (len(outputs) - len(bands))).rename(outputs)
            composite = ee.Image(
                ee.Algorithms.If(days.size().gt(0), built, empty)
            ).toFloat()
            if selected_only:
                selected = sampling_mask.selection_image(ee, {
                    "crs": feature.get("crs"), "transform": feature.get("transform"),
                    "width": ee.Number(feature.get("width")), "height": ee.Number(feature.get("height"))
                }, feature.get("selected_ids"))
                composite = composite.updateMask(selected)
            samples = composite.addBands(ee.Image.pixelCoordinates(projection)).sample(
                region=rectangle,
                projection=projection,
                dropNulls=selected_only,
                geometries=False,
            )
            return samples.map(
                lambda pixel: pixel.setGeometry(ee.Geometry.Point(PLACEHOLDER)).set(
                    {"region_id": feature.get("region_id"), "interval": parts.get(0)}
                )
            )

        return ee.FeatureCollection(
            ee.List(feature.get("intervals")).map(one_interval)
        ).flatten()

    return ee.FeatureCollection(regions).map(region_pixels).flatten()


# ----------------------------------------------------------------------------- submit


def chunk(items, size_of, limit):
    """Group items so each task holds about `limit` units of work."""
    groups, current, total = [], [], 0
    for item in items:
        if current and total + size_of(item) > limit:
            groups.append(current)
            current, total = [], 0
        current.append(item)
        total += size_of(item)
    if current:
        groups.append(current)
    return groups


def start_export(ee, collection, asset_id, description):
    task = ee.batch.Export.table.toAsset(
        collection=collection,
        description=description[:100],
        assetId=asset_id,
        overwrite=False,
    )
    task.start()
    return task.id


def missing_daily_checkpoints(plans, timeline_folder):
    """Plans without the checkpoint file retrieve_observed_timelines would reuse."""
    return [plan for plan in plans if not daily_checkpoint(plan, timeline_folder).exists()]


def daily_checkpoint(plan, folder):
    if plan.get("fragment", {}).get("properties", {}).get("sampling_mask"):
        selection = sampling_mask.load_selection(plan)
        contract = timelines.summary_contract(plan, selection, 0.8)
        return timelines.checkpoint_path(folder, plan["fragment_id"], contract)
    return Path(folder) / "observed_checkpoints" / (
        f"{plan['fragment_id']}_{settings_fingerprint(plan['settings'])}.json")


def submit_daily(
    ee,
    retriever,
    plans,
    timeline_folder,
    ledger_path,
    temp_folder,
    year,
    region_days_per_task=6000,
    checkpoint_folder=None,
):
    """Submit daily-summary tasks for regions without a local checkpoint; return at once."""
    ledger = load_ledger(ledger_path)
    covered = in_flight(ledger)
    missing = [
        plan
        for plan in plans
        if not daily_checkpoint(plan, checkpoint_folder(plan) if checkpoint_folder
                                else timeline_folder).exists()
        and work_item("daily", plan) not in covered
    ]
    ensure_folder(ee, temp_folder)
    submitted = []
    groups = chunk(
        missing,
        lambda plan: max(1, len(day_entries(plan["acquisitions"]))),
        region_days_per_task,
    )
    for group in groups:
        if outstanding_tasks(ledger) >= MAX_OUTSTANDING_TASKS:
            break
        key = hashlib.sha256(
            "|".join(work_item("daily", plan) for plan in group).encode()
        ).hexdigest()[:10]
        asset_id = f"{temp_folder}/daily_{year}_{key}"
        task_id = start_export(
            ee,
            daily_collection(ee, retriever, group),
            asset_id,
            f"cropharvest_daily_{year}_{key}",
        )
        entry = {
            "task_id": task_id,
            "kind": "daily",
            "year": year,
            "asset_id": asset_id,
            "items": [work_item("daily", plan) for plan in group],
            "request_fingerprints": {
                plan["fragment_id"]: request_fingerprint(plan) for plan in group
            },
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "state": "SUBMITTED",
            "collected": False,
        }
        ledger.append(entry)
        save_ledger(ledger_path, ledger)
        submitted.append(entry)
    save_ledger(ledger_path, ledger)
    return {
        "regions_needing_work": len(missing),
        "tasks_submitted": len(submitted),
        "already_running": sum(1 for p in plans if work_item("daily", p) in covered),
    }


def pending_composites(retriever, plans):
    """(plan, intervals) whose GeoTIFF is not yet a verified, reusable file."""
    work = []
    for plan in plans:
        settings = plan["settings"]
        folder = (
            retriever.imagery_folder
            / plan["fragment_id"]
            / settings_fingerprint(settings)
        )
        manifest_path = (
            retriever.report_folder
            / "composite_progress"
            / f"{plan['fragment_id']}_{settings_fingerprint(settings)}.json"
        )
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"files": {}}
        )
        intervals = [
            interval
            for interval in plan["download_intervals"]
            if not retriever.reusable_file(
                folder / f"{interval['name']}.tif",
                manifest["files"].get(interval["name"]),
                settings["grid"],
            )
        ]
        if intervals:
            work.append((plan, intervals))
    return work


def submit_composites(
    ee, retriever, plans, ledger_path, temp_folder, year, pixels_per_task=60000,
    selected_only=False,
):
    """Submit composite tasks for qualified bins not yet saved locally; return at once."""
    ledger = load_ledger(ledger_path)
    covered = in_flight(ledger)
    work = []
    from crop_pipeline.imagery import pixel_tables
    pending_work = pixel_tables.pending if selected_only else pending_composites
    kind = "pixel_table" if selected_only else "composite"
    for plan, intervals in pending_work(retriever, plans):
        intervals = [
            i for i in intervals if work_item(kind, plan, i) not in covered
        ]
        if intervals:
            work.append((plan, intervals))
    ensure_folder(ee, temp_folder)
    grid_pixels = lambda item: (
        (item[0]["fragment"]["properties"]["selected_pixels"] if selected_only else
         item[0]["settings"]["grid"]["width"] * item[0]["settings"]["grid"]["height"])
        * len(item[1])
    )
    submitted = []
    for group in chunk(work, grid_pixels, pixels_per_task):
        if outstanding_tasks(ledger) >= MAX_OUTSTANDING_TASKS:
            break
        names = [
            work_item(kind, plan, interval)
            for plan, intervals in group
            for interval in intervals
        ]
        key = hashlib.sha256("|".join(names).encode()).hexdigest()[:10]
        asset_id = f"{temp_folder}/{kind}_{year}_{key}"
        task_id = start_export(
            ee,
            composite_collection(ee, retriever, group, selected_only=True) if selected_only
            else composite_collection(ee, retriever, group),
            asset_id,
            f"cropharvest_composites_{year}_{key}",
        )
        entry = {
            "task_id": task_id,
            "kind": kind,
            "year": year,
            "asset_id": asset_id,
            "items": names,
            "request_fingerprints": {
                plan["fragment_id"]: request_fingerprint(plan) for plan, _ in group
            },
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "state": "SUBMITTED",
            "collected": False,
        }
        ledger.append(entry)
        save_ledger(ledger_path, ledger)
        submitted.append(entry)
    save_ledger(ledger_path, ledger)
    return {
        "composites_needing_work": sum(len(i) for _, i in work),
        "tasks_submitted": len(submitted),
    }


# ----------------------------------------------------------------------------- collect


def read_table(ee, asset_id, cache_dir=None):
    from crop_pipeline.imagery.table_download import download
    return download(ee, asset_id, cache_dir=cache_dir)


def read_pixel_table_csv(ee, asset_id):
    """Retrieve only measurement fields, avoiding per-feature GeoJSON overhead."""
    from io import BytesIO
    from zipfile import ZipFile, is_zipfile
    import pandas as pd
    import requests
    selectors = ["region_id", "interval", "x", "y", "B3", "B4", "B5", "B8", "B11", "B2", "clear_count"]
    url = ee.FeatureCollection(asset_id).getDownloadURL(filetype="CSV", selectors=selectors)
    with requests.get(url, timeout=(30, 180)) as response:
        response.raise_for_status()
        content = BytesIO(response.content)
    if is_zipfile(content):
        with ZipFile(content) as archive:
            files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(files) != 1:
                raise ValueError("Unexpected CSV download archive")
            with archive.open(files[0]) as stream:
                return pd.read_csv(stream, float_precision="round_trip")
    content.seek(0)
    return pd.read_csv(content, float_precision="round_trip")


def finite_or_none(value):
    return (
        None
        if value is None or (isinstance(value, float) and math.isnan(value))
        else float(value)
    )


def write_daily_checkpoints(
    plans_by_id, region_ids, table, timeline_folder, minimum_fraction
):
    """Rows identical to retrieve_observed_timelines, one checkpoint file per region.

    A region with no cloud-matched dates exports no rows; like the live code it still
    gets a checkpoint, with an empty row list.
    """
    written = 0
    groups = dict(tuple(table.groupby("region_id"))) if len(table) else {}
    for region_id in region_ids:
        rows = groups.get(region_id, table.iloc[0:0] if len(table) else None)
        plan = plans_by_id[region_id]
        properties = plan["fragment"]["properties"]
        expected = [entry.split(";")[0] for entry in day_entries(plan["acquisitions"])]
        records = [] if rows is None else rows.sort_values("date").to_dict("records")
        if [values["date"] for values in records] != expected:
            raise ValueError(f"{region_id}: exported dates differ from the plan")
        output = []
        for values in records:
            count = min(
                int(finite_or_none(values.get(name + "_count")) or 0)
                for name, _ in INDEX_BANDS
            )
            fraction = count / properties["selected_pixels"]
            if fraction > 1 + 1e-6:
                raise ValueError(
                    "Region reduction included more pixels than the sampling mask"
                )
            output.append(
                {
                    "region_id": region_id,
                    "partition_group": properties["partition_group"],
                    "crop": properties["crop"],
                    "date": values["date"],
                    "source_kind": "observed_daily",
                    "valid_pixels": count,
                    "valid_fraction": fraction,
                    "eligible": fraction >= minimum_fraction,
                    **{
                        name: finite_or_none(values.get(name + "_median"))
                        for name, _ in INDEX_BANDS
                    },
                }
            )
        checkpoint = {
            "settings": plan["settings"],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "rows": output,
            "retrieval": "earth_engine_batch_export",
        }
        if properties.get("sampling_mask"):
            selection = sampling_mask.load_selection(plan)
            checkpoint["contract"] = timelines.summary_contract(plan, selection, minimum_fraction)
            timelines.validate_cached_rows(output, plan, minimum_fraction)
        save_json(daily_checkpoint(plan, timeline_folder), checkpoint)
        written += 1
    # Regions whose plan has no matched dates export no rows; the live code saves them empty.
    return written


def write_composites(retriever, plans_by_id, table):
    """Rebuild each half-month GeoTIFF on its grid and record it like download_interval."""
    written = 0
    for (region_id, name), pixels in table.groupby(["region_id", "interval"]):
        plan = plans_by_id[region_id]
        settings, grid = plan["settings"], plan["settings"]["grid"]
        # pixelCoordinates in the grid's own projection gives column / row centres (0.5, 1.5 ...).
        columns = np.floor(pixels["x"].to_numpy()).astype(int)
        rows = np.floor(pixels["y"].to_numpy()).astype(int)
        if (
            len(pixels) != grid["width"] * grid["height"]
            or columns.min() < 0
            or rows.min() < 0
            or columns.max() >= grid["width"]
            or rows.max() >= grid["height"]
            or len(set(zip(columns, rows))) != len(pixels)
        ):
            raise ValueError(
                f"{region_id} {name}: exported pixels do not fill the grid"
            )
        values = np.full(
            (len(retriever.output_bands), grid["height"], grid["width"]),
            retriever.nodata_value,
            dtype="float32",
        )
        for band_index, band in enumerate(retriever.output_bands):
            values[band_index, rows, columns] = pixels[band].to_numpy(dtype="float32")

        request_id = settings_fingerprint(settings)
        folder = retriever.imagery_folder / region_id / request_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = (
            retriever.report_folder
            / "composite_progress"
            / f"{region_id}_{request_id}.json"
        )
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"fragment_id": region_id, "settings": settings, "files": {}}
        )
        interval = next(i for i in plan["download_intervals"] if i["name"] == name)
        granules = [
            row
            for row in plan["acquisitions"]
            if interval["start"] <= row["date"] < interval["end"]
        ]
        destination = folder / f"{name}.tif"
        record = {
            **interval,
            "path": str(destination.resolve()),
            "source_image_ids": [row["image_id"] for row in granules],
            "available_dates": sorted({row["date"] for row in granules}),
            "cloud_screened_available_dates": sorted(
                {row["date"] for row in granules if row.get("cloud_probability_id")}
            ),
        }
        temporary = destination.with_suffix(".tif.part")
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=grid["width"],
            height=grid["height"],
            count=len(retriever.output_bands),
            dtype="float32",
            crs=grid["crs"],
            transform=Affine(*grid["transform"]),
        ) as raster:
            raster.write(values)
        retriever.check_raster(temporary, grid)
        record.update(retriever.finish_raster(temporary))
        temporary.replace(destination)
        manifest["files"][name] = record
        save_json(manifest_path, manifest)
        written += 1
    return written


def collect(
    ee, project, ledger_path, handlers, kinds=None, year=None, plans_by_id=None
):
    """Save finished tasks locally, then delete their temporary assets.

    `handlers` maps a task kind to a function(entry, table) that writes and checks the
    local files; a kind mapped to None is a permanent asset, which is only marked done
    and never deleted. Only tasks of `kinds` (default: the handler kinds) and `year`
    are touched. Returns one status row per task, including EECU-seconds.
    """
    kinds = set(kinds or handlers)
    ledger = load_ledger(ledger_path)
    report = []
    for entry in ledger:
        download_cache = None
        if entry["kind"] not in kinds or (
            year is not None and entry.get("year") != year
        ):
            continue
        if entry.get("collected"):
            report.append({**_summary(entry), "note": "already collected"})
            continue
        if entry.get("state") in {"FAILED", "CANCELLED"}:
            report.append({**_summary(entry), "note": "previous failed/cancelled attempt; submit retries missing work"})
            continue
        if entry["kind"] != "permanent" and plans_by_id is not None:
            validate_task_requests(entry, plans_by_id)
        state, eecu = task_state(ee, project, entry["task_id"])
        entry["state"], entry["eecu_seconds"] = state, eecu
        note = ""
        if state in FINISHED_STATES:
            handler = handlers.get(entry["kind"])
            if handler is None:
                note = "permanent asset saved"
            else:
                print(f"Downloading {entry['kind']} result {entry['task_id']}…", flush=True)
                started = time.monotonic()
                download_cache = Path(ledger_path).parent / "download_pages" / entry["task_id"]
                table = read_table(ee, entry["asset_id"], cache_dir=download_cache)
                entry["download_seconds"] = time.monotonic() - started
                entry["download_rows"] = len(table)
                started = time.monotonic()
                note = handler(entry, table)
                entry["local_write_seconds"] = time.monotonic() - started
                print(f"Verified and saved {len(table):,} result rows.", flush=True)
                ee.data.deleteAsset(entry["asset_id"])
                note += "; temporary asset deleted"
            entry["collected"] = True
            entry["collected_at"] = datetime.now(timezone.utc).isoformat()
        elif state in FAILED_STATES:
            note = "failed: the submit cell will resubmit this work"
        save_ledger(ledger_path, ledger)
        report.append({**_summary(entry), "note": note})
        if entry.get("collected") and download_cache is not None:
            import shutil
            shutil.rmtree(download_cache, ignore_errors=True)
    return report


def remove_temp_folder(ee, ledger_path, temp_folder):
    """Delete the temporary asset folder once no temporary task is waiting to be collected."""
    waiting = [
        entry
        for entry in load_ledger(ledger_path)
        if entry["kind"] != "permanent"
        and not entry.get("collected")
        and entry.get("state") not in FAILED_STATES
    ]
    try:
        left = ee.data.listAssets({"parent": temp_folder}).get("assets", [])
    except ee.EEException:
        return "no temporary folder"
    if waiting or left:
        return f"kept: {len(waiting)} tasks waiting, {len(left)} temporary assets"
    ee.data.deleteAsset(temp_folder)
    return "temporary folder removed"


def submit_permanent(ee, collection, asset_id, ledger_path, content_key, description):
    """Export a permanent table asset unless the same content was already saved there."""
    asset_id = f"{asset_id}_{content_key}"
    ledger = load_ledger(ledger_path)
    for entry in ledger:
        if (
            entry["kind"] == "permanent"
            and entry["asset_id"] == asset_id
            and entry.get("content_key") == content_key
            and entry.get("state") not in FAILED_STATES
        ):
            return {
                "asset_id": asset_id,
                "submitted": False,
                "reason": "same content already exported",
            }
    ensure_folder(ee, asset_id.rsplit("/", 1)[0])
    task_id = start_export(ee, collection, asset_id, description)
    ledger.append(
        {
            "task_id": task_id,
            "kind": "permanent",
            "year": None,
            "asset_id": asset_id,
            "items": [asset_id],
            "content_key": content_key,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "state": "SUBMITTED",
            "collected": False,
        }
    )
    save_ledger(ledger_path, ledger)
    return {"asset_id": asset_id, "submitted": True, "task_id": task_id}


def content_key(records):
    """Fingerprint of what a permanent asset should contain."""
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _summary(entry):
    return {
        "task_id": entry["task_id"],
        "kind": entry["kind"],
        "year": entry.get("year"),
        "items": len(entry["items"]),
        "state": entry.get("state"),
        "eecu_seconds": round(float(entry.get("eecu_seconds") or 0), 4),
        "collected": bool(entry.get("collected")),
    }
