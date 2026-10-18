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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from crop_pipeline.common import save_json

INDEX_BANDS = [("NDVI", "B4"), ("NDRE", "B5"), ("NDMI", "B11")]
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
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


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
    metadata = ee.data.getOperation(f"projects/{project}/operations/{task_id}")["metadata"]
    return metadata.get("state", "UNKNOWN"), float(metadata.get("batchEecuUsageSeconds", 0) or 0)


def in_flight(ledger):
    """Work items covered by tasks that are submitted and not yet collected or failed."""
    covered = set()
    for entry in ledger:
        if (entry["kind"] != "permanent" and not entry.get("collected")
                and entry.get("state") not in FAILED_STATES):
            covered.update(entry["items"])
    return covered


# ----------------------------------------------------------------------------- server side


def granule_sources(ee, retriever, indices):
    """The region's own granules and their exact cloud-probability matches.

    Image IDs built inside a server-side map are not constants, so granules are looked
    up by system:index in these small collections; the images are the same.
    """
    in_region = ee.Filter.inList("system:index", indices)
    return (ee.ImageCollection(retriever.collection_name).filter(in_region),
            ee.ImageCollection(retriever.cloud_probability_collection).filter(in_region))


def masked_mosaic(ee, retriever, sources, indices_csv):
    """One date: masked granules (same order as live code) mosaicked like the live code."""
    granules, probabilities = sources

    def masked(index):
        same_index = ee.Filter.eq("system:index", ee.String(index))
        return retriever.mask_acquisition(ee.Image(granules.filter(same_index).first()),
                                          ee.Image(probabilities.filter(same_index).first()))

    images = ee.String(indices_csv).split(",").map(masked)
    return ee.ImageCollection.fromImages(images).mosaic()


def daily_index_image(ee, daily):
    """Explicit ratios preserve negative reflectance values and mask zero denominators."""
    indices = []
    for name, other in INDEX_BANDS:
        left, right = daily.select("B8"), daily.select(other)
        denominator = left.add(right)
        indices.append(
            left.subtract(right).divide(denominator).updateMask(denominator.neq(0)).rename(name))
    return ee.Image.cat(indices)


def region_indices(acquisitions):
    return sorted({row["image_index"] for row in acquisitions if row.get("cloud_probability_id")})


def day_entries(acquisitions, start=None, end=None):
    """'date;index,index' per date, granules sorted by image_index like the live code."""
    matched = [row for row in acquisitions if row.get("cloud_probability_id")
               and (start is None or start <= row["date"] < end)]
    entries = []
    for observation_date in sorted({row["date"] for row in matched}):
        same_date = sorted((row for row in matched if row["date"] == observation_date),
                           key=lambda row: row["image_index"])
        entries.append(f"{observation_date};" + ",".join(row["image_index"] for row in same_date))
    return entries


def daily_collection(ee, retriever, plans):
    """One feature per region-date with the median and count of each index."""
    reducer = ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True)
    regions = []
    for plan in plans:
        grid = plan["settings"]["grid"]
        footprint = ee.Geometry(plan["fragment"]["geometry"], proj="EPSG:4326", geodesic=False)
        regions.append(ee.Feature(footprint, {
            "region_id": plan["fragment_id"], "crs": grid["crs"],
            "transform": grid["transform"], "days": day_entries(plan["acquisitions"]),
            "indices": region_indices(plan["acquisitions"]),
        }))

    def region_days(feature):
        feature = ee.Feature(feature)
        geometry = feature.geometry()
        sources = granule_sources(ee, retriever, feature.get("indices"))

        def one_day(entry):
            parts = ee.String(entry).split(";")
            image = daily_index_image(ee, masked_mosaic(ee, retriever, sources, parts.get(1)))
            statistics = image.reduceRegion(
                reducer=reducer, geometry=geometry, crs=feature.get("crs"),
                crsTransform=feature.get("transform"), maxPixels=1000000)
            return ee.Feature(ee.Geometry.Point(PLACEHOLDER), statistics).set(
                {"date": parts.get(0), "region_id": feature.get("region_id")})

        return ee.FeatureCollection(ee.List(feature.get("days")).map(one_day))

    return ee.FeatureCollection(regions).map(region_days).flatten()


def composite_collection(ee, retriever, items):
    """One feature per grid pixel of each (region, interval) half-month composite."""
    bands, outputs, nodata = retriever.spectral_bands, retriever.output_bands, retriever.nodata_value
    regions = []
    for plan, intervals in items:
        grid = plan["settings"]["grid"]
        left, top = grid["transform"][2], grid["transform"][5]
        right, bottom = left + 10 * grid["width"], top - 10 * grid["height"]
        # Inset 1 m so every pixel centre is inside and no outside centre is.
        rectangle = ee.Geometry.Rectangle([left + 1, bottom + 1, right - 1, top - 1],
                                          proj=grid["crs"], geodesic=False)
        regions.append(ee.Feature(rectangle, {
            "region_id": plan["fragment_id"], "crs": grid["crs"], "transform": grid["transform"],
            "intervals": [
                interval["name"] + "".join(
                    ";" + entry.replace(";", "=")
                    for entry in day_entries(plan["acquisitions"], interval["start"], interval["end"]))
                for interval in intervals
            ],
            "indices": region_indices(plan["acquisitions"]),
        }))

    def region_pixels(feature):
        feature = ee.Feature(feature)
        projection = ee.Projection(feature.get("crs"), feature.get("transform"))
        rectangle = feature.geometry()
        sources = granule_sources(ee, retriever, feature.get("indices"))

        def one_interval(entry):
            parts = ee.String(entry).split(";")
            days = parts.slice(1)
            dailies = days.map(
                lambda day: masked_mosaic(ee, retriever, sources, ee.String(day).split("=").get(1)))
            daily_images = ee.ImageCollection.fromImages(dailies)
            median = daily_images.median().select(bands)
            clear_count = (daily_images.select("B3").count().rename("clear_count")
                           .unmask(0, sameFootprint=False))
            built = median.unmask(nodata, sameFootprint=False).addBands(clear_count)
            empty = ee.Image.constant([nodata] * len(bands) + [0]).rename(outputs)
            composite = ee.Image(ee.Algorithms.If(days.size().gt(0), built, empty)).toFloat()
            samples = composite.addBands(ee.Image.pixelCoordinates(projection)).sample(
                region=rectangle, projection=projection, dropNulls=False, geometries=False)
            return samples.map(lambda pixel: pixel.setGeometry(ee.Geometry.Point(PLACEHOLDER)).set(
                {"region_id": feature.get("region_id"), "interval": parts.get(0)}))

        return ee.FeatureCollection(ee.List(feature.get("intervals")).map(one_interval)).flatten()

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
    task = ee.batch.Export.table.toAsset(collection=collection, description=description[:100],
                                         assetId=asset_id, overwrite=True)
    task.start()
    return task.id


def missing_daily_checkpoints(plans, timeline_folder):
    """Plans without the checkpoint file retrieve_observed_timelines would reuse."""
    return [plan for plan in plans
            if not (Path(timeline_folder) / "observed_checkpoints"
                    / f"{plan['fragment_id']}_{settings_fingerprint(plan['settings'])}.json").exists()]


def submit_daily(ee, retriever, plans, timeline_folder, ledger_path, temp_folder, year,
                 region_days_per_task=6000):
    """Submit daily-summary tasks for regions without a local checkpoint; return at once."""
    ledger = load_ledger(ledger_path)
    covered = in_flight(ledger)
    missing = [plan for plan in missing_daily_checkpoints(plans, timeline_folder)
               if f"daily|{plan['fragment_id']}" not in covered]
    ensure_folder(ee, temp_folder)
    submitted = []
    groups = chunk(missing, lambda plan: max(1, len(day_entries(plan["acquisitions"]))),
                   region_days_per_task)
    for group in groups:
        key = hashlib.sha256("|".join(p["fragment_id"] for p in group).encode()).hexdigest()[:10]
        asset_id = f"{temp_folder}/daily_{year}_{key}"
        task_id = start_export(ee, daily_collection(ee, retriever, group), asset_id,
                               f"cropharvest_daily_{year}_{key}")
        entry = {"task_id": task_id, "kind": "daily", "year": year, "asset_id": asset_id,
                 "items": [f"daily|{plan['fragment_id']}" for plan in group],
                 "submitted_at": datetime.now(timezone.utc).isoformat(),
                 "state": "SUBMITTED", "collected": False}
        ledger.append(entry)
        submitted.append(entry)
    save_ledger(ledger_path, ledger)
    return {"regions_needing_work": len(missing), "tasks_submitted": len(submitted),
            "already_running": sum(1 for p in plans if f"daily|{p['fragment_id']}" in covered)}


def pending_composites(retriever, plans):
    """(plan, intervals) whose GeoTIFF is not yet a verified, reusable file."""
    work = []
    for plan in plans:
        settings = plan["settings"]
        folder = retriever.imagery_folder / plan["fragment_id"] / settings_fingerprint(settings)
        manifest_path = (retriever.report_folder / "composite_progress"
                         / f"{plan['fragment_id']}_{settings_fingerprint(settings)}.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
        intervals = [
            interval for interval in plan["download_intervals"]
            if not retriever.reusable_file(folder / f"{interval['name']}.tif",
                                           manifest["files"].get(interval["name"]),
                                           settings["grid"])
        ]
        if intervals:
            work.append((plan, intervals))
    return work


def submit_composites(ee, retriever, plans, ledger_path, temp_folder, year,
                      pixels_per_task=60000):
    """Submit composite tasks for qualified bins not yet saved locally; return at once."""
    ledger = load_ledger(ledger_path)
    covered = in_flight(ledger)
    work = []
    for plan, intervals in pending_composites(retriever, plans):
        intervals = [i for i in intervals
                     if f"composite|{plan['fragment_id']}|{i['name']}" not in covered]
        if intervals:
            work.append((plan, intervals))
    ensure_folder(ee, temp_folder)
    grid_pixels = lambda item: (item[0]["settings"]["grid"]["width"]
                                * item[0]["settings"]["grid"]["height"] * len(item[1]))
    submitted = []
    for group in chunk(work, grid_pixels, pixels_per_task):
        names = [f"composite|{plan['fragment_id']}|{i['name']}" for plan, ints in group for i in ints]
        key = hashlib.sha256("|".join(names).encode()).hexdigest()[:10]
        asset_id = f"{temp_folder}/composites_{year}_{key}"
        task_id = start_export(ee, composite_collection(ee, retriever, group), asset_id,
                               f"cropharvest_composites_{year}_{key}")
        entry = {"task_id": task_id, "kind": "composite", "year": year, "asset_id": asset_id,
                 "items": names, "submitted_at": datetime.now(timezone.utc).isoformat(),
                 "state": "SUBMITTED", "collected": False}
        ledger.append(entry)
        submitted.append(entry)
    save_ledger(ledger_path, ledger)
    return {"composites_needing_work": sum(len(i) for _, i in work),
            "tasks_submitted": len(submitted)}


# ----------------------------------------------------------------------------- collect


def read_table(ee, asset_id):
    return ee.data.computeFeatures({"expression": ee.FeatureCollection(asset_id),
                                    "fileFormat": "PANDAS_DATAFRAME"})


def finite_or_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def write_daily_checkpoints(plans_by_id, region_ids, table, timeline_folder, minimum_fraction):
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
            count = min(int(finite_or_none(values.get(name + "_count")) or 0)
                        for name, _ in INDEX_BANDS)
            fraction = count / properties["selected_pixels"]
            if fraction > 1 + 1e-6:
                raise ValueError("Region reduction included more pixels than the sampling mask")
            output.append({
                "region_id": region_id, "partition_group": properties["partition_group"],
                "crop": properties["crop"], "date": values["date"], "source_kind": "observed_daily",
                "valid_pixels": count, "valid_fraction": fraction,
                "eligible": fraction >= minimum_fraction,
                **{name: finite_or_none(values.get(name + "_median")) for name, _ in INDEX_BANDS},
            })
        path = (Path(timeline_folder) / "observed_checkpoints"
                / f"{region_id}_{settings_fingerprint(plan['settings'])}.json")
        save_json(path, {"settings": plan["settings"],
                         "retrieved_at": datetime.now(timezone.utc).isoformat(),
                         "rows": output, "retrieval": "earth_engine_batch_export"})
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
        if (len(pixels) != grid["width"] * grid["height"]
                or columns.min() < 0 or rows.min() < 0
                or columns.max() >= grid["width"] or rows.max() >= grid["height"]
                or len(set(zip(columns, rows))) != len(pixels)):
            raise ValueError(f"{region_id} {name}: exported pixels do not fill the grid")
        values = np.full((len(retriever.output_bands), grid["height"], grid["width"]),
                         retriever.nodata_value, dtype="float32")
        for band_index, band in enumerate(retriever.output_bands):
            values[band_index, rows, columns] = pixels[band].to_numpy(dtype="float32")

        request_id = settings_fingerprint(settings)
        folder = retriever.imagery_folder / region_id / request_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = retriever.report_folder / "composite_progress" / f"{region_id}_{request_id}.json"
        manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                    else {"fragment_id": region_id, "settings": settings, "files": {}})
        interval = next(i for i in plan["download_intervals"] if i["name"] == name)
        granules = [row for row in plan["acquisitions"]
                    if interval["start"] <= row["date"] < interval["end"]]
        destination = folder / f"{name}.tif"
        record = {
            **interval,
            "path": str(destination.resolve()),
            "source_image_ids": [row["image_id"] for row in granules],
            "available_dates": sorted({row["date"] for row in granules}),
            "cloud_screened_available_dates": sorted(
                {row["date"] for row in granules if row.get("cloud_probability_id")}),
        }
        temporary = destination.with_suffix(".tif.part")
        with rasterio.open(temporary, "w", driver="GTiff", width=grid["width"],
                           height=grid["height"], count=len(retriever.output_bands),
                           dtype="float32", crs=grid["crs"],
                           transform=Affine(*grid["transform"])) as raster:
            raster.write(values)
        retriever.check_raster(temporary, grid)
        record.update(retriever.finish_raster(temporary))
        temporary.replace(destination)
        manifest["files"][name] = record
        save_json(manifest_path, manifest)
        written += 1
    return written


def collect(ee, project, ledger_path, handlers, kinds=None, year=None):
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
        if entry["kind"] not in kinds or (year is not None and entry.get("year") != year):
            continue
        if entry.get("collected"):
            report.append({**_summary(entry), "note": "already collected"})
            continue
        state, eecu = task_state(ee, project, entry["task_id"])
        entry["state"], entry["eecu_seconds"] = state, eecu
        note = ""
        if state in FINISHED_STATES:
            handler = handlers.get(entry["kind"])
            if handler is None:
                note = "permanent asset saved"
            else:
                note = handler(entry, read_table(ee, entry["asset_id"]))
                ee.data.deleteAsset(entry["asset_id"])
                note += "; temporary asset deleted"
            entry["collected"] = True
            entry["collected_at"] = datetime.now(timezone.utc).isoformat()
        elif state in FAILED_STATES:
            note = "failed: the submit cell will resubmit this work"
        save_ledger(ledger_path, ledger)
        report.append({**_summary(entry), "note": note})
    return report


def remove_temp_folder(ee, ledger_path, temp_folder):
    """Delete the temporary asset folder once no temporary task is waiting to be collected."""
    waiting = [entry for entry in load_ledger(ledger_path)
               if entry["kind"] != "permanent" and not entry.get("collected")
               and entry.get("state") not in FAILED_STATES]
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
    ledger = load_ledger(ledger_path)
    for entry in ledger:
        if (entry["kind"] == "permanent" and entry["asset_id"] == asset_id
                and entry.get("content_key") == content_key
                and entry.get("state") not in FAILED_STATES):
            return {"asset_id": asset_id, "submitted": False, "reason": "same content already exported"}
    ensure_folder(ee, asset_id.rsplit("/", 1)[0])
    task_id = start_export(ee, collection, asset_id, description)
    ledger.append({"task_id": task_id, "kind": "permanent", "year": None, "asset_id": asset_id,
                   "items": [asset_id], "content_key": content_key,
                   "submitted_at": datetime.now(timezone.utc).isoformat(),
                   "state": "SUBMITTED", "collected": False})
    save_ledger(ledger_path, ledger)
    return {"asset_id": asset_id, "submitted": True, "task_id": task_id}


def content_key(records):
    """Fingerprint of what a permanent asset should contain."""
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _summary(entry):
    return {"task_id": entry["task_id"], "kind": entry["kind"], "year": entry.get("year"),
            "items": len(entry["items"]), "state": entry.get("state"),
            "eecu_seconds": round(float(entry.get("eecu_seconds") or 0), 4),
            "collected": bool(entry.get("collected"))}
