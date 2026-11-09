"""Lossless selected-pixel annual archives, independent of window selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from affine import Affine

from crop_pipeline.common import save_json
from crop_pipeline.imagery import batch_export as batch, sampling_mask, quality

FORMAT = "selected_pixel_parquet_v1"
COLUMNS = ["interval", "row", "column", *quality.STUDY_BANDS, "clear_count"]


def contract(plan):
    selection = sampling_mask.load_selection(plan)
    return {"format": FORMAT, "settings": plan["settings"],
            "acquisitions_key": plan.get("acquisition_fingerprint") or batch.content_key(plan["acquisitions"]), "mask_sha256": selection["mask_sha256"],
            "selected_pixels": selection["selected_pixels"],
            "intervals": plan["download_intervals"]}


def paths(root, plan):
    key = batch.content_key(contract(plan))[:20]
    folder = Path(root) / plan["fragment_id"]
    return folder / f"{key}.parquet", folder / f"{key}.json"


def validate_frame(frame, plan):
    """Require every selected cell for every bin, including cloudy/nodata records."""
    selection = sampling_mask.load_selection(plan)
    grid = plan["settings"]["grid"]
    expected_names = [i["name"] for i in plan["download_intervals"]]
    if set(frame["interval"]) != set(expected_names):
        raise ValueError("Pixel table periods differ from request")
    if len(frame) != len(expected_names) * selection["selected_pixels"]:
        raise ValueError("Pixel table has missing or extra selected pixels")
    for name, group in frame.groupby("interval", observed=True):
        rows = group["row"].to_numpy()
        columns = group["column"].to_numpy()
        if (not np.isfinite(rows).all() or not np.isfinite(columns).all()
                or np.any(rows != np.floor(rows)) or np.any(columns != np.floor(columns))
                or np.any(rows < 0) or np.any(rows >= grid["height"])
                or np.any(columns < 0) or np.any(columns >= grid["width"])):
            raise ValueError("Pixel coordinates are outside the saved grid")
        ids = rows.astype(np.int64) * grid["width"] + columns.astype(np.int64)
        if sorted(ids.tolist()) != selection["selected_ids"]:
            raise ValueError("Pixel table selection differs from saved mask")
    values = frame[quality.STUDY_BANDS + ["clear_count"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pixel tables must represent missing values with explicit nodata")
    counts = frame["clear_count"].to_numpy(dtype=float)
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Invalid clear-observation counts")


def reusable(root, plan, deep=False):
    path, receipt = paths(root, plan)
    if not path.exists() or not receipt.exists():
        return False
    try:
        saved = json.loads(receipt.read_text())
        if saved["contract"] != contract(plan) or saved["sha256"] != checksum(path):
            return False
        if deep:
            validate_frame(pq.read_table(path).to_pandas(), plan)
        return True
    except (ValueError, KeyError, OSError):
        return False


def checksum(path):
    from crop_pipeline.windows.seasons import file_sha256
    return file_sha256(path)


def pending(retriever, plans):
    root = retriever.imagery_folder / "pixel_tables"
    return [(plan, plan["download_intervals"]) for plan in plans if not reusable(root, plan)]


def write_tables(retriever, plans_by_id, table, region_ids):
    """Publish an annual partition only after full selected-pixel validation."""
    root = retriever.imagery_folder / "pixel_tables"
    unknown = set(table["region_id"]) - set(region_ids) if len(table) else set()
    if unknown:
        raise ValueError(f"Unexpected exported regions: {unknown}")
    total = 0
    for region_id in region_ids:
        plan = plans_by_id[region_id]
        source = table[table["region_id"] == region_id].copy() if len(table) else pd.DataFrame(columns=["interval", "x", "y", *quality.STUDY_BANDS, "clear_count"])
        source["column"] = np.floor(source["x"].to_numpy(dtype=float)).astype("int32")
        source["row"] = np.floor(source["y"].to_numpy(dtype=float)).astype("int32")
        frame = source[COLUMNS].sort_values(["interval", "row", "column"]).reset_index(drop=True)
        validate_frame(frame, plan)
        schema = pa.schema([("interval", pa.string()), ("row", pa.int32()),
                            ("column", pa.int32()),
                            *[(band, pa.float32()) for band in quality.STUDY_BANDS + ["clear_count"]]])
        path, receipt = paths(root, plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.part")
        try:
            pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False),
                           temporary, compression="zstd", use_dictionary=["interval"])
            validate_frame(pq.read_table(temporary).to_pandas(), plan)
            digest = checksum(temporary)
            temporary.replace(path)
            save_json(receipt, {"contract": contract(plan), "path": str(path.resolve()),
                                "sha256": digest, "rows": len(frame), "bytes": path.stat().st_size,
                                "fragment_id": region_id, "fragment": plan["fragment"]})
        finally:
            temporary.unlink(missing_ok=True)
        total += len(frame)
    return total


def audit_region(root, plan, minimum_fraction=0.8):
    """Adapt raw table values to the existing window/feature schema and scaling."""
    if not reusable(root, plan, deep=True):
        raise ValueError(f"Missing or invalid selected-pixel table for {plan['fragment_id']}")
    path, _ = paths(root, plan)
    frame = pq.read_table(path).to_pandas()
    return frame_to_window_rows(frame, plan, minimum_fraction, path)


def frame_to_window_rows(frame, plan, minimum_fraction, path):
    """Shared numeric adapter for Parquet and annual TIFF archives."""
    properties = plan["fragment"]["properties"]
    grid = plan["settings"]["grid"]
    transform = Affine(*grid["transform"])
    quality_rows, pixels = [], []
    for interval in plan["download_intervals"]:
        group = frame[frame["interval"] == interval["name"]]
        raw = group[quality.STUDY_BANDS].to_numpy(dtype=np.float64)
        counts = group["clear_count"].to_numpy()
        valid = (counts > 0) & np.isfinite(raw).all(axis=1) & (raw != -9999).all(axis=1)
        fraction = int(valid.sum()) / properties["selected_pixels"]
        quality_rows.append({"region_id": plan["fragment_id"], "crop": properties["crop"],
            "partition_group": properties["partition_group"], **interval,
            "selected_pixels": properties["selected_pixels"], "valid_pixels": int(valid.sum()),
            "valid_fraction": fraction,
            "status": "quality_pass" if fraction >= minimum_fraction else "low_coverage",
            "image_path": str(path), "error": ""})
        for index, record in enumerate(group.to_dict("records")):
            x, y = transform @ (record["column"] + 0.5, record["row"] + 0.5)
            pixels.append({"region_id": plan["fragment_id"], "crop": properties["crop"],
                "partition_group": properties["partition_group"], "name": interval["name"],
                "x": x, "y": y, "source_crs": grid["crs"], "valid": int(valid[index]),
                "clear_count": int(record["clear_count"]),
                **{band: float(raw[index, b] * 0.0001) if valid[index] else None
                   for b, band in enumerate(quality.STUDY_BANDS)}})
    return quality_rows, pixels


def plan_from_receipt(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    saved = receipt["contract"]
    return {"fragment_id": receipt["fragment_id"], "fragment": receipt["fragment"],
            "settings": saved["settings"], "acquisitions": [],
            "acquisition_fingerprint": saved["acquisitions_key"],
            "download_intervals": saved["intervals"]}


def import_existing_rasters(retriever, plan):
    """Reuse a complete verified raster year, preserving its files and manifests."""
    root = retriever.imagery_folder / "pixel_tables"
    if reusable(root, plan):
        return True
    key = batch.settings_fingerprint(plan["settings"])
    manifest_path = retriever.report_folder / "composite_progress" / f"{plan['fragment_id']}_{key}.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    records = []
    for interval in plan["download_intervals"]:
        record = manifest.get("files", {}).get(interval["name"])
        path = retriever.imagery_folder / plan["fragment_id"] / key / f"{interval['name']}.tif"
        if not retriever.reusable_file(path, record, plan["settings"]["grid"]):
            return False
        records.append((interval, path))
    import rasterio
    selection = sampling_mask.load_selection(plan)
    grid = plan["settings"]["grid"]
    rows, columns = np.divmod(selection["selected_ids"], grid["width"])
    frames = []
    for interval, path in records:
        with rasterio.open(path) as raster:
            values = raster.read()[:, rows, columns].T
        frame = pd.DataFrame(values, columns=retriever.output_bands)
        frame["region_id"] = plan["fragment_id"]
        frame["interval"] = interval["name"]
        frame["x"], frame["y"] = columns + .5, rows + .5
        frames.append(frame)
    write_tables(retriever, {plan["fragment_id"]: plan}, pd.concat(frames, ignore_index=True), [plan["fragment_id"]])
    return True
