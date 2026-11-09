"""Organized annual GeoTIFF outputs, reconstructed losslessly from selected pixels."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from affine import Affine

from crop_pipeline.common import save_json
from crop_pipeline.imagery import pixel_tables, quality
from crop_pipeline.windows.seasons import file_sha256

BANDS = quality.STUDY_BANDS + ["clear_count"]


def export_source(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    index = {}
    for identifier, record in source["pixel_table_index"].items():
        receipt_path = Path(record["path"])
        if file_sha256(receipt_path) != record["sha256"]:
            raise ValueError("Pixel receipt changed before annual TIFF export")
        plan = pixel_tables.plan_from_receipt(receipt_path)
        root = source["pixel_table_root"]
        if not pixel_tables.reusable(root, plan, deep=True):
            raise ValueError("Pixel table is incomplete; annual TIFF not published")
        table_path, _ = pixel_tables.paths(root, plan)
        path = destination / f"{identifier}_{source['year']}.tif"
        receipt = path.with_suffix(".json")
        table_sha = file_sha256(table_path)
        saved = json.loads(receipt.read_text()) if receipt.exists() else {}
        if not (path.exists() and saved.get("table_sha256") == table_sha
                and saved.get("sha256") == file_sha256(path)):
            grid = plan["settings"]["grid"]
            intervals = plan["download_intervals"]
            if len(intervals) != 24:
                raise ValueError("An annual TIFF requires all 24 half-month periods")
            frame = pq.read_table(table_path).to_pandas()
            values = np.full((len(intervals) * 7, grid["height"], grid["width"]), -9999, dtype="float32")
            descriptions = []
            for number, interval in enumerate(intervals):
                rows = frame[frame["interval"] == interval["name"]]
                values[number * 7:(number + 1) * 7, rows["row"].to_numpy(), rows["column"].to_numpy()] = rows[BANDS].to_numpy(dtype="float32").T
                descriptions.extend(f"{interval['name']}__{band}" for band in BANDS)
            temporary = path.with_suffix(".tif.part")
            try:
                with rasterio.open(temporary, "w", driver="GTiff", width=grid["width"], height=grid["height"],
                        count=len(descriptions), dtype="float32", crs=grid["crs"],
                        transform=Affine(*grid["transform"]), nodata=-9999,
                        compress="deflate", predictor=3, BIGTIFF="IF_SAFER") as image:
                    image.write(values)
                    image.descriptions = tuple(descriptions)
                    image.update_tags(archive="annual_selected_pixels_v1", region_id=identifier,
                                      year=str(source["year"]), pixels="saved parcel selection only")
                with rasterio.open(temporary) as image:
                    if image.descriptions != tuple(descriptions) or image.count != 168:
                        raise ValueError("Annual TIFF band labels differ from the requested year")
                    np.testing.assert_array_equal(image.read(), values)
                temporary.replace(path)
                save_json(receipt, {"table_sha256": table_sha, "sha256": file_sha256(path),
                                    "bands": descriptions, "grid": grid, "selected_pixels_only": True})
            finally:
                temporary.unlink(missing_ok=True)
        index[identifier] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return index


def audit_region(source, region, minimum_fraction=0.8):
    """Read the organized annual file without requiring the transport Parquet cache."""
    import pandas as pd
    from crop_pipeline.common import interval_bins
    from crop_pipeline.imagery import sampling_mask
    identifier = region["properties"]["fragment_id"]
    record = source["annual_tif_index"][identifier]
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise ValueError("Annual TIFF changed after source registration")
    grid = region["properties"]["sampling_grid"]
    intervals = interval_bins(int(source["year"]))
    plan = {"fragment_id": identifier, "fragment": region,
            "settings": {"grid": grid}, "download_intervals": intervals}
    selection = sampling_mask.load_selection(plan)
    rows, columns = np.divmod(selection["selected_ids"], grid["width"])
    descriptions = tuple(f"{interval['name']}__{band}" for interval in intervals for band in BANDS)
    frames = []
    with rasterio.open(path) as image:
        if (image.count != 168 or image.descriptions != descriptions
                or image.transform != Affine(*grid["transform"])
                or image.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                or image.shape != (grid["height"], grid["width"]) or image.nodata != -9999):
            raise ValueError("Annual TIFF bands or grid differ from source")
        for index, interval in enumerate(intervals):
            values = image.read(list(range(index*7+1, index*7+8)))[:, rows, columns].T
            frame = pd.DataFrame(values, columns=BANDS)
            frame["interval"], frame["row"], frame["column"] = interval["name"], rows, columns
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    pixel_tables.validate_frame(frame, plan)
    return pixel_tables.frame_to_window_rows(frame, plan, minimum_fraction, path)
