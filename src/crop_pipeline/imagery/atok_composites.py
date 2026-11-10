"""Export Atok half-month Sentinel-2 composites for a boundary, in the THS-SENTINEL2 format.

The recipe reproduces the existing shared files (checked on 2024-01 H1, 2024-07 H1,
2024-08 H2 and 2023-10 H2: 99.2-99.8% of shared pixels identical):

- Sentinel-2 L2A surface reflectance (COPERNICUS/S2_SR_HARMONIZED), digital numbers
- Cloud Score+ ``cs_cdf`` >= 0.6 per acquisition
- bands B3, B4, B5, B8, B11 resampled bilinearly to 10 m (B5 and B11 are 20 m)
- median of the acquisitions in each half-month: H1 = days 1-15, H2 = day 16 to month end
- float32 GeoTIFF on the 10 m UTM 51N lattice of the existing files, -9999 where masked

Unlike the old files, which were clipped to a KML outline displaced by about 1.2 km, the
region is the official boundary (here: Atok's eight barangays) plus a buffer.
Files are named ``Sentinel2_CSPlus_YYYY_MM_Hh.tif``; ``arrange_by_year`` copies them from
the flat Drive export folder into the ``YYYY/`` layout the notebooks read.
"""

import hashlib
import json
import math
import re
import shutil
from datetime import date
from pathlib import Path

from crop_pipeline.common import save_json

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
QA_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.6
BANDS = ["B3", "B4", "B5", "B8", "B11"]
RESAMPLING = "bilinear"
CRS = "EPSG:32651"
SCALE = 10
NODATA = -9999
BUFFER_M = 100
MAX_OUTSTANDING = 100  # leave room in the project's task queue
FINISHED = {"SUCCEEDED", "COMPLETED"}
FAILED = {"FAILED", "CANCELLED", "CANCELLING"}
FILE_PATTERN = re.compile(r"Sentinel2_CSPlus_(\d{4})_(\d{2})_H([12])\.tif$")

SETTINGS = {"collection": COLLECTION, "cloud_score": CLOUD_SCORE, "qa_band": QA_BAND,
            "clear_threshold": CLEAR_THRESHOLD, "bands": BANDS, "resampling": RESAMPLING,
            "reducer": "median", "crs": CRS, "scale": SCALE, "nodata": NODATA,
            "units": "Sentinel-2 L2A digital numbers (reflectance x 10000)"}


def file_stem(year, month, half):
    return f"Sentinel2_CSPlus_{year}_{month:02d}_H{half}"


def half_months(first, last, today=None):
    """(year, month, half, start, end) for each half-month from ``first`` to ``last`` (dates).

    ``end`` is exclusive. Half-months that have not finished by ``today`` are left out,
    so an export never freezes a composite before all its acquisitions exist.
    """
    today = today or date.today()
    out = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        for half, start, end in ((1, date(year, month, 1), date(year, month, 16)),
                                 (2, date(year, month, 16), next_month)):
            if first <= start <= last and end <= today:
                out.append((year, month, half, start, end))
        year, month = next_month.year, next_month.month
    return out


def load_boundary(path, buffer_m=BUFFER_M):
    """Union of the boundary polygons in UTM 51N, buffered by ``buffer_m`` metres."""
    import geopandas as gpd

    frame = gpd.read_file(path).to_crs(CRS)
    if frame.empty:
        raise ValueError(f"{path} has no polygons.")
    return frame.union_all().buffer(buffer_m)


def export_grid(region, scale=SCALE):
    """Pixel grid covering ``region`` on the lattice of the existing files (multiples of 10 m)."""
    minx, miny, maxx, maxy = region.bounds
    x0, y0 = math.floor(minx / scale) * scale, math.ceil(maxy / scale) * scale
    x1, y1 = math.ceil(maxx / scale) * scale, math.floor(miny / scale) * scale
    return {"crs": CRS, "transform": [scale, 0, x0, 0, -scale, y0],
            "width": int(round((x1 - x0) / scale)), "height": int(round((y0 - y1) / scale)),
            "bounds": [x0, y1, x1, y0]}


def plan_exports(boundary_path, first, last, drive_folder, today=None, buffer_m=BUFFER_M):
    """Everything needed to submit or check the exports; no Earth Engine calls."""
    region = load_boundary(boundary_path, buffer_m)
    grid = export_grid(region)
    boundary_key = hashlib.sha256(Path(boundary_path).read_bytes()).hexdigest()[:16]
    key = hashlib.sha256(json.dumps({"settings": SETTINGS, "grid": grid, "boundary": boundary_key,
                                     "buffer_m": buffer_m}, sort_keys=True).encode()).hexdigest()[:16]
    return {"region": region, "grid": grid, "drive_folder": drive_folder, "settings_key": key,
            "items": [{"stem": file_stem(y, m, h), "start": s.isoformat(), "end": e.isoformat()}
                      for y, m, h, s, e in half_months(first, last, today)]}


# ----------------------------------------------------------------------------- Earth Engine


def composite_image(ee, region, start, end):
    """Median Cloud Score+ composite of one half-month, masked outside ``region``."""
    scores = ee.ImageCollection(CLOUD_SCORE)

    def clear(image):
        return image.select(BANDS).resample(RESAMPLING).updateMask(image.select(QA_BAND).gte(CLEAR_THRESHOLD))

    collection = (ee.ImageCollection(COLLECTION).filterBounds(region).filterDate(start, end)
                  .linkCollection(scores, [QA_BAND]).map(clear))
    return collection.median().select(BANDS).toFloat().clip(region)


def ee_region(ee, region):
    """The buffered boundary as an Earth Engine geometry in UTM 51N."""
    from shapely.geometry import mapping

    return ee.Geometry(mapping(region), CRS, False)


def start_export(ee, plan, item):
    region = ee_region(ee, plan["region"])
    grid = plan["grid"]
    rectangle = ee.Geometry.Rectangle(grid["bounds"], ee.Projection(CRS), False)
    task = ee.batch.Export.image.toDrive(
        image=composite_image(ee, region, item["start"], item["end"]),
        description=f"atok_{item['stem']}"[:100],
        folder=plan["drive_folder"],
        fileNamePrefix=item["stem"],
        region=rectangle,
        crs=CRS,
        crsTransform=grid["transform"],
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    task.start()
    return task.id


def load_ledger(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def covered(ledger, settings_key):
    """Stems already exported, or still running, with these settings."""
    return {entry["stem"] for entry in ledger
            if entry["settings_key"] == settings_key and entry.get("state") not in FAILED}


def outstanding(ledger):
    return sum(entry.get("state") not in FINISHED | FAILED for entry in ledger)


def submit(ee, plan, ledger_path, existing_root=None, limit=MAX_OUTSTANDING):
    """Start exports for half-months not yet exported, running or present locally.

    Every task is recorded in the ledger as soon as it starts, so a rerun never submits
    the same half-month twice. Returns the stems submitted in this call.
    """
    ledger = load_ledger(ledger_path)
    done = covered(ledger, plan["settings_key"])
    if existing_root:
        done |= {stem for stem in present_files(existing_root, plan["grid"])}
    submitted = []
    for item in plan["items"]:
        if item["stem"] in done:
            continue
        if outstanding(ledger) >= limit:
            break
        task_id = start_export(ee, plan, item)
        ledger.append({"stem": item["stem"], "task_id": task_id, "state": "READY",
                       "settings_key": plan["settings_key"], "drive_folder": plan["drive_folder"],
                       "start": item["start"], "end": item["end"]})
        save_json(ledger_path, ledger)
        submitted.append(item["stem"])
    return submitted


def refresh(ee, ledger_path):
    """Update task states in the ledger; returns counts by state."""
    ledger = load_ledger(ledger_path)
    for entry in ledger:
        if entry.get("state") in FINISHED | FAILED:
            continue
        status = ee.data.getTaskStatus(entry["task_id"])[0]
        entry["state"] = status.get("state", "UNKNOWN")
        if status.get("error_message"):
            entry["error"] = status["error_message"]
    save_json(ledger_path, ledger)
    counts = {}
    for entry in ledger:
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    return counts


# ----------------------------------------------------------------------------- local files


def check_file(path, grid):
    """Problems with one downloaded composite against the planned grid (empty if fine)."""
    import rasterio

    problems = []
    with rasterio.open(path) as raster:
        if raster.crs is None or raster.crs.to_epsg() != int(CRS.split(":")[1]):
            problems.append(f"CRS {raster.crs}")
        if [round(v, 6) for v in list(raster.transform)[:6]] != [float(v) for v in grid["transform"]]:
            problems.append(f"transform {tuple(raster.transform)[:6]}")
        if (raster.width, raster.height) != (grid["width"], grid["height"]):
            problems.append(f"size {raster.width}x{raster.height}")
        if raster.count != len(BANDS):
            problems.append(f"{raster.count} bands")
        if raster.dtypes[0] != "float32":
            problems.append(f"dtype {raster.dtypes[0]}")
    return problems


def present_files(root, grid):
    """Stems already in ``root/YYYY/`` that match the planned grid."""
    found = set()
    for path in Path(root).glob("*/Sentinel2_CSPlus_*.tif"):
        if FILE_PATTERN.search(path.name) and not check_file(path, grid):
            found.add(path.stem)
    return found


def arrange_by_year(source, destination):
    """Copy ``Sentinel2_CSPlus_YYYY_MM_Hh.tif`` from a flat folder into ``destination/YYYY/``.

    Existing files of the same size are skipped; nothing is deleted. Returns copied paths.
    """
    copied = []
    for path in sorted(Path(source).glob("Sentinel2_CSPlus_*.tif")):
        match = FILE_PATTERN.search(path.name)
        if not match:
            continue
        target = Path(destination) / match.group(1) / path.name
        if target.is_file() and target.stat().st_size == path.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied
