"""Sentinel-1 half-month backscatter by small direct Earth Engine requests (no exports, no assets).

VV and VH from IW GRD scenes in both orbit directions are averaged in linear power per
half-month and returned in dB, NaN where a half-month has no acquisition. Requests are
chunked by year and CRS and cached by the exact points they sample, so reruns read the
cache and a shared cache can never return another sample's values.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

COLLECTION = "COPERNICUS/S1_GRD"
NODATA = -9999.
POLARIZATIONS = ("VV", "VH")
BINS = 24
MAX_REQUEST_BYTES = 32 * 2 ** 20  # computePixels refuses responses above ~48 MB


def initialize(project):
    import ee
    ee.Initialize(project=project)
    return ee


def half_months(year):
    bins = []
    for month in range(1, 13):
        end = date(year + (month == 12), month % 12 + 1, 1)
        bins += [(date(year, month, 1), date(year, month, 16)), (date(year, month, 16), end)]
    return bins


def band_names():
    return [f"{p}_{i:02d}" for p in POLARIZATIONS for i in range(BINS)]


def image(ee, year, region):
    """48 bands VV_00..VV_23, VH_00..VH_23 in dB; NODATA where a half-month has no acquisition."""
    collection = (ee.ImageCollection(COLLECTION).filterBounds(region)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    bands = []
    for polarization in POLARIZATIONS:
        # A fully masked placeholder keeps the band when a half-month has no acquisition.
        empty = ee.ImageCollection([ee.Image.constant(0).toFloat().rename(polarization).updateMask(0)])
        for index, (start, end) in enumerate(half_months(year)):
            linear = (collection.filterDate(start.isoformat(), end.isoformat()).select(polarization)
                      .map(lambda scene: ee.Image(10).pow(scene.divide(10)).rename(polarization).toFloat())
                      .merge(empty).mean())
            bands.append(linear.log10().multiply(10).rename(f"{polarization}_{index:02d}"))
    return ee.Image.cat(bands).unmask(NODATA).toFloat()


def chunks(rows, chunk=400):
    """``(cache name, row indices, year, crs)`` per request, named by the exact points."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["year"]), row["source_crs"]].append(index)
    for (year, crs), members in sorted(groups.items()):
        for start in range(0, len(members), chunk):
            part = members[start:start + chunk]
            digest = hashlib.sha1(json.dumps([[rows[i]["x"], rows[i]["y"]] for i in part]).encode()).hexdigest()[:12]
            yield f"s1_{year}_{crs.replace(':', '')}_{start}_{digest}.json", part, year, crs


def _request(ee, rows, part, year, crs):
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([rows[i]["x"], rows[i]["y"]], crs), {"i": i})
                                   for i in part])
    sampled = image(ee, year, points.geometry().bounds()).reduceRegions(points, ee.Reducer.first(), scale=10, crs=crs)
    return sampled.getInfo()


def sample(rows, cache, ee=None, chunk=400, workers=8, progress=print):
    """Per pixel row (``x``, ``y``, ``source_crs``, ``year``): array [n, 24, 2] VV, VH dB, NaN where missing.

    ``ee`` is only used for requests missing from ``cache``; pass ``None`` to require a complete cache.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    plan = list(chunks(rows, chunk))
    missing = [item for item in plan if not (cache / item[0]).is_file()]
    if missing and ee is None:
        raise RuntimeError(f"Sentinel-1 cache lacks {len(missing)} of {len(plan)} requests and no Earth Engine client was given")

    def fetch(item):
        name, part, year, crs = item
        result = _request(ee, rows, part, year, crs)
        temporary = cache / f".{name}.tmp"
        temporary.write_text(json.dumps(result))
        temporary.replace(cache / name)
        return name

    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(pool.map(fetch, missing), 1):
                if progress and (done % 10 == 0 or done == len(missing)):
                    progress(f"Sentinel-1 requests {done}/{len(missing)}")
    values = np.full((len(rows), BINS, len(POLARIZATIONS)), np.nan)
    names = band_names()
    for name, part, _, _ in plan:
        expected = set(part)
        for feature in json.loads((cache / name).read_text())["features"]:
            props = feature["properties"]
            if props["i"] not in expected:
                raise ValueError(f"Cached Sentinel-1 request {name} holds another sample")
            for j, band in enumerate(names):
                value = props.get(band)
                if value is not None and value != NODATA:
                    values[props["i"], j % BINS, j // BINS] = value
    return values


def row_tiles(height, width, bands, bytes_per_value=4, limit=MAX_REQUEST_BYTES):
    rows = max(1, min(height, limit // max(1, width * bands * bytes_per_value)))
    return [(start, min(rows, height - start)) for start in range(0, height, rows)]


def grid(ee, year, pixel_grid, cache):
    """Cube [24, 2, rows, cols] on a pixel grid (``window``, ``transform``, ``crs``), cached as .npy."""
    path = Path(cache) / f"s1_grid_{year}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        if ee is None:
            raise RuntimeError(f"Sentinel-1 grid cache {path} is missing and no Earth Engine client was given")
        _, _, width, height = pixel_grid["window"]
        a, b, c, d, e, f = pixel_grid["transform"]
        region = ee.Geometry.Rectangle([c, f + e * height, c + a * width, f], pixel_grid["crs"], False)
        expression = image(ee, year, region)
        names = band_names()
        strips = []
        for start, rows in row_tiles(height, width, len(names)):
            request = {"expression": expression, "fileFormat": "NUMPY_NDARRAY",
                       "grid": {"dimensions": {"width": width, "height": rows},
                                "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                                    "shearY": d, "scaleY": e, "translateY": f + e * start},
                                "crsCode": pixel_grid["crs"]}}
            structured = ee.data.computePixels(request)
            strips.append(np.stack([structured[name] for name in names]).astype("float32"))
        stacked = np.concatenate(strips, axis=1)
        stacked[stacked == NODATA] = np.nan
        np.save(path, stacked.reshape(len(POLARIZATIONS), BINS, height, width).transpose(1, 0, 2, 3))
    return np.load(path)
