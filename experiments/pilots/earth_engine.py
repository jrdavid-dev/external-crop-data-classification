"""Small direct Earth Engine requests for the pilots (no export tasks, no cloud assets).

Sentinel-1: half-month mean VV and VH backscatter (IW GRD, both orbit directions,
averaged in linear power, returned in dB). ERA5-Land: monthly mean 2 m air
temperature. Requests are chunked and cached in the pilot run folder.
"""

from collections import defaultdict
import hashlib
from datetime import date
import json
from pathlib import Path

import numpy as np

S1 = "COPERNICUS/S1_GRD"
ERA5 = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
NODATA = -9999.
POLARIZATIONS = ("VV", "VH")


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


def s1_image(ee, year, region):
    """48 bands: VV_00..VV_23, VH_00..VH_23 in dB; NODATA where a half-month has no acquisition."""
    collection = (ee.ImageCollection(S1).filterBounds(region)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    bands = []
    for polarization in POLARIZATIONS:
        # A fully masked placeholder keeps the band when a half-month has no acquisition.
        empty = ee.ImageCollection([ee.Image.constant(0).toFloat().rename(polarization).updateMask(0)])
        for index, (start, end) in enumerate(half_months(year)):
            linear = (collection.filterDate(start.isoformat(), end.isoformat()).select(polarization)
                      .map(lambda image: ee.Image(10).pow(image.divide(10)).rename(polarization).toFloat()).merge(empty).mean())
            bands.append(linear.log10().multiply(10).rename(f"{polarization}_{index:02d}"))
    return ee.Image.cat(bands).unmask(NODATA).toFloat()


def sample_s1(ee, rows, cache, chunk=400, progress=print):
    """Per pixel row (with x, y, source_crs, year): array [n, 24, 2] in dB, NaN where missing."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["year"], row["source_crs"]].append(index)
    values = np.full((len(rows), 24, 2), np.nan)
    names = [f"{p}_{i:02d}" for p in POLARIZATIONS for i in range(24)]
    for (year, crs), members in sorted(groups.items()):
        for start in range(0, len(members), chunk):
            part = members[start:start + chunk]
            # Name by the exact points so a shared cache can never return another sample's values.
            digest = hashlib.sha1(json.dumps([[rows[i]["x"], rows[i]["y"]] for i in part]).encode()).hexdigest()[:12]
            path = cache / f"s1_{year}_{crs.replace(':', '')}_{start}_{digest}.json"
            if not path.is_file():
                points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([rows[i]["x"], rows[i]["y"]], crs), {"i": i})
                                               for i in part])
                sampled = s1_image(ee, year, points.geometry().bounds()).reduceRegions(
                    points, ee.Reducer.first(), scale=10, crs=crs)
                path.write_text(json.dumps(sampled.getInfo()))
            for feature in json.loads(path.read_text())["features"]:
                props = feature["properties"]
                for j, name in enumerate(names):
                    value = props.get(name)
                    if value is not None and value != NODATA:
                        values[props["i"], j % 24, j // 24] = value
            if progress:
                progress(f"S1 {year} {crs}: {min(start + chunk, len(members))}/{len(members)}")
    return values


MAX_REQUEST_BYTES = 32 * 2 ** 20  # computePixels refuses responses above ~48 MB


def row_tiles(height, width, bands, bytes_per_value=4, limit=None):
    """(first row, rows) strips whose responses stay under ``limit``; one strip when the grid is small."""
    limit = MAX_REQUEST_BYTES if limit is None else limit
    rows = max(1, min(height, limit // max(1, width * bands * bytes_per_value)))
    return [(start, min(rows, height - start)) for start in range(0, height, rows)]


def s1_grid(ee, year, grid, cache):
    """Sentinel-1 cube [24, 2, rows, cols] on a pixel grid, via computePixels in row strips when large."""
    path = Path(cache) / f"s1_grid_{year}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        col, row, width, height = grid["window"]
        a, b, c, d, e, f = grid["transform"]
        region = ee.Geometry.Rectangle([c, f + e * height, c + a * width, f], grid["crs"], False)
        expression = s1_image(ee, year, region)
        names = [f"{p}_{i:02d}" for p in POLARIZATIONS for i in range(24)]
        strips = []
        for start, rows in row_tiles(height, width, len(names)):
            request = {"expression": expression, "fileFormat": "NUMPY_NDARRAY",
                       "grid": {"dimensions": {"width": width, "height": rows},
                                "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                                    "shearY": d, "scaleY": e, "translateY": f + e * start},
                                "crsCode": grid["crs"]}}
            structured = ee.data.computePixels(request)
            strips.append(np.stack([structured[name] for name in names]).astype("float32"))
        stacked = np.concatenate(strips, axis=1)
        stacked[stacked == NODATA] = np.nan
        np.save(path, stacked.reshape(2, 24, height, width).transpose(1, 0, 2, 3))
    return np.load(path)


def s1_features_cube(values):
    """[n, 24, 2] dB -> [n, 24, 3] VV, VH and VH/VV (dB difference)."""
    return np.concatenate([values, values[..., 1:2] - values[..., 0:1]], axis=-1)


def era5_monthly(ee, points, cache, chunk=500):
    """points: list of {"key", "lon", "lat", "year"} -> {key: [12 monthly mean degC]}."""
    path = Path(cache) / "era5_monthly.json"
    stored = json.loads(path.read_text()) if path.is_file() else {}
    todo = [p for p in points if p["key"] not in stored]
    by_year = defaultdict(list)
    for point in todo:
        by_year[point["year"]].append(point)
    for year, members in by_year.items():
        image = (ee.ImageCollection(ERA5).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                 .select("temperature_2m").toBands())
        for start in range(0, len(members), chunk):
            part = members[start:start + chunk]
            features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]), {"key": p["key"]})
                                             for p in part])
            result = image.reduceRegions(features, ee.Reducer.first(), scale=11132).getInfo()
            for feature in result["features"]:
                props = feature["properties"]
                months = [props.get(name) for name in sorted(k for k in props if k.endswith("temperature_2m"))]
                stored[props["key"]] = [None if m is None else m - 273.15 for m in months]
        Path(cache).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored))
    return stored


WORLDCOVER = "ESA/WorldCover/v200"
WORLDCOVER_CROPLAND = 40


def worldcover_grid(ee, grid, cache):
    """ESA WorldCover 2021 class codes on a pixel grid, nearest-neighbour, via computePixels."""
    path = Path(cache) / "worldcover_grid.npy"
    if not path.is_file():
        col, row, width, height = grid["window"]
        a, b, c, d, e, f = grid["transform"]
        request = {"expression": ee.ImageCollection(WORLDCOVER).first().select("Map"),
                   "fileFormat": "NUMPY_NDARRAY",
                   "grid": {"dimensions": {"width": width, "height": height},
                            "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                                "shearY": d, "scaleY": e, "translateY": f},
                            "crsCode": grid["crs"]}}
        structured = ee.data.computePixels(request)
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(structured["Map"]).astype("uint8"))
    return np.load(path)
