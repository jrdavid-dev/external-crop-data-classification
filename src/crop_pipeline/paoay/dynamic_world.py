"""Prepare Dynamic World probability inputs; class/eligibility rules belong later.

Local reuse is network-free. All publications are separate from existing Sentinel-2
inputs and immutable once complete. No export tasks or cloud assets are created.
"""

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio
import requests

from crop_pipeline.paoay import imagery, references

VERSION = 1
COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
PROBABILITIES = ["water", "trees", "grass", "flooded_vegetation", "crops",
                 "shrub_and_scrub", "built", "bare", "snow_and_ice"]
BANDS = PROBABILITIES + ["observation_count"]
NODATA = -9999
SOURCE_URL = "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1"
ATTRIBUTION = ("This dataset is produced for the Dynamic World Project by Google in partnership "
               "with National Geographic Society and the World Resources Institute.")


def prepare_request(imagery_handoff, output_root):
    """Inherit verified dates/extent; never select a different period silently."""
    bundle = imagery.load_imagery_bundle(imagery_handoff)
    source = bundle["manifest"]
    reference = references.load_reference_bundle(source["reference_handoff"])
    period, grid = source["period"], source["grid"]
    if max(grid["width"], grid["height"]) >= 10000 or grid["width"] * grid["height"] * 40 >= 30 * 1024**2:
        raise ValueError("Dynamic World ROI exceeds direct-download limits; explicit tiling is required")
    settings = {"collection": COLLECTION, "period": period, "grid": grid, "bands": BANDS,
                "geometry": reference["roi"]["features"][0]["geometry"], "nodata": NODATA,
                "mask": "provider_mask_and_all_nine_probability_bands_valid",
                "resampling": "nearest_before_daily_mosaic", "compositing": "same_date_mosaics_then_half_month_mean_probabilities",
                "observation_count": "number_of_valid_daily_mosaics_not_granules",
                "label_band": "not_aggregated", "probability_threshold": None}
    identity = {"version": VERSION, "imagery_sha256": references.checksum(imagery_handoff),
                "settings": settings, "implementation_sha256": references.checksum(__file__)}
    request_id = imagery._digest(identity)[:20]
    return {"request_id": request_id, "identity": identity, "settings": settings,
            "period": period, "fragment": reference["roi"]["features"][0],
            "imagery_handoff": str(Path(imagery_handoff).resolve()),
            "folder": Path(output_root) / "requests" / request_id}


def _verify_dependency(request):
    if (imagery._digest(request["identity"])[:20] != request["request_id"]
            or request["settings"] != request["identity"]["settings"]
            or request["period"] != request["settings"]["period"]
            or request["fragment"]["geometry"] != request["settings"]["geometry"]):
        raise ValueError("Dynamic World request changed after planning; prepare it again")
    if references.checksum(request["imagery_handoff"]) != request["identity"]["imagery_sha256"]:
        raise ValueError("Sentinel-2 handoff changed; prepare the request again")
    return imagery.load_imagery_bundle(request["imagery_handoff"])


def _paths(request, interval):
    folder = request["folder"] / "rasters"
    return folder / (interval["name"] + ".tif"), folder / (interval["name"] + ".json")


def inspect_raster(path, request, interval, available_dates=None):
    """Check full probabilities, daily counts and exact alignment without edits."""
    grid = request["settings"]["grid"]
    with rasterio.open(path) as raster:
        if ((raster.width, raster.height) != (grid["width"], grid["height"])
                or raster.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                or not raster.transform.almost_equals(Affine(*grid["transform"]))
                or list(raster.descriptions) != BANDS or raster.nodata != NODATA
                or any(dtype != "float32" for dtype in raster.dtypes)):
            raise ValueError("Dynamic World raster grid, bands or data format differs from its plan")
        data = raster.read()
    counts, probabilities = data[-1], data[:-1]
    maximum = (date.fromisoformat(interval["end"]) - date.fromisoformat(interval["start"])).days
    if available_dates is not None:
        maximum = min(maximum, len(set(available_dates)))
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts)) & (counts <= maximum)):
        raise ValueError("Dynamic World observation counts must count valid dates")
    support = counts > 0
    if np.any(probabilities[:, ~support] != NODATA):
        raise ValueError("Unobserved Dynamic World pixels must remain nodata")
    valid = probabilities[:, support]
    if (not np.all(np.isfinite(valid) & (valid >= 0) & (valid <= 1))
            or not np.allclose(valid.sum(axis=0), 1, atol=0.0005, rtol=0)):
        raise ValueError("Dynamic World probabilities must be finite, in [0, 1] and sum to one")
    roi = imagery._roi_mask(request)
    observed = roi & support
    return {"roi_pixels": int(roi.sum()), "observed_roi_pixels": int(observed.sum()),
            "observed_roi_fraction": float(observed.sum() / roi.sum()),
            "empty_after_masking": not bool(observed.any())}


def cached_interval(request, interval):
    path, receipt = _paths(request, interval)
    if not path.exists() and not receipt.exists():
        return None
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f"Incomplete Dynamic World pair for {interval['name']}; existing files preserved")
    record = json.loads(receipt.read_text())
    if (record.get("request_id") != request["request_id"] or record.get("interval") != interval
            or record.get("sha256") != references.checksum(path)):
        raise ValueError("Cached Dynamic World input changed; existing files preserved")
    if record.get("quality") != inspect_raster(path, request, interval, record["available_dates"]):
        raise ValueError("Dynamic World coverage receipt changed; existing files preserved")
    return record


def cache_status(request):
    return [{"interval": item["name"], "start": item["start"], "end_exclusive": item["end"],
             "status": "verified local" if cached_interval(request, item) else "missing"}
            for item in request["period"]["intervals"]]


class DynamicWorldRetriever:
    def __init__(self, earth_engine, request):
        self.ee, self.request = earth_engine, request

    def list_acquisitions(self):
        ee, settings = self.ee, self.request["settings"]
        collection = (ee.ImageCollection(COLLECTION)
                      .filterBounds(ee.Geometry(settings["geometry"], proj="EPSG:4326", geodesic=False))
                      .filterDate(settings["period"]["start"], settings["period"]["end"]))
        def describe(image):
            image = ee.Image(image)
            return ee.Feature(None, {"image_index": image.get("system:index"),
                "date": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
                "dynamicworld_algorithm_version": image.get("dynamicworld_algorithm_version"),
                "qa_algorithm_version": image.get("qa_algorithm_version")})
        payload = ee.FeatureCollection(collection.toList(collection.size()).map(describe)).getInfo()
        return sorted([{**item["properties"], "image_id": COLLECTION + "/" + item["properties"]["image_index"]}
                       for item in payload["features"]], key=lambda row: (row["date"], row["image_index"]))

    def make_composite(self, granules):
        ee, grid = self.ee, self.request["settings"]["grid"]
        daily = []
        for day in sorted({row["date"] for row in granules}):
            images = []
            for row in sorted((row for row in granules if row["date"] == day), key=lambda row: row["image_index"]):
                image = ee.Image(row["image_id"]).select(PROBABILITIES)
                # Keep provider masking, share all-band support and align before mosaics.
                image = image.updateMask(image.mask().reduce(ee.Reducer.min()))
                images.append(image.reproject(crs=grid["crs"], crsTransform=grid["transform"]))
            daily.append(ee.ImageCollection.fromImages(images).mosaic())
        if not daily:
            return ee.Image.constant([NODATA] * 9 + [0]).rename(BANDS).toFloat()
        collection = ee.ImageCollection.fromImages(daily)
        probabilities = collection.mean().select(PROBABILITIES).unmask(NODATA, sameFootprint=False)
        counts = collection.select(PROBABILITIES[0]).count().rename("observation_count").unmask(0, sameFootprint=False)
        return probabilities.addBands(counts).toFloat()

    def fetch_composite(self, interval, granules, destination):
        grid = self.request["settings"]["grid"]
        url = self.make_composite(granules).getDownloadURL({"name": interval["name"], "format": "GEO_TIFF",
            "filePerBand": False, "crs": grid["crs"], "crs_transform": grid["transform"],
            "dimensions": [grid["width"], grid["height"]]})
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with Path(destination).open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        with rasterio.open(destination, "r+") as raster:
            raster.nodata = NODATA
            if raster.count != len(BANDS):
                raise ValueError("Unexpected Dynamic World band count")
            raster.descriptions = tuple(BANDS)


def _inventory(request, retriever):
    path = request["folder"] / "acquisitions.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if (payload.get("identity") != request["identity"]
                or payload.get("acquisitions_sha256") != imagery._digest(payload["acquisitions"])):
            raise ValueError("Dynamic World acquisition inventory changed; existing files preserved")
        return payload["acquisitions"]
    acquisitions = retriever.list_acquisitions()
    indices = [row["image_index"] for row in acquisitions]
    period = request["period"]
    if len(indices) != len(set(indices)) or any(not period["start"] <= row["date"] < period["end"]
            or row["image_id"] != COLLECTION + "/" + row["image_index"] for row in acquisitions):
        raise ValueError("Dynamic World acquisition metadata is inconsistent with the request")
    path.parent.mkdir(parents=True, exist_ok=True)
    imagery._publish_json(path, {"identity": request["identity"], "acquisitions": acquisitions,
                              "acquisitions_sha256": imagery._digest(acquisitions),
                              "inventoried_at": datetime.now(timezone.utc).isoformat()})
    return acquisitions


def collect(request, mode="reuse", earth_engine=None, iterator=None):
    """Get eight separate probability inputs, reusing verified local pairs."""
    _verify_dependency(request)
    if mode not in {"reuse", "download_missing"}:
        raise ValueError("Dynamic World mode must be reuse or download_missing")
    missing = sum(row["status"] == "missing" for row in cache_status(request))
    if missing and mode == "reuse":
        raise FileNotFoundError(f"{missing} of 8 Dynamic World composites are missing. Select download_missing to obtain them")
    if missing and earth_engine is None:
        raise ValueError("download_missing requires an initialized Earth Engine connection")
    folder = request["folder"]
    handoff = folder / "dynamic_world_handoff.json"
    if handoff.exists():
        load_dynamic_world_bundle(handoff)
        return handoff
    retriever = DynamicWorldRetriever(earth_engine, request)
    acquisitions = _inventory(request, retriever) if missing else None
    items, records = request["period"]["intervals"], []
    for interval in (iterator(items) if iterator else items):
        record = cached_interval(request, interval)
        if record is None:
            granules = [row for row in acquisitions if interval["start"] <= row["date"] < interval["end"]]
            dates = sorted({row["date"] for row in granules})
            path, receipt = _paths(request, interval)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".download-", dir=path.parent) as temporary:
                pending = Path(temporary) / path.name
                retriever.fetch_composite(interval, granules, pending)
                quality = inspect_raster(pending, request, interval, dates)
                record = {"request_id": request["request_id"], "interval": interval,
                          "sha256": references.checksum(pending), "granules": granules,
                          "available_dates": dates, "quality": quality}
                os.link(pending, path)
                imagery._publish_json(receipt, record)
        records.append(record)
    manifest = {"format_version": VERSION, "kind": "paoay_dynamic_world", "status": "prepared",
                "request_id": request["request_id"], "identity": request["identity"],
                "period": request["period"], "grid": request["settings"]["grid"], "bands": BANDS,
                "imagery_handoff": request["imagery_handoff"], "summary": _summarize(request, records),
                "interpretation": "Predicted broad-cover inputs; no class decisions or independent ground truth",
                "license": "CC-BY-4.0", "attribution": ATTRIBUTION, "source_url": SOURCE_URL,
                "copernicus_notice": "Contains modified Copernicus Sentinel data [2015-present]",
                "files": {p.relative_to(folder).as_posix(): {"sha256": references.checksum(p)}
                          for item in items for p in _paths(request, item)}}
    if (folder / "acquisitions.json").exists():
        manifest["files"]["acquisitions.json"] = {"sha256": references.checksum(folder / "acquisitions.json")}
    imagery._publish_json(handoff, manifest)
    load_dynamic_world_bundle(handoff)
    return handoff


def _summarize(request, records):
    roi = imagery._roi_mask(request)
    any_observed, all_observed = np.zeros_like(roi), roi.copy()
    for interval in request["period"]["intervals"]:
        with rasterio.open(_paths(request, interval)[0]) as raster:
            observed = raster.read(10) > 0
        any_observed |= observed & roi
        all_observed &= observed
    return {"roi_pixels": int(roi.sum()), "any_bin_observed_pixels": int(any_observed.sum()),
            "all_eight_bins_observed_pixels": int(all_observed.sum()),
            "empty_bins": sum(row["quality"]["empty_after_masking"] for row in records),
            "note": "Coverage of predictions only; observations/masks differ from Sentinel-2 SR"}


def load_dynamic_world_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    identity = manifest.get("identity", {})
    settings = identity.get("settings", {})
    period = manifest.get("period", {})
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != "paoay_dynamic_world"
            or manifest.get("status") != "prepared" or manifest.get("bands") != BANDS
            or imagery._digest(identity)[:20] != manifest.get("request_id")
            or period != imagery.four_month_period(period.get("start"))
            or period != settings.get("period") or manifest.get("grid") != settings.get("grid")
            or settings.get("collection") != COLLECTION):
        raise ValueError("Unsupported or inconsistent Dynamic World handoff")
    required = {f"rasters/{item['name']}.{suffix}" for item in period["intervals"] for suffix in ("tif", "json")}
    if set(manifest.get("files", {})) not in (required, required | {"acquisitions.json"}):
        raise ValueError("Dynamic World handoff needs exactly eight raster/receipt pairs")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved Dynamic World file changed: {name}; existing files preserved")
    dependency = Path(manifest["imagery_handoff"])
    if references.checksum(dependency) != identity.get("imagery_sha256"):
        raise ValueError("Dynamic World Sentinel-2 dependency changed")
    source = imagery.load_imagery_bundle(dependency)["manifest"]
    if source["period"] != period or source["grid"] != settings["grid"]:
        raise ValueError("Dynamic World dates/grid differ from Sentinel-2")
    return {"manifest": manifest, "folder": path.parent,
            "rasters": [path.parent / f"rasters/{item['name']}.tif" for item in period["intervals"]]}
