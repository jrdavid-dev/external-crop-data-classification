"""Resumable Paoay imagery on the external model's existing acquisition recipe.

No network on import or in reuse mode. Raw external/legacy imagery is never edited.
"""

from datetime import date
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from crop_pipeline.common import interval_bins
from crop_pipeline.imagery.retrieval import SatelliteRetriever
from crop_pipeline.paoay import references

VERSION = 1
BANDS = ["B3", "B4", "B5", "B8", "B11", "B2", "clear_count"]


def four_month_period(start, today=None):
    """Require four complete calendar months, including across December."""
    if not isinstance(start, str):
        raise ValueError("Choose a start month as YYYY-MM-01")
    try:
        first = date.fromisoformat(start)
    except ValueError as error:
        raise ValueError("Choose a start month as YYYY-MM-01") from error
    if first.day != 1 or start != first.isoformat():
        raise ValueError("The period must start on the first day of a calendar month")
    number = first.year * 12 + first.month - 1 + 4
    end = date(number // 12, number % 12 + 1, 1)
    if first < date(2017, 3, 1) or end > (today or date.today()):
        raise ValueError("Choose a completed historical period within Sentinel-2 SR availability")
    bins = [item for year in range(first.year, end.year + 1) for item in interval_bins(year)
            if first.isoformat() <= item["start"] < end.isoformat()]
    assert len(bins) == 8
    return {"start": first.isoformat(), "end": end.isoformat(), "end_exclusive": True, "intervals": bins}


class PaoayRetriever(SatelliteRetriever):
    """Use shared masking/compositing with an explicit four-month date range."""

    def __init__(self, ee, period, folder):
        super().__init__(ee, int(period["start"][:4]), folder / "rasters", folder / "metadata")
        self.start_date, self.end_date = period["start"], period["end"]
        self.buffer_metres = 0  # Map the study boundary; no training parcel sampling buffer.


def prepare_request(reference_handoff, start, output_root, today=None):
    """Plan locally, without creating files or connecting to Earth Engine."""
    bundle = references.load_reference_bundle(reference_handoff)
    period = four_month_period(start, today)
    fragment = json.loads(json.dumps(bundle["roi"]["features"][0]))
    fragment["properties"] = {"fragment_id": "paoay_roi", "source_crs": bundle["manifest"]["metric_crs"]}
    retriever = PaoayRetriever(None, period, Path(output_root))
    settings = retriever.request_settings(fragment)
    grid = settings["grid"]
    if max(grid["width"], grid["height"]) >= 10000 or grid["width"] * grid["height"] * 28 >= 30 * 1024**2:
        raise ValueError("ROI exceeds the bounded direct-download limit; explicit tiling is required")
    identity = {"version": VERSION, "reference_sha256": references.checksum(reference_handoff),
                "period": period, "settings": settings,
                "implementation_sha256": references.checksum(__file__),
                "retrieval_sha256": references.checksum(inspect.getfile(SatelliteRetriever))}
    request_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"request_id": request_id, "identity": identity, "period": period, "settings": settings,
            "fragment": fragment, "reference_handoff": str(Path(reference_handoff).resolve()),
            "folder": Path(output_root) / "requests" / request_id}


def _publish_json(path, record):
    """Publish a complete local record atomically without replacing another file."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.link(staged, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _roi_mask(request):
    from pyproj import Transformer
    from shapely.geometry import shape, mapping
    from shapely import transform
    grid = request["settings"]["grid"]
    project = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True).transform
    geometry = mapping(transform(shape(request["fragment"]["geometry"]), project, interleaved=False))
    mask = geometry_mask([geometry], out_shape=(grid["height"], grid["width"]),
                         transform=Affine(*grid["transform"]), invert=True)
    if not mask.any():
        raise ValueError("No 10 m pixel centres lie within the ROI")
    return mask


def inspect_raster(path, request, interval=None):
    """Validate full payload and summarize clear pixels inside the study boundary."""
    retriever = PaoayRetriever(None, request["period"], request["folder"])
    retriever.check_raster(path, request["settings"]["grid"])
    with rasterio.open(path) as raster:
        if list(raster.descriptions) != BANDS or raster.nodata != -9999:
            raise ValueError("Raster band labels or nodata differ from the imagery recipe")
        data = raster.read()  # Bounded ROI: verify all values, not only a sample window.
    counts = data[-1]
    duration = max((date.fromisoformat(item["end"]) - date.fromisoformat(item["start"])).days
                   for item in ([interval] if interval else request["period"]["intervals"]))
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts)) & (counts <= duration)):
        raise ValueError("Clear counts must be finite daily observation counts")
    spectral = data[:-1]
    support = np.all(np.isfinite(spectral) & (spectral != -9999), axis=0)
    missing = np.all(spectral == -9999, axis=0)
    if np.any((counts > 0) != support) or np.any((counts == 0) & ~missing):
        raise ValueError("Spectral support and clear observation counts disagree")
    roi = _roi_mask(request)
    valid = support & (counts > 0) & roi
    return {"roi_pixels": int(roi.sum()), "clear_roi_pixels": int(valid.sum()),
            "clear_roi_fraction": float(valid.sum() / roi.sum()),
            "empty_after_screening": not bool(valid.any())}


def _paths(request, interval):
    folder = request["folder"] / "rasters"
    return folder / (interval["name"] + ".tif"), folder / (interval["name"] + ".json")


def cached_interval(request, interval):
    """Never repair or replace an existing unverified raster/receipt silently."""
    path, receipt = _paths(request, interval)
    if not path.exists() and not receipt.exists():
        return None
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f"Incomplete cached pair for {interval['name']}; preserve it and inspect before resuming")
    record = json.loads(receipt.read_text())
    if (record.get("request_id") != request["request_id"] or record.get("interval") != interval
            or record.get("sha256") != references.checksum(path)):
        raise ValueError(f"Cached imagery changed for {interval['name']}; existing files preserved")
    summary = inspect_raster(path, request, interval)
    if record.get("quality") != summary:
        raise ValueError("Cached imagery quality receipt changed; existing files preserved")
    return record


def cache_status(request):
    """Pure local inspection, suitable before deciding to connect."""
    return [{"interval": item["name"], "start": item["start"], "end_exclusive": item["end"],
             "status": "verified local" if cached_interval(request, item) else "missing"}
            for item in request["period"]["intervals"]]


def _inventory(request, ee):
    retriever = PaoayRetriever(ee, request["period"], request["folder"])
    path = request["folder"] / "acquisitions.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("identity") != request["identity"]:
            raise ValueError("Acquisition inventory differs from this request")
        if result.get("acquisitions_sha256") != _digest(result["acquisitions"]):
            raise ValueError("Cached acquisition inventory changed; existing files preserved")
        return retriever, result["acquisitions"]
    acquisitions = retriever.list_acquisitions(request["fragment"])
    matches = retriever.find_cloud_probability_matches(request["fragment"])
    for row in acquisitions:
        row["cloud_probability_id"] = (retriever.cloud_probability_collection + "/" + row["image_index"]
                                       if row["image_index"] in matches else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_json(path, {"identity": request["identity"], "acquisitions": acquisitions,
                         "acquisitions_sha256": _digest(acquisitions)})
    return retriever, acquisitions


def collect(request, mode="reuse", earth_engine=None, iterator=None):
    """Resume eight local pairs; download missing only when explicitly selected.

    Direct HTTPS downloads only: no exports, cloud assets, task cancellation or deletion.
    ``iterator`` allows the notebook's single task bar without one task per interval.
    """
    if (_digest(request["identity"])[:20] != request["request_id"]
            or request["settings"] != request["identity"]["settings"]
            or request["period"] != request["identity"]["period"]):
        raise ValueError("Imagery request changed after planning; prepare it again")
    if mode not in {"reuse", "download_missing"}:
        raise ValueError("Imagery mode must be reuse or download_missing")
    if references.checksum(request["reference_handoff"]) != request["identity"]["reference_sha256"]:
        raise ValueError("Reference handoff changed; run the plan again")
    references.load_reference_bundle(request["reference_handoff"])
    status = cache_status(request)
    missing = sum(row["status"] == "missing" for row in status)
    if missing and mode == "reuse":
        raise FileNotFoundError(f"{missing} of 8 matching composites are missing. Select download_missing to obtain them; existing imagery is preserved")
    if missing and earth_engine is None:
        raise ValueError("download_missing needs an initialized Earth Engine connection")
    folder = request["folder"]
    handoff = folder / "imagery_handoff.json"
    if handoff.exists():
        load_imagery_bundle(handoff)  # Corrupt completed bundles are never overwritten.
        return handoff
    retriever, acquisitions = _inventory(request, earth_engine) if missing else (None, None)
    items = request["period"]["intervals"]
    records = []
    for interval in (iterator(items) if iterator else items):
        record = cached_interval(request, interval)
        if record is None:
            granules = [row for row in acquisitions if interval["start"] <= row["date"] < interval["end"]]
            path, receipt = _paths(request, interval)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".download-", dir=path.parent) as temporary:
                pending = Path(temporary) / path.name
                retriever.fetch_composite({"settings": request["settings"], "fragment": request["fragment"]}, interval, granules, pending)
                retriever.finish_raster(pending)
                quality = inspect_raster(pending, request, interval)
                record = {"request_id": request["request_id"], "interval": interval,
                          "sha256": references.checksum(pending), "quality": quality,
                          "granules": granules, "matched_dates": sorted({row["date"] for row in granules if row.get("cloud_probability_id")})}
                # No replacement of existing files, even if another process completed first.
                os.link(pending, path)  # Atomic publication, fails if destination exists.
                _publish_json(receipt, record)
        records.append(record)
    summary = summarize(request, records)
    manifest = {"format_version": VERSION, "kind": "paoay_sentinel2", "status": "prepared",
                "request_id": request["request_id"], "identity": request["identity"],
                "period": request["period"], "grid": request["settings"]["grid"], "bands": BANDS,
                "spectral_scale_to_apply_later": 0.0001, "clear_counts_scaled": False,
                "reference_handoff": request["reference_handoff"], "summary": summary,
                "files": {str(p.relative_to(folder)): {"sha256": references.checksum(p)}
                          for interval in items for p in _paths(request, interval)}}
    if (folder / "acquisitions.json").exists():
        manifest["files"]["acquisitions.json"] = {"sha256": references.checksum(folder / "acquisitions.json")}
    _publish_json(handoff, manifest)
    load_imagery_bundle(handoff)
    return handoff


def summarize(request, records):
    roi = _roi_mask(request)
    complete = roi.copy()
    for interval in request["period"]["intervals"]:
        with rasterio.open(_paths(request, interval)[0]) as raster:
            data = raster.read()
        complete &= (data[-1] > 0) & np.all(np.isfinite(data[:-1]) & (data[:-1] != -9999), axis=0)
    return {"roi_pixels": int(roi.sum()), "all_eight_bins_clear_pixels": int(complete.sum()),
            "all_eight_bins_clear_fraction": float(complete.sum() / roi.sum()),
            "empty_bins": sum(row["quality"]["empty_after_screening"] for row in records),
            "interpretation": "Input coverage only; feature and model quality gates run in later notebooks",
            "crop_timing": "Fixed calendar demonstration; crop growth-stage alignment is unverified",
            "local_accuracy": "Not measured; calendar notes are not dated crop observations"}


def load_imagery_bundle(path):
    """Verify the exact eight-bin inventory before any downstream use."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    period = manifest.get("period", {})
    expected_period = four_month_period(period.get("start"))
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != "paoay_sentinel2"
            or manifest.get("status") != "prepared" or period != expected_period or manifest.get("bands") != BANDS
            or manifest.get("identity", {}).get("period") != period
            or manifest.get("grid") != manifest["identity"]["settings"]["grid"]):
        raise ValueError("Unsupported or inconsistent imagery handoff")
    if _digest(manifest["identity"])[:20] != manifest.get("request_id"):
        raise ValueError("Imagery request identity changed")
    required = {f"rasters/{item['name']}.{suffix}" for item in period["intervals"] for suffix in ("tif", "json")}
    actual = set(manifest.get("files", {}))
    if actual not in (required, required | {"acquisitions.json"}):
        raise ValueError("Imagery handoff must contain exactly eight raster/receipt pairs")
    for name, item in manifest["files"].items():
        if references.checksum(path.parent / name) != item["sha256"]:
            raise ValueError(f"Saved imagery changed: {name}; existing files preserved")
    reference = Path(manifest["reference_handoff"])
    if references.checksum(reference) != manifest["identity"]["reference_sha256"]:
        raise ValueError("Imagery reference dependency changed")
    references.load_reference_bundle(reference)
    return {"manifest": manifest, "folder": path.parent,
            "rasters": [path.parent / f"rasters/{item['name']}.tif" for item in period["intervals"]]}
