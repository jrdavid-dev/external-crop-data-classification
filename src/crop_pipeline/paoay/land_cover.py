"""Local land-cover guidance, separate from crop predictions and hard filtering."""

import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline.paoay import dynamic_world as dw, imagery, references

VERSION = 1
DEFAULT_RULES = {"minimum_dates": 2, "minimum_top_probability": 0.6, "minimum_margin": 0.1}
BANDS = ["top_class", "top_probability", "runner_up_probability", "probability_margin",
         "observation_count", "guidance"]
STATES = {0: "No observations", 1: "Too few dates", 2: "Uncertain probabilities",
          3: "Crop support", 4: "Other-cover support"}
NODATA = -9999


def validate_rules(rules):
    if set(rules) != set(DEFAULT_RULES):
        raise ValueError("Use exactly the three documented land-cover rules")
    count = rules["minimum_dates"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
        raise ValueError("minimum_dates must be an integer from 1 to 16")
    for key in ("minimum_top_probability", "minimum_margin"):
        value = rules[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite probability from 0 to 1")
    return dict(rules)


def classify(data, roi, rules):
    """Derive per-bin guidance; no observations never become water/class zero."""
    rules = validate_rules(rules)
    if data.shape != (10, *roi.shape) or roi.dtype != bool:
        raise ValueError("Expected nine probabilities and counts on a boolean ROI grid")
    counts = data[-1]
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))):
        raise ValueError("Observation counts must be finite nonnegative integers")
    observed = counts > 0
    probabilities = data[:-1, observed]
    if (not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1))
            or not np.allclose(probabilities.sum(axis=0), 1, atol=0.0005, rtol=0)
            or np.any(data[:-1, ~observed] != NODATA)):
        raise ValueError("Invalid Dynamic World probability support")
    result = np.full((6, *roi.shape), NODATA, dtype="float32")
    has_data = observed & roi
    result[4, roi] = counts[roi]
    result[5, roi] = 0
    values = data[:-1, has_data]
    ordered = np.sort(values, axis=0)
    leading = values.argmax(axis=0)
    top, runner = ordered[-1], ordered[-2]
    margin = top - runner
    result[0, has_data], result[1, has_data] = leading, top
    result[2, has_data], result[3, has_data] = runner, margin
    enough = counts[has_data] >= rules["minimum_dates"]
    # A tie is unresolved even when a user deliberately sets minimum_margin to zero.
    supported = enough & (top >= rules["minimum_top_probability"]) & (margin > 0) & (margin >= rules["minimum_margin"])
    states = np.where(enough, 2, 1)
    states[supported & (leading == dw.PROBABILITIES.index("crops"))] = 3
    states[supported & (leading != dw.PROBABILITIES.index("crops"))] = 4
    result[5, has_data] = states
    return result


def prepare_request(dynamic_world_handoff, output_root, rules=None):
    source = dw.load_dynamic_world_bundle(dynamic_world_handoff)
    manifest = source["manifest"]
    rules = validate_rules(DEFAULT_RULES if rules is None else rules)
    identity = {"version": VERSION, "dynamic_world_sha256": references.checksum(dynamic_world_handoff),
                "rules": rules, "period": manifest["period"], "grid": manifest["grid"],
                "implementation_sha256": references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "dynamic_world_handoff": str(Path(dynamic_world_handoff).resolve()),
            "folder": Path(output_root) / "bundles" / key}


def _source(request):
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Land-cover request changed after planning")
    path = request["dynamic_world_handoff"]
    if references.checksum(path) != request["identity"]["dynamic_world_sha256"]:
        raise ValueError("Dynamic World handoff changed; plan again")
    source = dw.load_dynamic_world_bundle(path)
    if any(source["manifest"][key] != request["identity"][key] for key in ("period", "grid")):
        raise ValueError("Land-cover dates/grid differ from Dynamic World")
    return source


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(request, iterator=None):
    """Publish the entire verified result atomically, never overwrite prior bundles."""
    source = _source(request)
    folder = request["folder"]
    manifest_path = folder / "land_cover_handoff.json"
    if folder.exists():
        load_land_cover_bundle(manifest_path)
        return manifest_path
    identity = request["identity"]
    grid, period = identity["grid"], identity["period"]
    dw_settings = source["manifest"]["identity"]["settings"]
    roi = imagery._roi_mask({"settings": dw_settings, "fragment": {"geometry": dw_settings["geometry"]}})
    crs = rasterio.crs.CRS.from_user_input(grid["crs"])
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Area summaries require a projected grid in metres")
    transform = Affine(*grid["transform"])
    hectares = abs(transform.a * transform.e - transform.b * transform.d) / 10000
    coverage, classes = [], []
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".land-cover-", dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        (staging / "rasters").mkdir(parents=True)
        intervals = period["intervals"]
        for interval in (iterator(intervals) if iterator else intervals):
            raw = source["folder"] / f"rasters/{interval['name']}.tif"
            receipt = json.loads(raw.with_suffix(".json").read_text())
            dw_request = {"settings": dw_settings, "period": period, "fragment": {"geometry": dw_settings["geometry"]}}
            dw.inspect_raster(raw, dw_request, interval, receipt["available_dates"])
            with rasterio.open(raw) as raster:
                result = classify(raster.read(), roi, identity["rules"])
            path = staging / "rasters" / f"{interval['name']}.tif"
            with rasterio.open(path, "w", driver="GTiff", width=grid["width"], height=grid["height"],
                               count=len(BANDS), dtype="float32", crs=grid["crs"], transform=transform,
                               nodata=NODATA, compress="deflate") as raster:
                raster.write(result)
                raster.descriptions = tuple(BANDS)
            state = result[5]
            row = {"interval": interval["name"], "roi_pixels": int(roi.sum())}
            row.update({label: int((state == code).sum()) for code, label in STATES.items()})
            coverage.append(row)
            for code, name in enumerate(dw.PROBABILITIES):
                count = int(((result[0] == code) & np.isin(state, [3, 4])).sum())
                classes.append({"interval": interval["name"], "category": name, "pixels": count,
                                "area_ha": count * hectares, "roi_percent": count / roi.sum() * 100})
            for code in (0, 1, 2):
                count = int((state == code).sum())
                classes.append({"interval": interval["name"], "category": STATES[code], "pixels": count,
                                "area_ha": count * hectares, "roi_percent": count / roi.sum() * 100})
        _write_csv(staging / "coverage.csv", coverage)
        _write_csv(staging / "class_areas.csv", classes)
        # Check dependencies again before publishing; a failed build leaves previous outputs intact.
        _source(request)
        files = [*sorted((staging / "rasters").glob("*.tif")), staging / "coverage.csv", staging / "class_areas.csv"]
        manifest = {"format_version": VERSION, "kind": "paoay_land_cover_guidance", "status": "prepared",
                    "bundle_id": request["bundle_id"], "identity": identity, "period": period, "grid": grid,
                    "dynamic_world_handoff": request["dynamic_world_handoff"], "bands": BANDS,
                    "guidance_codes": STATES, "class_codes": dict(enumerate(dw.PROBABILITIES)),
                    "nodata": NODATA, "pixel_area_ha": hectares, "roi_pixels": int(roi.sum()),
                    "rules_status": "illustrative_unvalidated_not_tuned_to_local_labels",
                    "downstream_policy": "guidance_only_preserve_all_valid_crop_model_inputs",
                    "limitations": ["Predicted broad cover, not ground truth or crop accuracy",
                                    "Bare is not verified fallow; crops does not identify a target crop",
                                    "Other-cover support is not permanent agricultural exclusion",
                                    "Per-bin guidance may change during crop rotations",
                                    "Top class for tied/weak vectors is descriptive only; use guidance state"],
                    "license": source["manifest"]["license"], "attribution": source["manifest"]["attribution"],
                    "source_url": dw.SOURCE_URL, "copernicus_notice": source["manifest"].get("copernicus_notice"),
                    "files": {p.relative_to(staging).as_posix(): {"sha256": references.checksum(p)} for p in files}}
        (staging / "land_cover_handoff.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(folder)
    load_land_cover_bundle(manifest_path)
    return manifest_path


def load_land_cover_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    identity = manifest.get("identity", {})
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != "paoay_land_cover_guidance"
            or manifest.get("status") != "prepared" or manifest.get("bands") != BANDS
            or imagery._digest(identity)[:20] != manifest.get("bundle_id")
            or manifest.get("period") != identity.get("period") or manifest.get("grid") != identity.get("grid")
            or manifest.get("downstream_policy") != "guidance_only_preserve_all_valid_crop_model_inputs"):
        raise ValueError("Unsupported or inconsistent land-cover handoff")
    validate_rules(identity["rules"])
    _source({"identity": identity, "bundle_id": manifest["bundle_id"],
             "dynamic_world_handoff": manifest["dynamic_world_handoff"]})
    expected = {f"rasters/{item['name']}.tif" for item in manifest["period"]["intervals"]} | {"coverage.csv", "class_areas.csv"}
    if set(manifest.get("files", {})) != expected:
        raise ValueError("Land-cover handoff needs exactly eight rasters and two summary tables")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved land-cover output changed: {name}; existing files preserved")
    tables = {}
    for name in ("coverage.csv", "class_areas.csv"):
        with (path.parent / name).open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
    return {"manifest": manifest, "folder": path.parent, "coverage": tables["coverage.csv"],
            "class_areas": tables["class_areas.csv"],
            "rasters": [path.parent / f"rasters/{item['name']}.tif" for item in manifest["period"]["intervals"]]}


def plot_guidance(bundle):
    """Preview evidence states, not a final crop/land-cover classification map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = ["#bdbdbd", "#e0c998", "#f5a742", "#5a9b54", "#547aaf"]
    cmap = ListedColormap(colors).with_extremes(bad="white")
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for axis, path, interval in zip(axes.flat, bundle["rasters"], bundle["manifest"]["period"]["intervals"]):
        with rasterio.open(path) as raster:
            data = raster.read(6, masked=True)
        axis.imshow(data, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 5), 5), interpolation="nearest")
        start = date.fromisoformat(interval["start"])
        last = date.fromisoformat(interval["end"]) - timedelta(days=1)
        axis.set_title(f"{start:%b} {start.day}–{last.day}, {start.year}", fontsize=10)
        axis.set_axis_off()
    figure.suptitle("Dynamic World evidence by half-month — illustrative rules, not crop accuracy")
    figure.legend(handles=[Patch(color=colors[code], label=label) for code, label in STATES.items()],
                  loc="outside lower center", ncol=3, fontsize=9)
    return figure
