"""Exploratory Paoay map with the aligned (green-up) model on local Cloud Score+ composites.

The main Paoay workflow applies the fixed January–April model. This module applies an
aligned model instead: each pixel's four-month window starts at its own detected green-up
month, as in training. It reuses the training detector, timing checks, feature formulas and
confidence rule unchanged. Two inputs differ from training and are recorded in every output:

* the imagery is Cloud Score+ half-month composites, not the training cloud mask; and
* each half-month median is passed to the detector as one observation dated at the bin
  midpoint, because no per-date observations exist for these composites.

The aligned model was chosen after the external test scores were known, so this map is
exploratory. It is not a replacement for the fixed-window map and carries no accuracy claim.
"""

from collections import Counter
from datetime import date
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import mapping, shape
from shapely.ops import transform as reproject
from pyproj import Transformer

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.common import interval_bins, save_json
from crop_pipeline.features import experiments
from crop_pipeline.imagery import quality
from crop_pipeline.paoay import crop_baseline, features as paoay_features, land_activity, references
from crop_pipeline.windows import seasons, timing_quality

BANDS = land_activity.BANDS
DETECTOR = {**quality.DEFAULT_DETECTOR, "relative_threshold": 0.2}  # as in windows.workflow.select
TIMING_POLICY = dict(timing_quality.DEFAULT_TIMING_POLICY)
CYCLE_POLICY = "earliest_complete"
CONFIDENCE = dict(crop_baseline.RULES)
CODES = {"carrot": 1, "cabbage": 2, "potato": 3, "Unknown crop type": 4,
         "No qualified green-up window": 5, "Window not cloud-free": 6}
NODATA = 0
DEVIATIONS = (
    "Imagery: Cloud Score+ half-month composites, not the training SCL/cloud-probability mask.",
    "Detector input: each half-month median is one observation dated at its bin midpoint; "
    "training required per-date observations (composite-only windows needed review).",
    "Completeness: a pixel's eight bins must each be clear in that pixel; composites carry no clear-date count.",
    "Model choice: the aligned matched recipe was chosen after the external test scores were known.",
)


def default_model(repository_root, cohort="aligned_matched"):
    """The aligned package from the current Training 02 run, verified against its stage manifest."""
    stage = resolve_stage_pointer(Path(repository_root) / "pipeline_outputs/training/current_models.json")
    record = json.loads(stage.read_text())
    path = stage.parent / cohort / "model_manifest.json"
    for name in ("model.joblib", "model_manifest.json", "recipe.json"):
        expected = record["files"].get(f"{cohort}/{name}")
        if expected is None or seasons.file_sha256(stage.parent / cohort / name) != expected:
            raise ValueError(f"{cohort}/{name} is missing or changed since Training 02")
    return path


def load_model(manifest_path):
    recipe = paoay_features.load_model_recipe(manifest_path)["recipe"]
    if recipe.get("window_recipe", {}).get("type") != "aligned":
        raise ValueError("This notebook needs an aligned-window model package")
    import joblib
    package = joblib.load(Path(manifest_path).parent / "model.joblib")
    if list(package["feature_columns"]) != list(recipe["feature_columns"]):
        raise ValueError("Model feature order differs from its recipe")
    return package["classifier"], recipe


def study_area(kmz_paths, roi_name, context_names, crs):
    """Study boundary and reference areas, projected to the composite CRS."""
    records, sources = references.read_inputs(kmz_paths)
    inspection = references.inspect_records(records, roi_name=roi_name, context_names=context_names)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    roles = {row["reference_id"]: row["role"] for row in inspection["audit"]}
    areas = [{"name": r["name"], "geometry": reproject(project, shape(r["geometry"]))} for r in records
             if roles[r["reference_id"]] == "Reference area" and r["geometry_status"] == "valid"]
    roi = reproject(project, shape(inspection["roi"]["geometry"]))
    return roi, areas, [{"file": s["name"], "sha256": s["sha256"]} for s in sources]


def _files(root, year):
    return [land_activity.composite_path(root, (year, month, half)) for month in range(1, 13) for half in (1, 2)]


def grid_for(root, roi):
    """The composite grid cropped outward to the study boundary; every file must share it."""
    sample = next(path for path in sorted(Path(root).glob("*/Sentinel2_CSPlus_*.tif")))
    with rasterio.open(sample) as source:
        exact = from_bounds(*roi.bounds, transform=source.transform)
        col, row = int(np.floor(exact.col_off)), int(np.floor(exact.row_off))
        window = Window(col, row, int(np.ceil(exact.col_off + exact.width)) - col,
                        int(np.ceil(exact.row_off + exact.height)) - row)
        window = window.intersection(Window(0, 0, source.width, source.height))
        grid = {"crs": source.crs.to_string(), "transform": tuple(source.window_transform(window))[:6],
                "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
                "source_transform": tuple(source.transform)[:6], "source_shape": [source.height, source.width]}
    mask = ~geometry_mask([mapping(roi)], out_shape=(grid["window"][3], grid["window"][2]),
                          transform=rasterio.Affine(*grid["transform"]))  # pixel centres inside
    return grid, mask


def read_year(root, year, grid):
    """Raw reflectance (24 bins, 5 bands, rows, cols); NaN where missing; None for absent files."""
    col, row, width, height = grid["window"]
    cube = np.full((24, len(BANDS), height, width), np.nan, dtype="float32")
    present = []
    for index, path in enumerate(_files(root, year)):
        if not path.is_file():
            present.append(False)
            continue
        with rasterio.open(path) as source:
            if (source.crs.to_string() != grid["crs"] or tuple(source.transform)[:6] != tuple(grid["source_transform"])
                    or [source.height, source.width] != grid["source_shape"] or list(source.descriptions) != list(BANDS)):
                raise ValueError(f"{path.name} uses a different grid or band order")
            values = source.read(window=Window(col, row, width, height)).astype("float32")
            nodata = source.nodata if source.nodata is not None else land_activity.NODATA
            values[(values == nodata) | ~np.isfinite(values) | (values <= 0)] = np.nan
        cube[index] = values
        present.append(True)
    return cube, present


def year_completeness(root, years, grid, mask):
    """Per year: half-month files present and the share of study pixels clear in each bin."""
    rows = []
    for year in years:
        cube, present = read_year(root, year, grid)
        clear = np.all(np.isfinite(cube), axis=1)[:, mask]  # (24, pixels)
        shares = clear.mean(axis=1)
        rows.append({"year": year, "half_months_present": sum(present),
                     "mean_clear_share": round(float(shares.mean()), 4),
                     "half_months_at_least_80pct_clear": int((shares >= .8).sum()),
                     "pixels_with_at_least_6_clear": int((clear.sum(axis=0) >= DETECTOR["minimum_points"]).sum())})
    return rows


def choose_year(rows):
    """Rule fixed before mapping: all 24 half-months present, then highest mean clear share."""
    complete = [row for row in rows if row["half_months_present"] == 24]
    if not complete:
        raise ValueError("No year has all 24 half-month composites")
    return max(complete, key=lambda row: (row["mean_clear_share"], row["year"]))["year"]


def _index(values):
    b3, b4, b5, b8, b11 = (float(v) * .0001 for v in values)
    return {"NDVI": quality.ratio(b8, b4), "NDRE": quality.ratio(b8, b5), "NDMI": quality.ratio(b8, b11)}


def observations(cube, year, pixels):
    """Detector rows: one row per pixel and half-month, dated at the bin midpoint (training convention)."""
    bins = interval_bins(year)
    rows = []
    for row, column in pixels:
        region = f"px_{row}_{column}"
        for index, interval in enumerate(bins):
            start, end = date.fromisoformat(interval["start"]), date.fromisoformat(interval["end"])
            values = cube[index, :, row, column]
            clear = bool(np.all(np.isfinite(values)))
            indices = _index(values) if clear else {"NDVI": None, "NDRE": None, "NDMI": None}
            rows.append({"region_id": region, "date": (start + (end - start) // 2).isoformat(),
                         "start": interval["start"], "end": interval["end"], "source_kind": "observed_daily",
                         "valid_fraction": 1.0 if clear else 0.0, **indices})
    return rows


def qualified_windows(rows, year):
    """Training's detector, onset-month rule and automatic timing checks, unchanged."""
    outcomes = seasons.identify_candidates(quality.detect_greenup(rows, DETECTOR))
    windows, audit = [], []
    for candidate in outcomes:
        reason = ""
        if candidate["status"] not in {"candidate", "provisional", "ambiguous"}:
            reason = "detector_" + candidate.get("reason", candidate["status"])
        else:
            try:
                start = seasons.choose_onset_month({}, candidate)
            except ValueError:
                reason = "month_boundary_requires_timing_review"
            else:
                end = seasons.add_calendar_months(start, 4)
                if end > date(year + 1, 1, 1):
                    reason = "window_extends_past_year"
                else:
                    windows.append({**candidate, "window_start": start.isoformat(), "window_end": end.isoformat()})
                    continue
        audit.append({"region_id": candidate["region_id"], "candidate_id": candidate.get("candidate_id", ""),
                      "outcome": reason})
    assessments = {row["candidate_id"]: row for row in timing_quality.assess_timing(
        windows, rows, DETECTOR, TIMING_POLICY, CYCLE_POLICY)}
    passed = []
    for window in windows:
        assessment = assessments[window["candidate_id"]]
        ok = assessment["automatic_timing_status"] == "passed"
        audit.append({"region_id": window["region_id"], "candidate_id": window["candidate_id"],
                      "outcome": "timing_passed" if ok else "timing_" + assessment["automatic_timing_reasons"].split(";")[0],
                      "window_start": window["window_start"]})
        if ok:
            passed.append(window)
    return passed, audit


def window_features(cube, window, row, column, grid):
    """The saved model's 72 inputs for one pixel window, via the Paoay/training formulas."""
    x, y = rasterio.Affine(*grid["transform"]) @ (column + .5, row + .5)
    base = {"window_start": window["window_start"], "window_end": window["window_end"], "field_id": "paoay_roi",
            "season_key": window["region_id"] + "|" + window["window_start"], "source_crs": grid["crs"], "x": x, "y": y}
    year = int(window["window_start"][:4])
    names = [interval["start"] for interval in interval_bins(year)]
    for number, interval in enumerate(seasons.window_intervals(window), start=1):
        values = cube[names.index(interval["start"]), :, row, column]
        if not np.all(np.isfinite(values)):
            return None
        pixel = {name: float(value) * .0001 for name, value in zip(BANDS, values)}
        pixel.update(valid=1, clear_count=1.0)
        features = seasons.pixel_features(pixel)
        if features is None:
            return None
        base.update({f"bin_{number:02d}_{key}": value for key, value in features.items()})
    try:
        return experiments.enhance_window(base, experiments.FEATURE_POLICY)
    except ValueError:
        return None


def earliest_complete(cube, windows, grid):
    """Per pixel, the earliest qualified window whose eight bins are all clear (training's cycle policy)."""
    by_pixel = {}
    for window in windows:
        by_pixel.setdefault(window["region_id"], []).append(window)
    selected, incomplete = {}, set()
    for region, candidates in by_pixel.items():
        _, row, column = region.split("_")
        for window in sorted(candidates, key=lambda w: w["window_start"]):
            row_features = window_features(cube, window, int(row), int(column), grid)
            if row_features is not None:
                selected[(int(row), int(column))] = (window, row_features)
                break
        else:
            incomplete.add((int(row), int(column)))
    return selected, incomplete


def classify(classifier, recipe, selected):
    """Raw probabilities, then the existing Paoay confidence rule (top >= 0.60, margin >= 0.10)."""
    keys = list(selected)
    if not keys:
        return {}
    matrix = np.asarray([[selected[key][1][column] for column in recipe["feature_columns"]] for key in keys], dtype=float)
    probabilities = classifier.predict_proba(matrix)
    ordered = np.sort(probabilities, axis=1)
    top, margin = ordered[:, -1], ordered[:, -1] - ordered[:, -2]
    labels = classifier.classes_[np.argmax(probabilities, axis=1)]
    accepted = (top >= CONFIDENCE["minimum_top_probability"]) & (margin >= CONFIDENCE["minimum_margin"]) & (margin > 0)
    return {key: {"raw": str(label), "label": str(label) if ok else "Unknown crop type",
                  "top": float(t), "margin": float(m), "window_start": selected[key][0]["window_start"],
                  **{f"p_{name}": float(p) for name, p in zip(classifier.classes_, probs)}}
            for key, label, ok, t, m, probs in zip(keys, labels, accepted, top, margin, probabilities)}


def code_raster(mask, results, incomplete):
    codes = np.full(mask.shape, NODATA, dtype="uint8")
    codes[mask] = CODES["No qualified green-up window"]
    for row, column in incomplete:
        codes[row, column] = CODES["Window not cloud-free"]
    onset = np.zeros(mask.shape, dtype="uint8")
    top = np.full(mask.shape, np.nan, dtype="float32")
    for (row, column), result in results.items():
        codes[row, column] = CODES[result["label"]]
        onset[row, column] = int(result["window_start"][5:7])
        top[row, column] = result["top"]
    return codes, onset, top


def area_table(codes, mask, areas, grid):
    names = {code: name for name, code in CODES.items()}
    rows = []
    for area in [{"name": "Whole study boundary", "geometry": None}, *areas]:
        inside = mask if area["geometry"] is None else mask & ~geometry_mask(
            [mapping(area["geometry"])], out_shape=mask.shape, transform=rasterio.Affine(*grid["transform"]))
        counts = Counter(codes[inside].tolist())
        total = int(inside.sum())
        rows.append({"area": area["name"], "pixels": total,
                     **{names[code]: counts.get(code, 0) for code in sorted(names)}})
    return rows


def run(kmz_paths, roi_name, context_names, imagery_root, model_manifest, output_root, years=range(2019, 2026)):
    """Build one keyed exploratory bundle; an existing bundle with the same inputs is reused, never overwritten."""
    classifier, recipe = load_model(model_manifest)
    sample = next(iter(sorted(Path(imagery_root).glob("*/Sentinel2_CSPlus_*.tif"))))
    with rasterio.open(sample) as source:
        crs = source.crs.to_string()
    roi, areas, sources = study_area(kmz_paths, roi_name, context_names, crs)
    grid, mask = grid_for(imagery_root, roi)
    completeness = year_completeness(imagery_root, list(years), grid, mask)
    year = choose_year(completeness)
    inventory = [{"file": str(path.relative_to(imagery_root)), "sha256": references.checksum(path)}
                 for path in _files(imagery_root, year)]
    identity = {"model_sha256": references.checksum(Path(model_manifest).parent / "model.joblib"),
                "recipe_sha256": references.checksum(Path(model_manifest).parent / "recipe.json"),
                "year": year, "imagery": inventory, "references": sources, "roi_name": roi_name,
                "context_names": sorted(context_names), "detector": DETECTOR, "timing_policy": TIMING_POLICY,
                "cycle_policy": CYCLE_POLICY, "confidence": CONFIDENCE, "module_sha256": references.checksum(__file__)}
    folder = Path(output_root) / seasons.record_fingerprint(identity)[:16]
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        return manifest_path
    cube, _ = read_year(imagery_root, year, grid)
    pixels = list(zip(*np.nonzero(mask)))
    rows = observations(cube, year, pixels)
    windows, audit = qualified_windows(rows, year)
    selected, incomplete = earliest_complete(cube, windows, grid)
    results = classify(classifier, recipe, selected)
    codes, onset, top = code_raster(mask, results, incomplete)
    staging = folder.with_name(folder.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    profile = {"driver": "GTiff", "width": mask.shape[1], "height": mask.shape[0], "crs": grid["crs"],
               "transform": rasterio.Affine(*grid["transform"]), "compress": "deflate"}
    with rasterio.open(staging / "aligned_map.tif", "w", count=2, dtype="uint8", nodata=NODATA, **profile) as raster:
        raster.write(codes, 1)
        raster.write(onset, 2)
        raster.descriptions = ("class_code", "window_start_month")
    with rasterio.open(staging / "top_probability.tif", "w", count=1, dtype="float32", nodata=np.nan, **profile) as raster:
        raster.write(top, 1)
    seasons.write_records(staging / "year_completeness.csv", completeness)
    seasons.write_records(staging / "window_audit.csv", audit)
    seasons.write_records(staging / "pixel_predictions.csv", [
        {"row": int(r), "column": int(c), **result} for (r, c), result in sorted(results.items())])
    seasons.write_records(staging / "area_summary.csv", area_table(codes, mask, areas, grid))
    pixel_ha = abs(grid["transform"][0] * grid["transform"][4]) / 10000
    save_json(staging / "manifest.json", {
        "kind": "paoay_aligned_exploratory_map", "status": "exploratory_unvalidated", "identity": identity,
        "model_manifest": str(Path(model_manifest).resolve()), "codes": CODES, "grid": grid,
        "pixel_ha": pixel_ha, "study_pixels": int(mask.sum()), "deviations_from_training": list(DEVIATIONS),
        "window_outcomes": dict(Counter(row["outcome"] for row in audit)),
        "pixels_with_qualified_window": len({w["region_id"] for w in windows}),
        "pixels_predicted": len(results), "pixels_window_not_cloud_free": len(incomplete),
        "files": {name: references.checksum(staging / name) for name in (
            "aligned_map.tif", "top_probability.tif", "year_completeness.csv", "window_audit.csv",
            "pixel_predictions.csv", "area_summary.csv")}})
    staging.rename(folder)
    return manifest_path


def load(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "paoay_aligned_exploratory_map":
        raise ValueError("Not an aligned exploratory Paoay bundle")
    for name, digest in manifest["files"].items():
        if references.checksum(manifest_path.parent / name) != digest:
            raise ValueError(f"Saved output changed: {name}")
    with rasterio.open(manifest_path.parent / "aligned_map.tif") as raster:
        codes, onset = raster.read(1), raster.read(2)
    tables = {name: seasons.read_table(manifest_path.parent / f"{name}.csv")
              for name in ("year_completeness", "window_audit", "area_summary")}
    return {"manifest": manifest, "codes": codes, "onset": onset, **tables}


def plot_map(bundle, areas=()):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = {1: land_activity.CROP_COLORS["Carrot"], 2: land_activity.CROP_COLORS["Cabbage"],
              3: land_activity.CROP_COLORS["Potato"], 4: land_activity.CROP_COLORS["Unknown crop type"],
              5: "#d9d9d9", 6: "#9ec5e8"}
    names = {code: name for name, code in bundle["manifest"]["codes"].items()}
    grid = bundle["manifest"]["grid"]
    transform = rasterio.Affine(*grid["transform"])
    height, width = bundle["codes"].shape
    extent = (transform.c, transform.c + transform.a * width, transform.f + transform.e * height, transform.f)
    shown = np.ma.masked_equal(bundle["codes"], NODATA)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(shown, cmap=ListedColormap([colors[c] for c in range(1, 7)]), vmin=.5, vmax=6.5,
                extent=extent, interpolation="nearest")
    for area in areas:
        for polygon in getattr(area["geometry"], "geoms", [area["geometry"]]):
            axis.plot(*polygon.exterior.xy, color="#0b0b0b", linewidth=.9)
    axis.legend(handles=[Patch(color=colors[c], label=names[c].capitalize()) for c in range(1, 7)],
                loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    year = bundle["manifest"]["identity"]["year"]
    axis.set_title(f"Exploratory aligned-model map, {year} (not validated)", loc="left", fontweight="bold")
    axis.set_xticks([]), axis.set_yticks([])
    figure.tight_layout()
    return figure


def plot_onsets(bundle):
    import matplotlib.pyplot as plt
    predicted = (bundle["codes"] >= 1) & (bundle["codes"] <= 4)
    months = bundle["onset"][predicted]
    counts = [int((months == month).sum()) for month in range(1, 13)]
    figure, axis = plt.subplots(figsize=(8, 3.2))
    axis.bar(range(1, 13), counts, color=land_activity.CROP_COLORS["Unknown crop type"])
    axis.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axis.set_ylabel("Predicted pixels")
    axis.set_title("Window start month (detected green-up) of predicted pixels", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
