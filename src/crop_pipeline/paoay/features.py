"""Exact saved-model feature inputs, without fitting, inference or land-cover filtering."""

import csv
import inspect
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline.features import experiments
from crop_pipeline.windows import seasons
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.paoay import imagery, dynamic_world, land_cover, references

VERSION = 1
PIXEL_STATES = {0: "Outside ROI", 1: "Missing bin observations", 2: "Undefined spectral index",
                3: "Nonfinite engineered features", 4: "Failed ROI composite coverage", 5: "Ready features"}


def find_model(analysis_root, explicit=None):
    """Choose only an explicit package or a unique candidate, never the newest/best score."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError("Selected model manifest missing; finish notebook 06")
        return path
    paths = sorted(Path(analysis_root).glob("**/selected_model/*/model_manifest.json"))
    if not paths:
        raise FileNotFoundError("No selected-model package found; finish notebook 06 first")
    if len(paths) != 1:
        raise ValueError("Multiple selected-model packages found; set model_manifest explicitly")
    return paths[0]


def load_model_recipe(path):
    """Verify package and formula provenance without deserializing executable model files."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "paoay_inference_model_package"
            or manifest.get("status") != "unvalidated_external_parcel_prototype"
            or set(manifest.get("files", {})) != {"model.joblib", "recipe.json"}):
        raise ValueError("Unsupported selected-model package; use notebook 06's verified export")
    for name, item in manifest["files"].items():
        if references.checksum(path.parent / name) != item["sha256"]:
            raise ValueError(f"Selected-model package changed: {name}; files preserved")
    recipe = json.loads((path.parent / "recipe.json").read_text())
    if (manifest.get("identity", {}).get("recipe") != recipe
            or manifest["identity"].get("model_sha256") != references.checksum(path.parent / "model.joblib")
            or recipe.get("status") != manifest["status"]):
        raise ValueError("Selected-model recipe and package identity differ")
    design = recipe.get("selected_design")
    if (design not in experiments.feature_sets() or recipe.get("feature_columns") != experiments.feature_sets()[design]
            or recipe.get("selection", {}).get("status") != "selected_on_validation"
            or recipe["selection"].get("selected_design") != design):
        raise ValueError("Saved feature design/order is unsupported or not validation-selected")
    experiments.validate_policy(recipe["feature_policy"])
    if (recipe.get("window_convention") != "four_calendar_months_eight_half_month_bins_end_exclusive"
            or recipe.get("quality", {}).get("missing_values") != "exclude_no_imputation"
            or recipe["quality"].get("finite_features_required") is not True):
        raise ValueError("Unsupported complete-window or missing-data policy")
    minimum = recipe["quality"].get("minimum_valid_fraction")
    if isinstance(minimum, bool) or not isinstance(minimum, (float, int)) or not .8 <= minimum <= 1:
        raise ValueError("Saved minimum valid fraction must be between 0.8 and 1")
    window = recipe.get("window_recipe", {})
    if (window.get("minimum_valid_fraction", minimum) != minimum
            or window.get("composites", 8) != 8
            or window.get("missing_values", "exclude_no_imputation") != "exclude_no_imputation"):
        raise ValueError("Saved window and quality recipes disagree")
    source = Path(manifest["feature_handoff"])
    if not source.is_file():
        raise FileNotFoundError("Original notebook 05 feature provenance is missing; restore it before feature preparation")
    if references.checksum(source) != recipe["source_feature_manifest_sha256"]:
        raise ValueError("Training feature provenance changed; preserve the original feature handoff")
    feature_manifest = json.loads(source.read_text())
    if manifest.get("experiment_cohort") in {"full", "matched"}:
        if feature_manifest.get("implementation_sha256") != references.checksum(inspect.getfile(experiments)):
            raise ValueError("Training feature implementation differs")
        return {"manifest": manifest, "recipe": recipe, "source_feature_handoff": source}
    if (feature_manifest.get("kind") != "local_parcel_feature_experiments" or feature_manifest.get("format_version") != 1
            or feature_manifest.get("feature_sets") != experiments.feature_sets()
            or feature_manifest.get("feature_policy") != recipe["feature_policy"]
            or feature_manifest.get("identity", {}).get("implementation_sha256") != references.checksum(inspect.getfile(experiments))):
        raise ValueError("Training feature implementation differs; do not compute a different recipe")
    return {"manifest": manifest, "recipe": recipe, "source_feature_handoff": source}


def _inputs(guidance_handoff):
    guidance = land_cover.load_land_cover_bundle(guidance_handoff)
    world = dynamic_world.load_dynamic_world_bundle(guidance["manifest"]["dynamic_world_handoff"])
    image_path = Path(world["manifest"]["imagery_handoff"])
    image = imagery.load_imagery_bundle(image_path)
    return guidance, image, image_path


def validate_imagery(recipe, image):
    """Require the exported acquisition/scaling recipe; never relabel legacy data."""
    window = recipe.get("window_recipe", {})
    if window.get("type") in {"fixed", "aligned"}:
        period = image["manifest"]["identity"]["period"]
        if window["type"] != "fixed" or period["start"][5:] != f"{window['start_month']:02d}-01":
            raise ValueError("The Paoay calendar inputs require a matching fixed-window model")
    saved = recipe["imagery"]
    settings = image["manifest"]["identity"]["settings"]
    for key in ("collection", "resampling", "cloud_probability_collection", "cloud_probability_threshold", "excluded_scl_classes"):
        if settings.get(key) != saved.get(key):
            raise ValueError(f"Paoay imagery differs from the model recipe: {key}")
    expected = {"request_version": REQUEST_VERSION, "resampling": RESAMPLING_POLICY,
        "spectral_scale": .0001, "collection_bands": imagery.BANDS[:-1],
        "feature_bands": list(seasons.FEATURE_BANDS), "bands": list(seasons.FEATURE_BANDS),
        "clear_counts_scaled": False, "output_resolution_metres": 10, "input_nodata_value": -9999,
        "cloud_probability_rule": "probability < threshold", "missing_cloud_probability": "exclude",
        "scene_edges": "valid_B8A_and_B9", "compositing": "same_date_mosaics_then_calendar_half_month_medians",
        "valid_data_rule": "all six collection bands must be valid before compositing",
        "nodata_handling": "mask nodata before reflectance scaling"}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported model imagery recipe; do not change its preprocessing")
    if (settings.get("version") != saved["request_version"] or settings.get("bands") != imagery.BANDS
            or settings.get("reflectance_scale_to_apply_later") != saved["spectral_scale"]
            or settings.get("nodata") != saved["input_nodata_value"]
            or image["manifest"].get("spectral_scale_to_apply_later") != saved["spectral_scale"]
            or image["manifest"].get("clear_counts_scaled") is not False):
        raise ValueError("Paoay band/scaling contract differs from the model")
    grid = image["manifest"]["grid"]
    if grid["transform"][0] != 10 or grid["transform"][4] != -10 or grid["transform"][1] != 0 or grid["transform"][3] != 0:
        raise ValueError("Expected a north-up 10 m model input grid")


def prepare_request(guidance_handoff, model_manifest, output_root):
    model = load_model_recipe(model_manifest)
    guidance, image, image_path = _inputs(guidance_handoff)
    validate_imagery(model["recipe"], image)
    identity = {"version": VERSION, "guidance_sha256": references.checksum(guidance_handoff),
                "model_manifest_sha256": references.checksum(model_manifest), "imagery_sha256": references.checksum(image_path),
                "feature_columns": model["recipe"]["feature_columns"], "feature_policy": model["recipe"]["feature_policy"],
                "implementation_sha256": references.checksum(__file__),
                "feature_implementation_sha256": references.checksum(inspect.getfile(experiments)),
                "base_implementation_sha256": references.checksum(inspect.getfile(seasons))}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "guidance_handoff": str(Path(guidance_handoff).resolve()),
            "model_manifest": str(Path(model_manifest).resolve()), "imagery_handoff": str(image_path.resolve()),
            "folder": Path(output_root) / "bundles" / key}


def _verify(request):
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Feature request changed after planning")
    for key, field in (("guidance_handoff", "guidance_sha256"), ("model_manifest", "model_manifest_sha256"), ("imagery_handoff", "imagery_sha256")):
        if references.checksum(request[key]) != request["identity"][field]:
            raise ValueError(f"Feature dependency changed: {key}")
    model = load_model_recipe(request["model_manifest"])
    guidance, image, path = _inputs(request["guidance_handoff"])
    if path.resolve() != Path(request["imagery_handoff"]).resolve():
        raise ValueError("Guidance and feature imagery dependencies differ")
    validate_imagery(model["recipe"], image)
    return model["recipe"], image


def _coverage(image, recipe):
    manifest = image["manifest"]
    reference = references.load_reference_bundle(manifest["reference_handoff"])
    request = {"period": manifest["period"], "settings": manifest["identity"]["settings"],
               "fragment": reference["roi"]["features"][0], "folder": image["folder"]}
    roi = imagery._roi_mask(request)
    rows, cube = [], []
    for interval, path in zip(manifest["period"]["intervals"], image["rasters"]):
        quality = imagery.inspect_raster(path, request, interval)
        with rasterio.open(path) as raster:
            cube.append(raster.read())
        rows.append({"interval": interval["name"], **quality,
                     "minimum_valid_fraction": recipe["quality"]["minimum_valid_fraction"],
                     "coverage_pass": quality["clear_roi_fraction"] >= recipe["quality"]["minimum_valid_fraction"]})
    return roi, np.asarray(cube), rows


def pixel_row(cube, row, column, period, grid, policy):
    """Reuse the training index/SAVI/shape functions without a second implementation."""
    x, y = Affine(*grid["transform"]) @ (column + .5, row + .5)
    base = {"window_start": period["start"], "window_end": period["end"],
            "field_id": "paoay_roi", "season_key": period["start"] + "/" + period["end"],
            "source_crs": grid["crs"], "x": x, "y": y}
    for number, values in enumerate(cube[:, :, row, column], start=1):
        if (not np.all(np.isfinite(values)) or np.any(values[:-1] == -9999)
                or values[-1] < 1 or values[-1] != np.floor(values[-1])):
            return None, 1
        pixel = {name: float(value) * .0001 for name, value in zip(imagery.BANDS[:-1], values[:-1])}
        pixel.update(valid=1, clear_count=float(values[-1]))
        features = seasons.pixel_features(pixel)
        if features is None:
            return None, 2
        base.update({f"bin_{number:02d}_{key}": value for key, value in features.items()})
    try:
        return experiments.enhance_window(base, policy), 5
    except ValueError:
        return None, 3


def build(request, iterator=None):
    """Create an immutable matrix, row/column locations and explicit exclusion audit."""
    recipe, image = _verify(request)
    folder, manifest_path = request["folder"], request["folder"] / "feature_handoff.json"
    if folder.exists():
        load_feature_bundle(manifest_path, require_ready=False)
        return manifest_path
    roi, cube, coverage = _coverage(image, recipe)
    period, grid = image["manifest"]["period"], image["manifest"]["grid"]
    complete = np.all((cube[:, -1] > 0) & np.all(np.isfinite(cube[:, :-1]) & (cube[:, :-1] != -9999), axis=1), axis=0)
    states = np.zeros(roi.shape, dtype="uint8")
    states[roi] = 1
    gate = all(row["coverage_pass"] for row in coverage)
    if not gate:
        states[roi & complete] = 4
    candidates = np.argwhere(roi & complete) if gate else np.empty((0, 2), dtype=int)
    matrix, rows, columns, xs, ys, ids = [], [], [], [], [], []
    ordered = recipe["feature_columns"]
    for row, column in (iterator(candidates) if iterator else candidates):
        feature, state = pixel_row(cube, int(row), int(column), period, grid, recipe["feature_policy"])
        states[row, column] = state
        if feature is not None:
            values = [feature[name] for name in ordered]
            if not np.all(np.isfinite(values)):
                states[row, column] = 3
                continue
            matrix.append(values); rows.append(row); columns.append(column)
            xs.append(feature["x"]); ys.append(feature["y"]); ids.append(feature["sample_id"])
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        staging.mkdir()
        np.savez_compressed(staging / "features.npz", matrix=np.asarray(matrix, dtype="float64").reshape(-1, len(ordered)),
            rows=np.asarray(rows, dtype="int32"), columns=np.asarray(columns, dtype="int32"),
            x=np.asarray(xs, dtype="float64"), y=np.asarray(ys, dtype="float64"), sample_ids=np.asarray(ids, dtype="U64"))
        with rasterio.open(staging / "pixel_status.tif", "w", driver="GTiff", width=grid["width"], height=grid["height"],
                           count=1, dtype="uint8", crs=grid["crs"], transform=Affine(*grid["transform"]), nodata=0, compress="deflate") as raster:
            raster.write(states, 1); raster.set_band_description(1, "feature_status")
        audit = [{"status": label, "pixels": int((states == code).sum())} for code, label in PIXEL_STATES.items() if code != 0]
        for name, records in (("coverage.csv", coverage), ("pixel_audit.csv", audit)):
            with (staging / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        _verify(request)
        manifest = {"format_version": VERSION, "kind": "paoay_crop_features", "status": "prepared" if matrix else "pending_inputs",
            "bundle_id": request["bundle_id"], "identity": request["identity"], "feature_columns": ordered,
            "guidance_handoff": request["guidance_handoff"], "model_manifest": request["model_manifest"], "imagery_handoff": request["imagery_handoff"],
            "period": period, "grid": grid, "row_count": len(matrix), "roi_pixels": int(roi.sum()), "pixel_status_codes": PIXEL_STATES,
            "quality_gate": "all_eight_bins_complete_and_each_bin_ROI_clear_fraction_meets_saved_minimum",
            "quality_note": "ROI clear coverage substitutes the training parcel footprint for this mapping demonstration",
            "land_cover_filter_applied": False, "local_timing": "fixed_calendar_growth_stage_alignment_unverified",
            "limitations": ["No crop labels assigned; no training or model inference performed",
                            "Numerical feature parity does not establish seasonal alignment or local accuracy"],
            "files": {name: {"sha256": references.checksum(staging / name)} for name in ("features.npz", "pixel_status.tif", "coverage.csv", "pixel_audit.csv")}}
        (staging / "feature_handoff.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(folder)
    load_feature_bundle(manifest_path, require_ready=False)
    return manifest_path


def load_feature_bundle(path, require_ready=True):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != "paoay_crop_features"
            or manifest.get("status") not in {"prepared", "pending_inputs"}
            or manifest.get("land_cover_filter_applied") is not False
            or set(manifest.get("files", {})) != {"features.npz", "pixel_status.tif", "coverage.csv", "pixel_audit.csv"}):
        raise ValueError("Unsupported or incomplete Paoay feature handoff")
    request = {key: manifest[key] for key in ("bundle_id", "identity", "guidance_handoff", "model_manifest", "imagery_handoff")}
    recipe, image = _verify(request)
    if (manifest.get("feature_columns") != recipe["feature_columns"] or manifest.get("feature_columns") != manifest["identity"]["feature_columns"]
            or manifest.get("period") != image["manifest"]["period"] or manifest.get("grid") != image["manifest"]["grid"]):
        raise ValueError("Saved feature schema, dates or grid changed")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved feature output changed: {name}; files preserved")
    with np.load(path.parent / "features.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    n = manifest["row_count"]
    if (set(values) != {"matrix", "rows", "columns", "x", "y", "sample_ids"}
            or values["matrix"].shape != (n, len(recipe["feature_columns"]))
            or not np.all(np.isfinite(values["matrix"]))
            or any(values[name].shape != (n,) for name in values if name != "matrix")
            or len(set(values["sample_ids"])) != n or manifest["status"] != ("prepared" if n else "pending_inputs")):
        raise ValueError("Feature matrix or pixel identities changed")
    grid = manifest["grid"]
    rows, columns = values["rows"], values["columns"]
    if (not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(columns.dtype, np.integer)
            or np.any((rows < 0) | (rows >= grid["height"]) | (columns < 0) | (columns >= grid["width"]))
            or len(set(zip(rows.tolist(), columns.tolist()))) != n):
        raise ValueError("Feature pixel locations are out of bounds or duplicated")
    transform = Affine(*grid["transform"])
    x, y = transform @ (columns + .5, rows + .5)
    if (not np.allclose(x, values["x"], atol=1e-8, rtol=0) or not np.allclose(y, values["y"], atol=1e-8, rtol=0)):
        raise ValueError("Feature coordinates differ from saved pixel centres")
    with rasterio.open(path.parent / "pixel_status.tif") as raster:
        if ((raster.height, raster.width) != (grid["height"], grid["width"])
                or raster.count != 1 or raster.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                or not raster.transform.almost_equals(transform)):
            raise ValueError("Feature status raster differs from the imagery grid")
        states = raster.read(1)
    if int((states == 5).sum()) != n or np.any(states[rows, columns] != 5):
        raise ValueError("Ready feature rows differ from the status raster")
    if require_ready and not n:
        raise ValueError("No ready feature pixels; inspect the coverage and exclusion audit")
    tables = {}
    for name in ("coverage.csv", "pixel_audit.csv"):
        with (path.parent / name).open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
    return {"manifest": manifest, "folder": path.parent, **values, "coverage": tables["coverage.csv"], "audit": tables["pixel_audit.csv"]}
