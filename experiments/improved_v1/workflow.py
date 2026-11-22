"""Improved pipeline: inputs -> selection -> models -> test -> Paoay map.

Kept from the main pipeline: the verified Preprocessing 01 quality handoff, locked
train/validation/test partitions, the crop/parcel/confidence training weights,
the Random Forest, parcel-level probability averaging, the 0.60/0.10 confidence
rule, sealed stage receipts and current-stage pointers, and a single external
test that never influences any choice.

Taken from the experiments: annual harmonic features (no window, no green-up
detection), Sentinel-1 backscatter and ratios, the omega 1.5 frequency option,
and model selection by spatial CV and dataset holdout on development data only.

Every stage writes a new folder under ``pipeline_outputs/improved/<stage>/`` and
publishes ``current_<stage>.json``; nothing in the main pipeline is changed.
"""

from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from crop_pipeline.analysis_workflow import resolve_stage_pointer, write_stage_pointer
from crop_pipeline.common import save_json
from experiments.improved_v1 import datasets
from experiments.pilots import data, earth_engine, evaluation, full
from crop_pipeline.training import earth_engine as scoring
from experiments.transfer import transfer
from crop_pipeline.windows import seasons

AREA = "improved"
CROPS = evaluation.CROPS
CANDIDATES = {
    "s2_harmonic": full.s2_harmonic,
    "s2_harmonic_all_radar": full.combine(full.s2_harmonic, full.s1_raw, full.s1_ratios),
    "s2_harmonic_omega1.5": full.s2_harmonic_omega(1.5),
    "s2_harmonic_omega1.5_all_radar": full.combine(full.s2_harmonic_omega(1.5), full.s1_raw, full.s1_ratios),
}
SELECTION_PARAMS = {**evaluation.PARAMS, "n_estimators": 300}
FINAL_PARAMS = {**evaluation.PARAMS, "n_estimators": 1000}
CONFIDENCE = {"minimum_top_probability": .60, "minimum_margin": .10}
MAP_CODES = {"carrot": 1, "cabbage": 2, "potato": 3, "Unknown crop type": 4, "No features": 5}


# Stage bookkeeping (same receipt idea as the main pipeline)

def _root(repository):
    return Path(repository).resolve()


def _pointer(repository, stage, variant=None):
    """``current_<stage>.json``; a dataset variant gets its own ``current_<stage>__<variant>.json``."""
    suffix = f"__{variant}" if variant else ""
    return _root(repository) / "pipeline_outputs" / AREA / f"current_{stage}{suffix}.json"


def _new_run(repository, stage):
    parent = _root(repository) / "pipeline_outputs" / AREA / stage
    for path in (parent.parent.parent, parent.parent, parent):
        if path.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {path}")
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"run_{uuid4().hex[:12]}"
    run.mkdir(exist_ok=False)
    return run


def seal(folder, stage, **metadata):
    receipt = Path(folder) / f"{stage}_receipt.json"
    save_json(receipt, {"kind": "improved_pipeline_stage", "stage": stage, **metadata,
                        "files": {str(p.relative_to(folder)): seasons.file_sha256(p)
                                  for p in sorted(Path(folder).rglob("*")) if p.is_file() and p != receipt}})
    return receipt


def verify(receipt, stage):
    receipt = Path(receipt).resolve()
    record = json.loads(receipt.read_text())
    if record.get("kind") != "improved_pipeline_stage" or record.get("stage") != stage or not record.get("files"):
        raise ValueError(f"Expected a completed improved-pipeline {stage} receipt")
    for name, digest in record["files"].items():
        path = receipt.parent / name
        if not path.is_file() or seasons.file_sha256(path) != digest:
            raise ValueError(f"Saved {stage} file changed: {name}")
    return receipt, record


def _start(repository, stage, variant=None):
    write_stage_pointer(_pointer(repository, stage, variant), status="pending", reason=f"{stage} attempt has not completed")


def _publish(repository, stage, receipt, variant=None):
    write_stage_pointer(_pointer(repository, stage, variant), receipt)
    return receipt


def _input(repository, explicit, stage, variant=None):
    path = Path(explicit) if explicit else resolve_stage_pointer(_pointer(repository, stage, variant))
    return verify(path if path.is_absolute() else _root(repository) / path, stage)


def _development_variant(repository, rows, arrays, variant, origins_handoff=None):
    """Rows and arrays for a dataset variant, with the receipt fields that record it.

    ``variant=None`` keeps every row, as before variants existed.
    """
    if variant is None:
        return rows, arrays, {}
    receipt, origins = datasets.load_origins(repository, origins_handoff)
    rows, arrays = datasets.subset(rows, arrays, datasets.variant_mask(rows, origins, variant))
    return rows, arrays, {"dataset_variant": variant, "origins_handoff": str(receipt)}


# Pure steps (tested without Earth Engine or the file system)

def development_index(rows, ok):
    return np.asarray([i for i, (r, good) in enumerate(zip(rows, ok)) if good and r["partition"] in ("train", "validation")])


def candidate_scores(rows, s2, s2_valid, s1, candidates=CANDIDATES, seeds=(1, 2), params=SELECTION_PARAMS, progress=print):
    """Development-only views per candidate on pixels usable by every candidate. Test rows are never fitted or scored."""
    built = {name: build(s2, s2_valid, s1) for name, build in candidates.items()}
    common = np.logical_and.reduce([ok for _, ok, _ in built.values()])
    development_only = common & np.asarray([r["partition"] in ("train", "validation") for r in rows])
    records = []
    for name, (X, _, _) in built.items():
        X = np.nan_to_num(X)
        for seed in seeds:
            random = evaluation.random_split(X, rows, development_only, params, seed)
            spatial = evaluation.spatial_cv(X, rows, development_only, params, seed)
            holdout = [h for h in evaluation.dataset_holdout(X, rows, development_only, params, seed) if h["status"] == "scored"]
            records.append({"candidate": name, "seed": seed, "predictors": X.shape[1], "pixels": int(development_only.sum()),
                            "random_split_macro_f1": random["macro_f1"], "spatial_cv_macro_f1": spatial["macro_f1"],
                            "dataset_holdout_macro_f1": float(np.mean([h["macro_f1_scored_crops"] for h in holdout]))})
        if progress:
            progress(f"{name} scored")
    return records


def choose(records):
    """Highest mean of spatial CV and dataset holdout (both measure transfer); ties go to fewer predictors."""
    by_name = {}
    for record in records:
        by_name.setdefault(record["candidate"], []).append(record)
    summary = {name: {"selection_score": float(np.mean([(r["spatial_cv_macro_f1"] + r["dataset_holdout_macro_f1"]) / 2 for r in rs])),
                      "predictors": rs[0]["predictors"]} for name, rs in by_name.items()}
    best = max(summary, key=lambda n: (round(summary[n]["selection_score"], 6), -summary[n]["predictors"]))
    return best, summary


def fit_final(rows, s2, s2_valid, s1, name, params=FINAL_PARAMS, seed=42):
    X, ok, columns = CANDIDATES[name](s2, s2_valid, s1)
    index = development_index(rows, ok)
    model = evaluation.fit_rows(np.nan_to_num(X), rows, index, params, seed)
    return model, columns, sorted({rows[i]["partition_group"] for i in index})


def meets_criteria(metrics, baseline_macro):
    """Main-pipeline rule: macro-F1 >= 0.50, above the baseline, and every crop F1 >= 0.50 (needs all crops)."""
    if metrics.get("macro_f1") is None or any(v["support"] == 0 for v in metrics["per_crop"].values()):
        return None
    return bool(metrics["macro_f1"] >= .5 and metrics["macro_f1"] > baseline_macro
                and all(v["f1"] >= .5 for v in metrics["per_crop"].values()))


def score_test(model, name, rows, s2, s2_valid, s1, development_groups):
    """Score each test population (JECAM, every supplement, and the combined pool) separately.

    Populations missing a crop report present-crop F1; their three-class macro-F1 is unavailable.
    """
    X, ok, _ = CANDIDATES[name](s2, s2_valid, s1)
    index = np.asarray([i for i, (r, good) in enumerate(zip(rows, ok)) if good and r["partition"] == "test"])
    test_rows = [rows[i] for i in index]
    if set(development_groups) & {r["partition_group"] for r in test_rows}:
        raise ValueError("Test parcel overlaps a development parcel")
    cohorts = {}
    for r in test_rows:
        if cohorts.setdefault(r["partition_group"], r.get("test_cohort") or "jecam") != (r.get("test_cohort") or "jecam"):
            raise ValueError("A test parcel appears in more than one test population")
    predictions = evaluation.parcel_predictions(model, np.nan_to_num(X[index]), test_rows)
    for p in predictions:
        p["test_cohort"] = cohorts[p["partition_group"]]
    majority = Counter({r["partition_group"]: r["crop"] for r in rows if r["partition_group"] in set(development_groups)}.values()).most_common(1)[0][0]
    populations = sorted({p["test_cohort"] for p in predictions}, key=lambda c: (c != "jecam", c))
    if len(populations) > 1:
        populations.append("combined")
    results = {}
    for population in populations:
        chosen = [p for p in predictions if population == "combined" or p["test_cohort"] == population]
        metrics = transfer.scores(chosen)
        complete = all(v["support"] > 0 for v in metrics["per_crop"].values())
        baseline = scoring.score_predictions(chosen, [majority] * len(chosen), CROPS)
        if not complete:
            metrics["macro_f1"] = None
            metrics["balanced_accuracy"] = None
        results[population] = {"parcels": len(chosen), "crop_coverage": "complete" if complete else "incomplete",
                               "metrics": metrics, "majority_baseline_macro_f1": baseline["macro_f1"] if complete else None,
                               "meets_f1_criteria": meets_criteria(metrics, baseline["macro_f1"])}
    return results, predictions, {"training_majority_class": majority}


def classify(model, name, s2, s2_valid, s1, rule=CONFIDENCE):
    """Labels and probabilities; low-confidence pixels become Unknown, pixels without features stay separate."""
    X, ok, _ = CANDIDATES[name](s2, s2_valid, s1)
    probabilities = model.predict_proba(np.nan_to_num(X))
    ordered = np.sort(probabilities, axis=1)
    accepted = ok & (ordered[:, -1] >= rule["minimum_top_probability"]) & (ordered[:, -1] - ordered[:, -2] >= rule["minimum_margin"])
    labels = np.where(~ok, "No features", np.where(accepted, model.classes_[np.argmax(probabilities, axis=1)], "Unknown crop type"))
    return labels, probabilities, accepted


# Stages

def build_inputs(repository, quality_handoff=None, pixels_per_parcel=5, ee=None, progress=print):
    """All parcels from the verified Preprocessing 01 handoff, annual Sentinel-2 series and Sentinel-1 at the same pixels."""
    _start(repository, "inputs")
    root = _root(repository)
    quality_path, quality = data.load_quality(root, quality_handoff)
    regions = data.sample_parcels(quality["regions"], {"train": None, "validation": None, "test": None},
                                  include_supplements=True)
    rows, s2_values, s2_valid = data.load_series(regions, quality["pixel_paths"], pixels_per_parcel)
    if ee is None:
        settings = json.loads((root / "settings.local.json").read_text())
        ee = earth_engine.initialize(settings["earth_engine_project"])
    cache = root / "pipeline_outputs" / AREA / "cache" / "sentinel1"
    s1_values = earth_engine.sample_s1(ee, rows, cache, progress=progress)
    run = _new_run(repository, "inputs")
    data.save(run, rows, {"s2_values": s2_values, "s2_valid": s2_valid, "s1_values": s1_values})
    receipt = seal(run, "inputs", quality_handoff=str(quality_path), quality_sha256=seasons.file_sha256(quality_path),
                   pixels_per_parcel=pixels_per_parcel, pixels=len(rows))
    return _publish(repository, "inputs", receipt)


def select(repository, inputs_handoff=None, seeds=(1, 2), progress=print, variant=None, origins_handoff=None):
    """Score candidates on development data; ``variant`` ("groupmate", "yours", "combined") limits it to that dataset."""
    _start(repository, "selection", variant)
    inputs, _ = _input(repository, inputs_handoff, "inputs")
    rows, arrays = data.load(inputs.parent)
    rows, arrays, chosen = _development_variant(repository, rows, arrays, variant, origins_handoff)
    records = candidate_scores(rows, arrays["s2_values"], arrays["s2_valid"], arrays["s1_values"], seeds=seeds, progress=progress)
    best, summary = choose(records)
    run = _new_run(repository, "selection")
    seasons.write_records(run / "candidate_scores.csv", records)
    save_json(run / "selection.json", {"selected": best, "rule": "max mean(spatial CV, dataset holdout) on development data",
                                       "summary": summary, "test_rows_used": False, **chosen})
    receipt = seal(run, "selection", inputs_handoff=str(inputs), **chosen)
    return _publish(repository, "selection", receipt, variant)


def freeze(repository, selection_handoff=None, variant=None):
    """Refit the selected candidate; the dataset variant always comes from the selection receipt."""
    _start(repository, "models", variant)
    selection, record = _input(repository, selection_handoff, "selection", variant)
    if record.get("dataset_variant") != variant:
        raise ValueError(f"Selection was made for variant {record.get('dataset_variant')!r}, not {variant!r}")
    choice = json.loads((selection.parent / "selection.json").read_text())["selected"]
    inputs, _ = verify(record["inputs_handoff"], "inputs")
    rows, arrays = data.load(inputs.parent)
    rows, arrays, chosen = _development_variant(repository, rows, arrays, variant, record.get("origins_handoff"))
    model, columns, groups = fit_final(rows, arrays["s2_values"], arrays["s2_valid"], arrays["s1_values"], choice)
    run = _new_run(repository, "models")
    joblib.dump({"classifier": model, "candidate": choice, "feature_columns": columns}, run / "model.joblib")
    save_json(run / "model.json", {"candidate": choice, "params": FINAL_PARAMS, "feature_columns": columns,
                                   "trained_on": "train + validation partitions", "development_groups": groups,
                                   "confidence_rule": CONFIDENCE, **chosen})
    receipt = seal(run, "models", selection_handoff=str(selection), inputs_handoff=str(inputs), **chosen)
    return _publish(repository, "models", receipt, variant)


def load_model(models_receipt):
    receipt, record = verify(models_receipt, "models")
    package = joblib.load(receipt.parent / "model.joblib")
    return package["classifier"], package["candidate"], json.loads((receipt.parent / "model.json").read_text()), record


def test(repository, models_handoff=None, variant=None):
    """Score once on every external test population; these are the same parcels for every variant."""
    _start(repository, "test", variant)
    models, _ = _input(repository, models_handoff, "models", variant)
    model, name, info, record = load_model(models)
    if record.get("dataset_variant") != variant:
        raise ValueError(f"Model was trained for variant {record.get('dataset_variant')!r}, not {variant!r}")
    inputs, _ = verify(record["inputs_handoff"], "inputs")
    rows, arrays = data.load(inputs.parent)
    results, predictions, baseline = score_test(model, name, rows, arrays["s2_values"], arrays["s2_valid"],
                                                arrays["s1_values"], info["development_groups"])
    chosen = {"dataset_variant": variant} if variant else {}
    run = _new_run(repository, "test")
    seasons.write_records(run / "test_parcels.csv", predictions)
    save_json(run / "test_scores.json", {"candidate": name, "populations": results, **baseline, **chosen})
    receipt = seal(run, "test", models_handoff=str(models), **chosen)
    return _publish(repository, "test", receipt, variant)


def paoay_inputs(repository, ee=None, roi_name=None, context_names=None, cache_name="paoay", cropland_source="land_activity"):
    """Study pixels (annual S2, S1 at the same grid), cropland flags and reference areas.

    Defaults give the Paoay study area with the land-activity cropland map. ``roi_name`` picks another KMZ
    boundary (for example "Atok"); ``cropland_source="worldcover"`` flags ESA WorldCover cropland instead,
    for areas the land-activity map does not cover. Radar is cached per ``cache_name`` and year.
    """
    import rasterio
    from rasterio.warp import transform as reproject_points
    from crop_pipeline.paoay import aligned

    root = _root(repository)
    settings = json.loads((root / "settings.local.json").read_text())

    def local(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    kmz_root = local(settings.get("paoay_kmz_root", "./data/paoay"))
    kmz_paths = [kmz_root / n for n in ("00 - Region of Interest.kmz", "00 - Atok Boundary.kmz", "Crop Area.kmz")]
    imagery_root = local(settings["atok_imagery_root"])
    with rasterio.open(next(iter(sorted(imagery_root.glob("*/Sentinel2_CSPlus_*.tif"))))) as source:
        crs = source.crs.to_string()
    roi_name = roi_name or settings.get("paoay_roi_name") or "Region of Interest"
    context_names = list(settings.get("paoay_context_names", []) if context_names is None else context_names)
    roi, areas, _ = aligned.study_area(kmz_paths, roi_name, [n for n in context_names if n != roi_name], crs)
    grid, mask = aligned.grid_for(imagery_root, roi)
    year = aligned.choose_year(aligned.year_completeness(imagery_root, list(range(2019, 2026)), grid, mask))
    cube, _ = aligned.read_year(imagery_root, year, grid)
    if ee is None:
        ee = earth_engine.initialize(settings["earth_engine_project"])
    s1_cube = earth_engine.s1_grid(ee, year, grid, root / "pipeline_outputs" / AREA / "cache" / f"{cache_name}_{year}")
    pixels = np.argwhere(mask)
    s2 = (cube.astype(float) * .0001)[:, :, mask].transpose(2, 0, 1)
    s1 = s1_cube.astype(float)[:, :, mask].transpose(2, 0, 1)
    if cropland_source == "worldcover":
        cover = earth_engine.worldcover_grid(ee, grid, root / "pipeline_outputs" / AREA / "cache" / cache_name)
        cropland = cover[tuple(pixels.T)] == earth_engine.WORLDCOVER_CROPLAND
    elif cropland_source == "land_activity":
        stage = resolve_stage_pointer(root / "sen4agrinet_work/paoay/land_activity/land_activity_stage.json")
        centres = [rasterio.Affine(*grid["transform"]) * (c + .5, r + .5) for r, c in pixels]
        with rasterio.open(stage.parent / "land_activity.tif") as raster:
            xs, ys = reproject_points(grid["crs"], raster.crs, [x for x, _ in centres], [y for _, y in centres])
            cropland = np.asarray([v[0] for v in raster.sample(zip(xs, ys))]) == 1
    else:
        raise ValueError(f"Unknown cropland source: {cropland_source}")
    return {"root": root, "grid": grid, "mask": mask, "pixels": pixels, "year": year, "areas": areas,
            "s2": s2, "s1": s1, "cropland": cropland, "roi_name": roi_name, "cropland_source": cropland_source}


def write_paoay_outputs(run, model, name, inputs):
    """Crop map, probabilities, per-area counts and a summary against Benguet statistics, written to ``run``."""
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    root, grid, mask, pixels = inputs["root"], inputs["grid"], inputs["mask"], inputs["pixels"]
    s2, s1, cropland, areas, year = inputs["s2"], inputs["s1"], inputs["cropland"], inputs["areas"], inputs["year"]
    labels, probabilities, accepted = classify(model, name, s2, np.isfinite(s2).all(axis=2), s1)
    profile = {"driver": "GTiff", "width": mask.shape[1], "height": mask.shape[0], "crs": grid["crs"],
               "transform": rasterio.Affine(*grid["transform"]), "compress": "deflate"}
    code = np.zeros(mask.shape, "uint8")
    code[tuple(pixels.T)] = [MAP_CODES[l] for l in labels]
    crop_band = np.zeros(mask.shape, "uint8")
    crop_band[tuple(pixels.T)] = cropland.astype("uint8") + 1
    with rasterio.open(run / "crop_map.tif", "w", count=2, dtype="uint8", nodata=0, **profile) as out:
        out.write(code, 1)
        out.write(crop_band, 2)
        out.descriptions = ("class_code", f"{inputs.get('cropland_source', 'land_activity')}_cropland (2=yes, 1=no)")
    probability = np.full((len(model.classes_),) + mask.shape, np.nan, "float32")
    for k in range(len(model.classes_)):
        probability[k][tuple(pixels.T)] = probabilities[:, k]
    with rasterio.open(run / "probabilities.tif", "w", count=len(model.classes_), dtype="float32", nodata=np.nan, **profile) as out:
        out.write(probability)
        out.descriptions = tuple(str(c) for c in model.classes_)
    statistics_path = root / "data/paoay/benguet_vegetable_area_2024.csv"
    rows = []
    raster_labels = np.full(mask.shape, "", dtype=object)
    raster_labels[tuple(pixels.T)] = labels
    for area in [{"name": "Whole study boundary", "geometry": None}, {"name": "Active cropland", "geometry": "cropland"}, *areas]:
        if area["geometry"] is None:
            inside = mask
        elif area["geometry"] == "cropland":
            inside = np.zeros(mask.shape, bool)
            inside[tuple(pixels[cropland].T)] = True
        else:
            inside = ~geometry_mask([mapping(area["geometry"])], out_shape=mask.shape,
                                    transform=rasterio.Affine(*grid["transform"])) & mask
        counts = Counter(raster_labels[inside].tolist())
        rows.append({"area": area["name"], "pixels": int(inside.sum()), **{k: counts.get(k, 0) for k in MAP_CODES}})
    seasons.write_records(run / "area_summary.csv", rows)
    confident = Counter(labels[cropland & accepted].tolist())
    total = sum(confident.values()) or 1
    statistics = {}
    if statistics_path.is_file():
        import csv
        with statistics_path.open() as stream:
            table = list(csv.DictReader(stream))
        area_total = sum(float(r["area_ha"]) for r in table)
        statistics = {r["crop"]: float(r["area_ha"]) / area_total for r in table}
    save_json(run / "paoay_summary.json", {
        "candidate": name, "year": year, "study_area": inputs.get("roi_name"),
        "cropland_source": inputs.get("cropland_source", "land_activity"), "study_pixels": int(mask.sum()),
        "share_with_features": float(np.mean(labels != "No features")),
        "cropland_pixels": int(cropland.sum()), "confident_share_in_cropland": float(accepted[cropland].mean()),
        "crop_shares_of_confident_cropland": {c: confident.get(c, 0) / total for c in CROPS},
        "benguet_2024_shares": statistics, "status": "exploratory_unvalidated_no_local_labels"})
    return run


def paoay_map(repository, models_handoff=None, ee=None):
    """Apply the frozen model to every Paoay study pixel; flag cropland with the land-activity map."""
    _start(repository, "paoay")
    models, _ = _input(repository, models_handoff, "models")
    model, name, _, _ = load_model(models)
    inputs = paoay_inputs(repository, ee)
    run = write_paoay_outputs(_new_run(repository, "paoay"), model, name, inputs)
    receipt = seal(run, "paoay", models_handoff=str(models))
    return _publish(repository, "paoay", receipt)


def atok_map(repository, models_handoff=None, ee=None, roi_name="Atok"):
    """The frozen model on the wider KMZ "Atok" boundary, with Sentinel-1 and WorldCover cropland.

    Writes ``pipeline_outputs/improved/atok/`` and ``current_atok.json``; the Paoay map stage is unchanged.
    The Paoay study area appears as the "Region of Interest" row of the area table.
    """
    _start(repository, "atok")
    models, _ = _input(repository, models_handoff, "models")
    model, name, _, _ = load_model(models)
    inputs = paoay_inputs(repository, ee, roi_name=roi_name, cache_name="atok_boundary", cropland_source="worldcover")
    run = write_paoay_outputs(_new_run(repository, "atok"), model, name, inputs)
    receipt = seal(run, "atok", models_handoff=str(models), study_area=roi_name, cropland_source="worldcover")
    return _publish(repository, "atok", receipt)
