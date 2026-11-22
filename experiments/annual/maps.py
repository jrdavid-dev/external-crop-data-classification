"""Preliminary Paoay and Atok maps from the frozen annual model.

Every study pixel in the chosen year gets annual features from the 24 half-month
Sentinel-2 composites and Sentinel-1 on the same grid. Pixels without enough clear
half-months stay "No features"; a prediction is accepted when its top probability is at
least 0.60 and leads the second by at least 0.10, otherwise "Unknown crop type". The
map also flags pixels with more than 20% of predictors outside the training 5th-95th
percentile range. Cropland is the land-activity map in Paoay and ESA WorldCover 2021
cropland for the wider Atok boundary. Without dated local labels the maps are unvalidated.
"""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.common import save_json
from experiments.annual import sentinel1
from experiments.annual import workflow as workflow
from crop_pipeline.windows import seasons

CROPS = workflow.CROPS
CONFIDENCE = {"minimum_top_probability": .60, "minimum_margin": .10}
RANGE_RULE = {"low_percentile": 5, "high_percentile": 95, "maximum_share_outside": .20}
MAP_CODES = {"carrot": 1, "cabbage": 2, "potato": 3, "Unknown crop type": 4, "No features": 5}
WORLDCOVER = "ESA/WorldCover/v200"
WORLDCOVER_CROPLAND = 40
AREAS = {
    "paoay": {"roi": None, "cropland": "land_activity", "cache": "paoay"},
    "atok": {"roi": "Atok", "cropland": "worldcover", "cache": "atok_boundary"},
}


def classify(model, X, ok, rule=CONFIDENCE):
    """Labels, class probabilities and the accepted flag for each pixel."""
    probabilities = np.zeros((len(X), len(CROPS)))
    probabilities[:, [CROPS.index(c) for c in model.classes_]] = model.predict_proba(np.nan_to_num(X))
    ordered = np.sort(probabilities, axis=1)
    accepted = ok & (ordered[:, -1] >= rule["minimum_top_probability"]) & (
        ordered[:, -1] - ordered[:, -2] >= rule["minimum_margin"])
    best = np.asarray(CROPS)[np.argmax(probabilities, axis=1)]
    labels = np.where(~ok, "No features", np.where(accepted, best, "Unknown crop type"))
    return labels, probabilities, accepted


def outside_training_range(X, training, rule=RANGE_RULE):
    """Share of predictors outside the training percentile range, and the flag above the allowed share."""
    low, high = np.percentile(training, [rule["low_percentile"], rule["high_percentile"]], axis=0)
    with np.errstate(invalid="ignore"):
        share = ((X < low) | (X > high)).mean(axis=1)
    return share, share > rule["maximum_share_outside"]


def worldcover_grid(ee, grid, cache):
    path = Path(cache) / "worldcover_grid.npy"
    if not path.is_file():
        if ee is None:
            raise RuntimeError(f"WorldCover cache {path} is missing and no Earth Engine client was given")
        _, _, width, height = grid["window"]
        a, b, c, d, e, f = grid["transform"]
        structured = ee.data.computePixels({
            "expression": ee.ImageCollection(WORLDCOVER).first().select("Map"), "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": width, "height": height},
                     "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e,
                                         "translateY": f}, "crsCode": grid["crs"]}})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(structured["Map"]).astype("uint8"))
    return np.load(path)


def study_inputs(root, area, ee=None):
    """Study pixels: annual Sentinel-2 [n, 24, 5], Sentinel-1 [n, 24, 2], cropland flags and reference areas."""
    import rasterio
    from rasterio.warp import transform as reproject_points
    from crop_pipeline.paoay import aligned

    spec = AREAS[area]
    settings = json.loads((root / "settings.local.json").read_text())

    def local(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    kmz_root = local(settings.get("paoay_kmz_root", "./data/paoay"))
    kmz = [kmz_root / n for n in ("00 - Region of Interest.kmz", "00 - Atok Boundary.kmz", "Crop Area.kmz")]
    imagery = local(settings["atok_imagery_root"])
    with rasterio.open(next(iter(sorted(imagery.glob("*/Sentinel2_CSPlus_*.tif"))))) as source:
        crs = source.crs.to_string()
    roi_name = spec["roi"] or settings.get("paoay_roi_name") or "Region of Interest"
    context = [n for n in settings.get("paoay_context_names", []) if n != roi_name]
    roi, areas, kmz_sources = aligned.study_area(kmz, roi_name, context, crs)
    grid, mask = aligned.grid_for(imagery, roi)
    year = aligned.choose_year(aligned.year_completeness(imagery, list(range(2019, 2026)), grid, mask))
    cube, _ = aligned.read_year(imagery, year, grid)
    cache = root / "pipeline_outputs" / workflow.AREA / "cache" / f"{spec['cache']}_{year}"
    if ee is None and not (cache / f"s1_grid_{year}.npy").is_file():
        ee = sentinel1.initialize(settings["earth_engine_project"])
    s1_cube = sentinel1.grid(ee, year, grid, cache)
    pixels = np.argwhere(mask)
    s2 = (cube.astype(float) * .0001)[:, :, mask].transpose(2, 0, 1)
    s1 = s1_cube.astype(float)[:, :, mask].transpose(2, 0, 1)
    if spec["cropland"] == "worldcover":
        if ee is None and not (cache / "worldcover_grid.npy").is_file():
            ee = sentinel1.initialize(settings["earth_engine_project"])
        cropland = worldcover_grid(ee, grid, cache)[tuple(pixels.T)] == WORLDCOVER_CROPLAND
    else:
        stage = resolve_stage_pointer(root / "sen4agrinet_work/paoay/land_activity/land_activity_stage.json")
        centres = [rasterio.Affine(*grid["transform"]) * (c + .5, r + .5) for r, c in pixels]
        with rasterio.open(stage.parent / "land_activity.tif") as raster:
            xs, ys = reproject_points(grid["crs"], raster.crs, [x for x, _ in centres], [y for _, y in centres])
            cropland = np.asarray([v[0] for v in raster.sample(zip(xs, ys))]) == 1
    return {"grid": grid, "mask": mask, "pixels": pixels, "year": year, "areas": areas, "roi_name": roi_name,
            "s2": s2, "s1": s1, "cropland": cropland, "cropland_source": spec["cropland"], "kmz_sources": kmz_sources}


def training_matrix(record, recipe):
    inputs, _ = workflow.verify(record["inputs_handoff"], "inputs")
    rows, arrays = workflow.load_inputs(inputs.parent)
    X, ok, _ = workflow.recipe_matrix(recipe, arrays)
    return X[[i for i, r in enumerate(rows) if ok[i] and r["partition"] == "train"]]


def write_outputs(run, package, training, inputs, statistics_path=None):
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    grid, mask, pixels, cropland = inputs["grid"], inputs["mask"], inputs["pixels"], inputs["cropland"]
    s2 = inputs["s2"]
    X, ok, columns = workflow.RECIPES[package["recipe"]](s2, np.isfinite(s2).all(axis=2), inputs["s1"])
    if columns != package["feature_columns"]:
        raise ValueError("Study features differ from the frozen feature columns")
    labels, probabilities, accepted = classify(package["classifier"], X, ok)
    share, flagged = outside_training_range(X, training)
    flagged &= ok
    profile = {"driver": "GTiff", "width": mask.shape[1], "height": mask.shape[0], "crs": grid["crs"],
               "transform": rasterio.Affine(*grid["transform"]), "compress": "deflate"}
    bands = np.zeros((3,) + mask.shape, "uint8")
    bands[0][tuple(pixels.T)] = [MAP_CODES[label] for label in labels]
    bands[1][tuple(pixels.T)] = cropland.astype("uint8") + 1
    bands[2][tuple(pixels.T)] = flagged.astype("uint8") + 1
    with rasterio.open(run / "crop_map.tif", "w", count=3, dtype="uint8", nodata=0, **profile) as out:
        out.write(bands)
        out.descriptions = ("class_code", f"{inputs['cropland_source']}_cropland (2=yes, 1=no)",
                            "outside_training_range (2=yes, 1=no)")
    probability = np.full((len(CROPS),) + mask.shape, np.nan, "float32")
    for k in range(len(CROPS)):
        probability[k][tuple(pixels.T)] = probabilities[:, k]
    with rasterio.open(run / "probabilities.tif", "w", count=len(CROPS), dtype="float32", nodata=np.nan,
                       **profile) as out:
        out.write(probability)
        out.descriptions = tuple(CROPS)
    raster_labels = np.full(mask.shape, "", dtype=object)
    raster_labels[tuple(pixels.T)] = labels
    rows = []
    for area in [{"name": "Whole study boundary", "geometry": None}, {"name": "Cropland", "geometry": "cropland"},
                 *inputs["areas"]]:
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
    if statistics_path and Path(statistics_path).is_file():
        with Path(statistics_path).open() as stream:
            table = list(csv.DictReader(stream))
        area_total = sum(float(r["area_ha"]) for r in table)
        statistics = {r["crop"]: float(r["area_ha"]) / area_total for r in table}
    summary = {
        "model_recipe": package["recipe"], "year": inputs["year"], "study_area": inputs["roi_name"],
        "cropland_source": inputs["cropland_source"], "study_pixels": int(mask.sum()),
        "share_with_features": float(ok.mean()), "cropland_pixels": int(cropland.sum()),
        "confident_share_in_cropland": float(accepted[cropland].mean()) if cropland.any() else None,
        "outside_training_range_share": float(flagged[ok].mean()) if ok.any() else None,
        "crop_shares_of_confident_cropland": {c: confident.get(c, 0) / total for c in CROPS},
        "benguet_2024_area_shares": statistics, "confidence_rule": CONFIDENCE, "range_rule": RANGE_RULE,
        "kmz_sources": inputs["kmz_sources"], "status": "preliminary_unvalidated_no_local_labels"}
    save_json(run / "map_summary.json", summary)
    return summary


def map_area(repository, area="paoay", models_handoff=None, ee=None):
    """Apply the frozen annual model to the Paoay study area or the wider Atok boundary."""
    if area not in AREAS:
        raise ValueError(f"Unknown study area {area!r}; use one of {sorted(AREAS)}")
    root = workflow._root(repository)
    workflow._start(repository, area)
    models, _ = workflow._input(repository, models_handoff, "models")
    package, _, record = workflow.load_model(models)
    training = training_matrix(record, package["recipe"])
    inputs = study_inputs(root, area, ee)
    run = workflow._new_run(repository, area)
    write_outputs(run, package, training, inputs, root / "data/paoay/benguet_vegetable_area_2024.csv")
    receipt = workflow.seal(run, area, models_handoff=str(models), study_area=inputs["roi_name"],
                            cropland_source=inputs["cropland_source"], year=inputs["year"])
    return workflow._publish(repository, area, receipt)
