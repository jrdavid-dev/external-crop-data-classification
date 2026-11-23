"""Barangay-wide imagery audit and farmland-first frozen-model application."""

from calendar import month_name
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import shape

from crop_pipeline.paoay import dynamic_world, features, imagery, land_cover, predictions, references

NAME = "Paoay, Atok, Benguet"
BOUNDARY_URL = "https://www.openstreetmap.org/relation/20601702"
PREDICTION_CODES = {
    1: "Carrot", 2: "Cabbage", 3: "Potato", 4: "Farmland, crop unresolved",
    5: "Farmland, imagery unavailable", 6: "Supported other cover", 7: "Land cover unresolved",
}
COVER_CODES = {1: "Supported farmland", 2: "Supported other cover", 3: "Land cover unresolved"}
MINIMUM_SUPPORTED_BINS = 4
REGIONAL_CROPLAND_URL = "https://zenodo.org/records/19303406/files/SEA_Cropland_2024.tif"
REGIONAL_CROPLAND_SOURCE = "https://doi.org/10.5281/zenodo.19303406"
NODATA = 0


def boundary_reference(boundary_path, output):
    """Create a separate reference bundle from one named barangay GeoJSON feature."""
    path = Path(boundary_path).resolve()
    digest = references.checksum(path)
    data = json.loads(path.read_text())
    items = data.get("features", [data])
    if len(items) != 1 or items[0].get("type") != "Feature":
        raise ValueError("Provide one GeoJSON feature for Barangay Paoay, Atok")
    feature = items[0]
    geometry = shape(feature["geometry"])
    west, south, east, north = geometry.bounds
    if (geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid
            or geometry.is_empty or not (120.65 < west < east < 120.85 and 16.50 < south < north < 16.75)):
        raise ValueError("Invalid or unexpected Paoay polygon; use WGS84 longitude/latitude")
    properties = feature.get("properties", {})
    label = properties.get("display_name", properties.get("name", ""))
    if not all(name in label.casefold() for name in ("paoay", "atok", "benguet")):
        raise ValueError("Boundary feature must name Paoay, Atok, and Benguet")
    record = {"reference_id": digest[:20], "name": NAME, "source_file": path.name,
              "geometry": feature["geometry"], "geometry_status": "valid", "notes": "",
              "source_url": data.get("source", properties.get("source_url", ""))}
    source = {"name": path.name, "path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return references.export_reference_bundle([record], [source], output, roi_name=NAME)


def date_candidates(today):
    return [f"{year}-{month:02d}-01" for year in range(2024, 2027) for month in (1, 5)
            if date(year, month + 4, 1) <= today]


def _audit_window(reference, start, output, ee, today):
    plan = imagery.prepare_request(reference, start, output / "sentinel2", today=today)
    retriever, acquisitions = imagery._inventory(plan, ee)
    retriever.download_observation_diagnostics = False
    geom = ee.Geometry(plan["fragment"]["geometry"], proj="EPSG:4326", geodesic=False)
    grid = plan["settings"]["grid"]
    names, valid, rows = [], [], []
    for number, interval in enumerate(plan["period"]["intervals"]):
        name = f"bin_{number + 1}"
        original = [r for r in acquisitions if interval["start"] <= r["date"] < interval["end"]]
        image = retriever.make_composite(plan["fragment"], interval, original)
        valid.append(image.select("clear_count").gt(0).rename(name))
        names.append(name)
        rows.append({"window_start": start, "window_end_exclusive": plan["period"]["end"],
                     "interval": interval["name"], "start": interval["start"], "end": interval["end"],
                     "sr_dates": len({r["date"] for r in original}),
                     "cloud_matched_dates": len({r["date"] for r in original if r.get("cloud_probability_id")})})
    cube = ee.Image.cat(*valid)
    stats = cube.addBands(cube.reduce(ee.Reducer.min()).rename("complete")).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, crs=grid["crs"],
        crsTransform=grid["transform"], maxPixels=10_000_000, tileScale=2,
    ).getInfo()
    for name, row in zip(names, rows, strict=True):
        row["clear_roi_fraction"] = float(stats[name])
    summary = {"start": start, "end_exclusive": plan["period"]["end"],
               "weakest_bin_clear_fraction": min(row["clear_roi_fraction"] for row in rows),
               "all_eight_clear_fraction": float(stats["complete"]),
               "bins_at_least_80pct_clear": sum(row["clear_roi_fraction"] >= .8 for row in rows),
               "scope": "whole_barangay_before_land_cover_mask"}
    print(f"{start}: weakest bin {summary['weakest_bin_clear_fraction']:.1%}, "
          f"all eight clear {summary['all_eight_clear_fraction']:.1%}", flush=True)
    return summary, rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(reference, output, ee, today=None, starts=None):
    """Evaluate real cloud-masked clear pixels, without requesting image downloads."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    starts = list(date_candidates(today) if starts is None else starts)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda start: _audit_window(reference, start, output, ee, today), starts))
    summaries = [item[0] for item in results]
    write_csv(output / "window_audit.csv", summaries)
    write_csv(output / "half_month_audit.csv", [row for _, rows in results for row in rows])
    return summaries


def farmland_mask(guidance, roi):
    """Require four supported bins and a strict majority of crop votes."""
    if roi.dtype != bool or guidance.shape != (8, *roi.shape):
        raise ValueError("Expected eight land-cover guidance bins on the barangay grid")
    crop = (guidance == 3).sum(axis=0)
    other = (guidance == 4).sum(axis=0)
    adequate = roi & (crop + other >= MINIMUM_SUPPORTED_BINS)
    cover = np.zeros(roi.shape, dtype=np.uint8)
    cover[roi] = 3
    cover[adequate & (crop > other)] = 1
    cover[adequate & (other > crop)] = 2
    return cover


def regional_cropland(grid, output, *, download=False):
    """Read only the local window of the published cropland TIFF via HTTP ranges."""
    import requests
    from rasterio import windows
    from rasterio.warp import reproject, Resampling, transform_bounds

    identity = {"grid": grid, "source": REGIONAL_CROPLAND_URL, "year": 2024}
    folder = Path(output) / imagery._digest(identity)[:20]
    target = folder / "cropland.tif"
    receipt = folder / "receipt.json"
    if folder.exists():
        meta = json.loads(receipt.read_text())
        if meta["identity"] != identity or meta["sha256"] != references.checksum(target):
            raise ValueError("Published regional cropland raster changed")
        return target
    if not download:
        raise FileNotFoundError("The 2024 cropland mask is missing; enable download to read the Paoay window")
    with requests.get(REGIONAL_CROPLAND_URL, headers={"Range": "bytes=0-1023"},
                      stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Regional cropland server did not honor a range request; full archive download skipped")
    transform = Affine(*grid["transform"])
    projected = np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_USE_HEAD="NO",
                      GDAL_HTTP_TIMEOUT="30"):
        with rasterio.open("/vsicurl/" + REGIONAL_CROPLAND_URL) as source:
            bounds = transform_bounds(grid["crs"], source.crs, *rasterio.transform.array_bounds(*projected.shape, transform))
            window = windows.from_bounds(*bounds, source.transform).round_offsets().round_lengths()
            window = windows.Window(window.col_off - 2, window.row_off - 2, window.width + 4, window.height + 4)
            data = source.read(1, window=window, masked=True)
            if not data.count() or np.any(data.compressed() != 1):
                raise ValueError("Unexpected class codes in the regional cropland layer")
            reproject(data.filled(0).astype("uint8"), projected,
                      src_transform=source.window_transform(window), src_crs=source.crs,
                      dst_transform=transform, dst_crs=grid["crs"], resampling=Resampling.nearest)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cropland-", dir=folder.parent) as temporary:
        stage = Path(temporary) / folder.name
        stage.mkdir()
        raster = stage / target.name
        with rasterio.open(raster, "w", driver="GTiff", height=projected.shape[0], width=projected.shape[1],
                           count=1, dtype="uint8", crs=grid["crs"], transform=transform,
                           nodata=0, compress="deflate") as dest:
            dest.write(projected, 1)
            dest.update_tags(source=REGIONAL_CROPLAND_SOURCE, dataset="SEA_Cropland10 2024")
        references._json(stage / receipt.name, {"identity": identity, "sha256": references.checksum(raster),
                                                "citation": REGIONAL_CROPLAND_SOURCE})
        stage.rename(folder)
    return target


def regional_cover(cropland, dynamic_cover, roi):
    """Use regional crop delineation as the eligibility gate; keep other cover as context."""
    if cropland.shape != roi.shape or dynamic_cover.shape != roi.shape or not np.isin(cropland, [0, 1]).all():
        raise ValueError("Expected a binary cropland raster on the barangay grid")
    cover = np.zeros(roi.shape, dtype=np.uint8)
    cover[roi] = 3
    cover[roi & (dynamic_cover == 2)] = 2
    cover[roi & (cropland == 1)] = 1
    return cover


def _imagery_contract(recipe, image, permit_shifted_fixed):
    try:
        features.validate_imagery(recipe, image)
    except ValueError:
        saved_window = recipe.get("window_recipe", {})
        if not permit_shifted_fixed or saved_window.get("type") != "fixed":
            raise
        # Only the calendar month can differ in an explicitly exploratory application.
        actual_month = int(image["manifest"]["period"]["start"][5:7])
        if actual_month == saved_window["start_month"]:
            raise
        features.validate_imagery({**recipe, "window_recipe": {**saved_window, "start_month": actual_month}}, image)
        return "exploratory_shifted_calendar"
    return "matching_fixed_calendar"


def map_bundle(sentinel_handoff, guidance_handoff, model_manifest, output, *, permit_shifted_fixed=False,
               cropland_path=None):
    """Apply the verified fixed model only where prior land-cover evidence supports crops."""
    image = imagery.load_imagery_bundle(sentinel_handoff)
    guide = land_cover.load_land_cover_bundle(guidance_handoff)
    world = dynamic_world.load_dynamic_world_bundle(guide["manifest"]["dynamic_world_handoff"])
    grid, period = image["manifest"]["grid"], image["manifest"]["period"]
    if (guide["manifest"]["grid"] != grid or guide["manifest"]["period"] != period
            or world["manifest"]["imagery_handoff"] != str(Path(sentinel_handoff).resolve())):
        raise ValueError("Land cover and Sentinel-2 must derive from the same barangay request")
    classifier, recipe = predictions.load_classifier(model_manifest)
    mode = _imagery_contract(recipe, image, permit_shifted_fixed)
    identity = {"sentinel": references.checksum(sentinel_handoff), "land_cover": references.checksum(guidance_handoff),
                "model": references.checksum(model_manifest), "mode": mode,
                "minimum_land_cover_bins": MINIMUM_SUPPORTED_BINS,
                "cropland_basis": "SEA_Cropland10_2024" if cropland_path else "Dynamic_World",
                "cropland_sha256": references.checksum(cropland_path) if cropland_path else None,
                "implementation": references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    output = Path(output)
    folder = output / key
    manifest = folder / "map_handoff.json"
    if folder.exists():
        saved = json.loads(manifest.read_text())
        if saved.get("identity") != identity or any(references.checksum(folder / name) != digest
                                                  for name, digest in saved["files"].items()):
            raise ValueError("Saved barangay map bundle changed")
        return manifest

    request = {"settings": image["manifest"]["identity"]["settings"], "fragment": {
        "geometry": image["manifest"]["identity"]["settings"]["geometry"]}}
    roi = imagery._roi_mask(request)
    guidance, rasters = [], []
    for interval, path in zip(period["intervals"], image["rasters"], strict=True):
        with rasterio.open(guide["folder"] / "rasters" / f"{interval['name']}.tif") as src:
            guidance.append(src.read(6))
        with rasterio.open(path) as src:
            rasters.append(src.read())
    dynamic_cover = farmland_mask(np.asarray(guidance), roi)
    cover = dynamic_cover
    if cropland_path:
        if not period["start"].startswith("2024-"):
            raise ValueError("Use the same-year regional cropland layer only for 2024 periods")
        with rasterio.open(cropland_path) as src:
            if (src.shape != roi.shape or src.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                    or src.transform != Affine(*grid["transform"])):
                raise ValueError("Regional cropland differs from the Sentinel-2 grid")
            cover = regional_cover(src.read(1), dynamic_cover, roi)
    crop = cover == 1
    cube = np.asarray(rasters)
    clear = (cube[:, -1] > 0) & np.all(np.isfinite(cube[:, :-1]) & (cube[:, :-1] != -9999), axis=1)
    complete = np.all(clear, axis=0) & crop
    minimum = recipe["quality"]["minimum_valid_fraction"]
    coverage = [{"interval": interval["name"], "farmland_pixels": int(crop.sum()),
                 "clear_farmland_pixels": int((clear[i] & crop).sum()),
                 "clear_farmland_fraction": float(clear[i, crop].mean()) if crop.any() else 0.0}
                for i, interval in enumerate(period["intervals"])]
    passes = crop.any() and all(row["clear_farmland_fraction"] >= minimum for row in coverage)
    codes = np.zeros(roi.shape, dtype=np.uint8)
    codes[cover == 2], codes[cover == 3], codes[crop] = 6, 7, 5
    probability_grid = np.full((3, *roi.shape), -9999, dtype="float32")
    if passes:
        positions, values = [], []
        for r, c in zip(*np.nonzero(complete)):
            row, _ = features.pixel_row(cube, r, c, period, grid, recipe["feature_policy"])
            if row is not None:
                positions.append((r, c))
                values.append([row[column] for column in recipe["feature_columns"]])
        if values:
            posterior = classifier.predict_proba(np.asarray(values))
            r, c = np.asarray(positions).T
            columns = [list(classifier.classes_).index(crop_name) for crop_name in ("carrot", "cabbage", "potato")]
            posterior = posterior[:, columns]
            probability_grid[:, r, c] = posterior.T
            top_two = np.sort(posterior, axis=1)[:, -2:]
            accepted = (top_two[:, 1] >= .6) & (np.diff(top_two, axis=1)[:, 0] >= .1)
            codes[r, c] = np.where(accepted, posterior.argmax(axis=1) + 1, 4)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".barangay-", dir=output) as temp:
        stage = Path(temp) / key
        stage.mkdir()
        profile = {"driver": "GTiff", "width": grid["width"], "height": grid["height"],
                   "crs": grid["crs"], "transform": Affine(*grid["transform"]), "compress": "deflate"}
        for filename, array in (("land_cover.tif", cover), ("dynamic_world_cover.tif", dynamic_cover), ("crop_map.tif", codes)):
            with rasterio.open(stage / filename, "w", **profile, count=1, dtype="uint8", nodata=0) as dst:
                dst.write(array, 1)
        with rasterio.open(stage / "crop_probabilities.tif", "w", **profile, count=3, dtype="float32", nodata=-9999) as dst:
            dst.write(probability_grid)
            dst.descriptions = ("carrot", "cabbage", "potato")
        write_csv(stage / "coverage.csv", coverage)
        summary = {"barangay_pixels": int(roi.sum()), "supported_farmland_pixels": int(crop.sum()),
                   "all_eight_clear_farmland_pixels": int(complete.sum()), "farmland_coverage_gate_passed": bool(passes),
                   "minimum_valid_fraction": minimum,
                   "dynamic_world_crop_support_pixels": int((dynamic_cover == 1).sum()),
                   "classes": {label: int((codes == n).sum()) for n, label in PREDICTION_CODES.items()}}
        write_csv(stage / "areas.csv", [{"label": label, "pixels": count, "area_ha": count * .01}
                                       for label, count in summary["classes"].items()])
        files = {path.name: references.checksum(path) for path in stage.iterdir()}
        references._json(stage / "map_handoff.json", {"identity": identity, "period": period, "grid": grid,
            "summary": summary, "sentinel_handoff": str(sentinel_handoff), "guidance_handoff": str(guidance_handoff),
            "model_manifest": str(model_manifest), "codes": PREDICTION_CODES, "files": files,
            "regional_cropland": str(cropland_path) if cropland_path else None,
            "regional_cropland_source": REGIONAL_CROPLAND_SOURCE if cropland_path else None})
        stage.rename(folder)
    return manifest


def apply_period(reference, start, model_manifest, output, *, ee=None, download=False,
                 permit_shifted_fixed=False, use_regional_cropland=False, today=None):
    """Reuse verified inputs or download missing inputs directly, with no cloud exports."""
    output = Path(output)
    mode = "download_missing" if download else "reuse"
    sentinel = imagery.prepare_request(reference, start, output / "sentinel2", today=today)
    recipe = features.load_model_recipe(model_manifest)["recipe"]
    settings = sentinel["settings"]
    _imagery_contract(recipe, {"manifest": {"identity": sentinel["identity"], "period": sentinel["period"],
        "grid": settings["grid"], "spectral_scale_to_apply_later": settings["reflectance_scale_to_apply_later"],
        "clear_counts_scaled": False}}, permit_shifted_fixed)
    sentinel_handoff = imagery.collect(sentinel, mode=mode, earth_engine=ee)
    world = dynamic_world.prepare_request(sentinel_handoff, output / "dynamic_world")
    world_handoff = dynamic_world.collect(world, mode=mode, earth_engine=ee)
    guidance = land_cover.build(land_cover.prepare_request(world_handoff, output / "land_cover"))
    cropland = (regional_cropland(settings["grid"], output / "regional_cropland", download=download)
                if use_regional_cropland else None)
    return map_bundle(sentinel_handoff, guidance, model_manifest, output / "maps",
                      permit_shifted_fixed=permit_shifted_fixed, cropland_path=cropland)


def fetch_boundary(path):
    """Save an indicative OSM barangay outline locally, with attribution."""
    import requests

    path = Path(path)
    if path.exists():
        return path
    response = requests.get("https://nominatim.openstreetmap.org/lookup",
                            params={"osm_ids": "R20601702", "format": "jsonv2", "polygon_geojson": 1},
                            headers={"User-Agent": "SakAI-research-Paoay-boundary/1.0"}, timeout=30)
    response.raise_for_status()
    rows = response.json()
    if len(rows) != 1 or rows[0].get("osm_id") != 20601702:
        raise ValueError("The Paoay boundary service did not return the expected OSM relation")
    row = rows[0]
    record = {"type": "Feature", "geometry": row["geojson"], "properties": {
        "display_name": row["display_name"], "source_url": BOUNDARY_URL, "license": row["licence"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    imagery._publish_json(path, record)
    return path


def fixed_training_period(handoff):
    """Recover the fixed training months when the map's verified model is available."""
    manifest = Path(handoff.get("model_manifest", ""))
    if not manifest.is_file() or references.checksum(manifest) != handoff["identity"].get("model"):
        return ""
    window = json.loads(manifest.read_text())["identity"]["recipe"]["window_recipe"]
    if window["type"] != "fixed":
        return ""
    start = window["start_month"]
    return f"{month_name[start]}–{month_name[(start + 2) % 12 + 1]}"


def plot_maps(handoff):
    """Show the land-cover mask next to its separate crop inference layer."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from rasterio.plot import plotting_extent

    handoff = Path(handoff)
    data = json.loads(handoff.read_text())
    panels = [
        ("land_cover.tif", COVER_CODES, ["#538b57", "#d8ddde", "#a5a5a9"], "Land-cover eligibility"),
        ("crop_map.tif", PREDICTION_CODES, ["#e08438", "#437f46", "#8b69af", "#edce72",
                                          "#8ebbd7", "#d8ddde", "#a5a5a9"], "Threshold-qualified crop prediction"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained")
    for ax, (filename, categories, colors, title) in zip(axes, panels, strict=True):
        with rasterio.open(handoff.parent / filename) as src:
            data_grid = src.read(1)
            extent = tuple(value / 1000 for value in plotting_extent(src))
        ax.imshow(np.ma.masked_equal(data_grid, 0), extent=extent, interpolation="nearest",
                  cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(.5, len(colors) + 1.5), len(colors)))
        ax.set(title=title, xlabel="UTM easting (km)", ylabel="UTM northing (km)")
        ax.legend(handles=[Patch(color=color, label=categories[index]) for index, color in enumerate(colors, 1)],
                  loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8, ncol=2)
    if data["identity"]["mode"] == "exploratory_shifted_calendar":
        training = fixed_training_period(data)
        mode = "Exploratory calendar shift"
        if training:
            mode += f" · {training} training"
    else:
        mode = "Calendar matches training"
    fig.suptitle(f"Barangay Paoay, Atok · {data['period']['start']} to {data['period']['end']} (exclusive)\n"
                 f"{mode} · local accuracy unvalidated", fontweight="bold")
    source = (" · Cropland: Cai and Zeng, SEA_Cropland10 2024."
              if data.get("regional_cropland") else " · Cropland: Dynamic World.")
    fig.supxlabel("Boundary © OpenStreetMap contributors (ODbL) when using the supplied OSM outline." + source, fontsize=8)
    return fig
