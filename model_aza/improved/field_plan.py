"""Field-visit plan from the improved Paoay map, and accuracy from the visit records.

Design: stratified random sampling (Olofsson et al. 2014). The population is the
land-activity cropland inside the study boundary; strata are the map's classes
there (carrot, cabbage, potato, Unknown crop type). Each stratum gets a fixed
number of primary points plus reserves for inaccessible points, kept at least
``spacing_m`` apart. A few points inside each KMZ reference area are added as a
separate, non-probability group so the team's known areas are also visited;
they never enter the accuracy estimate.

After the visit, ``estimate_accuracy`` weights each stratum by its share of
cropland pixels, so overall accuracy and per-class accuracy are not biased by
the unequal allocation. Points whose observed surface is not one of the three
crops are counted as errors for crop strata and reported separately.

The map stays exploratory; this module only prepares and scores the check.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from crop_pipeline.paoay import field_review
from crop_pipeline.sources import sampling

CROPS = ("carrot", "cabbage", "potato")
STRATA = CROPS + ("Unknown crop type",)
CODES = {1: "carrot", 2: "cabbage", 3: "potato", 4: "Unknown crop type"}
ALLOCATION = {"carrot": 5, "cabbage": 10, "potato": 30, "Unknown crop type": 30}
RESERVE_SHARE = 1.0
REFERENCE_POINTS = 3
SHEET_COLUMNS = ["visit_id", "group"] + field_review.OBSERVATION_COLUMNS  # no prediction: observe before comparing


def spaced_sample(cells, count, spacing_px, seed, label):
    """Up to ``count`` cells in stable-random order, each at least ``spacing_px`` from those already taken."""
    ranked = sorted(map(tuple, cells), key=lambda c: sampling.stable_rank(f"field|{seed}|{label}|{c[0]}|{c[1]}"))
    chosen = []
    for cell in ranked:
        if len(chosen) >= count:
            break
        if all((cell[0] - r) ** 2 + (cell[1] - c) ** 2 >= spacing_px ** 2 for r, c in chosen):
            chosen.append(cell)
    return chosen


def plan_points(codes, cropland, top_probability, areas=None, allocation=ALLOCATION, spacing_px=5, seed=1,
                reserve_share=RESERVE_SHARE, reference_points=REFERENCE_POINTS):
    """Visit points as dicts with pixel (row, col); also the stratum sizes used for weighting.

    ``areas`` maps reference-area names to boolean masks on the same grid.
    """
    sizes = {s: int(((codes == code) & cropland).sum()) for code, s in CODES.items()}
    points, taken = [], []
    for code, stratum in CODES.items():
        cells = np.argwhere((codes == code) & cropland)
        primary = allocation.get(stratum, 0)
        reserve = int(round(primary * reserve_share))
        chosen = spaced_sample([c for c in cells if all((c[0] - r) ** 2 + (c[1] - k) ** 2 >= spacing_px ** 2
                                                         for r, k in taken)], primary + reserve, spacing_px, seed, stratum)
        taken += chosen
        for rank, (r, c) in enumerate(chosen):
            points.append({"group": "stratified" if rank < primary else "reserve", "stratum": stratum,
                           "map_label": stratum, "row": int(r), "col": int(c),
                           "top_probability": float(top_probability[r, c]), "reference_area": None})
    for name, inside in (areas or {}).items():
        cells = np.argwhere(inside & np.isin(codes, list(CODES)))
        for r, c in spaced_sample(cells, reference_points, spacing_px, seed, f"area|{name}"):
            points.append({"group": "reference_area", "stratum": None, "map_label": CODES[int(codes[r, c])],
                           "row": int(r), "col": int(c), "top_probability": float(top_probability[r, c]),
                           "reference_area": name})
    order = {"stratified": 0, "reference_area": 1, "reserve": 2}
    points.sort(key=lambda p: (order[p["group"]], STRATA.index(p["map_label"]), p["row"], p["col"]))
    for number, point in enumerate(points, start=1):
        point["visit_id"] = f"V{number:03d}"
    return points, sizes


def add_coordinates(points, transform, crs):
    """Pixel-centre coordinates in the grid CRS and WGS84 longitude/latitude."""
    import rasterio
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for point in points:
        x, y = rasterio.Affine(*transform) * (point["col"] + .5, point["row"] + .5)
        lon, lat = to_wgs84.transform(x, y)
        point.update(x=x, y=y, longitude=round(lon, 6), latitude=round(lat, 6))
    return points


def write_outputs(folder, points, sizes, metadata):
    folder = Path(folder)
    columns = ["visit_id", "group", "stratum", "map_label", "top_probability", "reference_area",
               "latitude", "longitude", "row", "col"]
    with (folder / "visit_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    with (folder / "field_sheet.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, SHEET_COLUMNS)
        writer.writeheader()
        for p in points:
            writer.writerow({"visit_id": p["visit_id"], "group": p["group"], "latitude": p["latitude"],
                             "longitude": p["longitude"], "review_status": "unreviewed"})
    (folder / "visit_points.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["longitude"], p["latitude"]]},
         "properties": {k: p[k] for k in ("visit_id", "group", "stratum", "map_label", "top_probability", "reference_area")}}
        for p in points]}, indent=1))
    colors = {"stratified": "ff0000ff", "reference_area": "ffff0000", "reserve": "ff888888"}
    placemarks = "".join(
        f"<Placemark><name>{p['visit_id']}</name><description>{p['group']}"
        f"{'; area: ' + p['reference_area'] if p['reference_area'] else ''}</description>"
        f"<Style><IconStyle><color>{colors[p['group']]}</color></IconStyle></Style>"
        f"<Point><coordinates>{p['longitude']},{p['latitude']},0</coordinates></Point></Placemark>" for p in points)
    (folder / "visit_points.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>Paoay field visit plan</name>{placemarks}</Document></kml>")
    (folder / "plan.json").write_text(json.dumps({**metadata, "stratum_pixels": sizes,
                                                  "counts": Counter(p["group"] for p in points)}, indent=2))


def estimate_accuracy(sheet_rows, sizes, points):
    """Stratum-weighted accuracy from filled field-sheet rows of the stratified group (reserves that replaced a
    primary point are entered with group ``stratified``). Strata come from the plan's ``points`` by visit ID, so
    the field sheet never shows predictions. Unknown-stratum points report what was found there."""
    stratum_of = {p["visit_id"]: p["stratum"] for p in points}
    per = defaultdict(list)
    for row in sheet_rows:
        if row.get("group") != "stratified" or not (row.get("surface_state") or "").strip():
            continue
        row = {**row, "stratum": stratum_of[row["visit_id"]]}
        observed = (row.get("observed_crop") or "").strip().lower() if row["surface_state"] == "crop" else ""
        per[row["stratum"]].append(observed if observed in CROPS else f"other:{row['surface_state']}")
    crop_strata = [s for s in CROPS if per.get(s)]
    total = sum(sizes[s] for s in crop_strata)
    result = {"visited": {s: len(per.get(s, [])) for s in STRATA}, "per_class": {}}
    for s in crop_strata:
        hits = [o == s for o in per[s]]
        result["per_class"][s] = {"user_accuracy": float(np.mean(hits)), "points": len(hits),
                                  "weight": sizes[s] / total if total else None}
    if total:
        result["overall_accuracy_of_crop_labels"] = float(sum(v["user_accuracy"] * v["weight"]
                                                          for v in result["per_class"].values()))
    if per.get("Unknown crop type"):
        result["unknown_stratum_found"] = dict(Counter(per["Unknown crop type"]))
    return result


def reference_masks(repository, crs, transform, shape):
    """KMZ reference areas (Area 1–5, B1, B2, …) as masks on the map grid."""
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    from crop_pipeline.paoay import aligned
    root = Path(repository).resolve()
    settings = json.loads((root / "settings.local.json").read_text())
    kmz_root = Path(settings.get("paoay_kmz_root", "./data/paoay")).expanduser()
    kmz_root = kmz_root if kmz_root.is_absolute() else (root / kmz_root).resolve()
    kmz_paths = [kmz_root / n for n in ("00 - Region of Interest.kmz", "00 - Atok Boundary.kmz", "Crop Area.kmz")]
    _, areas, _ = aligned.study_area(kmz_paths, settings.get("paoay_roi_name", "Region of Interest"),
                                     list(settings.get("paoay_context_names", [])), crs)
    return {a["name"]: ~geometry_mask([mapping(a["geometry"])], out_shape=shape, transform=rasterio.Affine(*transform))
            for a in areas}


def build_plan(repository, paoay_handoff=None, seed=1, spacing_m=50, allocation=None):
    """Stage: visit plan from the improved Paoay map (current pointer unless a receipt is given)."""
    import rasterio
    from model_aza.improved import workflow
    receipt, record = workflow._input(repository, paoay_handoff, "paoay")
    with rasterio.open(receipt.parent / "crop_map.tif") as raster:
        codes, cropland = raster.read(1), raster.read(2) == 2
        crs, transform, pixel = raster.crs.to_string(), tuple(raster.transform)[:6], abs(raster.transform.a)
    with rasterio.open(receipt.parent / "probabilities.tif") as raster:
        probabilities = raster.read()
    finite = np.isfinite(probabilities).any(axis=0)
    top = np.where(finite, np.max(np.nan_to_num(probabilities, nan=-1.), axis=0), 0.)
    areas = reference_masks(repository, crs, transform, codes.shape)
    allocation = allocation or ALLOCATION
    points, sizes = plan_points(codes, cropland, top, areas, allocation, int(np.ceil(spacing_m / pixel)), seed)
    add_coordinates(points, transform, crs)
    run = workflow._new_run(repository, "field_plan")
    write_outputs(run, points, sizes, {"paoay_handoff": str(receipt), "seed": seed, "spacing_m": spacing_m,
                                       "allocation": allocation, "reserve_share": RESERVE_SHARE,
                                       "reference_points_per_area": REFERENCE_POINTS,
                                       "design": "stratified random by map class within land-activity cropland"})
    return workflow._publish(repository, "field_plan", workflow.seal(run, "field_plan", paoay_handoff=str(receipt)))
