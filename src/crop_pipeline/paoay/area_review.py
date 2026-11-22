"""Per-area predicted labels, crop-calendar context and NAMRIA ELUM farmland for saved maps.

Reads saved crop-map rasters (fixed January-April baseline, aligned exploratory map, and the
annual-recipe experiment's map when an experiment asks for it) and summarizes them inside each reference polygon and inside NAMRIA Existing Land
Use 2023 farmland. Nothing is refitted, masked or relabelled.

The calendar entries are transcriptions of the KMZ area notes. They describe when farmers say a
crop is usually in the field, not which crop occupied a polygon in the mapped year, so agreement
with them is context, not accuracy. ELUM 2023 farmland is a land-use designation digitized from
2015-2023 imagery and LGU maps; it does not say which crop grows where or when.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

CROPS = ("carrot", "cabbage", "potato")
LABELS = {1: "carrot", 2: "cabbage", 3: "potato", 4: "Unknown crop type", 5: "No prediction"}
COLORS = {1: "#eb6834", 2: "#1baf7a", 3: "#2a78d6", 4: "#c9c8c3", 5: "#efeeea"}
CALENDAR_SCOPES = {"crop_calendar", "mixed_area", "seedbed", "unconfirmed"}
# ELUM 2023 sub-classes that designate cultivated land. "General Production Agriculture" is
# excluded: the ELUM metadata uses it for closed forest on alienable land. Pasture and grazing
# land are excluded because they are not cropped.
FARMLAND_SUBCLASSES = ("Crops", "Orchard", "Agroforestry", "Agroforestry (within A and D)",
                       "Strategic Agriculture and Fisheries Development Zone")
MONTHS = "JFMAMJJASOND"


# --- Saved map products ----------------------------------------------------------------------

def _raster(path, bands):
    import rasterio
    with rasterio.open(path) as source:
        data = [source.read(band) for band in bands]
        grid = {"crs": source.crs.to_string(), "transform": tuple(source.transform)[:6],
                "shape": (source.height, source.width)}
    return data, grid


def product(kind, path, name=None):
    """Load one saved map as common label codes (0 outside, 1-3 crops, 4 Unknown, 5 no prediction).

    `kind` is "annual" (an experiments/annual map run folder), "fixed" (Paoay 04 crop-baseline handoff) or
    "aligned" (Paoay 07 bundle folder). Each product carries the months its prediction describes:
    one set for the whole map, or a per-pixel window start for the aligned map.
    """
    path = Path(path)
    start = None
    if kind == "annual":
        (codes,), grid = _raster(path / "crop_map.tif", [1])
        summary = json.loads((path / "map_summary.json").read_text())
        labels, months = codes.astype("uint8"), tuple(range(1, 13))
        period = f"{summary['year']} (whole year)"
        name = name or f"Annual model {summary['year']}"
    elif kind == "fixed":
        handoff = json.loads(path.read_text())
        (classes, status), grid = _raster(path.parent / "crop_baseline.tif", [1, 2])
        labels = np.zeros(classes.shape, "uint8")
        labels[status == 1] = classes[status == 1]
        labels[status == 2], labels[status == 3] = 4, 5
        first, end = handoff["period"]["start"], handoff["period"]["end"]
        months = tuple(sorted({int(item["start"][5:7]) for item in handoff["period"]["intervals"]}))
        period = f"{first} to {end} (end exclusive)"
        name = name or f"Fixed window {first[:7]}"
    elif kind == "aligned":
        manifest = json.loads((path / "manifest.json").read_text())
        (codes, start), grid = _raster(path / "aligned_map.tif", [1, 2])
        labels = np.where(codes >= 5, 5, codes).astype("uint8")
        months, year = None, manifest["identity"]["year"]
        period = f"{year}, four months from each pixel's green-up month"
        name = name or f"Aligned exploratory {year}"
    else:
        raise ValueError(f"Unknown map kind {kind!r}; use annual, fixed or aligned")
    return {"name": name, "kind": kind, "path": str(path), "labels": labels, "grid": grid,
            "months": months, "window_start": start, "period": period}


def aligned_bundle_for(aligned_root, model_manifest):
    """The Paoay 07 bundle made with `model_manifest`, or None when 07 has not run for that model."""
    target = str(Path(model_manifest).resolve())
    matches = [folder for folder in sorted(Path(aligned_root).glob("*/manifest.json"))
               if json.loads(folder.read_text()).get("model_manifest") == target]
    if len(matches) > 1:
        raise ValueError(f"Several aligned bundles use {target}; choose one explicitly")
    return matches[0].parent if matches else None


def pixel_months(item, rows, columns):
    """Months each selected pixel's prediction describes (empty where no window)."""
    if item["months"] is not None:
        return [set(item["months"])] * len(rows)
    starts = item["window_start"][rows, columns]
    return [set((int(s) - 1 + k) % 12 + 1 for k in range(4)) if s else set() for s in starts]


# --- Geometry --------------------------------------------------------------------------------

def area_masks(areas, grid):
    """Pixel-centre masks for each reference polygon on a product grid."""
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    transform = rasterio.Affine(*grid["transform"])
    return {area["name"]: ~geometry_mask([mapping(area["geometry"])], out_shape=grid["shape"], transform=transform)
            for area in areas}


def read_landuse(path, bounds, crs):
    """NAMRIA ELUM 2023 polygons intersecting `bounds` (in `crs`), projected to `crs`."""
    import geopandas as gpd
    from shapely.geometry import box
    frame = gpd.read_file(path, bbox=gpd.GeoSeries([box(*bounds)], crs=crs))
    return frame.to_crs(crs)[["main_elu", "sub_elu", "geometry"]]


def farmland(landuse):
    """ELUM polygons designated as cultivated land (see FARMLAND_SUBCLASSES)."""
    return landuse[landuse["sub_elu"].isin(FARMLAND_SUBCLASSES)]


def polygon_mask(frame, grid):
    import rasterio
    from rasterio.features import geometry_mask
    shapes = [geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty]
    if not shapes:
        return np.zeros(grid["shape"], bool)
    return ~geometry_mask(shapes, out_shape=grid["shape"], transform=rasterio.Affine(*grid["transform"]))


def landuse_by_area(landuse, areas):
    """Share of each reference polygon covered by each ELUM sub-class (polygon overlay, hectares)."""
    rows = []
    for area in areas:
        total = area["geometry"].area
        clipped = landuse.geometry.intersection(area["geometry"])
        shares = Counter()
        for sub, geometry in zip(landuse["sub_elu"], clipped):
            if geometry.area > 0:  # skip shared edges and corners
                shares[sub] += geometry.area
        for sub, size in sorted(shares.items(), key=lambda pair: -pair[1]):
            rows.append({"area": area["name"], "elu_subclass": sub, "farmland": sub in FARMLAND_SUBCLASSES,
                         "area_ha": round(size / 1e4, 3), "area_percent": round(100 * size / total, 1) if total else 0})
        unmapped = total - sum(shares.values())
        if unmapped > 1:
            rows.append({"area": area["name"], "elu_subclass": "(no ELUM polygon)", "farmland": False,
                         "area_ha": round(unmapped / 1e4, 3), "area_percent": round(100 * unmapped / total, 1)})
    return rows


# --- Label summaries -------------------------------------------------------------------------

def composition(item, masks, pixel_ha=None):
    """Label counts inside each mask; shares of mapped pixels and of confident crop pixels."""
    a, _, _, _, e, _ = item["grid"]["transform"]
    pixel_ha = pixel_ha or abs(a * e) / 1e4
    rows = []
    for name, mask in masks.items():
        values = item["labels"][mask & (item["labels"] > 0)]
        counts = Counter(values.tolist())
        mapped, confident = len(values), sum(counts.get(code, 0) for code in (1, 2, 3))
        for code, label in LABELS.items():
            count = counts.get(code, 0)
            rows.append({"product": item["name"], "area": name, "label": label, "pixels": count,
                         "area_ha": round(count * pixel_ha, 2),
                         "percent_of_mapped": round(100 * count / mapped, 1) if mapped else 0.0,
                         "percent_of_confident_crop": (round(100 * count / confident, 1)
                                                       if confident and code <= 3 else "")})
    return rows


def dominant(item, mask):
    """Most frequent confident crop inside a mask, its share of confident pixels and confident share."""
    values = item["labels"][mask & (item["labels"] > 0)]
    counts = Counter(v for v in values.tolist() if v <= 3)
    confident = sum(counts.values())
    if not confident:
        return None, 0.0, 0.0
    code, count = counts.most_common(1)[0]
    return LABELS[code], count / confident, confident / len(values)


# --- Calendar context ------------------------------------------------------------------------

def read_calendar(path, area_names=None):
    """Calendar entries keyed by area name: crop, months in the field, scope and source excerpt."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"name", "crop", "months", "scope", "source_excerpt"}.issubset(reader.fieldnames or []):
            raise ValueError("Calendar CSV needs name, crop, months, scope and source_excerpt columns")
        rows = list(reader)
    entries, seen = [], set()
    for row in rows:
        if not any((row.get(key) or "").strip() for key in ("crop", "months", "scope")):
            continue
        name, crop, scope = row["name"].strip(), row["crop"].strip().casefold(), row["scope"].strip()
        if area_names is not None and name not in area_names:
            raise ValueError(f"Calendar area {name!r} is not a reference area")
        if not crop or scope not in CALENDAR_SCOPES or not row["source_excerpt"].strip():
            raise ValueError(f"Calendar entry for {name!r} needs a crop, a known scope and a source excerpt")
        try:
            months = sorted({int(value) for value in row["months"].split(";")})
        except ValueError as error:
            raise ValueError(f"Calendar months for {name!r} must be semicolon-separated numbers") from error
        if not months or any(not 1 <= m <= 12 for m in months):
            raise ValueError(f"Calendar months for {name!r} must be numbers 1-12")
        if (name, crop) in seen:
            raise ValueError(f"Combine repeated {crop!r} entries for {name!r} into one month list")
        seen.add((name, crop))
        entries.append({"area": name, "crop": crop, "months": months, "scope": scope,
                        "modelled": crop in CROPS, "source_excerpt": row["source_excerpt"].strip(),
                        "review_notes": (row.get("review_notes") or "").strip()})
    return entries


def _expected(entries, area):
    """Modelled crops with a crop-calendar entry for an area: crop -> set of months."""
    return {e["crop"]: set(e["months"]) for e in entries
            if e["area"] == area and e["scope"] == "crop_calendar" and e["modelled"]}


def calendar_comparison(item, masks, entries, minimum_confident_pixels=5):
    """Per-area comparison of confident predictions with the calendar's crops for the map period.

    A confident pixel is calendar-consistent when its crop has a calendar entry for that area whose
    months overlap the months the prediction describes (the map period, or the pixel's own window).
    """
    rows = []
    for area, mask in masks.items():
        expected = _expected(entries, area)
        others = sorted({e["crop"] + ("" if e["scope"] == "crop_calendar" else f" ({e['scope'].replace('_', ' ')})")
                         for e in entries if e["area"] == area and not (e["scope"] == "crop_calendar" and e["modelled"])})
        r, c = np.nonzero(mask & (item["labels"] > 0))
        labels = item["labels"][r, c]
        windows = pixel_months(item, r, c)
        confident = labels <= 3
        consistent = np.array([LABELS[int(code)] in expected and bool(expected[LABELS[int(code)]] & months)
                               for code, months in zip(labels, windows)], bool)
        period = set().union(*windows) if windows else set()
        in_period = sorted(crop for crop, months in expected.items() if months & period)
        top, top_share, confident_share = dominant(item, mask)
        if not any(e["area"] == area for e in entries):
            status = "no_calendar_entry"
        elif not in_period:
            status = "no_modelled_crop_expected_in_period"
        elif confident.sum() < minimum_confident_pixels:
            status = "too_few_confident_pixels"
        else:
            status = "dominant_crop_in_calendar" if top in in_period else "dominant_crop_not_in_calendar"
        rows.append({
            "product": item["name"], "area": area, "period": item["period"],
            "calendar_crops_in_period": ";".join(in_period),
            "calendar_crops_outside_period": ";".join(sorted(set(expected) - set(in_period))),
            "other_or_context_crops": ";".join(others),
            "mapped_pixels": int(len(labels)), "confident_pixels": int(confident.sum()),
            "confident_percent": round(100 * confident_share, 1),
            "dominant_prediction": top or "", "dominant_percent_of_confident": round(100 * top_share, 1),
            "calendar_consistent_percent_of_confident": (round(100 * consistent[confident].mean(), 1)
                                                         if confident.any() else ""),
            "status": status})
    return rows


def calendar_detail(item, masks, entries):
    """One row per area and calendar crop: its months, overlap with the map period and predicted share."""
    rows = []
    for entry in entries:
        mask = masks.get(entry["area"])
        if mask is None:
            continue
        r, c = np.nonzero(mask & (item["labels"] > 0))
        labels = item["labels"][r, c]
        windows = pixel_months(item, r, c)
        period = set().union(*windows) if windows else set()
        code = {crop: k for k, crop in LABELS.items()}.get(entry["crop"])
        confident = int((labels <= 3).sum())
        count = int((labels == code).sum()) if code and code <= 3 else 0
        rows.append({"product": item["name"], "area": entry["area"], "crop": entry["crop"],
                     "scope": entry["scope"], "modelled": entry["modelled"],
                     "calendar_months": "".join(MONTHS[m - 1] if m in entry["months"] else "." for m in range(1, 13)),
                     "months_overlapping_period": ";".join(str(m) for m in sorted(set(entry["months"]) & period)),
                     "predicted_pixels": count if entry["modelled"] else "",
                     "percent_of_confident": round(100 * count / confident, 1) if entry["modelled"] and confident else "",
                     "source_excerpt": entry["source_excerpt"]})
    return rows


# --- Farmland --------------------------------------------------------------------------------

def farmland_composition(item, study_mask, farm_mask, other_cropland=None, other_name="Cropland"):
    """Label shares inside and outside ELUM farmland (and inside another cropland layer, if given)."""
    masks = {"ELUM farmland": study_mask & farm_mask, "Outside ELUM farmland": study_mask & ~farm_mask}
    if other_cropland is not None:
        masks[other_name] = study_mask & other_cropland
        masks[f"{other_name} within ELUM farmland"] = study_mask & other_cropland & farm_mask
    return composition(item, masks)


# --- Figures ---------------------------------------------------------------------------------

def _colormap():
    from matplotlib.colors import BoundaryNorm, ListedColormap
    return ListedColormap([COLORS[k] for k in sorted(COLORS)]).with_extremes(bad="white"), BoundaryNorm(np.arange(.5, 6), 5)


def legend_handles(farmland_shown=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[k], label=LABELS[k]) for k in sorted(LABELS)]
    handles.append(Line2D([], [], color="black", lw=1.4, label="Reference boundary"))
    if farmland_shown:
        handles.append(Line2D([], [], color="#7a4b00", lw=1, ls="--", label="ELUM 2023 farmland"))
    return handles


def _outline(axis, geometry, transform, **style):
    import rasterio
    inverse = ~rasterio.Affine(*transform)
    polygons = getattr(geometry, "geoms", [geometry])
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            xs, ys = ring.xy
            cols, rows = zip(*(inverse * (x, y) for x, y in zip(xs, ys)))
            axis.plot(np.asarray(cols) - .5, np.asarray(rows) - .5, **style)


def plot_area_panels(products, areas, study_mask=None, farmland_frame=None, padding_m=40):
    """One row per reference area, one column per map: predicted labels with the area boundary."""
    import matplotlib.pyplot as plt
    from shapely.geometry import box
    cmap, norm = _colormap()
    figure, axes = plt.subplots(len(areas), len(products), figsize=(3.6 * len(products), 3.3 * len(areas)),
                                layout="constrained", squeeze=False)
    for row, area in enumerate(areas):
        for column, item in enumerate(products):
            axis, grid = axes[row, column], item["grid"]
            mask = area_masks([area], grid)[area["name"]]
            a, _, x0, _, e, y0 = grid["transform"]
            minx, miny, maxx, maxy = area["geometry"].buffer(padding_m).bounds
            c0, c1 = int(np.floor((minx - x0) / a)), int(np.ceil((maxx - x0) / a))
            r0, r1 = int(np.floor((maxy - y0) / e)), int(np.ceil((miny - y0) / e))
            shown = np.ma.masked_equal(item["labels"], 0)
            if study_mask is not None and study_mask.shape == shown.shape:
                shown = np.ma.array(shown, mask=shown.mask | ~study_mask)
            axis.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest", alpha=.35)
            axis.imshow(np.ma.array(shown, mask=np.ma.getmaskarray(shown) | ~mask), cmap=cmap, norm=norm,
                        interpolation="nearest")
            if farmland_frame is not None:
                window = box(minx, miny, maxx, maxy)
                for geometry in farmland_frame.geometry[farmland_frame.intersects(window)]:
                    if geometry.geom_type in ("Polygon", "MultiPolygon"):
                        _outline(axis, geometry, grid["transform"], color="#7a4b00", lw=1, ls="--")
            _outline(axis, area["geometry"], grid["transform"], color="black", lw=1.4)
            axis.set_xlim(c0 - .5, c1 - .5)
            axis.set_ylim(r1 - .5, r0 - .5)
            axis.set_xticks([]), axis.set_yticks([])
            top, share, confident = dominant(item, mask)
            label = f"{top} {share:.0%} of confident" if top else "no confident crop"
            axis.set_title(f"{area['name']} · {int(mask.sum()) / 100:.1f} ha\n{label}; {confident:.0%} confident",
                           fontsize=9, loc="left")
            if row == 0:
                axis.annotate(item["name"], (.5, 1.32), xycoords="axes fraction", ha="center",
                              fontsize=10, weight="bold")
    figure.legend(handles=legend_handles(farmland_frame is not None), loc="outside lower center",
                  ncol=4, frameon=False, fontsize=9)
    figure.suptitle("Predicted labels inside each reference boundary (faded: surrounding pixels)\n"
                    "Preliminary model output; not verified crop occupancy", fontsize=11)
    return figure


def plot_calendar(entries, comparisons, products):
    """Calendar months per area and crop, with each map's period and its dominant prediction."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    crop_colors = {"carrot": COLORS[1], "cabbage": COLORS[2], "potato": COLORS[3]}
    areas = list(dict.fromkeys(e["area"] for e in entries))
    rows = [(area, e) for area in areas for e in entries if e["area"] == area]
    figure, (axis, text) = plt.subplots(1, 2, figsize=(14, .42 * len(rows) + 1.8), layout="constrained",
                                        gridspec_kw={"width_ratios": [3, 2.4]})
    fixed = [item for item in products if item["months"] is not None and len(item["months"]) < 12]
    for item in fixed:
        axis.axvspan(min(item["months"]) - .5, max(item["months"]) + .5, color="#ffe9a8", alpha=.6, zorder=0)
    for y, (area, entry) in enumerate(rows):
        color = crop_colors.get(entry["crop"], "#9a9a9a")
        hatch = None if entry["scope"] == "crop_calendar" else "///"
        for month in entry["months"]:
            axis.barh(y, 1, left=month - .5, color=color, alpha=.9 if entry["modelled"] else .45,
                      hatch=hatch, edgecolor="white", lw=.6)
        axis.text(12.7, y, entry["crop"] + ("" if entry["scope"] == "crop_calendar" else f" ({entry['scope']})"),
                  va="center", fontsize=8)
    boundaries = [i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]]
    for b in boundaries:
        axis.axhline(b - .5, color="#555", lw=.7)
    starts = [0, *boundaries]
    ends = [*boundaries, len(rows)]
    axis.set_yticks([(s + e - 1) / 2 for s, e in zip(starts, ends)], [rows[s][0] for s in starts], fontsize=9)
    axis.set_xticks(range(1, 13), list(MONTHS))
    axis.set_xlim(.5, 14.6)
    axis.set_ylim(len(rows) - .5, -.5)
    axis.set_title("KMZ calendar context: months each crop is reported in the field", loc="left", fontsize=10)
    handles = [Patch(color=c, label=k) for k, c in crop_colors.items()]
    handles += [Patch(facecolor="#9a9a9a", alpha=.45, label="not modelled"),
                Patch(facecolor="white", edgecolor="#555", hatch="///", label="seedbed / mixed / unconfirmed")]
    if fixed:
        handles.append(Patch(color="#ffe9a8", label="fixed map window"))
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.45, -.06), ncol=6, frameon=False, fontsize=8)
    text.set_axis_off()
    names = [item["name"] for item in products]

    def summary(name, area):
        for c in comparisons:
            if c["product"] == name and c["area"] == area:
                if not c["confident_pixels"]:
                    return "no confident crop"
                head = f"{c['dominant_prediction']} {c['dominant_percent_of_confident']:.0f}% (n={c['confident_pixels']})"
                if c["status"] in ("no_calendar_entry", "no_modelled_crop_expected_in_period"):
                    return head + "\nno modelled crop in calendar"
                return head + f"\nconsistent {c['calendar_consistent_percent_of_confident']:.0f}%"
        return ""

    table = [[summary(name, area) for name in names] for area in areas]
    cell = text.table(cellText=table, rowLabels=areas, colLabels=[n.replace(" ", "\n", 1) for n in names],
                      loc="upper center", cellLoc="center")
    cell.auto_set_font_size(False), cell.set_fontsize(8), cell.scale(1, 2.6)
    text.set_title("Dominant confident prediction and % of confident pixels\nconsistent with the calendar",
                   fontsize=10, loc="left")
    return figure


def plot_farmland(products, farmland_frame, rows, areas=()):
    """Each map with ELUM farmland outlines, and its label shares inside and outside farmland."""
    import matplotlib.pyplot as plt
    cmap, norm = _colormap()
    import textwrap
    figure, axes = plt.subplots(2, len(products), figsize=(max(8.5, 5.2 * len(products)), 7.4), layout="constrained",
                                squeeze=False, gridspec_kw={"height_ratios": [2.4, 1]})
    for column, item in enumerate(products):
        axis, grid = axes[0, column], item["grid"]
        axis.imshow(np.ma.masked_equal(item["labels"], 0), cmap=cmap, norm=norm, interpolation="nearest")
        for geometry in farmland_frame.geometry:
            if geometry is not None and geometry.geom_type in ("Polygon", "MultiPolygon"):
                _outline(axis, geometry, grid["transform"], color="#7a4b00", lw=.6, ls="--")
        for area in areas:
            _outline(axis, area["geometry"], grid["transform"], color="black", lw=1)
        height, width = grid["shape"]
        axis.set_xlim(-.5, width - .5), axis.set_ylim(height - .5, -.5)
        axis.set_xticks([]), axis.set_yticks([])
        axis.set_title(f"{item['name']}\n{item['period']}", fontsize=9, loc="left")
        bars = axes[1, column]
        groups = list(dict.fromkeys(r["area"] for r in rows if r["product"] == item["name"]))
        left = np.zeros(len(groups))
        for code, label in LABELS.items():
            values = np.array([next(r["percent_of_mapped"] for r in rows if r["product"] == item["name"]
                                    and r["area"] == g and r["label"] == label) for g in groups], float)
            bars.barh(range(len(groups)), values, left=left, color=COLORS[code], edgecolor="white", lw=.5)
            left += values
        bars.set_yticks(range(len(groups)), [textwrap.fill(g, 20) for g in groups], fontsize=8)
        bars.set_ylim(len(groups) - .5, -.5)
        bars.set_xlim(0, 100), bars.set_xlabel("% of mapped pixels", fontsize=8)
        bars.tick_params(axis="x", labelsize=8)
    figure.legend(handles=legend_handles(True), loc="outside lower center", ncol=4 if len(products) > 1 else 3,
                  frameon=False, fontsize=9)
    figure.suptitle("Predicted labels and NAMRIA ELUM 2023 farmland\n(Crops, Orchard, Agroforestry, SAFDZ; "
                    "a land-use designation, not a crop label)", fontsize=11)
    return figure


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path
