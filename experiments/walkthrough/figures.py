"""Figures for the pipeline walkthrough. Each function draws saved results only and returns the figure.

Colour roles are fixed across every figure: one hue per crop (validated for colour-vision
deficiency), neutral greys for 'no data' and 'Unknown', and separate hues for dataset roles,
so a colour never means two things.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from experiments.pilots import data, features
from experiments.walkthrough import metrics

CROPS = ("carrot", "cabbage", "potato")
CROP_COLORS = {"carrot": "#eb6834", "cabbage": "#1baf7a", "potato": "#2a78d6"}
ROLE_COLORS = {"Development (train/validation)": "#6250d6", "External test": "#eda100", "Paoay target": "#d55181"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NO_DATA, UNKNOWN, NOT_CROPLAND = "#d9d8d3", "#8a99a6", "#f1f0ec"
CORRESPONDENCE_COLORS = {metrics.UNAVAILABLE: "#bdbcb6", metrics.UNCERTAIN: "#eda100", metrics.IN_SEASON: "#2e7d32",
                         metrics.OFF_SEASON: "#f1f0ec"}
SOURCE_NAMES = {"eurocrops": "EuroCrops (NL)", "francecrops": "FranceCrops (FR)", "sen4agrinet": "Sen4AgriNet (FR/ES)",
                "cropharvest": "CropHarvest", "california_dwr": "California DWR (US)", "denmark_marker": "Denmark Marker (DK)"}
COHORT_NAMES = {"jecam": "JECAM (Madagascar, Senegal…)", "california_dwr_carrot_2023_v1": "California carrot",
                "denmark_marker_cabbage_2023_v1": "Denmark cabbage", "non_jecam_cabbage_supplement_v1": "CropHarvest cabbage",
                "non_jecam_carrot_potato_supplement_v1": "CropHarvest carrot/potato", "combined": "All test parcels"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": .8, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
    })


def _ygrid(axis):
    axis.yaxis.grid(True)
    axis.set_axisbelow(True)


def _note(figure, text):
    figure.text(0.01, -0.01, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def _month_axis(axis, labels=True):
    axis.set_xticks(np.arange(0, 24, 2))
    axis.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if labels else [])
    axis.set_xlim(-.5, 23.5)


def role(row):
    return "External test" if row["partition"] == "test" else "Development (train/validation)"


# 1. Where the labels come from

def ground_truth_map(frame, countries, paoay_lonlat=(120.6, 16.6)):
    """World and Europe maps of labelled parcels, sized by parcels per 1-degree cell, coloured by role."""
    parcels = frame.drop_duplicates("partition_group").copy()
    parcels["role"] = parcels.apply(role, axis=1)
    cells = parcels.groupby(["role", "site_cell"]).agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                                       parcels=("partition_group", "size")).reset_index()
    figure, (world, europe) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [2.1, 1]})
    for axis, bounds in ((world, (-130, 130, -35, 62)), (europe, (-10, 16, 35, 58))):
        countries.plot(ax=axis, color="#f3f2ee", edgecolor="#c9c8c2", linewidth=.4)
        for name, color in list(ROLE_COLORS.items())[:2]:
            chosen = cells[cells.role == name]
            axis.scatter(chosen.lon, chosen.lat, s=12 + 9 * np.sqrt(chosen.parcels), color=color, alpha=.85,
                         edgecolor="white", linewidth=.8, label=name, zorder=3)
        axis.scatter(*paoay_lonlat, marker="*", s=320, color=ROLE_COLORS["Paoay target"], edgecolor="white",
                     linewidth=1, zorder=4, label="Paoay, Atok (target, no labels)")
        axis.set_xlim(bounds[:2]); axis.set_ylim(bounds[2:])
        axis.set_xticks([]); axis.set_yticks([]); axis.set_xlabel(""); axis.set_ylabel("")
        for spine in axis.spines.values():
            spine.set_visible(True); spine.set_color(GRID)
    world.add_patch(Rectangle((-10, 35), 26, 23, fill=False, edgecolor=MUTED, linewidth=.8, linestyle="--"))
    totals = parcels.groupby("role").size()
    world.set_title(f"A. Ground-truth parcels: {totals.get('Development (train/validation)', 0):,} development, "
                    f"{totals.get('External test', 0):,} external test")
    europe.set_title("Europe (dashed box)")
    for label, (lon, lat) in {"Madagascar / Réunion": (47, -20), "Senegal": (-16, 14.5), "California": (-121, 37),
                              "Martinique": (-61, 14.6), "Paoay": paoay_lonlat}.items():
        world.annotate(label, (lon, lat), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    for size in (10, 100, 400):
        world.scatter([], [], s=12 + 9 * np.sqrt(size), color=MUTED, alpha=.5, label=f"{size} parcels per 1° cell")
    world.legend(loc="lower left", fontsize=8, ncol=2)
    _note(figure, "One marker per 1° × 1° cell, area proportional to the parcels it holds. Coordinates: parcel "
                  "representative points from the improved-pipeline inputs (current run).")
    figure.tight_layout()
    return figure


def crop_distribution(frame):
    """B. Parcels per crop by partition, and per crop by source."""
    parcels = frame.drop_duplicates("partition_group")
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    partitions = ["train", "validation", "test"]
    names = {"train": "Train", "validation": "Validation", "test": "External test"}
    width = .26
    for offset, crop in enumerate(CROPS):
        counts = [int(((parcels.partition == p) & (parcels.crop == crop)).sum()) for p in partitions]
        bars = left.bar(np.arange(3) + (offset - 1) * width, counts, width - .03, color=CROP_COLORS[crop], label=crop)
        left.bar_label(bars, labels=[f"{c:,}" for c in counts], fontsize=8, color=INK, padding=2)
    left.set_xticks(np.arange(3)); left.set_xticklabels([names[p] for p in partitions])
    left.set_ylabel("Parcels"); left.legend(); _ygrid(left)
    left.set_title("B. Crop class distribution by partition")
    table = parcels.assign(source=parcels.source_dataset.map(SOURCE_NAMES)).pivot_table(
        index="source", columns="crop", values="partition_group", aggfunc="count", fill_value=0)
    table = table.reindex(columns=list(CROPS), fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values().index]
    start = np.zeros(len(table))
    for crop in CROPS:
        right.barh(table.index, table[crop], left=start, color=CROP_COLORS[crop], edgecolor="white", linewidth=2, label=crop)
        for y, (s, v) in enumerate(zip(start, table[crop])):
            if v >= 60:
                right.text(s + v / 2, y, f"{v:,}", ha="center", va="center", fontsize=8, color="white")
        start += table[crop].to_numpy()
    right.set_xlabel("Parcels"); right.xaxis.grid(True); right.set_axisbelow(True)
    right.set_title("Crops per source: each source supplies only some crops")
    _note(figure, "Unique parcels in the current improved-pipeline inputs (all parcels that passed Preprocessing 01). "
                  "'External test' = JECAM plus the four supplements; the reserved development test split is not used.")
    figure.tight_layout()
    return figure


# 2. Raw imagery

def _raster(values, study, fill=np.nan):
    out = np.full(study["mask"].shape + values.shape[1:], fill, dtype=float)
    out[tuple(study["pixels"].T)] = values
    return out


def _extent(study):
    a, _, c, _, e, f = study["grid"]["transform"]
    height, width = study["mask"].shape
    return (c, c + a * width, f + e * height, f)


def _outline(axis, study, color=INK, areas=True):
    from shapely.geometry import shape
    if areas:
        for area in study["areas"]:
            geometry = area["geometry"]
            for part in getattr(geometry, "geoms", [geometry]):
                if hasattr(part, "exterior"):
                    axis.plot(*part.exterior.xy, color=color, linewidth=.9)
    axis.set_xticks([]); axis.set_yticks([])


def _scalebar(axis, study, metres=1000):
    left, right, bottom, top = _extent(study)
    x, y = left + .04 * (right - left), bottom + .05 * (top - bottom)
    axis.plot([x, x + metres], [y, y], color=INK, linewidth=3, solid_capstyle="butt")
    axis.text(x + metres / 2, y + .015 * (top - bottom), f"{metres // 1000} km", ha="center", fontsize=8, color=INK)
    axis.annotate("N", xy=(right - .05 * (right - left), top - .06 * (top - bottom)),
                  xytext=(right - .05 * (right - left), top - .16 * (top - bottom)),
                  ha="center", fontsize=9, color=INK, arrowprops={"arrowstyle": "-|>", "color": INK})


def false_colour(s2_pixels, study):
    """NIR-red-green composite (B8, B4, B3) stretched to the 2-98 % range of the study area."""
    rgb = s2_pixels[:, [data.BANDS.index("B8"), data.BANDS.index("B4"), data.BANDS.index("B3")]]
    low, high = np.nanpercentile(rgb, 2, axis=0), np.nanpercentile(rgb, 98, axis=0)
    scaled = np.clip((rgb - low) / (high - low), 0, 1)
    image = _raster(scaled, study)
    masked = ~np.isfinite(image).all(axis=2)
    image[masked] = np.array([0xd9, 0xd8, 0xd3]) / 255
    image[~study["mask"]] = 1.
    return image, masked


def paoay_sentinel2(study, bins=(2, 8, 14, 20)):
    figure, axes = plt.subplots(1, len(bins), figsize=(4 * len(bins), 4))
    extent = _extent(study)
    for axis, b in zip(axes, bins):
        image, masked = false_colour(study["s2"][:, b], study)
        axis.imshow(image, extent=extent, interpolation="nearest")
        _outline(axis, study)
        clear = 1 - masked[study["mask"]].mean()
        axis.set_title(f"{metrics.HALF_MONTHS[b]} {study['year']}: {clear:.0%} clear", fontsize=10)
    _scalebar(axes[0], study)
    figure.suptitle("Sentinel-2 over Paoay: false-colour composites (NIR, red, green); vegetation shows red",
                    x=.01, ha="left", fontweight="bold")
    figure.legend(handles=[Patch(color=NO_DATA, label="Masked: cloud, shadow or no acquisition"),
                           Line2D([], [], color=INK, label="Reference areas")], loc="lower center", ncol=2,
                  bbox_to_anchor=(.5, .03))
    _note(figure, "Cloud Score+ half-month median composites at 10 m (EPSG:32651). B2 (blue) is not collected, so a "
                  "true-colour image is not possible; the standard colour-infrared view is used instead.")
    figure.tight_layout(rect=(0, .1, 1, .95))
    return figure


def paoay_sentinel1(study, bins=(2, 8, 14, 20)):
    figure, axes = plt.subplots(2, len(bins), figsize=(4 * len(bins), 7.6))
    extent = _extent(study)
    for column, b in enumerate(bins):
        for row, (index, name) in enumerate(((0, "VV"), (1, "VH"))):
            image = _raster(study["s1"][:, b, index], study)
            shown = axes[row, column].imshow(image, extent=extent, cmap="Greys_r", vmin=-22 if index else -16,
                                             vmax=-8 if index else -2, interpolation="nearest")
            _outline(axes[row, column], study, color="#e87ba4")
            axes[row, column].set_title(f"{name}, {metrics.HALF_MONTHS[b]} {study['year']}", fontsize=10)
    for row, name in enumerate(("VV", "VH")):
        figure.colorbar(axes[row, 0].images[0], ax=list(axes[row]), fraction=.015, pad=.01, label=f"{name} (dB)")
    _scalebar(axes[0, 0], study)
    figure.suptitle("Sentinel-1 radar over Paoay: backscatter is measured through cloud, so no date is missing",
                    x=.01, ha="left", fontweight="bold")
    _note(figure, "Half-month median of Sentinel-1 GRD (IW, 10 m) on the same grid as the Sentinel-2 composites. "
                  "VV responds to surface roughness and soil moisture, VH (cross-polarised) to canopy volume.")
    return figure


# 3. Cloud masking and data availability

def pick_masking_example(quality, load, low=.3, high=.7, minimum_acquisitions=4, candidates=60):
    """The largest parcel composite that lost ``low``-``high`` of its pixels to masking and had several
    acquisitions in the half-month (so before/after differ per pixel). Returns (row, layers)."""
    chosen = quality[quality.valid_fraction.between(low, high)].sort_values("selected_pixels", ascending=False)
    for _, row in chosen.head(candidates).iterrows():
        layers = load(row.image_path)
        item = layers.get(row["name"], {})
        if "pre_mask_count" in item and np.nanmax(item["pre_mask_count"]) >= minimum_acquisitions:
            return row, layers
    raise ValueError("No parcel composite with several acquisitions and partial cloud loss")


def cloud_masking(layers, interval, row):
    """Before/after masking for one parcel composite: acquisitions, rejections, clear count and result."""
    item = layers[interval]
    before, removed, clear = item["pre_mask_count"], item["cloud_shadow_count"], item["clear_count"]
    inside = np.isfinite(before) & (before > 0)
    rgb = np.dstack([item["B8"], item["B4"], item["B3"]])
    low, high = np.nanpercentile(rgb[clear > 0], 2, axis=0), np.nanpercentile(rgb[clear > 0], 98, axis=0)
    result = np.clip((rgb - low) / (high - low), 0, 1)
    result[~(clear > 0)] = np.array([0xd9, 0xd8, 0xd3]) / 255
    result[~inside] = 1.
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    vmax = np.nanmax(before)
    for axis, layer, title in ((axes[0], before, "Before: acquisitions per pixel"),
                               (axes[1], removed, "Removed by cloud/shadow mask"),
                               (axes[2], clear, "After: clear acquisitions")):
        shown = np.where(inside, layer, np.nan)
        image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        axis.set_title(title, fontsize=10); axis.set_xticks([]); axis.set_yticks([])
    figure.colorbar(image, ax=axes[2], fraction=.046, label="Sentinel-2 acquisitions")
    axes[3].imshow(result, interpolation="nearest"); axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title("Composite kept (false colour); grey = no clear pixel", fontsize=10)
    figure.suptitle(f"Before and after cloud masking: {row['partition_group']} ({row['crop']}), {interval}: "
                    f"{row['valid_fraction']:.0%} of selected pixels clear "
                    f"({'passes' if row['valid_fraction'] >= .8 else 'fails'} the 80 % rule)", x=.01, ha="left",
                    fontweight="bold")
    _note(figure, "From the parcel's annual archive TIFF (Preprocessing 01 input): per pixel, the number of Sentinel-2 "
                  "acquisitions in the half-month before masking, those rejected as cloud or shadow, and those kept. "
                  "Only the parcel's saved pixels are stored, so the shape is the parcel, not a full scene.")
    figure.tight_layout(rect=(0, 0, 1, .9))
    return figure


def valid_observation_heatmap(shares):
    """B. Share of sampled pixels with a clear observation, per group and half-month."""
    figure, axis = plt.subplots(figsize=(14, .42 * len(shares) + 1.6))
    image = axis.imshow(shares.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(shares.to_numpy()):
        axis.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7,
                  color="white" if value > .6 else INK)
    axis.set_yticks(range(len(shares))); axis.set_yticklabels([" · ".join(map(str, i)) for i in shares.index])
    axis.set_xticks(range(24)); axis.set_xticklabels(metrics.HALF_MONTHS, rotation=60, ha="right", fontsize=8)
    axis.spines[:].set_visible(False)
    figure.colorbar(image, ax=axis, fraction=.02, label="Share of pixels clear")
    axis.set_title("B. Valid-observation heatmap: % of sampled pixels with a clear Sentinel-2 composite")
    _note(figure, "Improved-pipeline inputs: 5 stable-ranked pixels per parcel, clear flag from the annual archive. "
                  "Each source is shown in its own labelled year.")
    figure.tight_layout()
    return figure


def usable_coverage(quality, frame_rows, paoay_clear):
    """C. Per half-month: share of parcel composites passing the 80 % rule (dev vs test), and Paoay pixels clear."""
    quality = quality.copy()
    quality["bin"] = quality.name.map(data.bin_index)
    test = set(r["partition_group"] for r in frame_rows if r["partition"] == "test")
    quality["role"] = np.where(quality.partition_group.isin(test), "External test", "Development (train/validation)")
    passing = quality.assign(ok=quality.status.eq("quality_pass")).groupby(["role", "bin"]).ok.mean().unstack(0)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    width = .4
    for offset, name in ((-.2, "Development (train/validation)"), (.2, "External test")):
        top.bar(passing.index + offset, passing[name], width - .04, color=ROLE_COLORS[name], label=name)
    top.axhline(1, color=GRID, linewidth=.8)
    top.set_ylabel("Parcel composites usable"); top.set_ylim(0, 1.05)
    top.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); _ygrid(top)
    top.set_title("C. Usable coverage by interval: parcels with ≥ 80 % of pixels clear")
    bars = bottom.bar(np.arange(24), paoay_clear, .6, color=ROLE_COLORS["Paoay target"])
    bottom.bar_label(bars, labels=[f"{v:.0%}" for v in paoay_clear], fontsize=7, color=INK, padding=1)
    bottom.set_ylim(0, 1.2); bottom.set_ylabel("Study pixels clear")
    bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); _ygrid(bottom)
    bottom.set_title("Paoay study area, 2024 (the year the maps use)")
    bottom.axvspan(7.5, 15.5, color="#2a78d6", alpha=.06)
    bottom.text(11.5, 1.12, "main pipeline's fixed May–Aug window", ha="center", fontsize=8, color=MUTED)
    bottom.set_xticks(range(24)); bottom.set_xticklabels(metrics.HALF_MONTHS, rotation=60, ha="right", fontsize=8)
    _note(figure, "Top: every parcel composite in the current Preprocessing 01 quality table. Bottom: share of the "
                  "41,167 Paoay study pixels with all five bands present in the Cloud Score+ composite.")
    figure.tight_layout()
    return figure


def temporal_correspondence_heatmap(status, groups):
    """Rows: reference parcels grouped by source and crop, sorted by cycle start. Columns: half-months."""
    keys = (metrics.IN_SEASON, metrics.UNCERTAIN, metrics.UNAVAILABLE, metrics.OFF_SEASON)
    start = np.array([np.argmax(s == metrics.IN_SEASON) if (s == metrics.IN_SEASON).any() else 99 for s in status])
    order = np.lexsort((start, np.asarray(groups)))
    status, groups = status[order], np.asarray(groups)[order]
    colormap = ListedColormap([CORRESPONDENCE_COLORS[k] for k in range(4)])
    figure, (axis, side) = plt.subplots(1, 2, figsize=(14, max(6, .032 * len(status) + 2.5)),
                                        gridspec_kw={"width_ratios": [6, 1.3]}, sharey=True)
    axis.imshow(status, aspect="auto", cmap=colormap, norm=BoundaryNorm(np.arange(-.5, 4), 4), interpolation="nearest")
    boundaries = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    for first, _ in spans:
        axis.axhline(first - .5, color="white", linewidth=3)
    axis.set_yticks([(a + b - 1) / 2 for a, b in spans])
    axis.set_yticklabels([f"{groups[a]}  (n={b - a})" for a, b in spans], fontsize=8)
    axis.set_xticks(range(24)); axis.set_xticklabels(metrics.HALF_MONTHS, rotation=60, ha="right", fontsize=8)
    axis.spines[:].set_visible(False)
    for first, last in spans:
        observed = []
        for row in status[first:last]:
            inside = np.nonzero(row == metrics.IN_SEASON)[0]
            if len(inside):
                span = row[inside[0]:inside[-1] + 1]
                observed.append(np.mean(span == metrics.IN_SEASON))
            else:
                observed.append(0.)
        value = float(np.median(observed))
        side.barh((first + last - 1) / 2, value, height=(last - first) * .8, color=CORRESPONDENCE_COLORS[metrics.IN_SEASON])
        side.barh((first + last - 1) / 2, 1 - value, left=value, height=(last - first) * .8,
                  color=CORRESPONDENCE_COLORS[metrics.UNAVAILABLE])
        side.text(1.03, (first + last - 1) / 2, f"{value:.0%}", va="center", fontsize=8, color=INK)
    side.set_xlim(0, 1.3); side.set_xticks([0, .5, 1]); side.set_xticklabels(["0", "50%", "100%"])
    side.set_title("Crop cycle observed\n(median share clear)", fontsize=9); side.spines[["left"]].set_visible(False)
    axis.set_title("C. Temporal correspondence: does each label line up with a crop cycle the satellite can see?")
    figure.legend(handles=[Patch(color=CORRESPONDENCE_COLORS[k], label=metrics.CORRESPONDENCE_LABELS[k]) for k in keys],
                  loc="upper center", bbox_to_anchor=(.45, .985), ncol=4, fontsize=8.5)
    _note(figure, "Each row is one reference parcel (stable random sample, 30 per group) in its label year; rows are "
                  "sorted by cycle start. Cycle = the contiguous run around the parcel's NDVI peak where clear NDVI "
                  "stays in the upper half of its range (range ≥ 0.15); cloudy half-months inside the run do not end it.")
    figure.tight_layout(rect=(0, 0, 1, .955))
    return figure


# 4. Signatures and features

def spectral_signatures(frame, s2, valid, s1, variables=("NDVI", "NDRE", "NDMI", "B8", "B11", "VH")):
    """D. Median (line) and interquartile range (band) per crop and half-month, development parcels only."""
    development = (frame.partition != "test").to_numpy()
    stacked = features.stack(s2)
    names = list(features.VARIABLES)
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for axis, variable in zip(axes.flat, variables):
        if variable == "VH":
            values, ok = s1[..., 1], np.isfinite(s1[..., 1])
            unit = "dB"
        else:
            values, ok = stacked[..., names.index(variable)], valid
            unit = "reflectance" if variable.startswith("B") else "index"
        for crop in CROPS:
            chosen = development & (frame.crop == crop).to_numpy()
            series = np.where(ok[chosen], values[chosen], np.nan)
            with np.errstate(all="ignore"):
                q1, median, q3 = np.nanpercentile(series, [25, 50, 75], axis=0)
            axis.fill_between(range(24), q1, q3, color=CROP_COLORS[crop], alpha=.15, linewidth=0)
            axis.plot(range(24), median, color=CROP_COLORS[crop], linewidth=2, label=crop)
            axis.text(23.6, median[-1], crop, color=INK, fontsize=8, va="center")
        axis.set_title(variable if variable != "VH" else "Sentinel-1 VH"); axis.set_ylabel(unit); _ygrid(axis)
        _month_axis(axis)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.suptitle("D. Crop spectral signatures (development parcels): median and interquartile range per half-month",
                    x=.01, ha="left", fontweight="bold")
    _note(figure, "Clear-sky values only; Sentinel-1 has no cloud gaps. Carrot, cabbage and potato overlap for most "
                  "of the year: the differences are in timing and shape, which is what the harmonic features encode.")
    figure.tight_layout(rect=(0, .02, 1, .95))
    return figure


def signature_shift(frame, s2, valid, crop="potato"):
    """NDVI of one crop in each source region: the same label looks different in different places."""
    ndvi = metrics.ndvi(s2)
    frame = frame.assign(region=np.where(frame.partition == "test", frame.test_cohort.map(COHORT_NAMES).fillna("test"),
                                         frame.source_dataset.map(SOURCE_NAMES)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, crop in zip(axes, CROPS):
        regions = frame[frame.crop == crop].region.value_counts()
        regions = regions[regions >= 50].index
        for k, region in enumerate(regions):
            chosen = ((frame.crop == crop) & (frame.region == region)).to_numpy()
            series = np.where(valid[chosen], ndvi[chosen], np.nan)
            with np.errstate(all="ignore"):
                median = np.nanmedian(series, axis=0)
            test = "JECAM" in region or "California" in region or "Denmark" in region or "CropHarvest" in region
            axis.plot(range(24), median, linewidth=2, linestyle="--" if test else "-",
                      color=plt.cm.Greys(.35 + .6 * k / max(1, len(regions) - 1)), label=region)
        axis.set_title(f"{crop}: NDVI by region", color=INK); _month_axis(axis); _ygrid(axis)
        axis.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Median NDVI")
    figure.suptitle("Why transfer is hard: the same crop has a different calendar in each region "
                    "(solid = development, dashed = external test)", x=.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, .92))
    return figure


def feature_anatomy(pixel_s2, pixel_valid, pixel_s1, title, improved_columns):
    """How one pixel's year becomes model inputs in each pipeline."""
    stacked = features.stack(pixel_s2[None])[0]
    ndvi = stacked[:, features.VARIABLES.index("NDVI")]
    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1.25, 1, 1])
    series_axis = figure.add_subplot(grid[0, :2])
    t, curve, ok, coefficients = metrics.harmonic_curve(ndvi, pixel_valid)
    series_axis.axvspan(7.5, 15.5, color="#2a78d6", alpha=.07)
    series_axis.text(11.5, .02, "Main pipeline: fixed May–Aug window\n(8 half-months, all must be clear)",
                     ha="center", va="bottom", fontsize=8, color=MUTED, transform=series_axis.get_xaxis_transform())
    series_axis.plot(t, curve, color=INK, linewidth=2, label="Improved pipeline: 2-harmonic fit through clear points")
    series_axis.scatter(np.nonzero(pixel_valid)[0], ndvi[pixel_valid], s=46, color=CROP_COLORS["cabbage"],
                        edgecolor="white", zorder=3, label="Clear half-month composite")
    cloudy = np.nonzero(~pixel_valid)[0]
    series_axis.scatter(cloudy, np.full(len(cloudy), np.nanmin(curve) - .05), marker="x", s=40, color=MUTED,
                        label="Masked half-month (no value)")
    series_axis.set_ylabel("NDVI"); _month_axis(series_axis); _ygrid(series_axis)
    series_axis.legend(fontsize=8, loc="lower right")
    amp1, amp2 = np.hypot(*coefficients[1:3]), np.hypot(*coefficients[3:5])
    series_axis.set_title(f"One pixel's year ({title}): mean {coefficients[0]:.2f}, amp1 {amp1:.2f}, amp2 {amp2:.2f}")
    s1_axis = figure.add_subplot(grid[0, 2])
    for index, name, color in ((0, "VV", MUTED), (1, "VH", INK)):
        s1_axis.plot(range(24), pixel_s1[:, index], color=color, linewidth=2, label=name)
    s1_axis.set_ylabel("dB"); _month_axis(s1_axis); _ygrid(s1_axis); s1_axis.legend(fontsize=8)
    s1_axis.set_title("Same pixel, Sentinel-1 (no gaps)")
    window = features.fixed_interval(stacked[None], pixel_valid[None])
    main_axis = figure.add_subplot(grid[1, 0])
    block = window[0][0].reshape(8, len(features.VARIABLES)).T
    shown = (block - np.nanmean(block, axis=1, keepdims=True)) / (np.nanstd(block, axis=1, keepdims=True) + 1e-9)
    main_axis.imshow(shown, cmap="PuOr_r", vmin=-2, vmax=2, aspect="auto")
    for (i, j), value in np.ndenumerate(block):
        main_axis.text(j, i, "–" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=6.5)
    main_axis.set_yticks(range(len(features.VARIABLES))); main_axis.set_yticklabels(features.VARIABLES, fontsize=8)
    main_axis.set_xticks(range(8)); main_axis.set_xticklabels(metrics.HALF_MONTHS[8:16], rotation=60, ha="right", fontsize=7)
    main_axis.set_title(f"Main pipeline: 72 predictors (8 × 9){'' if window[1][0] else ' — window not clear: no prediction'}",
                        fontsize=10)
    improved_axis = figure.add_subplot(grid[1, 1:])
    terms = features.FULL_TERMS
    harmonic, _, columns = features.harmonic_features(stacked[None], pixel_valid[None], form="full")
    matrix = harmonic[0].reshape(len(features.VARIABLES), len(terms))
    shown = matrix / (np.nanmax(np.abs(matrix), axis=0, keepdims=True) + 1e-9)
    improved_axis.imshow(shown, cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        improved_axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    improved_axis.set_yticks(range(len(features.VARIABLES))); improved_axis.set_yticklabels(features.VARIABLES, fontsize=8)
    improved_axis.set_xticks(range(len(terms))); improved_axis.set_xticklabels(terms, fontsize=8)
    radar = len(improved_columns) - harmonic.shape[1]
    improved_axis.set_title(f"Improved pipeline: {harmonic.shape[1]} Sentinel-2 harmonic terms (9 variables × 7) "
                            f"+ {radar} Sentinel-1 terms = {len(improved_columns)} predictors", fontsize=10)
    _note(figure, "Cell colour is relative within each row (main) or column (improved), only to make patterns visible; "
                  "the numbers are the actual feature values the models receive.")
    figure.tight_layout()
    return figure


# 5. Model evidence

def selection_views(scores, selected):
    views = {"random_split_macro_f1": "Random split\n(optimistic)", "spatial_cv_macro_f1": "Spatial CV\n(new regions)",
             "dataset_holdout_macro_f1": "Dataset holdout\n(new source)"}
    means = scores.groupby("candidate")[list(views)].mean()
    figure, axis = plt.subplots(figsize=(12, 4.4))
    width = .8 / len(means)
    shades = [plt.cm.Greys(.35 + .55 * k / max(1, len(means) - 1)) for k in range(len(means))]
    for k, (name, row) in enumerate(means.iterrows()):
        color = "#2a78d6" if name == selected else shades[k]
        bars = axis.bar(np.arange(len(views)) + (k - (len(means) - 1) / 2) * width, row.to_numpy(), width - .02,
                        color=color, label=name + ("  ← selected" if name == selected else ""))
        axis.bar_label(bars, labels=[f"{v:.2f}" for v in row], fontsize=7.5, padding=2)
    axis.set_xticks(range(len(views))); axis.set_xticklabels(views.values()); axis.set_ylim(0, 1.05)
    axis.set_ylabel("Parcel macro-F1 (development data)"); _ygrid(axis); axis.legend(fontsize=8, ncol=2)
    axis.set_title("Experiment: which feature set transfers? Scores drop as the test moves further from training")
    figure.tight_layout()
    return figure


def model_comparison(records):
    """Per test population: present-crop macro-F1 (left) and share of parcels the model could score (right)."""
    table = pd.DataFrame(records)
    populations = list(dict.fromkeys(table.population))
    models = list(dict.fromkeys(table.model))
    colors = {m: c for m, c in zip(models, ["#bdbcb6", "#8a8984", "#2a78d6"])}
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    height = .8 / len(models)
    for k, model in enumerate(models):
        chosen = table[table.model == model].set_index("population").reindex(populations)
        positions = np.arange(len(populations)) + (k - (len(models) - 1) / 2) * height
        bars = left.barh(positions, chosen.f1, height - .02, color=colors[model], label=model)
        left.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "–" for v in chosen.f1], fontsize=7.5, padding=2)
        bars = right.barh(positions, chosen.scored / chosen.total, height - .02, color=colors[model])
        right.bar_label(bars, labels=[f"{s}/{t}" for s, t in zip(chosen.scored, chosen.total)], fontsize=7.5, padding=2)
    left.axvline(.5, color="#e34948", linestyle="--", linewidth=1)
    left.text(.51, len(populations) - .45, "target 0.50", color="#e34948", fontsize=8, va="top")
    left.set_yticks(range(len(populations))); left.set_yticklabels([COHORT_NAMES.get(p, p) for p in populations])
    left.invert_yaxis(); left.set_xlim(0, 1.1); left.set_xlabel("Macro-F1 over the crops present")
    left.legend(fontsize=8, loc="lower right"); left.xaxis.grid(True); left.set_axisbelow(True)
    left.set_title("Accuracy on each external test population")
    right.set_xlim(0, 1.25); right.set_xlabel("Parcels with a prediction"); right.xaxis.grid(True); right.set_axisbelow(True)
    right.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    right.set_title("Coverage: how many test parcels each model can classify")
    _note(figure, "Computed the same way for every model from the saved parcel predictions (current testing runs). "
                  "Single-crop populations report that crop's F1. The main-pipeline models skip parcels without a "
                  "clear 4-month window; the improved model scores every parcel.")
    figure.tight_layout()
    return figure


def confusion_panels(panels):
    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4))
    for axis, (title, table) in zip(np.atleast_1d(axes), panels):
        counts = table.to_numpy()
        shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(counts):
            axis.text(j, i, f"{value}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                      color="white" if shares[i, j] > .55 else INK)
        axis.set_xticks(range(counts.shape[1])); axis.set_xticklabels(table.columns, rotation=30, ha="right")
        axis.set_yticks(range(counts.shape[0])); axis.set_yticklabels(table.index)
        axis.set_xlabel("Predicted"); axis.set_ylabel("Reference label")
        axis.set_title(f"{title}\n(n = {counts.sum()})", fontsize=10); axis.spines[:].set_visible(False)
    figure.suptitle("A. Confusion matrices on JECAM (counts; shade = share of each reference row)",
                    x=.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, .92))
    return figure


def precision_recall_f1(panels):
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), sharey=True)
    metrics_order = (("precision", .45), ("recall", .7), ("f1", 1.))
    for axis, (title, table) in zip(np.atleast_1d(axes), panels):
        width = .26
        for k, (name, alpha) in enumerate(metrics_order):
            values = table[name].to_numpy()
            bars = axis.bar(np.arange(len(table)) + (k - 1) * width, np.nan_to_num(values), width - .03,
                            color=[CROP_COLORS[c] for c in table.crop], alpha=alpha, label=name)
            axis.bar_label(bars, labels=["–" if not np.isfinite(v) else f"{v:.2f}" for v in values], fontsize=7, padding=1)
        axis.axhline(.5, color="#e34948", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(table))); axis.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(table.crop, table.support)])
        axis.set_ylim(0, 1.1); axis.set_title(title, fontsize=10); _ygrid(axis)
    np.atleast_1d(axes)[0].set_ylabel("Score")
    figure.legend(handles=[Patch(color=MUTED, alpha=a, label=n) for n, a in metrics_order], loc="lower center", ncol=3)
    figure.suptitle("B. Per-crop precision (light), recall (medium) and F1 (solid) on JECAM; dashed = 0.50 target",
                    x=.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, .07, 1, .92))
    return figure


def feature_importance(grouped, single, top=15):
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2))
    grouped = grouped.sort_values("mean")
    left.barh(grouped.index, grouped["mean"], xerr=grouped["sd"], color="#2a78d6", error_kw={"ecolor": MUTED, "lw": 1})
    left.axvline(0, color=MUTED, linewidth=.8); left.xaxis.grid(True); left.set_axisbelow(True)
    left.set_xlabel("Drop in pixel macro-F1 when the variable's terms are shuffled")
    left.set_title("C. Grouped permutation importance (external test pixels)")
    single = single.sort_values().tail(top)
    right.barh(single.index, single.to_numpy(), color="#8a99a6")
    right.xaxis.grid(True); right.set_axisbelow(True); right.set_xlabel("Mean decrease in impurity")
    right.set_title(f"Top {top} single predictors (impurity, biased)")
    _note(figure, "Left: each variable's 7 (or 3) harmonic terms are permuted together, 5 repeats, on the external "
                  "test pixels; this diagnoses the frozen model and is not used for any choice. Right: the Random "
                  "Forest's built-in impurity importance, which favours continuous and correlated predictors (Strobl "
                  "et al. 2007), shown for reference only.")
    figure.tight_layout()
    return figure


# 6. Maps

MAP_CLASSES = [("carrot", CROP_COLORS["carrot"]), ("cabbage", CROP_COLORS["cabbage"]), ("potato", CROP_COLORS["potato"]),
               ("Unknown crop type (low confidence)", UNKNOWN), ("No prediction possible", NO_DATA),
               ("Not active cropland", NOT_CROPLAND)]


def _map_image(classes, inside):
    colors = dict(MAP_CLASSES)
    image = np.ones(classes.shape + (3,))
    for name, color in MAP_CLASSES:
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        image[classes == name] = rgb
    image[~inside] = 1.
    return image


def prediction_maps(panels, study, title):
    """Side-by-side class maps; each panel is (title, class-name raster, inside mask, footnote)."""
    figure, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.6))
    extent = _extent(study)
    for axis, (name, classes, inside, foot) in zip(np.atleast_1d(axes), panels):
        axis.imshow(_map_image(classes, inside), extent=extent, interpolation="nearest")
        _outline(axis, study)
        cropland = inside & (classes != "Not active cropland")
        shares = {c: (classes[cropland] == c).mean() for c, _ in MAP_CLASSES[:5]}
        axis.set_title(name, fontsize=10.5)
        axis.text(0.01, -0.02, " · ".join(f"{c.split(' (')[0]} {v:.1%}" for c, v in shares.items() if v > 0) + "\n" + foot,
                  transform=axis.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
        for spine in axis.spines.values():
            spine.set_visible(False)
    _scalebar(np.atleast_1d(axes)[0], study)
    figure.legend(handles=[Patch(color=c, label=n) for n, c in MAP_CLASSES] +
                  [Line2D([], [], color=INK, label="Reference areas")], loc="lower center", ncol=7, fontsize=8.5)
    figure.suptitle(title, x=.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, .07, 1, .94))
    return figure


def uncertainty_maps(paoay, study, threshold=.6, margin=.1):
    probabilities = paoay["probabilities"]
    ordered = np.sort(probabilities, axis=0)
    top, gap = ordered[-1], ordered[-1] - ordered[-2]
    cropland = paoay["cropland"] & paoay["inside"]
    figure = plt.figure(figsize=(16, 5.4))
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    extent = _extent(study)
    for k, (values, label, vmax) in enumerate(((top, "Top class probability", 1), (gap, "Margin to the next crop", .8))):
        axis = figure.add_subplot(grid[0, k])
        shown = np.where(cropland, values, np.nan)
        image = axis.imshow(shown, extent=extent, cmap="viridis", vmin=0 if k else 1 / 3, vmax=vmax, interpolation="nearest")
        _outline(axis, study, color="white")
        figure.colorbar(image, ax=axis, fraction=.046, label=label)
        axis.set_title(label + " (active cropland)", fontsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
    curve = figure.add_subplot(grid[0, 2])
    thresholds = np.linspace(1 / 3, .95, 60)
    chosen = cropland & np.isfinite(top)
    coverage = [np.mean((top[chosen] >= t) & (gap[chosen] >= margin)) for t in thresholds]
    curve.plot(thresholds, coverage, color="#2a78d6", linewidth=2, label=f"Share labelled (margin ≥ {margin})")
    curve.axvline(threshold, color="#e34948", linestyle="--", linewidth=1)
    current = np.mean((top[chosen] >= threshold) & (gap[chosen] >= margin))
    curve.annotate(f"rule in use: {current:.0%} labelled,\n{1 - current:.0%} 'Unknown'", (threshold, current),
                   xytext=(12, 18), textcoords="offset points", fontsize=8.5, color=INK,
                   arrowprops={"arrowstyle": "-", "color": MUTED})
    hist = curve.twinx()
    hist.hist(top[chosen], bins=30, range=(1 / 3, 1), color=GRID, zorder=0)
    hist.set_yticks([]); hist.spines[:].set_visible(False)
    curve.set_zorder(hist.get_zorder() + 1); curve.patch.set_visible(False)
    curve.set_xlabel("Top-probability threshold"); curve.set_ylabel("Active-cropland pixels labelled")
    curve.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); _ygrid(curve)
    curve.set_title("Confidence threshold vs coverage (grey: distribution of top probability)", fontsize=10)
    figure.suptitle("Prediction uncertainty, improved model in Paoay 2024: the model is rarely sure, and confidence is "
                    "not accuracy", x=.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, .93))
    return figure


def crop_shares(records):
    table = pd.DataFrame(records).set_index("map")
    figure, axis = plt.subplots(figsize=(11, 3.4))
    start = np.zeros(len(table))
    for crop in CROPS:
        axis.barh(table.index, table[crop], left=start, color=CROP_COLORS[crop], edgecolor="white", linewidth=2, label=crop)
        for y, (s, v) in enumerate(zip(start, table[crop])):
            if v >= .06:
                axis.text(s + v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8.5)
        start += table[crop].to_numpy()
    axis.set_xlim(0, 1); axis.invert_yaxis(); axis.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.set_title("Crop mix among confidently labelled pixels vs provincial statistics", pad=24)
    _note(figure, "Benguet 2024 shares are province-wide planted area for the three crops (PSA), not a Paoay ground "
                  "truth; a large gap is a warning sign, a match would not prove the map right.")
    figure.tight_layout()
    return figure
