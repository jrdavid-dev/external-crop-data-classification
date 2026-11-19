"""Figures for the active preprocessing, training and testing notebooks.

Every function draws saved results only. Nothing is fitted, scored or selected
here, and missing values stay visible instead of becoming zeros.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CROPS = ("carrot", "cabbage", "potato")
PARTITIONS = ("train", "validation", "test")
COHORTS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")
COLORS = {"carrot": "#eb6834", "cabbage": "#1baf7a", "potato": "#2a78d6"}
NEUTRAL, INK, MUTED, GRID = "#8a99a6", "#0b0b0b", "#52514e", "#e4e3df"
TARGET_COLOR = "#e34948"
F1_TARGET = 0.5
COHORT_LABELS = {"fixed_full": "Fixed calendar\nfull", "aligned_full": "Aligned green-up\nfull",
                 "fixed_matched": "Fixed calendar\nmatched", "aligned_matched": "Aligned green-up\nmatched"}


def read_table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure, folder, name):
    """Save next to the run outputs; the notebook still displays the figure."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return path


def _style(axis, title):
    axis.set_title(title, loc="left", fontweight="bold", color=INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _crop_bars(axis, groups, values, targets=None):
    """Grouped bars per crop; `values[crop]` aligns with `groups`, None means unavailable.

    With `targets` (one per group), each bar is labelled "count / target" and a dashed
    target line spans the group, so shortfalls read directly from the label.
    """
    positions = np.arange(len(groups))
    width = 0.26
    for offset, crop in enumerate(CROPS):
        heights = [0 if v is None else v for v in values[crop]]
        bars = axis.bar(positions + (offset - 1) * width, heights, width - 0.02,
                        color=COLORS[crop], label=crop)
        labels = ["n/a" if v is None else (f"{v:.2f}" if isinstance(v, float) else f"{v:,}")
                  for v in values[crop]]
        if targets is not None:
            labels = [label if label == "n/a" else f"{label}\n/ {target:,}"
                      for label, target in zip(labels, targets)]
        texts = axis.bar_label(bars, labels=labels, padding=2, fontsize=7 if targets else 8, color=MUTED,
                               linespacing=0.95)
        if targets is not None:
            for text in texts:
                text.set_bbox({"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
                text.set_zorder(4)
    if targets is not None:
        for position, target in zip(positions, targets):
            axis.hlines(target, position - 1.5 * width, position + 1.5 * width,
                        colors=TARGET_COLOR, linestyles="--", linewidth=1.5, zorder=3)
        axis.plot([], [], color=TARGET_COLOR, linestyle="--", linewidth=1.5, label="target per crop")
    axis.set_xticks(positions, groups)
    axis.margins(y=0.2 if targets else 0.15)
    return positions, width


def partition_targets(measure="parcels"):
    """Per-crop collection targets for each partition, from the adaptive collection policy."""
    from crop_pipeline.sources import adaptive
    return [adaptive.DEFAULT_POLICY["targets"][partition][measure] for partition in PARTITIONS]


def draw_reference_profile(rows):
    """Parcels per source and crop from reference_profile.csv (before window checks)."""
    sources = sorted({(r["source"], r["partition"] == "test") for r in rows}, key=lambda s: (s[1], s[0]))
    groups = [f"{name}\n({'external test' if test else 'development'})" for name, test in sources]
    values = {crop: [sum(int(r["parcels"]) for r in rows if r["crop"] == crop and r["source"] == name
                         and (r["partition"] == "test") == test) for name, test in sources]
              for crop in CROPS}
    figure, axis = plt.subplots(figsize=(9, 4.5))
    _crop_bars(axis, groups, values)
    axis.set_ylabel("Reference parcels")
    _style(axis, "Reference parcels by source and crop (before window checks)")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    return figure


def draw_quality_pass_rates(rows):
    """Share of audited tile intervals passing the 80% valid-pixel rule (quality_counts.csv)."""
    values = {}
    for crop in CROPS:
        values[crop] = []
        for partition in PARTITIONS:
            match = [r for r in rows if r["crop"] == crop and r["partition"] == partition]
            audited = sum(int(r["audited_tile_intervals"]) for r in match)
            passed = sum(int(r["quality_pass_tile_intervals"]) for r in match)
            values[crop].append(round(passed / audited, 2) if audited else None)
    figure, axis = plt.subplots(figsize=(8, 4))
    _crop_bars(axis, list(PARTITIONS), values)
    axis.set_ylim(0, 1.25)
    axis.set_ylabel("Half-month composites passing")
    _style(axis, "Imagery quality: composites with ≥ 80% valid pixels")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    return figure


def draw_observation_loss(rows):
    """Cloud/shadow share of acquired pixel-dates per calendar half-month (observation_loss_by_half_month.csv).

    Half-months without pre-mask counts are drawn as "n/a", not as zero loss.
    """
    labels = [r["half_month_start"] for r in rows]
    shares = [None if r.get("cloud_or_shadow_share") in (None, "") else float(r["cloud_or_shadow_share"])
              for r in rows]
    figure, axis = plt.subplots(figsize=(11, 3.8))
    positions = np.arange(len(rows))
    axis.bar(positions, [s or 0 for s in shares], 0.7, color=NEUTRAL)
    for position, share in zip(positions, shares):
        if share is None:
            axis.text(position, 0.02, "n/a", ha="center", fontsize=7, color=MUTED)
    axis.set_xticks(positions, labels, rotation=90, fontsize=8)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.set_xlabel("Half-month start (MM-DD), all source years combined")
    axis.set_ylabel("Cloud or shadow share")
    _style(axis, "Pixel-dates removed by the cloud/shadow mask before compositing")
    figure.tight_layout()
    return figure


def draw_window_counts(rows):
    """Parcel groups with a complete fixed or aligned window (window_counts.csv)."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, window in zip(axes, ("fixed", "aligned")):
        values = {crop: [int(next((r["parcel_groups"] for r in rows if r["window"] == window
                                   and r["partition"] == p and r["crop"] == crop), 0))
                         for p in PARTITIONS] for crop in CROPS}
        _crop_bars(axis, list(PARTITIONS), values, targets=partition_targets("parcels"))
        _style(axis, f"{'Fixed calendar' if window == 'fixed' else 'Aligned green-up'} window")
    axes[0].set_ylabel("Parcel groups with a complete window")
    axes[1].legend(frameon=False, ncol=4, fontsize=8)
    figure.suptitle("Parcels with a complete four-month window / parcel target (before pixel capping)", x=0.01, ha="left",
                    fontweight="bold", color=INK)
    figure.tight_layout()
    return figure


def draw_usable_vs_targets(rows, cohort):
    """Usable parcels and pixels against the collection targets for one cohort (usable_samples.csv)."""
    selected = [r for r in rows if r["cohort"] == cohort]
    if not selected:
        raise ValueError(f"No usable-sample rows for {cohort}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (measure, target_key, label) in zip(axes, (
            ("usable_parcel_groups", "target_parcels", "Parcels"),
            ("usable_pixels", "target_pixels", "Pixels"))):
        values = {crop: [int(next(r[measure] for r in selected if r["crop"] == crop and r["partition"] == p))
                         for p in PARTITIONS] for crop in CROPS}
        targets = [int(next(r[target_key] for r in selected if r["partition"] == p)) for p in PARTITIONS]
        _crop_bars(axis, list(PARTITIONS), values, targets=targets)
        _style(axis, label)
    axes[0].legend(frameon=False, ncol=4, loc="upper right", fontsize=8)
    figure.suptitle(f"Usable samples / target per crop: {cohort} (dashed = target)", x=0.01,
                    ha="left", fontweight="bold", color=INK)
    figure.tight_layout()
    return figure


def parcel_evidence_examples(quality):
    """Pick one parcel per source dataset and crop from a quality handoff (quality.json content).

    The choice is deterministic: among parcels with an annual image, take the one with the
    median selected-pixel count. It never looks at labels beyond the crop or at model results.
    """
    images = {}
    for row in quality.get("quality_rows", []):
        if row.get("image_path") and Path(row["image_path"]).suffix == ".tif":
            images.setdefault(row["region_id"], row["image_path"])
    groups = {}
    for region in quality.get("regions", []):
        properties = region["properties"]
        if properties["fragment_id"] in images:
            source = properties.get("source_dataset") or properties["partition_group"].split("|")[0]
            groups.setdefault((source, properties["crop"]), []).append(region)
    footprints = {}
    for source in quality.get("sources", []):
        path = Path(source.get("field_folder", "")) / "source_footprints.geojson"
        if path.is_file():
            for feature in json.loads(path.read_text())["features"]:
                footprints.setdefault(feature["properties"]["partition_group"], []).append(feature["geometry"])
    examples = []
    for (source, crop), regions in sorted(groups.items()):
        regions.sort(key=lambda r: (int(r["properties"]["selected_pixels"]), r["properties"]["fragment_id"]))
        region = regions[len(regions) // 2]
        examples.append({"source": source, "crop": crop, "properties": region["properties"],
                         "image_path": images[region["properties"]["fragment_id"]],
                         "footprints": footprints.get(region["properties"]["partition_group"], [])})
    return examples


def _interval_names(year, start_month):
    names = []
    for offset in range(4):
        month = (start_month - 1 + offset) % 12 + 1
        names += [f"{year}_{month:02d}_01_15", f"{year}_{month:02d}_16_end"]
    return names


def _bin_label(name):
    """'2022_01_16_end' -> 'Jan 16–end 2022'."""
    year, month, first, last = name.split("_")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{months[int(month) - 1]} {int(first)}–{last} {year}"


def draw_parcel_evidence(example, start_month=1):
    """Show a reference parcel outline over its downloaded Sentinel-2 NDVI for eight half-months.

    Only the saved parcel pixels exist in the annual TIFF; empty cells are outside the selection
    or failed the cloud mask. Nothing is filled or interpolated.
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.plot import plotting_extent
    import shapely
    from shapely.geometry import shape

    properties = example["properties"]
    year = int(properties.get("source_year") or properties["year"])
    with rasterio.open(example["image_path"]) as image:
        bands = {name: index + 1 for index, name in enumerate(image.descriptions)}
        extent = plotting_extent(image)
        to_image = Transformer.from_crs("EPSG:4326", image.crs, always_xy=True).transform
        outlines = [shapely.transform(shape(geometry), lambda xy: np.column_stack(to_image(xy[:, 0], xy[:, 1])))
                    for geometry in example["footprints"]]
        layers = []
        for name in _interval_names(year, start_month):
            if f"{name}__B8" not in bands:
                layers.append((name, None))
                continue
            red = image.read(bands[f"{name}__B4"], masked=True).astype(float) * 0.0001
            nir = image.read(bands[f"{name}__B8"], masked=True).astype(float) * 0.0001
            layers.append((name, (nir - red) / (nir + red)))
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for axis, (name, ndvi) in zip(axes.flat, layers):
        axis.set_facecolor("#f1f0ec")
        if ndvi is not None:
            shown = axis.imshow(ndvi, extent=extent, cmap="YlGn", vmin=0, vmax=0.9)
            valid = int(ndvi.count())
            axis.set_title(f"{_bin_label(name)}\n{valid} clear pixels", fontsize=9)
        else:
            axis.set_title(f"{_bin_label(name)}\nnot in archive", fontsize=9)
        for outline in outlines:
            for polygon in getattr(outline, "geoms", [outline]):
                axis.plot(*polygon.exterior.xy, color=TARGET_COLOR, linewidth=1.2)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks([])
        axis.set_yticks([])
    if any(ndvi is not None for _, ndvi in layers):
        figure.colorbar(shown, ax=axes, fraction=0.02, pad=0.01, label="NDVI")
    evidence = properties.get("label_evidence") or f"{properties.get('dataset', example['source'])} declared crop"
    figure.suptitle(f"{example['source']} · {properties['country']} {year} · parcel {properties['parcel_id']} · "
                    f"label: {example['crop']} ({properties.get('label_status', 'unknown')})\n"
                    f"Evidence: {evidence}. Red line = reference parcel outline; 10 m pixels.",
                    x=0.01, ha="left", fontsize=10, color=INK)
    return figure


def load_validation_scores(model_index):
    """Validation metrics stored in each frozen model manifest listed by models.json."""
    index = json.loads(Path(model_index).read_text())
    return {name: json.loads(Path(entry["path"]).read_text())["validation"] for name, entry in index.items()}


def baseline_macro_f1(baselines):
    """Read majority_baseline.json content into {cohort: macro-F1 or None}."""
    return {name: entry.get("metrics", {}).get("macro_f1") for name, entry in baselines.items()}


def draw_macro_f1(scores, title, baseline=None, comparison=None, comparison_label="Validation"):
    """Macro-F1 per cohort, optional comparison bars, majority baseline and the 0.50 target."""
    cohorts = [c for c in COHORTS if c in scores]
    positions = np.arange(len(cohorts))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.38 if comparison else 0.6
    main_offset = width / 2 if comparison else 0
    if comparison:
        bars = axis.bar(positions - width / 2, [comparison.get(c, {}).get("macro_f1") or 0 for c in cohorts],
                        width - 0.02, color=NEUTRAL, label=comparison_label)
        axis.bar_label(bars, labels=[_fmt(comparison.get(c, {}).get("macro_f1")) for c in cohorts],
                       padding=2, fontsize=9, color=MUTED)
    main = [scores[c].get("macro_f1") for c in cohorts]
    bars = axis.bar(positions + main_offset, [v or 0 for v in main], width - 0.02, color=COLORS["potato"],
                    label="External test" if comparison else "Macro-F1")
    axis.bar_label(bars, labels=[_fmt(v) for v in main], padding=2, fontsize=9, color=MUTED)
    if baseline:
        for index, cohort in enumerate(cohorts):
            if baseline.get(cohort) is not None:
                centre = positions[index] + main_offset
                axis.hlines(baseline[cohort], centre - width / 2, centre + width / 2,
                            colors=INK, linewidth=2, linestyles=(0, (2, 1.5)))
        axis.plot([], [], color=INK, linewidth=2, linestyle=(0, (2, 1.5)), label="Training-majority baseline")
    axis.axhline(F1_TARGET, color=TARGET_COLOR, linewidth=1, alpha=0.7)
    axis.annotate("0.50 target", (positions[-1] + 0.45, F1_TARGET), xytext=(0, 3),
                  textcoords="offset points", ha="right", fontsize=8, color=TARGET_COLOR)
    axis.set_xticks(positions, [COHORT_LABELS[c] for c in cohorts])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Parcel macro-F1")
    _style(axis, title)
    axis.legend(frameon=False, loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def draw_per_crop_f1(scores, title):
    cohorts = [c for c in COHORTS if c in scores]
    values = {crop: [scores[c].get("per_crop", {}).get(crop, {}).get("f1") for c in cohorts] for crop in CROPS}
    values = {crop: [None if v is None else round(v, 2) for v in vals] for crop, vals in values.items()}
    figure, axis = plt.subplots(figsize=(9, 4.2))
    _crop_bars(axis, [COHORT_LABELS[c] for c in cohorts], values)
    axis.axhline(F1_TARGET, color=TARGET_COLOR, linewidth=1, alpha=0.7)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    _style(axis, title)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    return figure


def draw_confusion_matrices(scores, title):
    """Parcel confusion matrices; shading is the row share so unequal supports compare fairly."""
    cohorts = [c for c in COHORTS if scores.get(c, {}).get("confusion_matrix")]
    if not cohorts:
        raise ValueError("No confusion matrices are available")
    figure, axes = plt.subplots(1, len(cohorts), figsize=(3.8 * len(cohorts), 4), squeeze=False)
    for axis, cohort in zip(axes[0], cohorts):
        metrics = scores[cohort]
        classes = metrics["classes"]
        matrix = np.array(metrics["confusion_matrix"])
        totals = matrix.sum(axis=1, keepdims=True)
        share = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
        axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                axis.text(j, i, matrix[i, j], ha="center", va="center", fontsize=12,
                          color="white" if share[i, j] > 0.55 else INK)
        axis.set_xticks(range(len(classes)), classes)
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Predicted")
        axis.set_title(f"{COHORT_LABELS[cohort].replace(chr(10), ' ')}\nmacro-F1 {_fmt(metrics.get('macro_f1'))}",
                       fontsize=10, color=INK)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0][0].set_ylabel("Reference crop")
    figure.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    figure.tight_layout()
    return figure
