"""Explain a completed pilot using saved records, without downloading or fitting models."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CROPS = ("carrot", "cabbage", "potato")
COLORS = ("#d97706", "#218559", "#7560a8")
PARTITIONS = ("train", "validation", "test")


def read_table(path):
    """Read saved results explicitly so missing inputs cannot become misleading zero counts."""
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def candidate_and_timing_counts(report_root, run_name):
    """Count each region once even when several qualified cycles share its pixels."""
    candidates, timing = Counter(), Counter()
    source_snapshot = (
        Path(report_root).parent / "analysis" / run_name / "source_inputs.json"
    )
    if source_snapshot.exists():
        sources = json.loads(source_snapshot.read_text())
    else:
        sources = [
            json.loads(
                (
                    Path(report_root).parent
                    / "source_handoffs"
                    / f"{name}_{run_name}.json"
                ).read_text()
            )
            for name in ("sen4agrinet", "francecrops")
        ]
    for source in sources:
        regions_path = Path(source["field_folder"]) / "sampled_download_regions.geojson"
        regions = json.loads(regions_path.read_text())["features"]
        gate_path = (
            Path(source["progress_folder"]).parent / "timing_first" / "timing_gate.json"
        )
        if source.get("collection") == "full_year_all_parcels":
            window_path = source_snapshot.parent / "seasons/estimated_windows.json"
            windows = json.loads(window_path.read_text())
        else:
            windows = json.loads(gate_path.read_text())["qualified_windows"]
        year = int(source["year"])
        eligible_regions = {
            window["region_id"]
            for window in windows
            if window["window_start"] >= f"{year}-01-01"
            and window["window_end"] <= f"{year + 1}-01-01"
        }
        for region in regions:
            properties = region["properties"]
            crop = properties["crop"]
            candidates[crop] += int(properties["selected_pixels"])
            if properties["fragment_id"] in eligible_regions:
                timing[crop] += int(properties["selected_pixels"])
    return candidates, timing


def save_figure(figure, folder, name):
    figure.savefig(
        folder / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    return figure


def label_bars(axis, bars):
    axis.bar_label(bars, padding=3, fontsize=9, fmt="%.0f")
    axis.margins(y=0.18)


def draw_workflow(annual_collection=False):
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.set_axis_off()
    stages = [
        "Crop labels\nand parcel locations",
        "Candidate pixels\ninside parcels",
        "Cloud-filtered\ndaily timelines",
        "Timing checks\nand proposed windows",
        "Median images\nfor qualified windows",
        "Complete windows\nand retained pixels",
        "Features and\nparcel-separated splits",
        "Models and\nsaved evaluation",
    ]
    if annual_collection:
        stages[1:5] = ["All eligible interior pixels\nwith parcel groups", "Earth Engine masks\nand daily mosaics", "Full-year median images\nand dated summaries", "Timing checks\non the saved archive"]
    positions = [(0.12 + index * 0.25, 0.72) for index in range(4)]
    positions += [(0.87 - index * 0.25, 0.24) for index in range(4)]
    for label, position in zip(stages, positions):
        axis.text(
            *position,
            label,
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": "#edf4f8",
                "edgecolor": "#527487",
            },
        )
    for start, end in zip(positions, positions[1:]):
        vertical = start[0] == end[0]
        start_point = (
            (start[0], start[1] - 0.1)
            if vertical
            else (start[0] + (0.105 if end[0] > start[0] else -0.105), start[1])
        )
        end_point = (
            (end[0], end[1] + 0.1)
            if vertical
            else (end[0] - (0.105 if end[0] > start[0] else -0.105), end[1])
        )
        axis.annotate(
            "",
            xy=end_point,
            xytext=start_point,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#527487", "lw": 1.7},
        )
    figure.suptitle(
        "From labelled parcels to saved evaluation", fontsize=15
    )
    figure.text(
        0.5,
        0.01,
        "Annual imagery is saved before window selection; one parcel stays in one split." if annual_collection else "Timing is estimated before image downloads; one parcel stays in one split.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return figure


def draw_pixel_counts(candidates, timing, selection, annual_collection=False):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for axis, crop, color in zip(axes, CROPS, COLORS):
        record = selection[crop]
        values = [
            candidates[crop],
            timing[crop],
            int(record["available_usable_pixels"]),
            int(record["selected_usable_pixels"]),
        ]
        bars = axis.bar(range(4), values, color=color)
        label_bars(axis, bars)
        axis.set_xticks(
            range(4),
            [
                "Candidates",
                "Timing\nqualified",
                "Complete\nunique eligible",
                "Retained",
            ],
            fontsize=9,
        )
        axis.set_title(crop.capitalize())
        axis.set_ylabel("Sampled pixel locations")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Where pixels are filtered and selected", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Archive pixels are counted once per tile. Complete counts exclude upstream holdouts; all usable development pixels are retained." if annual_collection else "Timing counts cover in-year regions, once each. Complete counts include deduplication; retention applies the crop cap.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


def draw_split_counts(summary):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    positions = np.arange(len(PARTITIONS))
    for axis, field, title in zip(
        axes,
        ("pixels", "parcel_groups"),
        ("Pixels per split", "Parcel groups per split"),
    ):
        for index, (crop, color) in enumerate(zip(CROPS, COLORS)):
            values = [
                next(
                    row[field]
                    for row in summary
                    if row["crop"] == crop and row["partition"] == partition
                )
                for partition in PARTITIONS
            ]
            bars = axis.bar(
                positions + (index - 1) * 0.24,
                values,
                width=0.24,
                color=color,
                label=crop.capitalize(),
            )
            label_bars(axis, bars)
        axis.set_xticks(positions, ["Training", "Validation", "Test"])
        axis.set_title(title)
        axis.set_ylabel("Count")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle("One source parcel stays entirely within one split", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def draw_training_weights(rows):
    totals = Counter()
    for row in rows:
        totals[row["crop"]] += float(row["weight"])
    total_weight = sum(totals.values())
    figure, axis = plt.subplots(figsize=(7, 3.6))
    bars = axis.bar(
        [crop.capitalize() for crop in CROPS],
        [100 * totals[crop] / total_weight for crop in CROPS],
        color=COLORS,
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Share of total training weight (%)")
    axis.set_title("Training influence by crop")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def selected_design(metrics):
    return metrics["feature_selection"]["external_parcel_holdout"]["selected_design"]


def draw_confusion(metrics):
    design = selected_design(metrics)
    result = metrics["test"][design]
    matrix = np.asarray(result["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6.2, 5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    threshold = matrix.max() / 2
    for actual, predicted in np.ndindex(matrix.shape):
        axis.text(
            predicted,
            actual,
            str(matrix[actual, predicted]),
            ha="center",
            va="center",
            color="white" if matrix[actual, predicted] > threshold else "black",
            fontsize=12,
        )
    labels = [crop.capitalize() for crop in result["classes"]]
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted crop")
    axis.set_ylabel("Reported crop")
    axis.set_title(
        f"Test confusion matrix: {design}\nValidation-selected model; counts are pixels"
    )
    figure.colorbar(image, ax=axis, label="Pixel count", shrink=0.8)
    figure.tight_layout()
    return figure


def draw_model_comparison(metrics):
    designs = list(metrics["validation"])
    selected = selected_design(metrics)
    labels = [
        design.replace("interval_and_shape", "Interval + shape")
        .replace("interval_only", "Interval only")
        .replace("shape_only", "Shape only")
        + ("\n(selected on validation)" if design == selected else "")
        for design in designs
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = np.arange(len(designs))
    for axis, metric, title in zip(
        axes,
        ("macro_f1", "accuracy"),
        ("Macro-F1 (selection uses validation)", "Accuracy"),
    ):
        for offset, stage, color in [
            (-0.18, "validation", "#218559"),
            (0.18, "test", "#527ab5"),
        ]:
            values = [metrics[stage][design][metric] for design in designs]
            bars = axis.bar(
                positions + offset,
                values,
                width=0.36,
                color=color,
                label=stage.capitalize(),
            )
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_xticks(positions, labels, fontsize=9)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score (0–1)")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    return figure


def generate_report(report_root, run_name, training_run_name):
    """Render only existing results; absence raises instead of triggering processing."""
    seasons = Path(report_root).parent / "analysis" / run_name / "seasons"
    training = seasons / training_run_name
    metrics = json.loads((training / "experiment_metrics.json").read_text())
    weights = read_table(training / "training_weights.csv")
    selection = {
        row["crop"]: row for row in read_table(seasons / "usable_pixel_selection.csv")
    }
    candidates, timing = candidate_and_timing_counts(report_root, run_name)
    output = training / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    annual_collection = run_name.startswith("full_year_all_parcels_")
    figures = {
        "01_pipeline": draw_workflow(annual_collection),
        "02_pixel_counts": draw_pixel_counts(candidates, timing, selection, annual_collection),
        "03_parcel_splits": draw_split_counts(metrics["split_summary"]),
        "04_training_weights": draw_training_weights(weights),
        "05_test_confusion": draw_confusion(metrics),
        "06_model_comparison": draw_model_comparison(metrics),
    }
    for name, figure in figures.items():
        save_figure(figure, output, name)
    return figures, output
