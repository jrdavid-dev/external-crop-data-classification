"""Read saved preprocessing artifacts and explain one deterministic example per crop."""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.dates as date_labels
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.plot import plotting_extent
from shapely.geometry import shape
from shapely.ops import transform as project_geometry

from crop_pipeline.windows import seasons as season_workflow

CROPS = ("carrot", "cabbage", "potato")


def read_table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def choose_examples(windows, retained_rows):
    """Use median retained-pixel count, not model performance or a hand-picked curve."""
    retained_counts = Counter(
        (row["crop"], row["partition_group"]) for row in retained_rows
    )
    examples = {}
    for crop in CROPS:
        candidates = [
            window
            for window in windows
            if window["crop"] == crop
            and retained_counts[(crop, window["partition_group"])]
        ]
        candidates.sort(
            key=lambda window: (
                retained_counts[(crop, window["partition_group"])],
                window["partition_group"],
                window["region_id"],
            )
        )
        if not candidates:
            raise ValueError(
                f"No saved selected window with retained pixels for {crop}."
            )
        examples[crop] = candidates[len(candidates) // 2]
    return examples


def load_reference_shapes(source, region):
    """Use actual saved reference geometry; sampled footprints are not parcel outlines."""
    field_folder = Path(source["field_folder"])
    if source.get("collection") == "full_year_all_parcels":
        features = json.loads((field_folder / "source_footprints.geojson").read_text())["features"]
        return [shape(feature["geometry"]) for feature in features
                if feature["properties"]["partition_group"] == region["properties"]["partition_group"]]
    filename = field_folder / "cabbage_training_parcels.geojson"
    if source["source_name"] == "sen4agrinet":
        filename = field_folder.parent / "download_candidates.geojson"
    if source["source_name"].startswith("cropharvest_"):
        filename = field_folder / "cropharvest_parcels.geojson"
    features = json.loads(filename.read_text())["features"]
    identifiers = set(region["properties"]["source_fragments"])
    return [
        shape(feature["geometry"])
        for feature in features
        if feature["properties"].get("fragment_id") in identifiers
    ]


def load_context(report_root, run_name):
    base = Path(report_root).parent
    analysis = base / "analysis" / run_name
    seasons = analysis / "seasons"
    windows = json.loads((seasons / "estimated_windows.json").read_text())
    retained = read_table(seasons / "estimated_features" / "complete_features.csv")
    examples = choose_examples(windows, retained)
    observations = json.loads(
        (analysis / "timelines" / "observed_timelines.json").read_text()
    )
    contexts = {}
    snapshot_path = analysis / "source_inputs.json"
    if snapshot_path.exists():
        sources = json.loads(snapshot_path.read_text())
    else:
        sources = [
            json.loads(
                (base / "source_handoffs" / f"{name}_{run_name}.json").read_text()
            )
            for name in ("sen4agrinet", "francecrops")
        ]
    for source in sources:
        source_name = source["source_name"]
        regions_path = Path(source["field_folder"]) / "sampled_download_regions.geojson"
        regions = {
            region["properties"]["fragment_id"]: region
            for region in json.loads(regions_path.read_text())["features"]
        }
        quality = read_table(
            analysis / "quality" / source_name / "composite_quality.csv"
        )
        for crop, window in examples.items():
            if window["region_id"] not in regions:
                continue
            region = regions[window["region_id"]]
            quality_map = {
                row["name"]: row
                for row in quality
                if row["region_id"] == window["region_id"]
            }
            intervals = season_workflow.window_intervals(window)
            images = [
                load_image(quality_map[interval["name"]]["image_path"], region, interval["name"])
                for interval in intervals
            ]
            contexts[crop] = {
                "window": window,
                "region": region,
                "source": source,
                "images": images,
                "intervals": intervals,
                "observations": [
                    row
                    for row in observations
                    if row["region_id"] == window["region_id"]
                ],
                "retained": [
                    row
                    for row in retained
                    if row["crop"] == crop
                    and row["partition_group"] == window["partition_group"]
                ],
                "reference_shapes": load_reference_shapes(source, region),
            }
    return contexts


def load_image(path, region, interval=None):
    if not path:
        raise FileNotFoundError(
            "The example's selected window has a missing image path."
        )
    if Path(path).suffix == ".parquet":
        if interval is None:
            raise ValueError("Choose a half-month period for the annual pixel table")
        import pyarrow.parquet as parquet
        from affine import Affine
        from crop_pipeline.imagery.quality import STUDY_BANDS
        grid = region["properties"]["sampling_grid"]
        frame = parquet.read_table(path, filters=[("interval", "=", interval)]).to_pandas()
        if not len(frame):
            raise ValueError("The chosen period is absent from the pixel table")
        values = np.full((7, grid["height"], grid["width"]), np.nan)
        values[:, frame["row"].to_numpy(), frame["column"].to_numpy()] = frame[STUDY_BANDS + ["clear_count"]].to_numpy().T
        values[values == -9999] = np.nan
        transform = Affine(*grid["transform"])
        crs = rasterio.crs.CRS.from_user_input(grid["crs"])
        extent = plotting_extent(values[0], transform=transform)
    else:
        with rasterio.open(path) as image:
            if image.count == 168:
                if interval is None:
                    raise ValueError("Choose a half-month period for the annual TIFF")
                from crop_pipeline.imagery.quality import STUDY_BANDS
                indices = [image.descriptions.index(f"{interval}__{band}") + 1
                           for band in STUDY_BANDS + ["clear_count"]]
                values = image.read(indices, masked=True).filled(np.nan).astype(float)
            else:
                values = image.read(masked=True).filled(np.nan).astype(float)
            extent = plotting_extent(image)
            transform = image.transform
            crs = image.crs
    with rasterio.open(region["properties"]["sampling_mask"]) as mask:
        if (
            mask.crs != crs
            or mask.transform != transform
            or mask.shape != values.shape[1:]
        ):
            raise ValueError("Example image and sampling-mask grids differ.")
        selected = mask.read(1) == 1
    valid = np.isfinite(values[:6]).all(axis=0) & (values[6] > 0)
    # The fixed stretch keeps brightness comparable across all eight intervals.
    rgb = np.moveaxis(values[[1, 0, 5]] * 0.0001, 0, -1)
    rgb = np.nan_to_num(np.clip(rgb / 0.3, 0, 1))
    rgb[~valid] = [0.9, 0.9, 0.9]
    return {
        "rgb": rgb,
        "values": values,
        "valid": valid,
        "selected": selected,
        "extent": extent,
        "transform": transform,
        "crs": crs,
    }


def pixel_centers(image, mask):
    rows, columns = np.where(mask)
    return rasterio.transform.xy(image["transform"], rows, columns)


def show_image(axis, image):
    axis.imshow(
        image["rgb"], extent=image["extent"], origin="upper", interpolation="nearest"
    )
    axis.set_aspect("equal")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.tick_params(labelsize=8)
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(axis="x", labelrotation=25)
    axis.set_xlabel("Easting (m)", fontsize=9)
    axis.set_ylabel("Northing (m)", fontsize=9)


def draw_boundaries(axis, geometry):
    if geometry.geom_type == "Polygon":
        x_coordinates, y_coordinates = geometry.exterior.xy
        axis.plot(x_coordinates, y_coordinates, color="#ffcf33", linewidth=1.6)
        return
    for component in geometry.geoms:
        draw_boundaries(axis, component)


def draw_sampling(context):
    image = context["images"][0]
    figure, axis = plt.subplots(figsize=(7.5, 5.6))
    show_image(axis, image)
    transformer = Transformer.from_crs("EPSG:4326", image["crs"], always_xy=True)
    for geometry in context["reference_shapes"]:
        draw_boundaries(axis, project_geometry(transformer.transform, geometry))
    x_coordinates, y_coordinates = pixel_centers(image, image["selected"])
    axis.scatter(
        x_coordinates,
        y_coordinates,
        s=23,
        facecolors="none",
        edgecolors="#ffcf33",
        label="Candidate pixels",
    )
    axis.scatter(
        [float(row["x"]) for row in context["retained"]],
        [float(row["y"]) for row in context["retained"]],
        s=13,
        color="#00c7e6",
        label="Retained pilot pixels",
        zorder=4,
    )
    left, right, bottom, top = image["extent"]
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_title(
        "Reference outline and sampled pixels\nImage extent may clip the reference boundary"
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    figure.tight_layout()
    return figure


def draw_screening(context):
    # The interval with the most excluded candidates best explains the saved screening.
    index = min(
        range(8),
        key=lambda position: np.sum(
            context["images"][position]["selected"]
            & context["images"][position]["valid"]
        ),
    )
    image = context["images"][index]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis in axes:
        show_image(axis, image)
    axes[0].set_title("Saved composite after cloud/quality masking")
    for accepted, color, label in [
        (True, "#00c7e6", "Valid sampled pixels"),
        (False, "#e13a50", "No valid saved observation"),
    ]:
        mask = image["selected"] & (image["valid"] if accepted else ~image["valid"])
        x_coordinates, y_coordinates = pixel_centers(image, mask)
        axes[1].scatter(
            x_coordinates,
            y_coordinates,
            color=color,
            s=20,
            label=f"{label}: {int(mask.sum())}",
        )
    axes[1].set_title("Sampled-pixel eligibility in the saved composite")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    figure.suptitle(
        context["intervals"][index]["name"] + " — no unmasked original scene is stored",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_observed_segments(axis, observations, index, color):
    valid = [
        row
        for row in observations
        if row.get("eligible") and row.get(index) is not None
    ]
    segments = []
    current = []
    for row in valid:
        observation_date = date.fromisoformat(row["date"])
        if current and (observation_date - current[-1][0]).days > 20:
            segments.append(current)
            current = []
        current.append((observation_date, float(row[index])))
    segments.append(current)
    for segment in segments:
        axis.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            ".-",
            color=color,
            linewidth=1,
            label=index if segment is segments[0] else None,
        )


def draw_timeline(context):
    window = context["window"]
    observations = sorted(context["observations"], key=lambda row: row["date"])
    start, end = date.fromisoformat(window["window_start"]), date.fromisoformat(
        window["window_end"]
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(12, 5.7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for index, color in [("NDVI", "#218559"), ("NDRE", "#8658a8"), ("NDMI", "#347fb5")]:
        plot_observed_segments(axes[0], observations, index, color)
    axes[0].axhline(
        float(window["ndvi_threshold"]),
        color="#218559",
        linestyle=":",
        linewidth=1,
        label="NDVI rise threshold",
    )
    onset_start = date.fromisoformat(window["onset_interval_start"])
    onset_end = date.fromisoformat(window["onset_interval_end"])
    axes[0].axvspan(
        onset_start,
        onset_end,
        color="#dd7b37",
        alpha=0.3,
        label="Estimated onset interval",
    )
    for axis in axes:
        axis.axvspan(
            start,
            end,
            color="#f6ce63",
            alpha=0.2,
            label="Selected four-month window" if axis is axes[0] else None,
        )
        axis.axvline(start, color="#b18112", linewidth=1)
        axis.axvline(end, color="#b18112", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)
    dates = [date.fromisoformat(row["date"]) for row in observations]
    axes[1].scatter(
        dates,
        [float(row["valid_fraction"]) for row in observations],
        s=12,
        color="#4e6470",
    )
    axes[1].axhline(
        0.8, color="#b04343", linestyle="--", label="80% observation coverage"
    )
    axes[0].set_ylabel("Vegetation index")
    axes[1].set_ylabel("Valid sampled\npixel fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Actual observation date (gaps are not filled)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    axes[1].xaxis.set_major_locator(date_labels.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(date_labels.DateFormatter("%b %Y"))
    figure.suptitle(
        f"{window['crop'].capitalize()}: selected {start:%d %b %Y} to {end:%d %b %Y} (end excluded)\nShading is an estimated observation window, not recorded planting/harvest",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def draw_eight_intervals(context):
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.7))
    for axis, interval, image in zip(
        axes.flat, context["intervals"], context["images"]
    ):
        axis.imshow(image["rgb"], origin="upper", interpolation="nearest")
        rows, columns = np.where(image["selected"])
        axis.scatter(
            columns, rows, s=7, facecolors="none", edgecolors="#ffcf33", linewidths=0.5
        )
        valid_count = int(np.sum(image["selected"] & image["valid"]))
        axis.set_title(
            f"{interval['start']} to {interval['end']}\n{valid_count}/{int(image['selected'].sum())} valid candidates",
            fontsize=9,
        )
        axis.set_axis_off()
    figure.suptitle(
        "Eight half-month composites — yellow rings: candidate pixels; grey: no valid saved observation",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.01,
        "Same RGB stretch (B4/B3/B2, reflectance 0–0.3) in every panel. Images are post-mask composites.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    return figure


def generate_examples(report_root, run_name):
    """Generate PNGs and an example manifest entirely from saved local files."""
    contexts = load_context(report_root, run_name)
    output = Path(report_root).parent / "analysis" / run_name / "preprocessing_visuals"
    output.mkdir(parents=True, exist_ok=True)
    figures, manifest = {}, []
    for crop in CROPS:
        context = contexts[crop]
        figures[crop] = {
            "sampling": draw_sampling(context),
            "screening": draw_screening(context),
            "timeline": draw_timeline(context),
            "eight_intervals": draw_eight_intervals(context),
        }
        for name, figure in figures[crop].items():
            figure.savefig(
                output / f"{crop}_{name}.png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )
        window = context["window"]
        manifest.append(
            {
                key: window[key]
                for key in (
                    "crop",
                    "region_id",
                    "partition_group",
                    "window_start",
                    "window_end",
                    "selected_pixels",
                )
            }
        )
    (output / "examples.json").write_text(
        json.dumps(
            {
                "selection_rule": "median_retained_pixel_count_then_parcel_and_region_id; no model scores used",
                "examples": manifest,
                "limitations": "Post-mask composites only; no unmasked scenes or original per-scene cloud masks stored. Reference outlines may be clipped by the downloaded extent.",
            },
            indent=2,
        )
        + "\n"
    )
    return figures, output
