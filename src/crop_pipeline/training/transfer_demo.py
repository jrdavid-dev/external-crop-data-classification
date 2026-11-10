"""Compact helpers for the notebook 07 v3 demonstration.

Training and scoring reuse ``external_combined``; this module only loads the locked
external data, tabulates model comparisons and draws the demo figures. The long
diagnostic analyses stay in notebook 07 v2, whose saved figures v3 displays.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.training import external_combined as ext

CLASSES = list(ext.CLASSES)
SOURCE_NAMES = {"cropharvest": "CropHarvest", "sen4agrinet": "Sen4AgriNet", "francecrops": "FranceCrops"}
COLORS = {"carrot": "#F58518", "cabbage": "#54A24B", "potato": "#B279A2"}
# Name -> (external_combined model name, feature design); the RF row is the prespecified prototype.
MODEL_SPECS = {
    "RF, interval (prespecified prototype)": ("random_forest", "interval_only"),
    "LR, interval": ("logistic_regression", "interval_only"),
    "RF, shape": ("random_forest", "shape_only"),
    "LR, shape (v1 CV-selected)": ("logistic_regression", "shape_only"),
}


def add_source_regions(data):
    """Readable source and source-region names, as in notebook 07 v2."""
    data = data.copy()
    data["source"] = data["domain"].map(SOURCE_NAMES)
    data["region_name"] = np.select(
        [data["domain"] == "sen4agrinet", data["domain"] == "francecrops", data["region"] == "France"],
        ["Catalonia (ES)", "France", "Île-de-France"],
        data["region"],
    )
    data["source_region"] = data["source"] + ": " + data["region_name"]
    return data


def load_external(v1_folder):
    """Rebuild the pooled external table from v1's cached inputs without writing anything.

    ``lock_test_split`` only verifies v1's saved split here, so the test partition is the
    same one every earlier result used.
    """
    v1_folder = Path(v1_folder)
    split_path = v1_folder / "test_split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"{split_path} is missing; run notebook 07 (v1) first.")
    paths = {table: v1_folder / "inputs" / f"{table}.csv" for table in ext.ASSETS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cached input tables are missing; run notebook 07 (v1) first: " + ", ".join(missing))
    # Same reader as ext.cached_tables, without rewriting its input manifest.
    tables = {table: pd.read_csv(path, dtype={"partition_group": str, "asset_index": str})
              for table, path in paths.items()}
    data = ext.add_derived_features(ext.harmonize(tables))
    data, _ = ext.link_split_units(data)
    data = ext.add_spatial_blocks(data)
    return add_source_regions(ext.lock_test_split(data, split_path))


def coverage_table(data):
    """Parcels per crop and source region, split into development and test."""
    return (
        data.groupby(["source_region", "partition", "crop"])["partition_group"].nunique()
        .unstack(["partition", "crop"], fill_value=0)
        .reindex(columns=pd.MultiIndex.from_product([["development", "test"], CLASSES]), fill_value=0)
    )


def _row(name, metrics, interval=None):
    row = {"model": name, "accuracy": metrics["accuracy"],
           "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
           "macro_f1_95ci": "" if interval is None
           else f"{interval['macro_f1_low']:.3f}-{interval['macro_f1_high']:.3f}"}
    row.update({f"recall_{crop}": metrics["per_crop"][crop]["recall"] for crop in CLASSES})
    return row


def baseline_rows(test):
    """Always-majority and class-share guessing, the floor every model must beat."""
    shares = test["crop"].value_counts(normalize=True).reindex(CLASSES, fill_value=0.0)
    majority = shares.idxmax()
    one_hot = np.zeros((len(test), len(CLASSES)))
    one_hot[:, CLASSES.index(majority)] = 1
    present = int((shares > 0).sum())
    guess = {"model": "Baseline: guess with class shares (expected)", "accuracy": float((shares ** 2).sum()),
             "balanced_accuracy": 1 / present, "macro_f1": 1 / present, "macro_f1_95ci": ""}
    guess.update({f"recall_{crop}": float(shares[crop]) for crop in CLASSES})
    return [_row(f"Baseline: always {majority}", ext.evaluate(test, one_hot)), guess]


def train_and_score(data, specs=None):
    """Fit each model on development parcels and score it once on the locked test set.

    Returns the comparison table and ``{name: (model, columns, test probabilities)}``.
    """
    specs = MODEL_SPECS if specs is None else specs
    designs = feature_experiments.feature_sets()
    development = data[data["partition"] == "development"]
    test = data[data["partition"] == "test"].reset_index(drop=True)
    rows, fitted = baseline_rows(test), {}
    for name, (model_name, design) in specs.items():
        columns = designs[design]
        model = ext.fit_model(model_name, development, columns)
        probabilities = ext.class_probabilities(model, test, columns)
        fitted[name] = (model, columns, probabilities)
        rows.append(_row(name, ext.evaluate(test, probabilities), ext.bootstrap_interval(test, probabilities)))
    return pd.DataFrame(rows).set_index("model"), fitted


def parcel_predictions(test, probabilities):
    """One row per test parcel: label, predicted crop and mean class probabilities."""
    frame = pd.DataFrame(probabilities, columns=CLASSES).assign(
        parcel=test["partition_group"].to_numpy(), source_region=test["source_region"].to_numpy(),
        label=test["crop"].to_numpy())
    parcels = frame.groupby("parcel").agg(
        {"source_region": "first", "label": "first", "parcel": "size", **{crop: "mean" for crop in CLASSES}}
    ).rename(columns={"parcel": "pixels"})
    parcels["predicted"] = parcels[CLASSES].idxmax(axis=1)
    parcels["correct"] = parcels["predicted"] == parcels["label"]
    return parcels


def plot_confusions(test, fitted, names):
    """Row-normalised test confusion matrices, side by side."""
    figure, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4), squeeze=False)
    for axis, name in zip(axes[0], names):
        matrix = np.asarray(ext.evaluate(test, fitted[name][2])["confusion_matrix"])
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                axis.text(j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center",
                          color="white" if shares[i, j] > 0.5 else "black")
        axis.set_xticks(range(len(CLASSES)), CLASSES)
        axis.set_yticks(range(len(CLASSES)), CLASSES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Label")
        axis.set_title(name, fontsize=10)
    figure.suptitle("Locked external test set (pixels, row %)")
    figure.tight_layout()
    return figure


def importance_grid(model, columns):
    """Forest impurity importance arranged as half-month bin x variable."""
    names = pd.Series(columns).str.extract(r"bin_(\d+)_(.+)")
    if names.isna().any().any():
        raise ValueError("importance_grid expects interval columns named bin_NN_<variable>.")
    grid = pd.DataFrame({"bin": names[0].astype(int), "variable": names[1],
                         "importance": model.feature_importances_})
    return grid.pivot(index="variable", columns="bin", values="importance").reindex(
        list(dict.fromkeys(names[1])))


def plot_importance_grid(grid):
    figure, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(grid.to_numpy(), cmap="viridis", aspect="auto")
    axis.set_xticks(range(grid.shape[1]), [f"H{b}" for b in grid.columns])
    axis.set_yticks(range(grid.shape[0]), grid.index)
    axis.set_xlabel("Half-month in the 4-month window")
    axis.set_title("Which bands and dates the prototype RF relies on (impurity importance)")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure


def text_table(frame, columns=None):
    """Left-aligned, wrapped display for tables of prose (specifications, limitations)."""
    frame = frame if columns is None else frame[columns]
    return frame.style.hide(axis="index").set_properties(
        **{"text-align": "left", "white-space": "pre-wrap", "vertical-align": "top"})


def split_check(data):
    """Parcels and pixels per partition; fails if any split unit or parcel is in both."""
    shared = {column: sorted(set(data.loc[data["partition"] == "development", column])
                             & set(data.loc[data["partition"] == "test", column]))
              for column in ("split_unit", "partition_group")}
    if any(shared.values()):
        raise ValueError(f"Development and test share units: {shared}")
    return (data.groupby(["partition", "crop"])
            .agg(parcels=("partition_group", "nunique"), pixels=("sample_id", "size"))
            .unstack("partition").swaplevel(axis=1).sort_index(axis=1))


def plot_class_distribution(inventory):
    """Parcels (fields for Atok) and samples per crop and source, from v2's inventory table."""
    used = inventory[inventory["samples"] > 0]
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, column, title in zip(axes, ("parcels", "samples"), ("Parcels / fields", "Samples (pixels or pixel-windows)")):
        table = used.pivot_table(index="source", columns="crop", values=column, aggfunc="sum", fill_value=0)
        table = table.reindex(columns=CLASSES, fill_value=0)
        table.plot.barh(ax=axis, stacked=True, color=[COLORS[c] for c in CLASSES], legend=axis is axes[0])
        axis.set_title(title)
        axis.set_ylabel("")
    figure.tight_layout()
    return figure


def read_kml(path, crs):
    """Named polygons from a KML file, reprojected to ``crs``."""
    import xml.etree.ElementTree as ET

    import geopandas as gpd
    from shapely.geometry import Polygon

    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    rows = []
    for placemark in ET.parse(path).getroot().iter(f"{{{namespace['k']}}}Placemark"):
        coordinates = placemark.find(".//k:coordinates", namespace).text.split()
        rows.append({"name": placemark.find("k:name", namespace).text,
                     "geometry": Polygon([tuple(map(float, c.split(",")[:2])) for c in coordinates])})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(crs)


def atok_grid(reference_tif, kml_path, landuse_path):
    """ROI and ELUM 'Crops' masks on the Atok analysis grid (taken from a v2 map GeoTIFF)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from shapely import union_all
    from shapely.geometry import mapping

    with rasterio.open(reference_tif) as reference:
        crs, transform, shape = reference.crs, reference.transform, reference.shape

    def rasterize(geometry):
        return geometry_mask([mapping(geometry)], out_shape=shape, transform=transform, invert=True)

    kml = read_kml(kml_path, crs)
    roi = kml.loc[kml["name"] == "Region of Interest", "geometry"].iloc[0]
    landuse = gpd.read_file(landuse_path, bbox=gpd.GeoSeries([roi], crs=crs), engine="pyogrio").to_crs(crs)
    crops = landuse.loc[(landuse["sub_elu"] == "Crops") & landuse.intersects(roi), "geometry"]
    roi_mask = rasterize(roi)
    return {"roi": roi_mask, "cropland": rasterize(union_all(crops.to_numpy())) & roi_mask,
            "transform": transform, "crs": crs, "kml": kml}


def load_cube(paths):
    """The cached Atok half-month cube from notebook 08 or 09."""
    for path in map(Path, paths):
        if path.is_file():
            return np.load(path)["cube"]
    raise FileNotFoundError("No Atok ROI cube cache; run notebook 08 or 09 first.")


def plot_validity(grid):
    """Year x half-month heatmap of valid observations; hatched cells are missing intervals."""
    from crop_pipeline.imagery.observation_quality import MONTHS

    figure, axis = plt.subplots(figsize=(11, 0.45 * len(grid) + 1.4))
    image = axis.imshow(grid.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i, j in zip(*np.nonzero(grid.isna().to_numpy())):
        axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="xx", edgecolor="#E45756", lw=0))
    axis.set_xticks(np.arange(0, 24, 2) + 0.5, MONTHS)
    axis.set_yticks(range(len(grid)), grid.index)
    axis.set_title("Share of Atok ROI pixels with a valid (cloud-free) observation per half-month; "
                   "hatched = no composite", fontsize=10)
    figure.colorbar(image, ax=axis, shrink=0.8, label="valid share")
    figure.tight_layout()
    return figure


def plot_valid_share_map(cube, grid, years_slice):
    """Per-pixel share of valid half-months, with the imagery gap and cropland outline."""
    from crop_pipeline.imagery.observation_quality import valid_mask

    share = valid_mask(cube[years_slice]).mean(axis=0).astype(float)
    share[~grid["roi"]] = np.nan
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(share, cmap="YlGnBu", vmin=0, vmax=1)
    axis.contour(grid["cropland"], levels=[0.5], colors="black", linewidths=0.4)
    axis.set_axis_off()
    axis.set_title("Valid half-months per pixel (black: ELUM cropland)", fontsize=10)
    figure.colorbar(image, ax=axis, shrink=0.7, label="share valid")
    figure.tight_layout()
    return figure


def plot_usable_coverage(coverage):
    figure, axis = plt.subplots(figsize=(8, 3.5))
    coverage.plot.bar(ax=axis, color=["#B279A2", "#54A24B", "#F58518"], width=0.8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share of cropland")
    axis.set_xlabel("")
    axis.set_title("Usable cropland: >= 6 of 8 valid half-months in the crop window", fontsize=10)
    axis.legend(frameon=False, fontsize=8, loc="upper left", ncols=3)
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    return figure


def external_qc(windows_run):
    """Observation-quality checks from the notebook 04 run behind Sen4AgriNet/FranceCrops."""
    windows_run = Path(windows_run)
    timing = pd.read_csv(windows_run / "automatic_timing_audit.csv")
    usable = pd.read_csv(windows_run / "usable_pixel_selection.csv").set_index("crop")
    return {
        "timing": timing["automatic_timing_status"].value_counts().rename("parcel-seasons").to_frame(),
        "usable": usable[["available_usable_pixels", "selected_usable_pixels", "status"]],
    }


def show_figure(folder, name, width=1000):
    """Display a figure saved by notebook 07 v2."""
    from IPython.display import Image, display

    path = Path(folder) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run notebook 07 v2 to produce it.")
    display(Image(filename=str(path), width=width))
