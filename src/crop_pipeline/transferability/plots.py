"""Figures for notebook 07. Each function draws one figure, saves it under `figures` and shows it."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from shapely import union_all
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from .atok import INTERVAL_COLUMNS, LABEL_ROWS
from .climate import TMAX, TMIN
from .config import CLASSES, COLORS, INTERVAL, OOD_QUANTILE, SEED, SOURCE_COLORS, TARGET_SELECTIVE_ACCURACY, VARS
from .experiments import ATOK, BIN_GROUPS, VARIABLE_GROUPS
from .external import short_region
from .maps import MAP_WINDOWS
from .models import predicted, risk_coverage

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUP_COLORS = {ATOK: SOURCE_COLORS["Atok"], **{k: v for k, v in SOURCE_COLORS.items() if k != "Atok"}}
PAIR_COLORS = {"carrot vs cabbage": "#E45756", "carrot vs potato": "#F58518", "cabbage vs potato": "#54A24B"}
UNKNOWN_RGBA = to_rgba("#F3DE8A")
CAUSE_COLORS = {"classified": "#4C78A8", "uncertain only": "#F3DE8A", "out of distribution only": "#B279A2", "both": "#C44E52"}
LABEL_OFFSETS = {"Sen4AgriNet: Catalonia (ES)": (8, -18), "CropHarvest: Île-de-France": (8, 8), "FranceCrops: France": (-150, -12),
                 "CropHarvest: Madagascar": (-150, 4), "CropHarvest: Réunion (FR)": (10, -16)}


def _finish(figure, path, tight=True, **savefig):
    if tight:
        figure.tight_layout()
    figure.savefig(path, **{"dpi": 150, **savefig})
    plt.show()


def _swatch(color):
    return plt.Rectangle((0, 0), 1, 1, color=color)


def _bins(variable):
    return [f"bin_{b:02d}_{variable}" for b in range(1, 9)]


def _annotate_cells(axis, values, fmt, color=lambda v: "black", fontsize=8):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            axis.text(j, i, "" if np.isnan(v) else fmt(v), ha="center", va="center", fontsize=fontsize, color=color(v))


# Section 1: data profile

def parcel_centroids(data, region_counts, atok, grid, countries_zip, figures):
    world = gpd.read_file(f"zip://{countries_zip.resolve()}")
    centroids = data.groupby(["source", "source_region", "partition_group"])[["lon", "lat"]].mean().reset_index()
    figure, axis = plt.subplots(figsize=(14, 6.5))
    world.plot(ax=axis, color="#EEEEEE", edgecolor="#BBBBBB", linewidth=0.4)
    for source, part in centroids.groupby("source"):
        axis.scatter(part["lon"], part["lat"], s=12, color=SOURCE_COLORS[source], alpha=0.85, edgecolor="none",
                     label=f"{source} ({len(part)} parcels)")
    axis.scatter([grid.lon], [grid.lat], marker="*", s=260, color=SOURCE_COLORS["Atok"], edgecolor="black", zorder=5,
                 label=f"Atok ROI ({atok['field'].nunique()} labelled fields, provisional)")
    for name, row in region_counts.iterrows():
        axis.annotate(f"{name}\n{int(row['parcels'])} parcels, {int(row['pixels'])} px", (row["lon"], row["lat"]),
                      xytext=LABEL_OFFSETS.get(name, (8, 6)), textcoords="offset points", fontsize=7.5,
                      arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5})
    axis.set_xlim(-75, 140)
    axis.set_ylim(-35, 62)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend(loc="lower left", frameon=True, fontsize=8)
    axis.set_title("Training parcels by source region, and the Atok target")
    _finish(figure, figures / "parcel_centroids.png")


def region_crop_coverage(coverage, dev_test, figures):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(np.log1p(coverage.to_numpy()), cmap="Blues", aspect="auto")
    for i, region in enumerate(coverage.index):
        for j, crop in enumerate(CLASSES):
            value = coverage.loc[region, crop]
            if value == 0:
                axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#C44E52", lw=0))
                axis.text(j, i, "none", ha="center", va="center", fontsize=8, color="#C44E52")
            else:
                split = (f"\n({dev_test.loc[(region, crop), 'development']} dev / {dev_test.loc[(region, crop), 'test']} test)"
                         if (region, crop) in dev_test.index else "")
                axis.text(j, i, f"{value}{split}", ha="center", va="center", fontsize=7.5,
                          color="white" if np.log1p(value) > 3 else "black")
    axis.set_xticks(range(3), CLASSES)
    axis.set_yticks(range(len(coverage)), coverage.index, fontsize=8)
    axis.set_title("Parcels (fields for Atok) per region and crop")
    _finish(figure, figures / "region_crop_coverage.png")


def half_month_coverage(rows, clear_by_half, figures):
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 2 + 0.38 * len(rows)), sharex=True,
                                         gridspec_kw={"height_ratios": [1.2, 0.38 * len(rows)]})
    top.bar(np.arange(24) + 0.5, clear_by_half, width=0.9, color="#4C78A8")
    top.set_ylabel("Atok clear\nshare")
    top.set_ylim(0, 1)
    top.set_title("Atok cloud-free coverage (top) and window coverage per source vs the Atok calendar")
    for i, (_, vector, color, note) in enumerate(rows):
        for k in np.nonzero(vector)[0]:
            bottom.barh(i, 1, left=k, color=color, alpha=float(np.clip(vector[k], 0.12, 1)), edgecolor="none")
        bottom.text(24.3, i, note, va="center", fontsize=7.5, color="#444444")
    bottom.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    bottom.invert_yaxis()
    bottom.set_xticks(np.arange(0, 24, 2) + 1, MONTHS)
    bottom.set_xlim(0, 24)
    for boundary in range(0, 25, 2):
        bottom.axvline(boundary, color="#DDDDDD", lw=0.5, zorder=0)
    bottom.axvspan(8, 16, color="#30557A", alpha=0.06, zorder=0)  # May-Aug
    _finish(figure, figures / "half_month_coverage.png", bbox_inches="tight")


def temperature_profiles(climate, distances, figures):
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for axis, columns, title in ((axes[0], TMAX, "Mean daily maximum (°C)"), (axes[1], TMIN, "Mean daily minimum (°C)")):
        for i, (region, profile) in enumerate(climate.profiles.loc[distances.index].iterrows()):
            axis.plot(range(1, 13), profile[columns], color=palette[i % 10], lw=1.4,
                      label=f"{distances.loc[region, 'rank']}. {region} (d={distances.loc[region, 'climate_distance']:.1f})")
        axis.plot(range(1, 13), climate.target[columns], color="black", lw=3, label="Atok target")
        axis.plot(range(1, 13), climate.atok_raw[columns], color="black", lw=1, ls="--", label="Atok, TerraClimate unadjusted")
        axis.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        axis.set_title(title)
    axes[1].legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Temperature profiles: " + climate.note, fontsize=9)
    _finish(figure, figures / "temperature_profiles.png", bbox_inches="tight")


def temperature_differences(climate, distances, figures):
    figure, axes = plt.subplots(1, 2, figsize=(15, 3.8), sharey=True)
    for axis, columns, title in ((axes[0], TMAX, "TMAX: region - Atok (°C)"), (axes[1], TMIN, "TMIN: region - Atok (°C)")):
        diff = climate.profiles.loc[distances.index, columns].to_numpy() - climate.target[columns].to_numpy()
        shown = axis.imshow(diff, cmap="RdBu_r", vmin=-12, vmax=12, aspect="auto")
        _annotate_cells(axis, diff, lambda v: f"{v:.0f}", fontsize=7)
        axis.set_xticks(range(12), list("JFMAMJJASOND"))
        axis.set_title(title)
    axes[0].set_yticks(range(len(distances)), [f"{r} (d={d:.0f}, aligned {a:.0f})" for r, d, a in
                                               zip(distances.index, distances["climate_distance"], distances["aligned_distance"])], fontsize=8)
    figure.colorbar(shown, ax=axes, shrink=0.8)
    _finish(figure, figures / "temperature_differences.png", tight=False, bbox_inches="tight")


def _band(axis, rows, variable, color, label):
    values = rows[_bins(variable)].to_numpy()
    bins = np.arange(1, 9)
    axis.fill_between(bins, *np.nanpercentile(values, [25, 75], axis=0), color=color, alpha=0.18, lw=0)
    axis.plot(bins, np.nanmedian(values, axis=0), color=color, lw=2, label=f"{label} (n={len(rows):,})")


def _signatures(data, atok):
    return pd.concat([data.assign(group=data["source"])[["group", "crop"] + INTERVAL_COLUMNS],
                      atok.assign(group=ATOK)[["group", "crop"] + INTERVAL_COLUMNS]], ignore_index=True)


def domain_shift_by_crop(data, atok, figures, variables=("NDVI", "NDRE", "NDMI")):
    signatures = _signatures(data, atok)
    figure, axes = plt.subplots(len(CLASSES), len(variables), figsize=(14, 9), sharex=True)
    for i, crop in enumerate(CLASSES):
        for j, variable in enumerate(variables):
            for group, rows in signatures[signatures["crop"] == crop].groupby("group"):
                _band(axes[i, j], rows, variable, GROUP_COLORS[group], group)
            axes[i, j].set_title(f"{crop}: {variable}", fontsize=9)
            if j == 0:
                axes[i, j].legend(fontsize=7, frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Half-month bin from window start")
    figure.suptitle("Domain shift: each crop's trajectory by source (median, IQR)")
    _finish(figure, figures / "domain_shift_by_crop.png")


def signatures_by_source(data, atok, figures, variables=("NDVI", "NDRE", "NDMI")):
    signatures = _signatures(data, atok)
    groups = list(GROUP_COLORS)
    figure, axes = plt.subplots(len(groups), len(variables), figsize=(14, 2.6 * len(groups)), sharex=True)
    for i, group in enumerate(groups):
        for j, variable in enumerate(variables):
            for crop, rows in signatures[signatures["group"] == group].groupby("crop"):
                _band(axes[i, j], rows, variable, COLORS[crop], crop)
            axes[i, j].set_title(f"{group}: {variable}", fontsize=9)
            if j == 0:
                axes[i, j].legend(fontsize=7, frameon=False)
    figure.suptitle("Crop separability within each source (median, IQR)")
    _finish(figure, figures / "temporal_signatures_by_source.png")


def feature_distributions(both_means, region_order, figures):
    figure, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for axis, variable in zip(axes.ravel(), VARS):
        parts = axis.boxplot([both_means.loc[both_means["region"] == r, variable] for r in region_order], showfliers=False,
                             patch_artist=True, widths=0.6)
        for patch, region in zip(parts["boxes"], region_order):
            patch.set_facecolor(SOURCE_COLORS["Atok"] if region.startswith("Atok") else SOURCE_COLORS[region.split(":")[0]])
            patch.set_alpha(0.75)
        axis.set_title(f"{variable} (mean of 8 half-months)", fontsize=9)
    for axis in axes[-1]:
        axis.set_xticks(range(1, len(region_order) + 1), [short_region(r) for r in region_order], rotation=40, ha="right", fontsize=8)
    figure.suptitle("Feature distributions: each external region vs Atok (boxes 25-75%, whiskers 1.5 IQR)")
    _finish(figure, figures / "feature_distributions_by_region.png")


def atok_vs_region_shift(shift, figures):
    figure, axis = plt.subplots(figsize=(12, 4.2))
    columns = VARS + ["mean"]
    values = shift[columns].to_numpy()
    shown = axis.imshow(values, cmap="magma_r", vmin=0, vmax=max(1.5, float(np.nanmax(values))), aspect="auto")
    _annotate_cells(axis, values, lambda v: f"{v:.2f}", color=lambda v: "white" if v > 1 else "black")
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(shift)), [f"{r} ({n} shared crops)" for r, n in zip(shift.index, shift["shared_crops"])], fontsize=8)
    figure.colorbar(shown, ax=axis, label="Wasserstein distance (external SD units)")
    axis.set_title("How far Atok lies from each region, same crop")
    _finish(figure, figures / "atok_vs_region_shift.png")


def crop_growth_curves(groups, figures, variables=("NDVI", "NDRE", "NDMI", "B11")):
    figure, axes = plt.subplots(len(groups), len(variables), figsize=(16, 3 * len(groups)), sharex=True)
    for i, (group, rows) in enumerate(groups.items()):
        for j, variable in enumerate(variables):
            axis = axes[i, j]
            for crop, crop_rows in rows.groupby("crop"):
                values = crop_rows[_bins(variable)].to_numpy()
                mean, sd = values.mean(0), values.std(0)
                axis.fill_between(range(1, 9), mean - sd, mean + sd, color=COLORS[crop], alpha=0.15, lw=0)
                axis.plot(range(1, 9), mean, color=COLORS[crop], lw=2, label=f"{crop} (n={len(crop_rows):,})")
            axis.set_title(f"{group}: {variable}", fontsize=8.5)
            if j == 0:
                axis.legend(fontsize=7, frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Half-month from window start")
    figure.suptitle("Crop growth curves (mean ± 1 SD)")
    _finish(figure, figures / "crop_growth_curves.png")


def crop_separability(separability, groups, figures):
    figure, axes = plt.subplots(1, len(groups), figsize=(18, 3.8), sharey=True)
    for axis, group in zip(axes, groups):
        for pair, part in separability[separability["group"] == group].groupby("pair"):
            axis.plot(part["bin"], part["JM"], "-o", ms=3, color=PAIR_COLORS.get(pair, "grey"), label=pair)
        axis.axhline(1.9, color="black", lw=0.7, ls="--")
        axis.axhline(1.0, color="grey", lw=0.7, ls=":")
        axis.set_ylim(0, 2.05)
        axis.set_xlabel("Half-month from window start")
        axis.set_title(group, fontsize=9)
        axis.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Jeffries-Matusita distance (9 variables)")
    figure.suptitle("Crop separability by growth stage (dashed: 1.9 well separable; dotted: 1.0)")
    _finish(figure, figures / "crop_separability.png")


# Section 3: exclusion funnel

def exclusion_funnel(funnel, figures):
    panels = [(s, u) for s, u in funnel[["source", "unit"]].drop_duplicates().itertuples(index=False)]
    heights = [len(funnel[(funnel["source"] == s) & (funnel["unit"] == u)]) + 0.6 for s, u in panels]
    figure, axes = plt.subplots(len(panels), 1, figsize=(10, 0.9 + 0.5 * len(funnel) + 0.35 * len(panels)),
                                gridspec_kw={"height_ratios": heights})
    for axis, (source, unit) in zip(axes, panels):
        part = funnel[(funnel["source"] == source) & (funnel["unit"] == unit)].reset_index(drop=True)
        axis.barh(range(len(part)), part["retained_%"], color="#4C78A8")
        for i, r in part.iterrows():
            axis.text(r["retained_%"] + 1, i, f"{r['count']:,} ({r['retained_%']:.0f}%)" + (f", -{r['dropped']:,}" if r["dropped"] > 0 else ""),
                      va="center", fontsize=8)
        axis.set_yticks(range(len(part)), part["step"], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, 135)
        axis.set_title(f"{source}: {unit}", fontsize=9, loc="left")
    axes[-1].set_xlabel("% retained relative to the first step of this unit")
    _finish(figure, figures / "exclusion_funnel.png")


# Section 4: external models

def learning_curves(curve, figures):
    summary = curve.groupby("fraction").agg(["mean", "std"])
    x = summary[("train_parcels", "mean")]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4))
    left.plot(x, summary[("train_macro_F1", "mean")], "-o", color="#BBBBBB", label="training folds")
    left.errorbar(x, summary[("validation_macro_F1", "mean")], yerr=summary[("validation_macro_F1", "std")],
                  fmt="-o", capsize=3, color="#4C78A8", label="validation folds (mean ± sd over folds and seeds)")
    left.set_ylim(0, 1.02)
    left.set_xlabel("Training parcels (mean per fold)")
    left.set_ylabel("Macro-F1")
    left.legend(frameon=False, fontsize=8)
    left.set_title("Overall")
    for crop in CLASSES:
        right.errorbar(x, summary[(f"recall_{crop}", "mean")], yerr=summary[(f"recall_{crop}", "std")],
                       fmt="-o", capsize=3, color=COLORS[crop], label=crop)
    right.set_ylim(0, 1.02)
    right.set_xlabel("Training parcels (mean per fold)")
    right.set_ylabel("Validation recall")
    right.legend(frameon=False, fontsize=8)
    right.set_title("By crop")
    _finish(figure, figures / "learning_curves.png")


def importance_ablation(importance, ablation, figures):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for axis, kind in zip(axes[:2], ("variable", "bin")):
        summary = importance[importance["kind"] == kind].groupby("group")["drop"].agg(["mean", "std"])
        summary = summary.reindex(list(VARIABLE_GROUPS) if kind == "variable" else list(BIN_GROUPS))
        axis.barh(summary.index, summary["mean"], xerr=summary["std"], color="#4C78A8")
        axis.axvline(0, color="black", lw=0.6)
        axis.invert_yaxis()
        axis.set_xlabel("Drop in validation macro-F1 (mean ± sd)")
        axis.set_title(f"Grouped permutation: by {kind}")
    summary = ablation.groupby("removed")["change"].agg(["mean", "std"]).reindex(VARS)
    axes[2].barh(summary.index, summary["mean"], xerr=summary["std"], color="#72B7B2")
    axes[2].axvline(0, color="black", lw=0.6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Change in validation macro-F1 when removed")
    axes[2].set_title("Feature-family ablation (refit)")
    _finish(figure, figures / "grouped_importance_ablation.png")


def calibration_selective_ood(curves, test_distance, atok_distance, ood_threshold, figures):
    figure = plt.figure(figsize=(15, 4.3))
    reliability_axis, selective_axis, hist_axis = figure.subplots(1, 3)
    for (name, label), curve in curves.items():
        style = "-o" if label == "test" else "--s"
        short = f"{name.split(' (')[0]}, {label}"
        reliability_axis.plot(curve["centres"], curve["observed"], style, ms=4, label=short)
        selective_axis.plot(curve["coverage"], curve["accuracy"], style, ms=3, label=short)
    reliability_axis.plot([0, 1], [0, 1], color="grey", lw=0.8)
    reliability_axis.set_xlabel("Mean top-class probability")
    reliability_axis.set_ylabel("Observed accuracy")
    reliability_axis.set_title("Reliability (bins with >= 15 samples)")
    reliability_axis.legend(fontsize=7, frameon=False)
    selective_axis.set_xlabel("Share of predictions kept (coverage)")
    selective_axis.set_ylabel("Accuracy of kept predictions")
    selective_axis.set_title("Selective prediction")
    selective_axis.legend(fontsize=7, frameon=False)
    hist_axis.hist(test_distance, bins=40, alpha=0.6, density=True, label="external test")
    hist_axis.hist(atok_distance, bins=40, alpha=0.6, density=True, label=ATOK, color=SOURCE_COLORS["Atok"])
    hist_axis.axvline(ood_threshold, color="black", lw=1, label=f"OOD threshold ({OOD_QUANTILE:.0%} of test)")
    hist_axis.set_xlabel("Mean distance to 5 nearest development samples")
    hist_axis.set_title("Distance to the training data")
    hist_axis.legend(fontsize=7, frameon=False)
    _finish(figure, figures / "calibration_selective_ood.png")


def threshold_transfer(test_cal, test, transfer_curves, transfer, atok_cal, atok, figures):
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 4.6))
    cov, acc = risk_coverage(test_cal, test["crop"])
    left.plot(cov, acc, color="black", lw=2.2, label="locked test (pooled model, calibrated)")
    for region, (cov, acc) in transfer_curves.items():
        left.plot(cov, acc, lw=1, alpha=0.8, label=f"unseen: {short_region(region)}")
    cov, acc = risk_coverage(atok_cal, atok["crop"])
    left.plot(cov, acc, color=SOURCE_COLORS["Atok"], lw=2, ls="--", label="Atok (provisional labels, exploratory)")
    left.axhline(TARGET_SELECTIVE_ACCURACY, color="grey", lw=0.8, ls=":")
    left.axvline(0.5, color="grey", lw=0.8, ls=":")
    left.set_xlabel("Coverage (share of samples classified, most confident first)")
    left.set_ylabel("Accuracy of classified samples")
    left.set_ylim(0, 1.02)
    left.set_title("Risk-coverage curves (dotted: target accuracy, 50% coverage)")
    left.legend(fontsize=7, frameon=False, ncol=2)
    positions = range(len(transfer))
    right.bar(positions, transfer["coverage at threshold"], color="#BBBBBB", label="coverage at threshold")
    right.plot(positions, transfer["accuracy of kept"], "o", color="#C44E52", label="accuracy of kept")
    right.axhline(TARGET_SELECTIVE_ACCURACY, color="#C44E52", lw=0.8, ls=":")
    right.set_xticks(positions, [short_region(r) for r in transfer["held_out_region"]], rotation=30, ha="right")
    right.set_ylim(0, 1.02)
    right.set_title("Threshold set on other regions, applied to the unseen one")
    right.legend(fontsize=8, frameon=False)
    _finish(figure, figures / "calibration_fitted_threshold_transfer.png")


PANEL_NOTES = {1: "1 shared crop: recall of that crop\n(depends on how many crops the model knows)",
               2: "2 shared crops: balanced accuracy\n(chance 0.50)", 3: "3 shared crops: balanced accuracy\n(chance 0.33)"}


def transfer_matrix(transfer, figures):
    """Scores over 1, 2 or 3 shared crops are different tasks: separate panels, never one scale."""
    matrix, shared_count, train_regions = transfer["matrix"], transfer["shared_count"], transfer["train_regions"]
    panel_counts = [k for k in (1, 2, 3) if (shared_count == k).to_numpy().any()]
    figure, axes = plt.subplots(1, len(panel_counts), figsize=(7 * len(panel_counts), 0.55 * len(train_regions) + 2.6))
    for axis, k in zip(np.atleast_1d(axes), panel_counts):
        columns = [c for c in transfer["targets"] if (shared_count[c] == k).any()]
        panel = matrix[columns].where(shared_count[columns] == k).to_numpy(dtype=float)
        shown = axis.imshow(panel, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _annotate_cells(axis, panel, lambda v: f"{v:.2f}", color=lambda v: "white" if v < 0.6 else "black")
        axis.set_xticks(range(len(columns)), [short_region(c) for c in columns], rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(len(train_regions)), [f"train: {short_region(r)}" for r in train_regions], fontsize=8)
        axis.set_title(PANEL_NOTES[k], fontsize=9)
        figure.colorbar(shown, ax=axis, shrink=0.8)
    figure.suptitle("Single-region transfer, split by number of shared crops (not comparable across panels)")
    _finish(figure, figures / "transfer_matrix_by_shared_crops.png")


def one_vs_rest_auc(ovr, transfer, figures):
    train_regions, targets = transfer["train_regions"], transfer["targets"]
    figure, axes = plt.subplots(1, len(CLASSES), figsize=(6.5 * len(CLASSES), 0.55 * len(train_regions) + 2.6))
    for axis, crop in zip(axes, CLASSES):
        table = ovr[ovr["crop"] == crop].pivot(index="train", columns="test", values="AUC")
        table = table.reindex(index=[r for r in train_regions if r in table.index], columns=[t for t in targets if t in table.columns])
        values = table.to_numpy(dtype=float)
        shown = axis.imshow(values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
        _annotate_cells(axis, values, lambda v: f"{v:.2f}")
        axis.set_xticks(range(table.shape[1]), [short_region(c) for c in table.columns], rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(table.shape[0]), [f"train: {short_region(r)}" for r in table.index], fontsize=8)
        axis.set_title(f"{crop} vs rest: ROC AUC (0.5 = chance)", fontsize=9)
        figure.colorbar(shown, ax=axis, shrink=0.8)
    figure.suptitle("One-vs-rest transfer per crop (both regions contain the crop and another crop)")
    _finish(figure, figures / "transfer_one_vs_rest_auc.png")


def transfer_vs_similarity(ovr, pair_stats, figures):
    external = ovr[~ovr["test"].str.startswith("Atok")]
    atok_part = ovr[ovr["test"].str.startswith("Atok")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for axis, measure, label in ((axes[0], "climate_distance", "Season-aligned climate distance"),
                                 (axes[1], "geographic_km", "Distance between centroids (km)")):
        for crop in CLASSES:
            part = external[external["crop"] == crop]
            axis.scatter(part[measure], part["AUC"], s=40, alpha=0.8, color=COLORS[crop], label=f"{crop}, external target")
        axis.scatter(atok_part[measure], atok_part["AUC"], marker="*", s=160, c=[COLORS[c] for c in atok_part["crop"]],
                     edgecolor="black", label=f"target = {ATOK}")
        axis.axhline(0.5, color="grey", lw=0.8, ls="--")
        rho = pair_stats[(pair_stats["points"] == "external targets only") & (pair_stats["measure"] == measure)]["spearman_rho"].iloc[0]
        axis.set_xlabel(label)
        axis.set_title(f"External targets: Spearman rho = {rho:.2f}")
    axes[0].set_ylabel("One-vs-rest ROC AUC")
    axes[0].legend(fontsize=7, frameon=False)
    figure.suptitle("Single-region transfer (one-vs-rest AUC) vs climate and geographic distance")
    _finish(figure, figures / "transfer_vs_similarity.png")


# Section 5: climate matching

def climate_external_targets(data, targets, external_sets, target_stats, figures):
    figure, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.8), sharey=True)
    stats = target_stats.set_index("target")
    for axis, target in zip(np.atleast_1d(axes), targets):
        part = external_sets[external_sets["target"] == target]
        shown = axis.scatter(part["climate_distance"], part["balanced_acc"], s=10 + part["train_parcels"] / 3, c=part["n_sources"],
                             cmap="viridis", alpha=0.75, edgecolor="none")
        axis.axhline(1 / data.loc[data["source_region"] == target, "crop"].nunique(), color="grey", lw=0.8, ls="--")
        row = stats.loc[target]
        axis.set_title(f"target: {short_region(target)}\nrho = {row.get('rho_climate', np.nan):.2f} ({int(row['eligible_sets'])} sets)", fontsize=9)
        axis.set_xlabel("Climate distance to target")
    np.atleast_1d(axes)[0].set_ylabel("Balanced accuracy on target (verified labels)")
    figure.colorbar(shown, ax=axes, label="source regions in set", shrink=0.8)
    figure.suptitle("Climate distance vs transfer, external targets (dashed: chance)")
    _finish(figure, figures / "climate_external_targets.png", tight=False, bbox_inches="tight")


def climate_experiment(region_sets, correlations, control_scores, climate_score, figures):
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for axis, distance_name in zip(axes[:2], ("calendar_distance", "aligned_distance")):
        shown = axis.scatter(region_sets[distance_name], region_sets["atok_balanced_acc"], s=12 + region_sets["train_parcels"] / 2,
                             c=region_sets["n_regions"], cmap="viridis", alpha=0.75, edgecolor="none")
        flagged = region_sets["is_climate_set"] if distance_name == "calendar_distance" else region_sets["is_aligned_set"]
        axis.scatter(region_sets.loc[flagged, distance_name], region_sets.loc[flagged, "atok_balanced_acc"], s=160, facecolor="none",
                     edgecolor="red", lw=1.5, label="climate-ranked set")
        rho = correlations.loc[correlations["distance"] == distance_name, "spearman_rho"].iloc[0]
        axis.set_xlabel(f"Parcel-weighted climate distance ({distance_name.split('_')[0]})")
        axis.set_ylabel("Atok balanced accuracy (provisional)")
        axis.set_title(f"Spearman rho = {rho:.2f}")
        axis.axhline(1 / 3, color="grey", lw=0.8, ls="--")
        axis.legend(fontsize=8, frameon=False)
    figure.colorbar(shown, ax=axes[:2], label="regions in set", shrink=0.8)
    axes[2].hist(control_scores, bins=10, color="#BBBBBB", label="size-matched random parcels")
    axes[2].axvline(climate_score, color="red", lw=2, label="climate-ranked set")
    axes[2].set_xlabel("Atok balanced accuracy (provisional)")
    axes[2].set_title("Climate set vs size-matched controls")
    axes[2].legend(fontsize=8, frameon=False)
    _finish(figure, figures / "climate_experiment.png", tight=False, bbox_inches="tight")


# Section 6: Atok evaluation

def atok_confusion_matrices(atok, predictions, figures):
    figure, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.8))
    for axis, (name, proba) in zip(axes, predictions.items()):
        counts = confusion_matrix(atok["crop"], predicted(proba), labels=CLASSES)
        shares = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        axis.imshow(shares, cmap="Oranges", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{counts[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=7,
                          color="white" if shares[i, j] > 0.5 else "black")
        axis.set_xticks(range(3), CLASSES, fontsize=7)
        axis.set_yticks(range(3), CLASSES, fontsize=7)
        axis.set_xlabel("Predicted", fontsize=8)
        axis.set_title(name, fontsize=7.5)
    axes[0].set_ylabel("Provisional calendar label")
    figure.suptitle("Atok confusion matrices (provisional labels; rows sum to 100%)", fontsize=10)
    _finish(figure, figures / "atok_confusion_matrices.png")


def atok_domain_shift(development, atok, distance_to_training, figures):
    """External development data in its first two principal components, with Atok projected in; OOD share by field."""
    scaler = distance_to_training.scaler
    pca = PCA(n_components=2, random_state=SEED).fit(scaler.transform(development[INTERVAL]))
    external_xy = pca.transform(scaler.transform(development[INTERVAL]))
    sample = atok.sample(min(3000, len(atok)), random_state=SEED)
    atok_xy = pca.transform(scaler.transform(sample[INTERVAL]))
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    for source, color in ((s, c) for s, c in SOURCE_COLORS.items() if s != "Atok"):
        mask = (development["source"] == source).to_numpy()
        left.scatter(external_xy[mask, 0], external_xy[mask, 1], s=6, alpha=0.5, color=color, label=source)
    left.scatter(atok_xy[:, 0], atok_xy[:, 1], s=4, alpha=0.25, color=SOURCE_COLORS["Atok"], label="Atok (sample of 3,000)")
    left.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    left.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    left.legend(fontsize=8, frameon=False, markerscale=3)
    left.set_title("Interval features in external principal components")
    ood_table = atok.groupby(["field", "crop"])["ood"].mean().rename("OOD share")
    ood_table.plot.barh(ax=right, color=SOURCE_COLORS["Atok"])
    right.axvline(1 - OOD_QUANTILE, color="black", lw=0.8, ls="--", label="external test rate")
    right.set_xlim(0, 1)
    right.set_xlabel("Share of pixel-windows out of distribution")
    right.legend(fontsize=8, frameon=False)
    right.set_title("Out-of-distribution share by field")
    _finish(figure, figures / "atok_domain_shift.png")
    return ood_table.to_frame()


def atok_failure_analysis(failure, failure_models, figures, model="climate-set RF"):
    by_year = failure.pivot_table(index=["field", "crop", "start_month"], columns="year", values=f"correct_{model}", aggfunc="mean")
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    shown = left.imshow(by_year.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    left.set_xticks(range(by_year.shape[1]), by_year.columns)
    left.set_yticks(range(len(by_year)), [f"{f} {c} (from {m:02d})" for f, c, m in by_year.index], fontsize=8)
    left.set_title("Climate-set RF: share correct by field-season and year (blank = no clear window)")
    figure.colorbar(shown, ax=left)
    for label in failure_models:
        by_clear = failure.groupby("n_clear")[f"correct_{label}"].mean()
        right.plot(by_clear.index, by_clear.to_numpy(), "-o", label=label)
    right.set_xlabel("Clear composites in the window (of 8)")
    right.set_ylabel("Share correct")
    right.set_ylim(0, 1)
    right.legend(frameon=False, fontsize=8)
    right.set_title("Agreement vs clear composites")
    _finish(figure, figures / "atok_failure_analysis.png")


# Section 7: maps

def draw_outlines(axis, grid):
    for geom, color, width in [(g, "black", 0.8) for g in grid.polygons["grid_geometry"]] + [(grid.roi_grid, "#555555", 0.6)]:
        points = [grid.to_pixel(x, y) for x, y in zip(*geom.exterior.xy)]
        axis.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=width)


def map_background(grid, cropland, never_clear):
    background = np.full((grid.height, grid.width, 4), 0.0)
    background[grid.roi_mask] = (0.86, 0.86, 0.86, 1)
    background[cropland] = (0.62, 0.62, 0.62, 1)
    background[never_clear & grid.roi_mask] = (0.25, 0.25, 0.25, 1)
    return background


def _map_axis(axis, rgba, grid, title, fontsize=9):
    axis.imshow(rgba, interpolation="nearest")
    draw_outlines(axis, grid)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=fontsize)


def crop_maps(raw, map_models, year, best_year, grid, background, figures):
    class_rgba = np.array([to_rgba(COLORS[c]) for c in CLASSES])
    figure, axes = plt.subplots(len(map_models), len(MAP_WINDOWS), figsize=(6 * len(MAP_WINDOWS), 5.6 * len(map_models)),
                                layout="constrained")
    for row_axes, model_name in zip(axes, map_models):
        for axis, month in zip(row_axes, MAP_WINDOWS):
            codes, _, _, low, ood = raw[(model_name, year, month)]
            unknown = low | ood
            rgba = background.copy()
            mapped = np.isfinite(codes)
            known = mapped & ~unknown
            rgba[known] = class_rgba[codes[known].astype(int)]
            rgba[mapped & unknown] = UNKNOWN_RGBA
            _map_axis(axis, rgba, grid, f"{model_name}\n{MAP_WINDOWS[month]}, {year}")
    handles = [_swatch(COLORS[c]) for c in CLASSES] + [_swatch(UNKNOWN_RGBA)] + [_swatch(c) for c in ((0.62,) * 3, (0.86,) * 3, (0.25,) * 3)]
    figure.legend(handles, CLASSES + ["unknown (low probability or out of distribution)", "cropland, < 6 clear composites",
                                      "outside the ELUM Crops mask", "no Sentinel-2 data"],
                  loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(f"Exploratory Atok maps, {year} ({'best' if year == best_year else 'worst'} coverage year)", fontsize=12)
    _finish(figure, figures / f"atok_maps_{year}.png", tight=False, dpi=130, bbox_inches="tight")


def rejection_cause_maps(cause_masks, map_models, year, grid, background, figures):
    figure, axes = plt.subplots(len(map_models), len(MAP_WINDOWS), figsize=(6 * len(MAP_WINDOWS), 5.4 * len(map_models)),
                                layout="constrained")
    for row_axes, model_name in zip(axes, map_models):
        for axis, month in zip(row_axes, MAP_WINDOWS):
            rgba = background.copy()
            for category, mask in cause_masks[(model_name, month)].items():
                rgba[mask] = to_rgba(CAUSE_COLORS[category])
            _map_axis(axis, rgba, grid, f"{model_name}\n{MAP_WINDOWS[month]}, {year}")
    figure.legend([_swatch(c) for c in CAUSE_COLORS.values()], list(CAUSE_COLORS), loc="lower center", ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle(f"Rejection causes, {year}", fontsize=12)
    _finish(figure, figures / f"atok_rejection_causes_{year}.png", tight=False, dpi=130, bbox_inches="tight")


def operating_points(points, map_models, year, figures):
    figure, axes = plt.subplots(1, len(map_models), figsize=(7 * len(map_models), 4.2), sharey=True)
    for axis, (model_name, (_, _, threshold)) in zip(np.atleast_1d(axes), map_models.items()):
        for month, window in MAP_WINDOWS.items():
            for use_ood, style in ((True, "-"), (False, "--")):
                part = points[(points["model"] == model_name) & (points["window"] == window) & (points["ood_check"] == use_ood)]
                axis.plot(part["threshold"], part["coverage"], style, label=f"{window.split(' (')[0]}{'' if use_ood else ', no OOD check'}")
        axis.axvline(threshold, color="black", lw=0.8, label=f"chosen threshold {threshold:.2f}")
        axis.axhline(0.5, color="grey", lw=0.8, ls=":", label="minimum coverage 50%")
        axis.set_xlabel("Calibrated probability threshold")
        axis.set_title(model_name, fontsize=9)
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=7, frameon=False)
    np.atleast_1d(axes)[0].set_ylabel("Share of mappable cropland classified")
    figure.suptitle(f"Operating points, {year}: coverage vs threshold")
    _finish(figure, figures / f"atok_operating_points_{year}.png")


def model_disagreement(agreement_masks, year, grid, background, figures):
    colors = {"agree": "#4C9F70", "disagree": "#D1495B", "either_unknown": UNKNOWN_RGBA}
    figure, axes = plt.subplots(1, len(MAP_WINDOWS), figsize=(6 * len(MAP_WINDOWS), 5.6), layout="constrained")
    for axis, month in zip(axes, MAP_WINDOWS):
        masks = agreement_masks[month]
        rgba = background.copy()
        for key, color in colors.items():
            rgba[masks[key]] = to_rgba(color)
        _map_axis(axis, rgba, grid, f"{MAP_WINDOWS[month]}, {year}: agree {masks['agree'].sum():,} / "
                                    f"disagree {masks['disagree'].sum():,} / unknown {masks['either_unknown'].sum():,}")
    figure.legend([_swatch(c) for c in colors.values()], ["same crop", "different crop", "unknown in either"],
                  loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    _finish(figure, figures / f"atok_model_disagreement_{year}.png", tight=False, dpi=130, bbox_inches="tight")


def _zoom(grid, fields, margin_m=120):
    """Pixel bounds (c0, r0, c1, r1) around the labelled fields, and a panel width matching their shape."""
    zoom = union_all([grid.polygons.loc[f, "grid_geometry"] for f in fields]).buffer(margin_m)
    (c0, r0), (c1, r1) = grid.to_pixel(zoom.bounds[0], zoom.bounds[3]), grid.to_pixel(zoom.bounds[2], zoom.bounds[1])
    return (c0, r0, c1, r1), float(np.clip(5 * (c1 - c0) / (r1 - r0), 2.5, 6))


def _label_field(axis, grid, field, text):
    x, y = grid.to_pixel(*grid.polygons.loc[field, "grid_geometry"].representative_point().coords[0])
    axis.annotate(text, (x, y), fontsize=8, ha="center", color="black", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "lw": 0})


def field_errors(atok, called, unknown, year, grid, fields, figures):
    (c0, r0, c1, r1), panel_w = _zoom(grid, fields)
    category = {"correct": to_rgba("#1565C0"), "unknown": UNKNOWN_RGBA, "too cloudy": to_rgba("#9E9E9E")}
    figure, axes = plt.subplots(len(called), len(MAP_WINDOWS), figsize=(max(panel_w * len(MAP_WINDOWS), 12), 5.4 * len(called)),
                                layout="constrained")
    for row_axes, name in zip(axes, called):
        for axis, month in zip(row_axes, MAP_WINDOWS):
            rgba = np.ones((grid.height, grid.width, 4))
            rgba[grid.roi_mask] = (0.93, 0.93, 0.93, 1)
            for field, _, start, _ in LABEL_ROWS:
                if start == month:
                    rgba[grid.interiors[field]] = category["too cloudy"]
            selected = ((atok["year"] == year) & (atok["start_month"] == month)).to_numpy()
            if selected.any():
                truth, calls, abstain = atok.loc[selected, "crop"].to_numpy(), called[name][selected], unknown[name][selected]
                rgba[atok.loc[selected, "row"].to_numpy(), atok.loc[selected, "col"].to_numpy()] = np.array(
                    [category["unknown"] if u else (category["correct"] if c == t else to_rgba(COLORS[c]))
                     for t, c, u in zip(truth, calls, abstain)])
            _map_axis(axis, rgba, grid, f"{name}\n{MAP_WINDOWS[month]}, {year}", fontsize=8.5)
            for field, crop, start, _ in LABEL_ROWS:
                if start == month:
                    _label_field(axis, grid, field, f"{field}: {crop}")
            axis.set_xlim(c0, c1)
            axis.set_ylim(r1, r0)
    handles = [_swatch(category["correct"])] + [_swatch(COLORS[c]) for c in CLASSES] + [_swatch(category[k]) for k in ("unknown", "too cloudy")]
    figure.legend(handles, ["correct (matches calendar)"] + [f"wrong: predicted {c}" for c in CLASSES] + ["unknown (abstained)", "too cloudy (< 6 clear)"],
                  loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(f"Errors on the provisional label fields, {year} (calendar labels, not verified)", fontsize=11)
    _finish(figure, figures / f"atok_field_errors_{year}.png", tight=False, bbox_inches="tight")


def field_consistency(per_pixel_shares, grid, fields, figures):
    (c0, r0, c1, r1), panel_w = _zoom(grid, fields)
    figure, axes = plt.subplots(1, len(per_pixel_shares), figsize=(panel_w * len(per_pixel_shares) + 1.5, 5.6), layout="constrained")
    for axis, (name, per_pixel) in zip(axes, per_pixel_shares.items()):
        share_grid = np.full((grid.height, grid.width), np.nan)
        share_grid[per_pixel["row"], per_pixel["col"]] = per_pixel["share"]
        base = np.ones((grid.height, grid.width, 4))
        base[grid.roi_mask] = (0.93, 0.93, 0.93, 1)
        axis.imshow(base, interpolation="nearest")
        shown = axis.imshow(share_grid, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        draw_outlines(axis, grid)
        for field in fields:
            _label_field(axis, grid, field, field)
        axis.set_xlim(c0, c1)
        axis.set_ylim(r1, r0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{name}\nshare of each pixel's labelled windows correct, 2019-2025", fontsize=9)
    figure.colorbar(shown, ax=axes, shrink=0.7, label="share correct (unknown counts as not correct)")
    _finish(figure, figures / "atok_field_consistency.png", tight=False, bbox_inches="tight")
