"""Draw the figures for docs/pipeline/ from the current run outputs.

Every figure reads saved run files through the current stage pointers (or the runs named on the
command line), never numbers copied from documents. ``docs/pipeline/figure_sources.json`` lists
each figure with the repository-relative files it read and their SHA-256, so a reader can trace
every value to a run.

    .venv/bin/python tools/build_pipeline_guide.py
    .venv/bin/python tools/build_pipeline_guide.py --experiments pipeline_outputs/experiments/rerun_20261122
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from crop_pipeline.analysis_workflow import resolve_stage_pointer  # noqa: E402

GUIDE = ROOT / "docs" / "pipeline"
CROPS = ("carrot", "cabbage", "potato")
# Reference palette: the first three categorical slots validate as a set (all pairs).
CROP_COLORS = {"carrot": "#eb6834", "cabbage": "#1baf7a", "potato": "#2a78d6"}
SERIES = {"fixed": "#2a78d6", "aligned": "#eb6834", "annual": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e1", "#fcfcfb", "#c9c8c3"
MODELS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")
MODEL_LABELS = {"fixed_full": "Fixed full", "aligned_full": "Aligned full",
                "fixed_matched": "Fixed matched", "aligned_matched": "Aligned matched"}
SOURCES = {}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": .8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
        "font.family": "DejaVu Sans"})


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def used(figure, *paths):
    SOURCES.setdefault(figure, {}).update({rel(p): sha(p) for p in paths})


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, stage, name):
    path = GUIDE / stage / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(Path(stage) / "figures" / f"{name}.png")


def crop_legend(ax, extra=(), **kwargs):
    handles = [Patch(color=CROP_COLORS[c], label=c) for c in CROPS] + list(extra)
    ax.legend(handles=handles, **{"loc": "upper right", "fontsize": 8.5, **kwargs})


# --- Overview ----------------------------------------------------------------------------------

def overview():
    stages = [("01 Data", "Labelled parcels\nand annual\nSentinel-2"),
              ("02 Pre-\nprocessing", "Quality, four-month\nwindows, 72 predictors,\nlocked split"),
              ("03 Models", "Train, select on\nvalidation, freeze,\nexternal test"),
              ("04 Limits\n& errors", "Comparators, tuning,\nerror analysis,\napplicability"),
              ("05 Atok\napplication", "Paoay maps, calendar\ncontext, land use,\nfield checks"),
              ("06 Experi-\nments", "Annual recipe and\nvariants, label-free\nAtok checks")]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    for i, (title, text) in enumerate(stages):
        experimental = i == 5
        ax.add_patch(FancyBboxPatch((i + .06, .2), .82, .62, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor="#eef3fb" if not experimental else "#eaf6f1",
                                    edgecolor=SERIES["fixed"] if not experimental else SERIES["annual"],
                                    linewidth=1.4, linestyle="-" if not experimental else "--"))
        ax.text(i + .47, .69, title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
        ax.text(i + .47, .38, text, ha="center", va="center", fontsize=8, color=INK_2)
        if i < 5:
            ax.annotate("", (i + 1.06, .51), (i + .9, .51),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.4})
    ax.text(0, .04, "Solid: reported pipeline (src/crop_pipeline, notebooks/).   "
                    "Dashed: experiments (experiments/), which read the reported runs but never change them.",
            fontsize=8.3, color=INK_2)
    return save(fig, ".", "pipeline_overview")


# --- 01 Data -------------------------------------------------------------------------------------

def data_figures(features_dir):
    usable = features_dir / "usable_samples.csv"
    population = features_dir / "test_population_coverage.csv"
    rows = read_csv(usable)
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    cohorts = [("fixed_full", "Fixed window"), ("aligned_full", "Aligned window")]
    groups = [(c, p) for c, _ in cohorts for p in ("train", "validation", "test")]
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for crop in CROPS:
        values = np.array([next(int(r["usable_parcel_groups"]) for r in rows
                                if r["cohort"] == c and r["partition"] == p and r["crop"] == crop) for c, p in groups])
        axes[0].bar(x, values, .7, bottom=bottom, color=CROP_COLORS[crop], edgecolor=SURFACE, linewidth=1.5)
        bottom += values
    for xi, total in zip(x, bottom):
        axes[0].text(xi, total + 15, f"{int(total)}", ha="center", fontsize=8, color=INK_2)
    axes[0].set_xticks(x, [f"{p}\n{label.split()[0].lower()}" for c, label in cohorts for p in ("train", "validation", "test")],
                       fontsize=8.5)
    axes[0].set_ylabel("Usable parcels")
    axes[0].set_title("Usable parcels per split (full cohorts, JECAM test)")
    crop_legend(axes[0])
    tests = [r for r in read_csv(population) if r["cohort"] == "fixed_full"]
    campaigns = list(dict.fromkeys(r["test_population"] for r in tests))
    names = {"jecam": "JECAM (primary)", "combined": "Combined pool"}
    campaigns = [c for c in campaigns if c != "combined"]
    y = np.arange(len(campaigns))
    left = np.zeros(len(campaigns))
    for crop in CROPS:
        values = np.array([sum(int(r["usable_parcel_groups"]) for r in tests if r["test_population"] == c and r["crop"] == crop)
                           for c in campaigns])
        axes[1].barh(y, values, .6, left=left, color=CROP_COLORS[crop], edgecolor=SURFACE, linewidth=1.5)
        left += values
    axes[1].set_yticks(y, [names.get(c, c.replace("_", " ").replace(" v1", "")) for c in campaigns], fontsize=8.3)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Usable test parcels (fixed window)")
    axes[1].set_title("External test campaigns")
    fig.tight_layout()
    out.append(save(fig, "01_data", "usable_parcels"))
    used(out[-1], usable, population)

    development = json.loads((features_dir / "features_development.json").read_text())
    test = json.loads((features_dir / "features_test.json").read_text())
    cells = Counter()
    for role, data in (("development", development), ("test", test)):
        seen = set()
        for row in data["rows"]["fixed_full"]:
            if row["partition_group"] not in seen:
                seen.add(row["partition_group"])
                cells[role, row["site_cell"]] += 1
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for role, color, marker in (("development", SERIES["fixed"], "o"), ("test", SERIES["aligned"], "D")):
        points = [(tuple(map(int, cell.split(","))), n) for (r, cell), n in cells.items() if r == role]
        lat = [p[0][0] + .5 for p in points]
        lon = [p[0][1] + .5 for p in points]
        ax.scatter(lon, lat, s=[12 + 3 * np.sqrt(n) for _, n in points], c=color, marker=marker, alpha=.75,
                   edgecolors=SURFACE, linewidths=.8, label=f"{role} parcels")
    ax.scatter([120.7], [16.6], s=120, marker="*", c=INK, label="Atok, Benguet (application)")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-30, 65)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where the labelled parcels are (one-degree site cells; size = parcels)")
    ax.legend(loc="lower left", fontsize=8.5)
    out.append(save(fig, "01_data", "parcel_locations"))
    used(out[-1], features_dir / "features_development.json", features_dir / "features_test.json")
    return out


# --- 02 Preprocessing --------------------------------------------------------------------------

def preprocessing_figures(features_dir):
    out = []
    loss = read_csv(features_dir / "observation_loss_by_half_month.csv")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    labels = [r["half_month_start"] for r in loss]
    clear = [100 * float(r["clear_share"]) for r in loss]
    ax.bar(range(len(labels)), clear, .75, color=SERIES["fixed"], edgecolor=SURFACE)
    ax.axvspan(7.5, 15.5, color="#f3d9c9", alpha=.45, zorder=0)
    ax.set_ylim(0, max(clear) * 1.18)
    ax.text(11.5, max(clear) * 1.08, "fixed window: May–August", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(labels)), [l if l.endswith("-01") else "" for l in labels], fontsize=8)
    ax.set_ylabel("Clear pixel-dates (%)")
    ax.set_title("Share of clear Sentinel-2 observations by half-month (all audited parcels)")
    out.append(save(fig, "02_preprocessing", "clear_observations"))
    used(out[-1], features_dir / "observation_loss_by_half_month.csv")

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3)
    for half in range(25):
        ax.axvline(half, color=GRID, lw=.6, zorder=0)
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    ax.set_xticks(np.arange(1, 24, 2), months)
    ax.set_yticks([.75, 1.95], ["Aligned", "Fixed"])
    ax.grid(False)
    for k in range(8):
        ax.add_patch(Rectangle((8 + k + .05, 1.6), .9, .7, color=SERIES["fixed"], alpha=.85))
        ax.add_patch(Rectangle((5 + k + .05, .4), .9, .7, color=SERIES["aligned"], alpha=.85))
    ax.annotate("green-up start", (5, 1.15), (2.2, 1.35), fontsize=8.5, color=INK_2,
                arrowprops={"arrowstyle": "-|>", "color": MUTED})
    ax.set_title("Both windows: four months as eight half-month bins (9 values each = 72 predictors)")
    ax.text(16.3, 1.95, "May 1 – Sep 1 (end exclusive), every year", va="center", fontsize=8.5, color=INK_2)
    ax.text(13.3, .75, "four months from each parcel's detected green-up", va="center", fontsize=8.5, color=INK_2)
    out.append(save(fig, "02_preprocessing", "window_recipes"))

    counts = read_csv(features_dir / "window_counts.csv")
    usable = read_csv(features_dir / "usable_samples.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, crop in zip(axes, CROPS):
        stages = []
        for window in ("fixed", "aligned"):
            complete = next(int(r["parcel_groups"]) for r in counts
                            if r["window"] == window and r["partition"] == "development" and r["crop"] == crop)
            final = sum(int(r["usable_parcel_groups"]) for r in usable
                        if r["cohort"] == f"{window}_full" and r["partition"] in ("train", "validation") and r["crop"] == crop)
            matched = sum(int(r["usable_parcel_groups"]) for r in usable
                          if r["cohort"] == f"{window}_matched" and r["partition"] in ("train", "validation") and r["crop"] == crop)
            stages.append((window, complete, final, matched))
        x = np.arange(3)
        for i, (window, *values) in enumerate(stages):
            ax.bar(x + (i - .5) * .36, values, .34, color=SERIES[window], edgecolor=SURFACE, label=window)
        ax.axhline(80, color=INK_2, lw=1, ls="--")
        ax.set_xticks(x, ["complete\nwindow", "usable\n(full)", "matched"], fontsize=8.3)
        ax.set_title(crop)
    axes[0].set_ylabel("Development parcels")
    axes[-1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Development parcels surviving each step, by window (dashed: target 80 = 60 train + 20 validation)", x=.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    out.append(save(fig, "02_preprocessing", "window_funnel"))
    used(out[-1], features_dir / "window_counts.csv", features_dir / "usable_samples.csv")
    return out + split_figures(features_dir)


# Partition slots (violet, yellow, magenta) validate as a set; yellow and magenta sit below 3:1 on the
# surface, so every bar carries a direct count label.
PARTITION_COLORS = {"train": "#6250d6", "validation": "#eda100", "test": "#e87ba4"}
SOURCE_LABELS = {"sen4agrinet": "Sen4AgriNet", "francecrops": "FranceCrops", "eurocrops": "EuroCrops"}
TEST_LABELS = {"jecam": "JECAM (Madagascar)", "non_jecam_carrot_potato_supplement_v1": "CropHarvest France / Réunion",
               "non_jecam_cabbage_supplement_v1": "CropHarvest France / Antilles",
               "california_dwr_carrot_2023_v1": "California DWR", "denmark_marker_cabbage_2023_v1": "Denmark"}


def split_strata(features_dir):
    """Rebuild the matched_first strata and hash order from the saved split, and check they reproduce it."""
    from math import floor
    from crop_pipeline.sources import sampling

    split = json.loads((features_dir / "development_split.json").read_text())
    rows = json.loads((features_dir / "features_development.json").read_text())["rows"]
    info, usable = {}, defaultdict(set)
    for cohort, cohort_rows in rows.items():
        for r in cohort_rows:
            info[r["partition_group"]] = (r["crop"], r["source_dataset"], r["partition_group"].split("|")[1])
            usable[cohort].add(r["partition_group"])
    matched = set(split["matched_parcel_groups"])
    strata = defaultdict(list)
    for group in split["assignments"]:
        crop = info[group][0]
        strata[crop, group in matched, group in usable["fixed_full"], group in usable["aligned_full"]].append(group)
    for groups in strata.values():
        groups.sort(key=lambda g: (sampling.stable_rank(g, split["seed"]), g))
        n = len(groups)
        if n > 1:
            k = max(1, min(n - 1, floor(n * split["validation_fraction"] + .5)))
            if any((split["assignments"][g] == "validation") != (i < k) for i, g in enumerate(groups)):
                raise ValueError("Hash order does not reproduce development_split.json")
    return split, info, strata


STRIP_SOURCES = {"sen4agrinet": ("Sen4AgriNet", "#2a78d6"), "francecrops": ("FranceCrops", "#eb6834"),
                 "eurocrops": ("EuroCrops", "#1baf7a"), "jecam": ("JECAM Madagascar", "#eda100"),
                 "cropharvest": ("CropHarvest", "#e87ba4"), "california": ("California", "#008300"),
                 "denmark": ("Denmark", "#6250d6")}


def strip_source(row_source, cohort=None):
    if cohort is None:
        return row_source
    return {"jecam": "jecam", "california_dwr_carrot_2023_v1": "california",
            "denmark_marker_cabbage_2023_v1": "denmark"}.get(cohort, "cropharvest")


def split_strip(features_dir, info, assignment):
    """Slide-style strip: parcels as collected, then pooled, shuffled and cut; the test places stay apart."""
    from crop_pipeline.sources import sampling

    test = {}
    for cohort_rows in json.loads((features_dir / "features_test.json").read_text())["rows"].values():
        for r in cohort_rows:
            test[r["partition_group"]] = strip_source(r["source_dataset"], r["test_cohort"])
    source = {g: i[1] for g, i in info.items()} | test
    rank = {g: sampling.stable_rank(g, 42) for g in source}
    order = list(STRIP_SOURCES)
    collected = sorted(source, key=lambda g: (order.index(source[g]), g))
    shuffled = {p: sorted((g for g in info if assignment[g] == p), key=rank.get) for p in ("train", "validation")}
    test_blocks = sorted(test, key=lambda g: (order.index(test[g]), g))
    ours = shuffled["train"] + shuffled["validation"] + test_blocks
    everything = sorted(source, key=rank.get)
    n = len(source)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(-n * .01, n * 1.01)
    ax.set_ylim(.1, 9.2)
    ax.axis("off")

    def strip(y, groups, alpha=1.0):
        ax.bar(np.arange(len(groups)) + .5, .9, 1, bottom=y, linewidth=0, alpha=alpha,
               color=[STRIP_SOURCES[source[g]][1] for g in groups])

    def heading(y, bold, text, color):
        label = ax.text(0, y, bold, fontsize=12.5, weight="bold", color=color, va="bottom")
        ax.annotate("  " + text, xy=(1, 0), xycoords=label, fontsize=12, color=INK, va="bottom")

    def bracket(y, start, end, label, sub):
        ax.plot([start + 4, start + 4, end - 4, end - 4], [y + .18, y, y, y + .18], color=INK, lw=1.3)
        ax.text((start + end) / 2, y - .12, label, ha="center", va="top", fontsize=10.5, weight="bold",
                color="white", bbox={"boxstyle": "square,pad=.35", "fc": INK, "ec": INK})
        ax.text((start + end) / 2, y - .78, sub, ha="center", va="top", fontsize=8.5, color=INK_2)

    heading(8.55, "1.", f"All {n:,} labelled parcels, as collected from each source", INK)
    strip(7.5, collected)
    left, chips = 0, []
    for key in order:
        count = sum(source[g] == key for g in collected)
        name, color = STRIP_SOURCES[key]
        if key in ("sen4agrinet", "francecrops", "eurocrops"):
            ax.text(left + count / 2, 7.95, f"{name}\n{count}", ha="center", va="center", fontsize=8.5,
                    color="white", weight="bold")
        chips.append(Patch(color=color, label=f"{name} ({count})"))
        left += count
    europe = sum(source[g] in ("sen4agrinet", "francecrops", "eurocrops") for g in source)
    ax.plot([4, 4, europe - 4, europe - 4], [7.38, 7.25, 7.25, 7.38], color=MUTED, lw=1)
    ax.plot([europe + 4, europe + 4, n - 4, n - 4], [7.38, 7.25, 7.25, 7.38], color=MUTED, lw=1)
    ax.text(europe / 2, 7.15, "Europe", ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.text((europe + n) / 2, 7.15, "other places", ha="center", va="top", fontsize=8.5, color=INK_2)

    heading(6.15, "2. What we did:", "pool the three European sources, shuffle by hash, cut; keep other places for test", "#2a78d6")
    strip(5.1, ours)
    cuts = np.cumsum([0, len(shuffled["train"]), len(shuffled["validation"]), len(test_blocks)])
    for x in cuts[1:-1]:
        ax.plot([x, x], [5.05, 6.05], color=SURFACE, lw=3)
    bracket(4.95, cuts[0], cuts[1], "training set", f"{cuts[1]:,} parcels · Europe")
    bracket(4.95, cuts[1], cuts[2], "validation set", f"{cuts[2] - cuts[1]} parcels · Europe")
    bracket(4.95, cuts[2], cuts[3], "test set", f"{cuts[3] - cuts[2]} parcels · new places")

    heading(2.75, "3. Not done:", "shuffle everything together, then cut", MUTED)
    strip(1.7, everything, alpha=.45)
    third = [round(n * f) for f in (.62, .83)]
    for x in third:
        ax.plot([x, x], [1.65, 2.65], color=SURFACE, lw=3)
    ax.text(n / 2, 1.5, "Most test parcels would come from the same European regions as training, so the test score "
            "would hide how the model does somewhere new, like Atok.", ha="center", va="top", fontsize=9, color=INK_2)
    ax.legend(handles=chips, loc="lower center", bbox_to_anchor=(.5, -.04), ncol=7, fontsize=8.5,
              handlelength=1.2, columnspacing=1.2)
    out = save(fig, "02_preprocessing", "split_strip")
    used(out, features_dir / "development_split.json", features_dir / "features_development.json",
         features_dir / "features_test.json")
    return out


def split_figures(features_dir):
    split, info, strata = split_strata(features_dir)
    assignment = split["assignments"]
    out = []
    out.append(split_strip(features_dir, info, assignment))

    # How the hash order works: one crop group in ID order, then sorted by hash score.
    key = max(strata, key=lambda k: len(strata[k]))
    groups = strata[key]
    by_id = sorted(groups)
    fig = plt.figure(figsize=(11, 6.4))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3.1], hspace=.75)
    for row, (order, title) in enumerate((
            (by_id, f"1. Before: the {len(groups)} {key[0]} parcels with only a fixed window, in collection (ID) order"),
            (groups, "2. After: the same parcels sorted by their hash score (SHA-256 of seed 42 + parcel ID)"))):
        ax = fig.add_subplot(grid[row])
        colors = [PARTITION_COLORS[assignment[g]] for g in order]
        ax.bar(range(len(order)), 1, 1, color=colors, linewidth=0)
        ax.set_xlim(-.5, len(order) - .5)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("parcel position", fontsize=8.5)
        if row == 1:
            n_val = sum(assignment[g] == "validation" for g in groups)
            ax.text(n_val / 2, .5, f"first {n_val} → validation", ha="center", va="center", fontsize=9, color=INK)
            ax.text(n_val + (len(groups) - n_val) / 2, .5, f"remaining {len(groups) - n_val} → train",
                    ha="center", va="center", fontsize=9, color="white")
    ax = fig.add_subplot(grid[2])
    names = sorted(strata, key=lambda k: (k[0], -len(strata[k])))
    labels = []
    for i, k in enumerate(names):
        counts = Counter(assignment[g] for g in strata[k])
        windows = "+".join(w for w, ok in (("fixed", k[2]), ("aligned", k[3])) if ok)
        labels.append(f"{k[0]} · {windows}{' · matched' if k[1] else ''}")
        left = 0
        for part in ("validation", "train"):
            ax.barh(i, counts[part], .7, left=left, color=PARTITION_COLORS[part], edgecolor=SURFACE, linewidth=2)
            left += counts[part]
        share = 100 * counts["validation"] / len(strata[k])
        note = (f"{counts['validation']} / {len(strata[k])} validation ({share:.0f}%)" if len(strata[k]) > 1
                else "1 parcel, kept in train")
        ax.text(left + 4, i, note, va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(names)), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(len(v) for v in strata.values()) * 1.32)
    ax.set_xlabel("Development parcels")
    ax.set_title("3. Every crop × window group is cut the same way: about 25% validation, 75% train", fontsize=10)
    ax.legend(handles=[Patch(color=PARTITION_COLORS[p], label=p) for p in ("train", "validation")],
              loc="lower right", fontsize=8.5)
    fig.suptitle("How development parcels are shuffled into train and validation", x=.01, ha="left",
                 fontsize=12, weight="bold")
    out.append(save(fig, "02_preprocessing", "split_shuffle"))
    used(out[-1], features_dir / "development_split.json", features_dir / "features_development.json")

    # What ends up where: sources, crops and countries per partition.
    test_rows = json.loads((features_dir / "features_test.json").read_text())["rows"]
    test = {}
    for cohort_rows in test_rows.values():
        for r in cohort_rows:
            test[r["partition_group"]] = (r["crop"], r["test_cohort"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    bars = [(SOURCE_LABELS[s], p, sum(1 for g, i in info.items() if i[1] == s and assignment[g] == p))
            for s in SOURCE_LABELS for p in ("train", "validation")]
    bars += [(TEST_LABELS[c], "test", sum(1 for v in test.values() if v[1] == c)) for c in TEST_LABELS]
    for i, (label, part, n) in enumerate(bars):
        ax.barh(i, n, .72, color=PARTITION_COLORS[part])
        ax.text(n + 8, i, f"{n}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(bars)), [f"{label} · {part}" for label, part, _ in bars], fontsize=8.3)
    ax.invert_yaxis()
    ax.axhline(5.5, color=MUTED, lw=.8, ls="--")
    ax.legend(handles=[Patch(color=PARTITION_COLORS[p], label=l) for p, l in (
        ("train", "train (Europe)"), ("validation", "validation (Europe)"),
        ("test", "test (other places, never mixed in)"))], loc="lower right", fontsize=8)
    ax.set_xlabel("Parcels")
    ax.set_title("Parcels per source and partition")

    ax = axes[1]
    parts = {"train": [info[g][0] for g in info if assignment[g] == "train"],
             "validation": [info[g][0] for g in info if assignment[g] == "validation"],
             "test": [v[0] for v in test.values()]}
    for i, (part, crops) in enumerate(parts.items()):
        counts, left = Counter(crops), 0
        for crop in CROPS:
            share = 100 * counts[crop] / len(crops)
            ax.barh(i, share, .62, left=left, color=CROP_COLORS[crop], edgecolor=SURFACE, linewidth=2)
            ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center", fontsize=8.3, color="white")
            left += share
    ax.set_yticks(range(3), [f"{p}\n{len(c)} parcels" for p, c in parts.items()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of parcels (%)")
    ax.set_title("Crop mix per partition")
    crop_legend(ax, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)

    ax = axes[2]
    countries = ("FR", "ES", "NL")
    x = np.arange(len(countries))
    for j, part in enumerate(("train", "validation")):
        counts = Counter(info[g][2] for g in info if assignment[g] == part)
        values = [counts[c] for c in countries]
        ax.bar(x + (j - .5) * .38, values, .36, color=PARTITION_COLORS[part], label=part)
        for xi, v in zip(x, values):
            ax.text(xi + (j - .5) * .38, v + 10, str(v), ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, ["France", "Spain", "Netherlands"])
    ax.set_ylabel("Development parcels")
    ax.set_title("Countries in train and validation")
    ax.legend(fontsize=8.5)
    fig.suptitle("Where the parcels end up", x=.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    out.append(save(fig, "02_preprocessing", "split_distribution"))
    used(out[-1], features_dir / "development_split.json", features_dir / "features_development.json",
         features_dir / "features_test.json")
    return out


# --- 03 Models -----------------------------------------------------------------------------------

def model_figures(training_run, testing_run):
    out = []
    validation = {m: json.loads((training_run / "models" / m / "model_manifest.json").read_text())["validation"]["macro_f1"]
                  for m in MODELS}
    scopes = {"jecam": testing_run / "reports/test_scores.json",
              "combined": testing_run / "reports/combined/test_scores.json"}
    scores = {k: json.loads(p.read_text()) for k, p in scopes.items()}
    majority = json.loads((testing_run / "reports/majority_baseline.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 3.6))
    series = [("Locked validation (development)", validation, "#6250d6"),
              ("JECAM external test", {m: scores["jecam"][m]["macro_f1"] for m in MODELS}, SERIES["fixed"]),
              ("Combined external pool", {m: scores["combined"][m]["macro_f1"] for m in MODELS}, SERIES["annual"])]
    x = np.arange(len(MODELS))
    for i, (label, values, color) in enumerate(series):
        bars = ax.bar(x + (i - 1) * .26, [values[m] for m in MODELS], .24, color=color, edgecolor=SURFACE, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .012, f"{bar.get_height():.2f}",
                    ha="center", fontsize=7.6, color=INK_2)
    base = np.mean([majority[m]["metrics"]["macro_f1"] for m in MODELS])
    ax.axhline(base, color=INK_2, lw=1, ls="--")
    ax.text(-.45, base + .015, f"training-majority baseline ≈ {base:.2f}", ha="left", fontsize=8, color=INK_2)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Parcel macro-F1")
    ax.set_title("Development validation is far above the external tests")
    ax.legend(loc="upper right", fontsize=8.5, ncol=3)
    out.append(save(fig, "03_models", "validation_vs_external"))
    used(out[-1], scopes["jecam"], scopes["combined"], testing_run / "reports/majority_baseline.json",
         *(training_run / "models" / m / "model_manifest.json" for m in MODELS))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, m in zip(axes, MODELS):
        matrix = np.array(scores["jecam"][m]["confusion_matrix"], float)
        share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), n in np.ndenumerate(matrix):
            ax.text(j, i, int(n), ha="center", va="center", fontsize=9, color="white" if share[i, j] > .55 else INK)
        ax.set_xticks(range(3), CROPS, fontsize=8)
        ax.set_yticks(range(3), CROPS if m == MODELS[0] else [], fontsize=8)
        ax.grid(False)
        ax.set_title(f"{MODEL_LABELS[m]}  (F1 {scores['jecam'][m]['macro_f1']:.2f})", fontsize=9.5)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("reference")
    fig.suptitle("JECAM confusion matrices (parcels; shading = share of each reference crop)", x=.01, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    out.append(save(fig, "03_models", "jecam_confusion"))
    used(out[-1], scopes["jecam"])
    return out


# --- 04 Limits and errors ----------------------------------------------------------------------

def limits_figures(testing_run, paoay_root):
    out = []
    batch2 = testing_run.parent / f"{testing_run.name}_batch2"
    files = [batch2 / "comparison_metrics.csv", batch2 / "shape_comparison/comparison_metrics.csv",
             batch2 / "indices_only_comparison/comparison_metrics.csv", batch2 / "tuned_comparison/comparison_metrics.csv"]
    metrics = {}
    for path in files:
        for r in read_csv(path):
            if r["macro_f1"] and r["population"] in ("jecam", "combined"):
                metrics[r["cohort"], r["model"], r["population"]] = (float(r["macro_f1"]), int(r["predictors"]))
    order = ["interval_only", "indices_only", "spectral_bands_only", "shape_only", "interval_and_shape", "tuned",
             "single_composite"]
    names = {"interval_only": "72 interval (baseline)", "indices_only": "32 indices only",
             "spectral_bands_only": "40 bands only", "shape_only": "48 shape only",
             "interval_and_shape": "120 interval + shape", "tuned": "72 tuned forest",
             "single_composite": "9 one half-month"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, population in zip(axes, ("jecam", "combined")):
        y = np.arange(len(order))
        for cohort, offset in (("fixed_matched", -.17), ("aligned_matched", .17)):
            values = [metrics.get((cohort, m, population), (np.nan,))[0] for m in order]
            ax.barh(y + offset, values, .32, color=SERIES[cohort.split("_")[0]], edgecolor=SURFACE,
                    label=cohort.replace("_", " "))
        ax.set_yticks(y, [names[m] for m in order], fontsize=8.5)
        ax.set_xlim(0, .8)
        ax.set_xlabel("Parcel macro-F1")
        ax.set_title("JECAM (83 parcels)" if population == "jecam" else "Combined pool (175 parcels)")
    axes[0].invert_yaxis()  # shared y axis: invert once
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, fontsize=8.5)
    fig.suptitle("Comparison variants on the matched parcels (Testing 03, Training 03)", x=.01, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    out.append(save(fig, "04_limits_and_errors", "comparison_variants"))
    used(out[-1], *files)

    batch3 = testing_run.parent / f"{testing_run.name}_batch3"
    errors = [r for r in read_csv(batch3 / "campaign_crop_errors.csv") if r["population"] == "jecam"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(MODELS))
    for i, crop in enumerate(CROPS):
        rate = [next(int(r["errors"]) / int(r["parcels"]) for r in errors if r["model"] == m and r["crop"] == crop)
                for m in MODELS]
        axes[0].bar(x + (i - 1) * .26, [100 * v for v in rate], .24, color=CROP_COLORS[crop], edgecolor=SURFACE, label=crop)
    axes[0].set_xticks(x, [MODEL_LABELS[m] for m in MODELS], fontsize=8.5)
    axes[0].set_ylabel("Parcels misclassified (%)")
    axes[0].set_title("JECAM errors by reference crop")
    axes[0].set_ylim(0, 118)
    axes[0].legend(fontsize=8.5, loc="upper left", ncol=3)
    inside = [sum(int(r["reference_dates_in_window"]) for r in errors if r["model"] == m) for m in MODELS]
    total = [sum(int(r["reference_dates_available"]) for r in errors if r["model"] == m) for m in MODELS]
    axes[1].bar(x, [100 * a / b for a, b in zip(inside, total)], .5,
                color=[SERIES[m.split("_")[0]] for m in MODELS], edgecolor=SURFACE)
    for xi, a, b in zip(x, inside, total):
        axes[1].text(xi, 100 * a / b + 2, f"{a}/{b}", ha="center", fontsize=8, color=INK_2)
    axes[1].set_xticks(x, [MODEL_LABELS[m] for m in MODELS], fontsize=8.5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Reference dates inside window (%)")
    axes[1].set_title("Does the window cover the JECAM observation date?")
    fig.tight_layout()
    out.append(save(fig, "04_limits_and_errors", "error_analysis"))
    used(out[-1], batch3 / "campaign_crop_errors.csv")

    oob = read_csv(batch3 / "oob_summary.csv")
    methods = [("pixel_oob", "pixel out-of-bag"), ("parcel_from_pixel_oob", "parcel from pixel OOB"),
               ("whole_parcel_oob", "whole parcel held out")]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    x = np.arange(len(MODELS))
    for i, (method, label) in enumerate(methods):
        values = [next((100 * float(r["error_rate"]) for r in oob if r["model"] == m and r["method"] == method), np.nan)
                  for m in MODELS]
        ax.bar(x + (i - 1) * .26, values, .24, color=["#c9c8c3", "#8fb6e8", SERIES["fixed"]][i], edgecolor=SURFACE, label=label)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Same-field pixels make internal error look optimistic")
    ax.legend(fontsize=8.5, loc="upper left")
    out.append(save(fig, "04_limits_and_errors", "oob_optimism"))
    used(out[-1], batch3 / "oob_summary.csv")

    applicability = resolve_stage_pointer(paoay_root / "applicability/applicability_stage.json").parent / "summary.csv"
    season = resolve_stage_pointer(paoay_root / "season_shift/season_shift_stage.json").parent / "curves.csv"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    rows = read_csv(applicability)
    axes[0].barh(range(len(rows)), [100 * float(r["inside_fraction"]) for r in rows], .55,
                 color=[NEUTRAL, NEUTRAL, SERIES["aligned"]], edgecolor=SURFACE)
    for i, r in enumerate(rows):
        axes[0].text(100 * float(r["inside_fraction"]) - 1, i, f"{100 * float(r['inside_fraction']):.0f}%",
                     ha="right", va="center", fontsize=8.5, color=INK)
    axes[0].set_yticks(range(len(rows)), [r["pixels"].replace(" (", "\n(") for r in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Inside area of applicability (%)")
    axes[0].set_title("Paoay pixels like the training data?")
    curves = read_csv(season)
    for source, group, color, ls in (("training", "potato", CROP_COLORS["potato"], "-"),
                                     ("training", "cabbage", CROP_COLORS["cabbage"], "-"),
                                     ("training", "carrot", CROP_COLORS["carrot"], "-")):
        series = [r for r in curves if r["source"] == source and r["group"] == group and r["index"] == "NDVI"]
        axes[1].plot([int(r["bin"]) for r in series], [float(r["median"]) for r in series], color=color, lw=2,
                     ls=ls, label=f"training {group}")
    study = [r for r in curves if r["source"] != "training" and r["group"] == "Reference areas" and r["index"] == "NDVI"]
    axes[1].plot([int(r["bin"]) for r in study], [float(r["median"]) for r in study], color=INK, lw=2, ls="--",
                 label="Paoay reference areas")
    axes[1].fill_between([int(r["bin"]) for r in study], [float(r["q25"]) for r in study],
                         [float(r["q75"]) for r in study], color=INK, alpha=.08)
    axes[1].set_xticks(range(1, 9), [r["bin_label"] for r in study], fontsize=7.5, rotation=30)
    axes[1].set_ylabel("Median NDVI")
    axes[1].set_title("Seasonal curves: training crops vs Paoay (map window)")
    axes[1].legend(fontsize=7.8, loc="upper right")
    fig.tight_layout()
    out.append(save(fig, "04_limits_and_errors", "applicability_and_season"))
    used(out[-1], applicability, season)
    return out


# --- 05 Atok application ------------------------------------------------------------------------

def atok_figures(settings, paoay_root):
    from crop_pipeline.paoay import aligned, area_review
    out = []
    baseline = resolve_stage_pointer(paoay_root / "crop_baseline/crop_baseline_stage.json")
    products = [area_review.product("fixed", baseline)]
    folder = area_review.aligned_bundle_for(paoay_root / "aligned_exploratory", aligned.default_model(ROOT))
    if folder is not None:
        products.append(area_review.product("aligned", folder))
    grid = products[0]["grid"]

    def local(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (ROOT / path).resolve()

    kmz_root = local(settings.get("paoay_kmz_root", "./data/paoay"))
    kmz = [kmz_root / n for n in ("00 - Region of Interest.kmz", "00 - Atok Boundary.kmz", "Crop Area.kmz")]
    roi_name = settings.get("paoay_roi_name") or "Region of Interest"
    roi, areas, _ = aligned.study_area(kmz, roi_name, [n for n in settings.get("paoay_context_names", []) if n != roi_name],
                                       grid["crs"])
    study = area_review.area_masks([{"name": "study", "geometry": roi}], grid)["study"]
    landuse = area_review.read_landuse(local(settings["atok_landuse_path"]), roi.buffer(200).bounds, grid["crs"])
    farmland = area_review.farmland(landuse)
    masks = area_review.area_masks(areas, grid)
    calendar = local(settings.get("paoay_calendar_file", paoay_root / "review/calendars/kmz_notes_calendar.csv"))
    entries = area_review.read_calendar(calendar, set(masks))
    land = resolve_stage_pointer(paoay_root / "land_activity/land_activity_stage.json").parent / "land_activity.tif"
    import rasterio
    with rasterio.open(land) as source:
        active = source.read(1) == 1
    farm_rows = [r for item in products for r in area_review.farmland_composition(
        item, study, area_review.polygon_mask(farmland, grid), active, "Land-activity cropland")]
    comparison = [r for item in products for r in area_review.calendar_comparison(item, masks, entries)]
    for name, figure in (("farmland_and_maps", area_review.plot_farmland(products, farmland, farm_rows, areas)),
                         ("area_boundaries", area_review.plot_area_panels(products, areas, study, farmland)),
                         ("calendar_context", area_review.plot_calendar(entries, comparison, products))):
        path = GUIDE / "05_atok_application" / "figures" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(figure)
        out.append(f"05_atok_application/figures/{name}.png")
        used(out[-1], baseline.parent / "crop_baseline.tif", calendar, land,
             *([Path(folder) / "aligned_map.tif"] if folder is not None else []))
    return out


# --- 06 Experiments ------------------------------------------------------------------------------

def experiment_figures(experiments_dir):
    out = []
    summary_path = experiments_dir / "experiments.json"
    summary = json.loads(summary_path.read_text())
    app_path = experiments_dir / "application/application_atok.csv"
    reference_path = experiments_dir / "application/province_reference.json"
    app = {r["map"]: r for r in read_csv(app_path)}
    reference = json.loads(reference_path.read_text())
    labels = list(summary["experiments"])
    short = {k: k.split(" (")[0] for k in labels}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    y = np.arange(len(labels))
    for ax, scope, title in ((axes[0], "jecam", "JECAM"), (axes[1], "combined", "Combined pool")):
        values = [summary["experiments"][k]["external"][scope]["macro_f1"] for k in labels]
        reference_value = summary["experiments"][labels[0]]["external"][scope]["macro_f1"]
        ax.barh(y, values, .62, color=[SERIES["annual"] if k == labels[0] else "#8fd3ba" for k in labels], edgecolor=SURFACE)
        ax.axvline(reference_value, color=INK_2, lw=1, ls="--")
        for yi, v in zip(y, values):
            ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=7.8, color=INK_2)
        ax.set_xlim(0, .75)
        ax.set_title(f"{title} macro-F1")
    axes[0].set_yticks(y, [short[k] for k in labels], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle("Annual-family experiments on the external tests (dashed: annual recipe A)", x=.01, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    out.append(save(fig, "06_experiments", "experiment_external_scores"))
    used(out[-1], summary_path)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    rows = labels + ["PSA Benguet 2024 (reference)"]
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for crop in CROPS:
        values = np.array([float(app[k][f"{crop}_share_of_confident_cropland"]) for k in labels] + [reference[crop]])
        ax.barh(y, 100 * values, .62, left=100 * left, color=CROP_COLORS[crop], edgecolor=SURFACE, linewidth=1.5)
        left += values
    for yi, k in enumerate(labels):
        ax.text(101, yi, f"confident {100 * float(app[k]['confident_share_of_cropland']):.0f}% · "
                         f"distance {float(app[k]['crop_mix_distance_to_benguet_2024']):.2f}", va="center",
                fontsize=7.8, color=INK_2)
    ax.set_yticks(y, [short[k] for k in labels] + ["PSA Benguet 2024"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of confident cropland predictions in the Atok boundary (%)")
    ax.set_title("Crop mix each experiment maps for Atok, against provincial harvested area")
    crop_legend(ax, loc="upper center", bbox_to_anchor=(.4, -.14), ncol=3)
    out.append(save(fig, "06_experiments", "atok_crop_mix"))
    used(out[-1], app_path, reference_path)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    points = sorted(((summary["experiments"][k]["external"]["combined"]["macro_f1"],
                      float(app[k]["crop_mix_distance_to_benguet_2024"]), k) for k in labels))
    for xv, yv, k in points:
        ax.scatter(xv, yv, s=60, color=SERIES["annual"] if k == labels[0] else SERIES["fixed"], edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
    renderer = fig.canvas.get_renderer()
    placed = []
    # Try four positions per label; keep the first whose box overlaps no placed label.
    for xv, yv, k in points:
        for dx, dy, ha in ((5, 5, "left"), (5, -12, "left"), (-5, 5, "right"), (-5, -12, "right"), (5, 16, "left")):
            text = ax.annotate(short[k], (xv, yv), (dx, dy), textcoords="offset points", fontsize=7.8,
                               color=INK_2, ha=ha)
            box = text.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            text.remove()
        else:
            placed.append(ax.annotate(short[k], (xv, yv), (5, 5), textcoords="offset points", fontsize=7.8,
                                      color=INK_2).get_window_extent(renderer))
    ax.set_xlabel("Combined external macro-F1 (higher is better, retrospective)")
    ax.set_ylabel("Atok crop-mix distance to PSA 2024 (lower is closer)")
    ax.set_title("No experiment is best on both")
    out.append(save(fig, "06_experiments", "external_vs_atok"))
    used(out[-1], summary_path, app_path)
    return out


# Mixed training: Europe sources keep the strip colours; test places get the remaining slots, ordered so
# the adjacent pairs validate (the red/green pair sits in the 6-8 CVD band, so every chunk is labelled).
MIXED_PLACES = {"Madagascar": "#eda100", "Senegal / Réunion / Antilles": "#6250d6", "France (CropHarvest)": "#e87ba4",
                "California": "#008300", "Denmark": "#e34948"}
EUROPE_SCALE = .2


def mixed_place(place):
    if place == "France":
        return "France (CropHarvest)"
    return place if place in MIXED_PLACES else "Senegal / Réunion / Antilles"


def mixed_split_strip(mixed_dir, seed=1):
    """Slide-style strip of the weighted mixed-training split, recomputed from its inputs and checked."""
    from crop_pipeline.sources import sampling
    from experiments.annual import workflow as aw
    from experiments.mixed_training import mixing, weighted

    experiment = json.loads((mixed_dir / "experiment.json").read_text())
    inputs = Path(experiment["inputs_handoff"]).parent
    rows, arrays = aw.load_inputs(inputs)
    _, ok, _ = aw.recipe_matrix(experiment["recipe"], arrays)
    rows = weighted.add_coordinates(rows)
    side = weighted.split_by_place(rows, seed=seed)
    places = {r["partition_group"]: mixing.place_of(r) for r in rows if r["partition"] == "test"}
    saved = {(r["place"], r["side"]): int(r["parcels"]) for r in read_csv(mixed_dir / "split.csv")
             if r["split_seed"] == str(seed)}
    if Counter((places[g], v) for g, v in side.items()) != Counter(saved):
        raise ValueError("Recomputed mixed-training split does not match split.csv")
    added_index = [i for i, good in enumerate(ok) if good and side.get(rows[i]["partition_group"]) == "added"]
    fold = weighted.cv_folds(rows, np.asarray(added_index), seed=seed)
    block = {}
    for r in rows:
        if r["partition"] == "test":
            block.setdefault(r["partition_group"], mixing.block_of(r))
    europe = sorted({r["partition_group"] for r in rows if r["partition"] != "test"},
                    key=lambda g: sampling.stable_rank(g, 42))
    source = {r["partition_group"]: r["source_dataset"] for r in rows}
    color = {g: STRIP_SOURCES[source[g]][1] for g in europe}
    color |= {g: MIXED_PLACES[mixed_place(p)] for g, p in places.items()}
    order = list(MIXED_PLACES)
    by_place = lambda g: (order.index(mixed_place(places[g])), block[g], g)  # noqa: E731
    added = sorted((g for g, v in side.items() if v == "added"), key=by_place)
    held = sorted((g for g, v in side.items() if v == "held_out"), key=by_place)
    collected_europe = sorted(europe, key=lambda g: (list(STRIP_SOURCES).index(source[g]), g))
    test_all = sorted(places, key=by_place)
    E = len(europe) * EUROPE_SCALE
    gap = 12

    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    width = E + gap + len(test_all) + gap
    ax.set_xlim(-width * .01, width * 1.01)
    ax.set_ylim(-.3, 12.4)
    ax.axis("off")

    def strip(y, groups, x0, scale=1.0, alpha=None):
        if not groups:
            return
        alphas = alpha or [1.0] * len(groups)
        for a in sorted(set(alphas)):
            idx = [i for i, v in enumerate(alphas) if v == a]
            ax.bar(x0 + (np.asarray(idx) + .5) * scale, .9, scale, bottom=y, linewidth=0, alpha=a,
                   color=[color[groups[i]] for i in idx])

    def heading(y, bold, text, tone=INK):
        label = ax.text(0, y, bold, fontsize=12.5, weight="bold", color=tone, va="bottom")
        ax.annotate("  " + text, xy=(1, 0), xycoords=label, fontsize=11.5, color=INK, va="bottom")

    def bracket(y, start, end, label, sub, box=INK):
        ax.plot([start + 2, start + 2, end - 2, end - 2], [y + .16, y, y, y + .16], color=INK, lw=1.3)
        ax.text((start + end) / 2, y - .1, label, ha="center", va="top", fontsize=10, weight="bold",
                color="white", bbox={"boxstyle": "square,pad=.32", "fc": box, "ec": box})
        if sub:
            ax.text((start + end) / 2, y - .72, sub, ha="center", va="top", fontsize=8.3, color=INK_2)

    def label_places(y, groups, x0):
        left = x0
        for name in order:
            n = sum(mixed_place(places[g]) == name for g in groups)
            if n >= 28:
                ax.text(left + n / 2, y + .45, str(n), ha="center", va="center", fontsize=8, color="white",
                        weight="bold")
            left += n

    t0 = E + gap
    heading(11.75, "1.", f"All parcels in the annual inputs: Europe {len(europe):,} and test places {len(places)}")
    strip(10.7, collected_europe, 0, EUROPE_SCALE)
    left = 0
    for key in ("sen4agrinet", "francecrops", "eurocrops"):
        n = sum(source[g] == key for g in europe)
        ax.text(left + n * EUROPE_SCALE / 2, 11.15, f"{STRIP_SOURCES[key][0]}\n{n:,}", ha="center", va="center",
                fontsize=8, color="white", weight="bold")
        left += n * EUROPE_SCALE
    strip(10.7, test_all, t0)
    label_places(10.7, test_all, t0)
    ax.text(E / 2, 10.45, f"Europe (drawn at {EUROPE_SCALE:.0%} width)", ha="center", va="top", fontsize=8.3,
            color=INK_2)
    ax.text(t0 + len(test_all) / 2, 10.45, "test places (full width)", ha="center", va="top", fontsize=8.3,
            color=INK_2)

    heading(9.2, "2. Mixed training:", f"about half of every test place joins training, in whole 0.1° blocks (split seed {seed})",
            "#2a78d6")
    strip(8.15, europe, 0, EUROPE_SCALE)
    strip(8.15, added, t0)
    strip(8.15, held, t0 + len(added) + gap)
    label_places(8.15, added, t0)
    label_places(8.15, held, t0 + len(added) + gap)
    weight = experiment["weight_choice"][str(seed)]["chosen"]
    bracket(8.0, 0, t0 + len(added), "training set",
            f"Europe train + validation {len(europe):,}  +  added blocks {len(added)} (each counts {weight:g}×)")
    bracket(8.0, t0 + len(added) + gap, t0 + len(added) + gap + len(held), "test set",
            f"held-out blocks {len(held)}", box="#c2185b")

    heading(5.6, "3. Validation inside the added half:", "its blocks are dealt into 3 folds that take turns")
    fold_order = sorted(added, key=lambda g: (fold.get(g, -1), by_place(g)))
    sizes = Counter(fold.get(g, -1) for g in added)
    for k in range(3):
        y = 4.55 - k * 1.25
        alphas = [1.0 if fold.get(g) == k else .18 for g in fold_order]
        strip(y, fold_order, t0, alpha=alphas)
        start = t0 + sum(sizes[j] for j in range(-1, k))
        ax.plot([start, start + sizes[k]], [y - .08, y - .08], color=INK, lw=2.2)
        ax.text(t0 - 8, y + .45, f"turn {k + 1}", ha="right", va="center", fontsize=9, color=INK_2)
        ax.text(start + sizes[k] / 2, y - .17, f"validation fold ({sizes[k]})", ha="center", va="top",
                fontsize=8, color=INK)
    choice = experiment["weight_choice"][str(seed)]["cv_macro_f1"]
    ax.text(t0 - 8, 2.05 - .25, "Each turn trains on Europe + the faded folds and scores the dark fold.\n"
            "Pooled scores by weight: " + ", ".join(f"{float(w):g}× {v:.3f}" for w, v in choice.items())
            + f"  →  {weight:g}× chosen.", ha="right", va="top", fontsize=8.5, color=INK_2)

    scores = defaultdict(list)
    for r in read_csv(mixed_dir / "summary.csv"):
        scores[r["model"]].append(float(r["macro_f1_jecam"]))
    names = [("europe_only", "Europe only"), ("europe_plus_added_w1", "+ added 1×"),
             ("europe_plus_added_w2", "+ added 2×"), ("europe_plus_added_w5", "+ added 5×"),
             ("new_place_chosen_weight", "new place")]
    ax.text(0, .55, f"Held-out JECAM macro-F1, mean of {len(scores['europe_only'])} splits:   " + "   ·   ".join(
        f"{label} {np.mean(scores[key]):.3f}" for key, label in names), fontsize=9.5, color=INK, va="bottom")
    ax.text(0, .05, "\"New place\" trains on the added blocks of the other places only, the way Atok would be "
            "mapped.", fontsize=8.5, color=INK_2, va="bottom")
    handles = [Patch(color=STRIP_SOURCES[k][1], label=STRIP_SOURCES[k][0]) for k in ("sen4agrinet", "francecrops", "eurocrops")]
    handles += [Patch(color=c, label=n) for n, c in MIXED_PLACES.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .02), ncol=8, fontsize=8.3,
              handlelength=1.1, columnspacing=1.0)
    out = save(fig, "06_experiments", "mixed_split_strip")
    used(out, mixed_dir / "experiment.json", mixed_dir / "split.csv", mixed_dir / "summary.csv",
         inputs / "pixels.json")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=ROOT / "pipeline_outputs/experiments/rerun_20261122")
    parser.add_argument("--mixed", type=Path, default=ROOT / "pipeline_outputs/mixed_training/weighted_20261122")
    args = parser.parse_args()
    style()
    settings = json.loads((ROOT / "settings.local.json").read_text())
    report_root = Path(settings.get("report_root", "./data/pipeline_work/inventory")).expanduser()
    paoay_root = (report_root if report_root.is_absolute() else (ROOT / report_root).resolve()).parent / "paoay"
    features_dir = resolve_stage_pointer(ROOT / "pipeline_outputs/preprocessing/current_features.json").parent
    training_run = resolve_stage_pointer(ROOT / "pipeline_outputs/training/current_models.json").parent.parent
    testing_run = resolve_stage_pointer(ROOT / "pipeline_outputs/testing/current_scores.json").parent
    figures = [overview(), *data_figures(features_dir), *preprocessing_figures(features_dir),
               *model_figures(training_run, testing_run), *limits_figures(testing_run, paoay_root),
               *atok_figures(settings, paoay_root), *experiment_figures(args.experiments.resolve()),
               mixed_split_strip(args.mixed.resolve())]
    (GUIDE / "figure_sources.json").write_text(json.dumps({
        "runs": {"features": rel(features_dir), "training": rel(training_run), "testing": rel(testing_run),
                 "experiments": rel(args.experiments), "mixed_training": rel(args.mixed)},
        "figures": {f: SOURCES.get(f) or {"schematic": "diagram of the documented recipe; no run data"}
                    for f in figures}}, indent=2) + "\n")
    print("\n".join(figures))


if __name__ == "__main__":
    main()
