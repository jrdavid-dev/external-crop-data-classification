"""Seasonal-curve comparison between the model's training pixels and the study area.

Cross-region crop models mostly fail because the growing season is shifted between regions
(Nyborg et al. 2022, "TimeMatch"; Wang et al. 2021, "Phenology Alignment Network"). This
diagnostic puts the training pixels' half-month index curves, per crop, next to the study
area's curves for the same calendar bins, and estimates the bin shift that would best align
them. It describes; it does not adapt the model, move the window or change any prediction.

Limits: the window has eight half-month bins, so only shifts of a few bins can be tested and a
crop grown mostly outside the window cannot be aligned at all. Study-area groups are not
labelled fields: "reference areas" are farmer-described polygons with rotations, and pixels
grouped by raw prediction were chosen by the model because they resemble that crop.
"""

import csv
from datetime import date
import json
from pathlib import Path
import tempfile

import numpy as np

from crop_pipeline.paoay import applicability, crop_baseline, features, imagery, predictions, references

VERSION = 1
KIND = "paoay_season_shift"
INDICES = ("NDVI", "NDRE", "NDMI", "SAVI")
RULES = {"indices": list(INDICES), "bins": 8, "statistic": "median_with_interquartile_range",
         "maximum_lag_bins": 3, "minimum_overlap_bins": 5, "lag_metric": "rmse_of_median_curves",
         "minimum_group_pixels": 20}
FILES = {"curves.csv", "comparison.csv"}
CURVE_COLUMNS = ["source", "group", "index", "bin", "bin_label", "pixels", "q25", "median", "q75"]
COMPARISON_COLUMNS = ["crop", "study_group", "index", "training_pixels", "study_pixels", "training_peak_bin", "study_peak_bin",
                      "training_amplitude", "study_amplitude", "training_mean", "study_mean", "correlation",
                      "rmse_same_bins", "best_lag_bins", "best_lag_days", "rmse_best_lag"]


def bin_columns(index):
    return [f"bin_{number:02d}_{index}" for number in range(1, RULES["bins"] + 1)]


def available(columns):
    """The comparison needs the per-bin index predictors (interval designs)."""
    return all(name in columns for index in INDICES for name in bin_columns(index))


def bin_labels(period):
    labels = []
    for interval in period["intervals"]:
        start = date.fromisoformat(interval["start"])
        labels.append(f"{start:%b} {'1–15' if start.day == 1 else '16–end'}")
    return labels


def summarize(values):
    """Median and interquartile range per bin for an (n pixels, 8 bins) array."""
    values = np.asarray(values, dtype="float64").reshape(-1, RULES["bins"])
    q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
    return {"pixels": len(values), "q25": q25, "median": median, "q75": q75}


def best_lag(training, study, maximum_lag=RULES["maximum_lag_bins"], minimum_overlap=RULES["minimum_overlap_bins"]):
    """Bin shift minimizing RMSE between two median curves.

    A positive lag means the study-area curve runs later than the training curve by that many
    half-month bins: study[bin + lag] is compared with training[bin].
    """
    training, study = np.asarray(training, dtype="float64"), np.asarray(study, dtype="float64")
    if training.shape != study.shape or training.ndim != 1:
        raise ValueError("Curves must be one-dimensional and equally long")
    scores = {}
    for lag in range(-maximum_lag, maximum_lag + 1):
        left = training[max(0, -lag):len(training) - max(0, lag)]
        right = study[max(0, lag):len(study) - max(0, -lag)]
        if len(left) >= minimum_overlap:
            scores[lag] = float(np.sqrt(np.mean((left - right) ** 2)))
    lag = min(scores, key=lambda value: (scores[value], abs(value)))
    return {"lag": lag, "rmse": scores[lag], "rmse_same_bins": scores[0]}


def compare(training, study):
    """Describe two median curves with the statistics reported in the comparison table."""
    shift = best_lag(training, study)
    varying = np.std(training) > 0 and np.std(study) > 0
    return {"training_peak_bin": int(np.argmax(training)) + 1, "study_peak_bin": int(np.argmax(study)) + 1,
            "training_amplitude": float(np.ptp(training)), "study_amplitude": float(np.ptp(study)),
            "training_mean": float(np.mean(training)), "study_mean": float(np.mean(study)),
            "correlation": float(np.corrcoef(training, study)[0, 1]) if varying else "",
            "rmse_same_bins": shift["rmse_same_bins"], "best_lag_bins": shift["lag"],
            "best_lag_days": shift["lag"] * 15, "rmse_best_lag": shift["rmse"]}


def _training_curves(path, cohort, model_manifest, crops):
    manifest = json.loads(Path(model_manifest).read_text())
    rows = json.loads(Path(path).read_text())["rows"][cohort]
    split = manifest["development_split"]
    if any(split.get(row["partition_group"]) != row["partition"] for row in rows):
        raise ValueError("Development rows differ from the model's recorded training/validation split")
    train = [row for row in rows if row["partition"] == "train"]
    return {crop: {index: np.asarray([[float(row[name]) for name in bin_columns(index)] for row in train if row["crop"] == crop],
                                     dtype="float64").reshape(-1, RULES["bins"]) for index in INDICES} for crop in crops}


def _reference_pixels(feature):
    """Feature rows inside the union of reference areas (context polygons excluded)."""
    image = imagery.load_imagery_bundle(feature["manifest"]["imagery_handoff"])
    reference = references.load_reference_bundle(image["manifest"]["reference_handoff"])
    grid = feature["manifest"]["grid"]
    mask = crop_baseline.reference_mask(reference, grid, np.ones((grid["height"], grid["width"]), dtype=bool))
    return mask[feature["rows"], feature["columns"]]


def prepare_request(prediction_handoff, output_root, training_rows=None):
    prediction = predictions.load_prediction_bundle(prediction_handoff)
    feature_handoff = prediction["manifest"]["feature_handoff"]
    feature = features.load_feature_bundle(feature_handoff)
    if not available(feature["manifest"]["feature_columns"]):
        raise ValueError("Season comparison needs the per-bin index predictors of an interval feature design")
    base = applicability.prepare_request(feature_handoff, output_root, training_rows)
    identity = {"version": VERSION, "prediction_sha256": references.checksum(prediction_handoff),
                "feature_sha256": base["identity"]["feature_sha256"],
                "model_manifest_sha256": base["identity"]["model_manifest_sha256"],
                "training_rows_sha256": base["identity"]["training_rows_sha256"], "cohort": base["identity"]["cohort"],
                "reference_sha256": references.checksum(
                    imagery.load_imagery_bundle(feature["manifest"]["imagery_handoff"])["manifest"]["reference_handoff"]),
                "rules": RULES, "implementation_sha256": references.checksum(__file__),
                "reference_mask_sha256": references.checksum(crop_baseline.__file__)}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "prediction_handoff": str(Path(prediction_handoff).resolve()),
            "feature_handoff": str(Path(feature_handoff).resolve()), "model_manifest": base["model_manifest"],
            "training_rows": base["training_rows"], "folder": Path(output_root) / "bundles" / key}


def build(request):
    """Save per-bin curves and the training-versus-study comparison; reuse a verified bundle."""
    folder, manifest_path = request["folder"], request["folder"] / "season_shift_handoff.json"
    if folder.exists():
        load_bundle(manifest_path)
        return manifest_path
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Season comparison request changed after planning")
    prediction = predictions.load_prediction_bundle(request["prediction_handoff"])
    feature = features.load_feature_bundle(request["feature_handoff"])
    columns = list(feature["manifest"]["feature_columns"])
    class_codes = prediction["manifest"]["class_codes"]
    crops = list(class_codes)
    training = _training_curves(request["training_rows"], request["identity"]["cohort"], request["model_manifest"], crops)
    labels = bin_labels(feature["manifest"]["period"])
    in_reference = _reference_pixels(feature)
    study = {"All study-area pixels": np.ones(len(feature["matrix"]), dtype=bool), "Reference areas": in_reference}
    study.update({f"Predicted {crop}": prediction["class_codes"] == code for crop, code in class_codes.items()})
    study = {name: mask for name, mask in study.items() if int(mask.sum()) >= RULES["minimum_group_pixels"]}
    curves, medians = [], {}
    def add(source, group, index, values):
        summary = summarize(values)
        medians[(source, group, index)] = summary
        curves.extend({"source": source, "group": group, "index": index, "bin": number + 1, "bin_label": labels[number],
                       "pixels": summary["pixels"], "q25": float(summary["q25"][number]),
                       "median": float(summary["median"][number]), "q75": float(summary["q75"][number])}
                      for number in range(RULES["bins"]))
    for index in INDICES:
        positions = [columns.index(name) for name in bin_columns(index)]
        for crop in crops:
            if len(training[crop][index]) >= RULES["minimum_group_pixels"]:
                add("training", crop, index, training[crop][index])
        for name, mask in study.items():
            add("study", name, index, feature["matrix"][mask][:, positions])
    comparison = []
    for index in INDICES:
        for crop in crops:
            if ("training", crop, index) not in medians:
                continue
            for name in ("Reference areas", "All study-area pixels", f"Predicted {crop}"):
                if ("study", name, index) not in medians:
                    continue
                first, second = medians[("training", crop, index)], medians[("study", name, index)]
                comparison.append({"crop": crop, "study_group": name, "index": index, "training_pixels": first["pixels"],
                                   "study_pixels": second["pixels"], **compare(first["median"], second["median"])})
    if not comparison:
        raise ValueError("No training crop or study group has enough pixels for a season comparison")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".season-shift-", dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        staging.mkdir()
        for name, rows, header in (("curves.csv", curves, CURVE_COLUMNS), ("comparison.csv", comparison, COMPARISON_COLUMNS)):
            with (staging / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=header); writer.writeheader(); writer.writerows(rows)
        manifest = {"format_version": VERSION, "kind": KIND, "status": "prepared", "bundle_id": request["bundle_id"],
            "identity": request["identity"], "prediction_handoff": request["prediction_handoff"],
            "feature_handoff": request["feature_handoff"], "model_manifest": request["model_manifest"],
            "training_rows": request["training_rows"], "period": feature["manifest"]["period"], "bin_labels": labels,
            "crops": crops, "study_groups": {name: int(mask.sum()) for name, mask in study.items()},
            "predictions_changed": False,
            "lag_convention": "positive lag: the study-area curve runs later than the training curve by that many half-month bins",
            "method": "Median half-month index curves; shift by minimum RMSE within the window (cf. Nyborg et al. 2022, TimeMatch)",
            "limitations": ["Only shifts of up to three half-month bins can be tested inside one four-month window",
                            "Study-area groups are not labelled single-crop fields",
                            "Pixels grouped by raw prediction were selected by the model and resemble training by construction",
                            "A good curve match is not evidence that a prediction is correct"],
            "files": {name: {"sha256": references.checksum(staging / name)} for name in sorted(FILES)}}
        (staging / "season_shift_handoff.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(folder)
    load_bundle(manifest_path)
    return manifest_path


def load_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != KIND or manifest.get("status") != "prepared"
            or manifest.get("predictions_changed") is not False or set(manifest.get("files", {})) != FILES
            or imagery._digest(manifest.get("identity", {}))[:20] != manifest.get("bundle_id")):
        raise ValueError("Unsupported or incomplete season comparison handoff")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved season comparison output changed: {name}; files preserved")
    if references.checksum(manifest["prediction_handoff"]) != manifest["identity"]["prediction_sha256"]:
        raise ValueError("Season comparison belongs to different predictions")
    tables = {}
    for name in FILES:
        with (path.parent / name).open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
    return {"manifest": manifest, "folder": path.parent, "curves": tables["curves.csv"], "comparison": tables["comparison.csv"]}


def comparison_rows(bundle, index="NDVI", study_group="Reference areas"):
    """Display rows for one index and study group, falling back to all study pixels."""
    groups = {row["study_group"] for row in bundle["comparison"]}
    study_group = study_group if study_group in groups else "All study-area pixels"
    rows = []
    for row in bundle["comparison"]:
        if row["index"] != index or row["study_group"] != study_group:
            continue
        labels = bundle["manifest"]["bin_labels"]
        rows.append({"crop": row["crop"].title(), "compared_with": study_group,
                     "training_peak": labels[int(row["training_peak_bin"]) - 1], "study_peak": labels[int(row["study_peak_bin"]) - 1],
                     "training_amplitude": f"{float(row['training_amplitude']):.2f}", "study_amplitude": f"{float(row['study_amplitude']):.2f}",
                     "training_mean": f"{float(row['training_mean']):.2f}", "study_mean": f"{float(row['study_mean']):.2f}",
                     "correlation": f"{float(row['correlation']):.2f}" if row["correlation"] != "" else "",
                     "rmse_same_bins": f"{float(row['rmse_same_bins']):.3f}",
                     "best_shift": (f"{int(row['best_lag_bins']):+d} bins ({int(row['best_lag_days']):+d} days)"
                                    + (", at the test limit" if abs(int(row["best_lag_bins"])) == RULES["maximum_lag_bins"] else "")),
                     "rmse_best_shift": f"{float(row['rmse_best_lag']):.3f}"})
    return rows


def plot(bundle, index="NDVI"):
    """One panel per crop: training curve against the study area's curves for the same bins."""
    import matplotlib.pyplot as plt

    manifest = bundle["manifest"]
    curves = {}
    for row in bundle["curves"]:
        if row["index"] == index:
            curves.setdefault((row["source"], row["group"]), []).append(row)
    def series(key):
        rows = sorted(curves[key], key=lambda row: int(row["bin"]))
        return tuple(np.array([float(row[name]) for row in rows]) for name in ("q25", "median", "q75")) + (int(rows[0]["pixels"]),)
    crops = [crop for crop in manifest["crops"] if ("training", crop) in curves]
    bins = np.arange(1, RULES["bins"] + 1)
    figure, axes = plt.subplots(1, len(crops), figsize=(5.2 * len(crops), 4.6), sharey=True, layout="constrained", squeeze=False)
    styles = (("Reference areas", "#3a8a5c", "-"), ("All study-area pixels", "#777777", ":"))
    for axis, crop in zip(axes[0], crops):
        low, middle, high, count = series(("training", crop))
        axis.fill_between(bins, low, high, color="#4b77a9", alpha=.2)
        axis.plot(bins, middle, color="#4b77a9", linewidth=2.2, marker="o", label=f"Training {crop} (n={count:,})")
        for name, color, line in styles:
            if ("study", name) in curves:
                low, middle, high, count = series(("study", name))
                if line == "-":
                    axis.fill_between(bins, low, high, color=color, alpha=.15)
                axis.plot(bins, middle, color=color, linewidth=2, linestyle=line, marker="s" if line == "-" else None,
                          label=f"{name} (n={count:,})")
        if ("study", f"Predicted {crop}") in curves:
            _, middle, _, count = series(("study", f"Predicted {crop}"))
            axis.plot(bins, middle, color="#c0504d", linewidth=1.6, linestyle="--", label=f"Predicted {crop} (n={count:,})")
        axis.set_title(crop.title())
        axis.set_xticks(bins, manifest["bin_labels"], rotation=45, ha="right", fontsize=8)
        axis.grid(alpha=.25)
        axis.legend(fontsize=7.5, loc="best")
    axes[0][0].set_ylabel(f"{index} (median; band = middle 50% of pixels)")
    figure.suptitle(f"Seasonal {index} curves: European training pixels and the study area, same calendar bins "
                    f"({manifest['period']['start']} to {manifest['period']['end']}, end exclusive)")
    return figure
