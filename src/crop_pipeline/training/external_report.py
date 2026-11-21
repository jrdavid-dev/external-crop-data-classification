"""Report parcel-level uncertainty from a frozen external testing run."""

import argparse
import csv
import json
from collections import defaultdict
from math import isclose
from pathlib import Path

import numpy as np

from crop_pipeline.training.source_holdout import CROPS

MODELS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")
POPULATION_NAMES = {
    "jecam": "JECAM (primary test)",
    "california_dwr_carrot_2023_v1": "California carrot",
    "denmark_marker_cabbage_2023_v1": "Denmark cabbage",
    "non_jecam_cabbage_supplement_v1": "Non-JECAM cabbage",
    "non_jecam_carrot_potato_supplement_v1": "Non-JECAM carrot and potato",
    "combined": "Combined external test",
}
REVISED_POPULATION_NAMES = {
    scope: ("Non-JECAM potato" if scope == "non_jecam_carrot_potato_supplement_v1" else name)
    for scope, name in POPULATION_NAMES.items()
    if scope != "non_jecam_cabbage_supplement_v1"
}
REVISION_NOTE = (
    "The reported scope was revised after score review. It excludes the non-JECAM cabbage "
    "campaign and reference carrot parcels from the non-JECAM carrot/potato campaign. "
    "The original frozen evaluation remains available."
)
DEFAULT_SEED = 42
DEFAULT_REPETITIONS = 10_000
F1_FLOOR = 0.50


def read_predictions(path, campaign=None):
    parcels = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parcel = row["partition_group"]
            actual, predicted, source = row["crop"], row["prediction"], row["test_cohort"]
            if (not parcel or parcel in parcels or actual not in CROPS or predicted not in CROPS
                    or not source or (campaign is not None and source != campaign)):
                raise ValueError(f"Invalid or duplicate parcel {parcel!r} in {path}")
            parcels[parcel] = (source, actual, predicted)
    return parcels


def require_matched(fixed, aligned):
    if {key: row[:2] for key, row in fixed.items()} != {
        key: row[:2] for key, row in aligned.items()
    }:
        raise ValueError("Matched models must have identical parcel IDs, campaigns and reference crops")


def metric_values(matrix):
    """Evaluate one or more confusion matrices with reference crops on rows."""
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    correct = np.diagonal(matrix, axis1=-2, axis2=-1)
    total = support.sum(axis=-1)
    observed = correct.sum(axis=-1)
    chance = (support * predicted).sum(axis=-1)

    def ratio(numerator, denominator, undefined=0.0):
        return np.divide(numerator, denominator, where=denominator != 0,
                         out=np.full(np.shape(denominator), undefined))

    precision = ratio(correct, predicted)
    recall = ratio(correct, support)
    f1 = ratio(2 * correct, support + predicted)
    complete = (support > 0).all(axis=-1)
    result = {
        ("overall_accuracy", ""): ratio(observed, total, np.nan),
        ("kappa", ""): ratio(observed * total - chance, total**2 - chance, np.nan),
        ("macro_f1", ""): np.where(complete, f1.mean(axis=-1), np.nan),
        ("balanced_accuracy", ""): np.where(complete, recall.mean(axis=-1), np.nan),
    }
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        for index, crop in enumerate(CROPS):
            result[name, crop] = values[..., index]
    return result


def parcel_draws(rows, repetitions, rng):
    """Return row indices for paired, campaign-by-reference-crop resampling."""
    strata = defaultdict(list)
    for index, (campaign, actual, _) in enumerate(rows):
        strata[campaign, actual].append(index)
    return np.concatenate([
        rng.choice(indices, size=(repetitions, len(indices)), replace=True)
        for _, indices in sorted(strata.items())
    ], axis=1)


def confusion(actual, predicted):
    size = len(CROPS)
    pairs = size * actual + predicted
    counts = np.stack([(pairs == cell).sum(axis=-1) for cell in range(size**2)], axis=-1)
    return counts.reshape(*counts.shape[:-1], size, size)


def evaluate(parcels, majority, repetitions, rng):
    rows = [parcels[key] for key in sorted(parcels)]
    actual = np.array([CROPS.index(row[1]) for row in rows], dtype=int)
    predicted = np.array([CROPS.index(row[2]) for row in rows], dtype=int)
    baseline = np.full(len(rows), CROPS.index(majority))
    draws = parcel_draws(rows, repetitions, rng)
    matrices = {name: confusion(actual, labels) for name, labels in
                (("model", predicted), ("majority", baseline))}
    points = {name: metric_values(matrix) for name, matrix in matrices.items()}
    samples = {name: metric_values(confusion(actual[draws], labels[draws]))
               for name, labels in (("model", predicted), ("majority", baseline))}
    return matrices, points, samples


def interval(point, samples):
    valid = samples[np.isfinite(samples)]
    result = {"estimate": float(point) if np.isfinite(point) else None,
              "ci_low": None, "ci_high": None, "valid_replicates": len(valid)}
    if result["estimate"] is not None and len(valid):
        result["ci_low"], result["ci_high"] = map(float, np.quantile(valid, [0.025, 0.975]))
    return result


def verify_saved(saved, matrix, points, context):
    if saved["classes"] != CROPS or saved["confusion_matrix"] != matrix.tolist():
        raise ValueError(f"Saved confusion matrix differs for {context}")
    for i, crop in enumerate(CROPS):
        if saved["per_crop"][crop]["support"] != matrix[i].sum():
            raise ValueError(f"Saved support differs for {context}/{crop}")
    for (metric, crop), value in points.items():
        if crop:
            expected = saved["per_crop"][crop][metric]
        elif metric in ("macro_f1", "balanced_accuracy"):
            expected = saved.get(metric)
        else:
            continue
        if not (np.isnan(value) if expected is None else isclose(value, expected, abs_tol=1e-12)):
            raise ValueError(f"Saved {metric} differs for {context}/{crop}")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retained_reference(scope, crop):
    """Apply the revised reporting scope using campaign and reference label."""
    return (scope in REVISED_POPULATION_NAMES
            and (scope != "non_jecam_carrot_potato_supplement_v1" or crop == "potato"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_status(coverage):
    if any(not counts["parcels"] for counts in coverage.values()):
        return "incomplete_class_coverage"
    return ("targets_met" if all(counts["target_met"] for counts in coverage.values())
            else "pilot_below_targets")


def revised_coverage(reports):
    """Pool saved pixel counts from retained campaign/crop cells."""
    coverage = {}
    for scope in REVISED_POPULATION_NAMES:
        if scope == "combined":
            break
        folder = reports if scope == "jecam" else reports / scope
        scores = read_json(folder / "test_scores.json")
        coverage[scope] = {
            model: {
                crop: (scores[model]["coverage"][crop] if retained_reference(scope, crop)
                       else {"parcels": 0, "pixels": 0, "target_met": False})
                for crop in CROPS
            }
            for model in MODELS
        }
    combined = {}
    for model in MODELS:
        combined[model] = {}
        for crop in CROPS:
            parcels = sum(group[model][crop]["parcels"] for group in coverage.values())
            pixels = sum(group[model][crop]["pixels"] for group in coverage.values())
            combined[model][crop] = {"parcels": parcels, "pixels": pixels,
                                     "target_met": parcels >= 20 and pixels >= 400}
    coverage["combined"] = combined
    return coverage


def coverage_rows(run, *, revised=False):
    """Read the frozen model-specific parcel and usable-pixel counts."""
    reports = run / "reports"
    names = REVISED_POPULATION_NAMES if revised else POPULATION_NAMES
    if revised:
        populations = revised_coverage(reports)
    else:
        populations = {}
        for scope in read_json(reports / "test_scopes.json"):
            folder = reports if scope == "jecam" else reports / scope
            populations[scope] = {
                model: score.get("coverage", {})
                for model, score in read_json(folder / "test_scores.json").items()
            }
    for scope, models in populations.items():
        for model in MODELS:
            for crop, counts in models[model].items():
                if counts["parcels"]:
                    yield {"Population": names.get(scope, scope), "Model": model,
                           "Crop": crop, "Parcels": counts["parcels"], "Usable pixels": counts["pixels"],
                           "Meets both targets": counts["target_met"]}


def report(run, output, seed=DEFAULT_SEED, repetitions=DEFAULT_REPETITIONS, *, revised=False):
    run, output = run.resolve(), output.resolve()
    if output.is_relative_to(run) or output.exists():
        raise ValueError("Use a new output directory outside the frozen run")
    if repetitions < 2:
        raise ValueError("Use at least two bootstrap repetitions")
    reports, predictions = run / "reports", run / "predictions"

    scopes = read_json(reports / "test_scopes.json")
    if (len(scopes) != 6 or scopes[0] != "jecam" or scopes[-1] != "combined"
            or len(set(scopes)) != len(scopes)):
        raise ValueError("Expected JECAM, four distinct supplements and the combined pool")
    reported_coverage = revised_coverage(reports) if revised else None
    figure_scores = {}
    rng = np.random.default_rng(seed)
    pooled = {model: {} for model in MODELS}
    majority_crops = {}
    metrics, differences, cells = [], [], []
    for scope in scopes:
        subfolder = "" if scope == "jecam" else scope
        saved_scores = read_json(reports / subfolder / "test_scores.json")
        saved_baselines = read_json(reports / subfolder / "majority_baseline.json")
        predictions_by_model = {
            model: read_predictions(predictions / subfolder / f"{model}_test_parcels.csv",
                                    None if scope == "combined" else scope)
            for model in MODELS
        }
        require_matched(predictions_by_model["fixed_matched"], predictions_by_model["aligned_matched"])
        for model, parcels in predictions_by_model.items():
            if scope == "combined":
                if parcels != pooled[model]:
                    raise ValueError(f"Saved combined parcel rows differ from the campaign pool for {model}")
                parcels = pooled[model]
            else:
                if pooled[model].keys() & parcels.keys():
                    raise ValueError(f"Duplicate parcel across campaigns for {model}")
                pooled[model].update(parcels)
            baseline = saved_baselines[model]
            majority = baseline["training_majority_class"]
            if majority_crops.setdefault(model, majority) != majority:
                raise ValueError(f"Frozen majority crop differs across populations for {model}")
            actual = np.array([CROPS.index(row[1]) for row in parcels.values()])
            predicted = np.array([CROPS.index(row[2]) for row in parcels.values()])
            model_matrix = confusion(actual, predicted)
            majority_matrix = confusion(actual, np.full(len(actual), CROPS.index(majority)))
            verify_saved(saved_scores[model], model_matrix, metric_values(model_matrix), f"{scope}/{model}")
            verify_saved(baseline["metrics"], majority_matrix, metric_values(majority_matrix),
                         f"{scope}/{model}/majority")
            if revised:
                if scope not in reported_coverage:
                    continue
                parcels = {key: row for key, row in parcels.items() if retained_reference(*row[:2])}
                coverage = reported_coverage[scope][model]
            else:
                coverage = saved_scores[model]["coverage"]
            matrices, points, samples = evaluate(parcels, majority, repetitions, rng)
            context = {"population": scope, "model": model, "parcels": len(parcels),
                       "coverage_status": coverage_status(coverage)}
            model_points = points["model"]
            macro = model_points["macro_f1", ""]
            figure_scores.setdefault(scope, {})[model] = {
                "classes": CROPS, "confusion_matrix": matrices["model"].tolist(),
                "macro_f1": float(macro) if np.isfinite(macro) else None,
                "per_crop": {crop: {"f1": float(model_points["f1", crop])}
                             for crop in CROPS if coverage[crop]["parcels"]},
            }
            if any(coverage[crop]["parcels"] != matrices["model"][i].sum()
                   for i, crop in enumerate(CROPS)):
                raise ValueError(f"Saved coverage differs for {scope}/{model}")
            for predictor, matrix in matrices.items():
                for i, actual in enumerate(CROPS):
                    for j, predicted in enumerate(CROPS):
                        cells.append({"population": scope, "model": model, "predictor": predictor,
                                      "actual": actual, "predicted": predicted, "count": matrix[i, j]})
                for (metric, crop), point in points[predictor].items():
                    limits = interval(point, samples[predictor][metric, crop])
                    flag = metric == "f1" and predictor == "model"
                    support = matrix[CROPS.index(crop)].sum() if crop else len(parcels)
                    pixels = coverage[crop]["pixels"] if crop else sum(
                        values["pixels"] for values in coverage.values())
                    metrics.append({**context, "predictor": predictor, "metric": metric,
                                    "crop": crop, "support": support, "usable_pixels": pixels, **limits,
                                    "f1_below_floor": bool(point < F1_FLOOR) if flag else None,
                                    "ci_extends_below_floor": below_floor(limits["ci_low"]) if flag else None})
            key = ("macro_f1", "")
            model_f1, baseline_f1 = points["model"][key], points["majority"][key]
            limits = interval(model_f1 - baseline_f1, samples["model"][key] - samples["majority"][key])
            available = limits["estimate"] is not None
            reaches_floor = bool(model_f1 >= F1_FLOOR) if available else None
            exceeds_majority = bool(model_f1 > baseline_f1) if available else None
            interval_positive = limits["ci_low"] > 0 if limits["ci_low"] is not None else None
            differences.append({
                **context, "majority_crop": majority,
                "model_macro_f1": float(model_f1) if available else None,
                "majority_macro_f1": float(baseline_f1) if available else None,
                **limits, "macro_f1_at_least_0_50": reaches_floor,
                "exceeds_majority": exceeds_majority, "ci_entirely_above_zero": interval_positive,
                "meets_aggregate_criteria": all((reaches_floor, exceeds_majority, interval_positive))
                if available else None,
            })

    metadata = {
        "source_run": run.name, "seed": seed, "repetitions": repetitions, "confidence_level": 0.95,
        "random_generator": "NumPy default_rng (PCG64)",
        "numpy_version": np.__version__,
        "sampling_unit": "partition_group",
        "sampling": "With replacement within campaign and reference-crop strata, preserving observed counts",
        "combined_assumption": "Conditional on observed campaign and reference-crop counts for each model",
        "interval_method": "Percentile, linear 2.5th and 97.5th quantiles of finite replicates",
        "undefined_metrics": (
            "Zero-denominator precision, recall and crop F1 are scored as zero, matching saved scores. "
            "Kappa with zero denominator and three-class macro-F1 and balanced accuracy with missing "
            "reference crops are unavailable. Undefined replicates are excluded and valid counts are reported."
        ),
        "verified_model_confusion_matrices": len(scopes) * len(MODELS),
        "verified_majority_confusion_matrices": len(scopes) * len(MODELS),
        "verified_saved_point_metrics": ["support", "precision", "recall", "f1", "macro_f1", "balanced_accuracy"],
        "matched_parcel_ids_and_reference_labels_verified": scopes,
        "reported_populations": list(figure_scores),
        "reporting_scope_revision": REVISION_NOTE if revised else None,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "parcel_metrics.csv", metrics)
    write_csv(output / "paired_macro_f1.csv", differences)
    write_csv(output / "confusion_matrices.csv", cells)
    (output / "figure_scores.json").write_text(json.dumps(figure_scores, indent=2) + "\n", encoding="utf-8")
    (output / "methods.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(metadata, metrics, differences), encoding="utf-8")


def below_floor(value):
    return None if value is None else value < F1_FLOOR


def render_report(metadata, metrics, differences):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    def with_ci(row):
        return f"{number(row['estimate'])} [{number(row['ci_low'])}, {number(row['ci_high'])}]"

    def decision(value):
        return "—" if value is None else ("yes" if value else "no")

    model_metrics = {(row["population"], row["model"], row["metric"], row["crop"]): row
                     for row in metrics if row["predictor"] == "model"}
    revision = metadata["reporting_scope_revision"]
    names = REVISED_POPULATION_NAMES if revision else POPULATION_NAMES
    supplement_count = "three" if revision else "four"
    lines = [
        "# External parcel evaluation",
        "",
        f"Frozen predictions from `{metadata['source_run']}`. JECAM is the primary benchmark. "
        "Its Madagascar site provides the closest documented comparison to Paoay’s highland vegetable farming. "
        f"The {supplement_count} supplements broaden the evaluation across growing regions. The non-JECAM potato parcels "
        f"come from Île-de-France and Réunion. The combined pool includes JECAM and all {supplement_count} supplements, "
        "so it overlaps the campaign results. "
        "The full models each use their own eligible parcels. Matched parcel IDs, campaigns and reference "
        "crops agree in every population.",
        "",
        f"95% percentile intervals use {metadata['repetitions']:,} parcel bootstrap repetitions, "
        f"seed {metadata['seed']}, and {metadata['random_generator']}. Each draw samples with replacement "
        "within campaign and reference crop. Model and frozen training-majority predictions use the same "
        "drawn parcel IDs with the same multiplicities. Combined intervals are conditional on each model’s "
        "observed campaign and reference-crop counts. Intervals assume independent parcels within strata. "
        "They do not quantify geographic clustering, campaign-to-campaign variation, or model-fitting uncertainty.",
        "",
        metadata["undefined_metrics"],
        "A zero for a crop with no reference parcels follows that reporting convention and provides no evidence "
        "about detection of that crop. A stratum with one parcel contributes the same parcel to every draw.",
        "",
        "Coverage targets are 20 reference parcels and 400 usable pixels per crop. "
        "Each crop cell lists parcels / pixels and F1 [95% CI]. "
        "**†** marks F1 below 0.50. **‡** marks an interval extending below 0.50. "
        "A dash indicates an absent reference crop or unavailable statistic.",
    ]
    if revision:
        lines[3:3] = [revision, ""]
    for scope in metadata["reported_populations"]:
        lines += [
            "",
            f"## {names.get(scope, scope)}",
            "",
            "| Model | Total parcels / pixels | Carrot | Cabbage | Potato | Macro-F1 [95% CI] |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        rows = [row for row in differences if row["population"] == scope]
        for row in rows:
            key = (scope, row["model"])
            macro = model_metrics[*key, "macro_f1", ""]
            crops = []
            for crop in CROPS:
                score = model_metrics[*key, "f1", crop]
                flags = (" †" if score["f1_below_floor"] else "") + (
                    " ‡" if score["ci_extends_below_floor"] else "")
                crops.append(f"{score['support']} / {score['usable_pixels']} · {with_ci(score)}{flags}"
                             if score["support"] else "—")
            totals = f"{row['parcels']} / {macro['usable_pixels']}"
            lines.append("| " + " | ".join([row["model"], totals, *crops, with_ci(macro)]) + " |")

        lines += [
            "",
            "| Model | Accuracy [95% CI] | Kappa [95% CI] | Majority macro-F1 | "
            "Paired difference [95% CI] | F1 ≥ 0.50 | Above majority | CI above zero | All three |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            key = (scope, row["model"])
            scores = [with_ci(model_metrics[*key, name, ""]) for name in ("overall_accuracy", "kappa")]
            criteria = [decision(row[name]) for name in ("macro_f1_at_least_0_50", "exceeds_majority",
                                                         "ci_entirely_above_zero", "meets_aggregate_criteria")]
            values = [row["model"], *scores, number(row["majority_macro_f1"]), with_ci(row), *criteria]
            lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        f"Verified {metadata['verified_model_confusion_matrices']} saved model confusion matrices, "
        f"{metadata['verified_majority_confusion_matrices']} majority matrices and all available saved "
        "performance scores and support counts.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Frozen testing run containing predictions/ and reports/")
    parser.add_argument("output", type=Path, help="New output directory outside the frozen run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--revised-scope", action="store_true", help="Use the revised notebook test scope")
    args = parser.parse_args()
    report(args.run, args.output, args.seed, args.repetitions, revised=args.revised_scope)


if __name__ == "__main__":
    main()
