"""Indices-only and development-only tuning comparisons for the matched forests."""

from csv import DictReader
from pathlib import Path
from shutil import copyfile

import joblib
import numpy as np
from sklearn.base import clone

from crop_pipeline.common import save_json
from crop_pipeline.paoay.predictions import load_classifier
from crop_pipeline.training import geographic_validation, model_comparisons as comparison, notebook_workflow, source_holdout
from crop_pipeline.training.external_report import (
    DEFAULT_REPETITIONS, DEFAULT_SEED, REVISED_POPULATION_NAMES, confusion, interval,
    metric_values, parcel_draws, write_csv,
)
from crop_pipeline.windows.seasons import file_sha256

INDICES = {"NDVI", "NDRE", "NDMI", "SAVI"}
TUNING_CANDIDATES = {
    "original": {},
    "leaf_4": {"min_samples_leaf": 4},
    "leaf_8": {"min_samples_leaf": 8},
    "depth_12": {"max_depth": 12},
    "half_features": {"max_features": 0.5},
}


def select_settings(baseline, train, columns, folds):
    fold_rows = comparison.spatial_fold_rows(train, folds)
    scores = {
        label: comparison.spatial_scores(clone(baseline).set_params(**params), columns, fold_rows)
        for label, params in TUNING_CANDIDATES.items()
    }
    # max keeps the earliest candidate on exact ties, favoring the original.
    return max(scores, key=lambda label: scores[label]["macro_f1"]), scores


def freeze_followup(training_run, batch2_receipt, experiment):
    """Read only development inputs, fit the requested models, and seal their recipes."""
    if experiment not in ("indices_only", "tuned"):
        raise ValueError("Expected indices_only or tuned")
    training_run, batch2_receipt = training_run.resolve(), batch2_receipt.resolve()
    original = comparison.read_json(batch2_receipt)
    models = training_run / "models"
    stage, _ = notebook_workflow._verify(models / "models_stage.json", "models_stage")
    features = training_run / "features_development.json"
    if (file_sha256(stage) != original["training_models_stage_sha256"]
            or file_sha256(features) != original["development"]["feature_sha256"]):
        raise ValueError("Follow-up requires the original frozen development snapshot")
    data = comparison.read_json(features)
    source_holdout.require_matching(data["rows"])
    output = batch2_receipt.parent / f"{experiment}_comparison"
    if output.is_relative_to(training_run):
        raise ValueError("Place comparison outputs outside the frozen training run")
    output.mkdir(exist_ok=False)
    cohorts = {}
    for cohort in comparison.COHORTS:
        base_manifest = models / cohort / "model_manifest.json"
        previous = original["cohorts"][cohort]
        rows = source_holdout.checked_rows(data["rows"][cohort], False)
        split = {row["partition_group"]: row["partition"] for row in rows}
        train = [row for row in rows if row["partition"] == "train"]
        validation = [row for row in rows if row["partition"] == "validation"]
        if (file_sha256(base_manifest) != previous["baseline_manifest_sha256"]
                or split != previous["development_split"]
                or {row["partition_group"] for row in train} & {row["partition_group"] for row in validation}):
            raise ValueError(f"{cohort}: frozen development split or forest changed")
        baseline, baseline_recipe = load_classifier(base_manifest)
        columns = baseline_recipe["feature_columns"]
        selection = {}
        candidate = baseline
        folder = output / cohort
        folder.mkdir()
        baseline_validation = Path(cohort) / "interval_only_validation_predictions.csv"
        copyfile(batch2_receipt.parent / baseline_validation, output / baseline_validation)
        artifacts = {}
        if experiment == "indices_only":
            columns = [name for name in columns if name.rsplit("_", 1)[-1] in INDICES]
            if len(columns) != 32:
                raise ValueError("Expected four indices in each of eight bins")
            band = previous["models"]["spectral_bands_only"]
            recipe_path = Path(band["recipe"])
            original_band = batch2_receipt.parent / recipe_path
            if file_sha256(original_band) != band["sha256"]:
                raise ValueError("Frozen bands-only recipe changed")
            (output / recipe_path.parent).mkdir()
            for name in ("recipe.json", "model.joblib", "validation_predictions.csv"):
                copyfile(original_band.with_name(name), output / recipe_path.with_name(name))
            artifacts["spectral_bands_only"] = band
        else:
            folds = list(geographic_validation.spatial_folds(train))
            held = [sorted(ids) for _, ids in folds]
            if held != previous["fold_held_ids"]:
                raise ValueError("Tuning must use the original development spatial folds")
            label, scores = select_settings(baseline, train, columns, folds)
            candidate = clone(baseline).set_params(**TUNING_CANDIDATES[label])
            selection = {
                "selected": label, "settings_overrides": TUNING_CANDIDATES,
                "training_spatial_oof_scores": scores, "fold_held_ids": held,
                "selection_rule": "highest pooled training spatial out-of-fold parcel macro-F1; "
                                  "ties use candidate order, starting with the original",
            }
            print(f"{cohort}: selected {label} (training OOF macro-F1 {scores[label]['macro_f1']:.3f})",
                  flush=True)
        model = comparison.fit(candidate, train, columns)
        metrics, predictions = source_holdout.parcel_metrics(model, validation, columns)
        folder = folder / experiment
        folder.mkdir()
        model_path = folder / "model.joblib"
        joblib.dump(model, model_path)
        write_csv(folder / "validation_predictions.csv", predictions)
        recipe_path = folder / "recipe.json"
        save_json(recipe_path, {
            "cohort": cohort, "variant": experiment, "feature_columns": columns,
            "window_recipe": baseline_recipe["window_recipe"], "quality": baseline_recipe["quality"],
            "label_weights": comparison.WEIGHTS, "random_forest": model.get_params(),
            "model_sha256": file_sha256(model_path), "validation": metrics,
            "selection": selection,
        })
        artifacts[experiment] = {"recipe": str(recipe_path.relative_to(output)),
                                 "sha256": file_sha256(recipe_path)}
        cohorts[cohort] = {"development_split": split, "models": artifacts, **selection}
    receipt = output / "comparators.json"
    save_json(receipt, {**original, "cohorts": cohorts, "experiment": experiment,
                        "source_batch2_receipt_sha256": file_sha256(batch2_receipt)})
    return receipt


def paired_macro_difference(original, tuned, repetitions, rng):
    comparison.require_shared_parcels(original, tuned)
    if {row["crop"] for row in original} != set(comparison.CROPS):
        return interval(np.nan, np.empty(0))
    original = sorted(original, key=lambda row: row["partition_group"])
    tuned = sorted(tuned, key=lambda row: row["partition_group"])
    draws = parcel_draws([(r["test_cohort"], r["crop"], r["prediction"]) for r in original],
                         repetitions, rng)
    encode = comparison.CROPS.index
    actual = np.array([encode(row["crop"]) for row in original])
    labels = [np.array([encode(row["prediction"]) for row in rows]) for rows in (original, tuned)]
    key = ("macro_f1", "")
    points = [metric_values(confusion(actual, prediction))[key] for prediction in labels]
    samples = [metric_values(confusion(actual[draws], prediction[draws]))[key] for prediction in labels]
    return interval(points[1] - points[0], samples[1] - samples[0])


def paired_results(output):
    with (output / "external_predictions.csv").open(newline="", encoding="utf-8") as stream:
        records = list(DictReader(stream))
    rng = np.random.default_rng(DEFAULT_SEED)
    results = []
    for scope in REVISED_POPULATION_NAMES:
        for cohort in comparison.COHORTS:
            baseline, tuned = [
                comparison.scoped_predictions(
                    [r for r in records if r["cohort"] == cohort and r["model"] == label], scope)
                for label in ("interval_only", "tuned")
            ]
            results.append({"population": scope, "cohort": cohort, "parcels": len(baseline),
                            **paired_macro_difference(baseline, tuned, DEFAULT_REPETITIONS, rng)})
    write_csv(output / "paired_macro_f1.csv", results)
    return results


def render_followup(receipt, scores):
    frozen = comparison.read_json(receipt)
    output = receipt.parent
    if frozen["experiment"] == "indices_only":
        lines = [
            "# Eight-bin spectral feature comparison", "",
            "The matched forests compare 32 indices-only, 40 bands-only, and 72 bands-plus-indices "
            "predictors with the same training weights, forest settings, parcels and pixel locations. "
            "Indices-only retains NDMI and its moisture-sensitive information.",
        ]
    else:
        lines = [
            "# Development-only matched forest tuning", "",
            "The 72 predictors, training pixels, weights, 1,000 trees, seed 42, Gini criterion and "
            "half-size bootstrap were held fixed. Each candidate changed one setting from the original. "
            "Selection used five geographic folds inside development training and pooled out-of-fold "
            "parcel macro-F1. Locked validation was scored after selection.",
            "", "| Cohort | Candidate | Selected | Training spatial macro-F1 |",
            "| --- | --- | --- | ---: |",
        ]
        for cohort, details in frozen["cohorts"].items():
            for label, metrics in details["training_spatial_oof_scores"].items():
                chosen = "yes" if label == details["selected"] else ""
                lines.append(f"| {cohort} | {label} | {chosen} | {metrics['macro_f1']:.3f} |")
    lines += [
        "", "External scores use the revised Batch 1 scope and identical matched parcel IDs. "
        "This is a follow-up on an already examined benchmark.",
        *comparison.comparison_sections(scores),
    ]
    if frozen["experiment"] == "tuned":
        lines += [
            "", "## Paired external macro-F1 differences", "",
            f"Tuned minus original. Percentile 95% intervals use {DEFAULT_REPETITIONS:,} paired parcel "
            f"resamples with seed {DEFAULT_SEED}, stratified by campaign and reference crop. "
            "The combined interval is conditional on the observed campaign and crop counts. "
            "The three-class difference is unavailable for supplements missing a reference crop.",
            "", "| Population | Cohort | Parcels | Difference [95% CI] |",
            "| --- | --- | ---: | --- |",
        ]
        for row in paired_results(output):
            estimate = row["estimate"]
            value = ("—" if estimate is None else
                     f"{estimate:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
            lines.append(f"| {REVISED_POPULATION_NAMES[row['population']]} | {row['cohort']} | "
                         f"{row['parcels']} | {value} |")
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
