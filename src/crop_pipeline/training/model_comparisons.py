"""Batch 2 comparisons on the frozen matched development and external snapshots."""

from csv import DictReader
import json
from pathlib import Path
from shutil import copyfile

import joblib
import numpy as np
import sklearn
from sklearn.base import clone

from crop_pipeline.common import save_json
from crop_pipeline.paoay.predictions import load_classifier
from crop_pipeline.training import earth_engine, exploratory, geographic_validation, notebook_workflow, source_holdout
from crop_pipeline.training.external_report import (
    REVISED_POPULATION_NAMES, read_predictions, retained_reference, write_csv,
)
from crop_pipeline.windows.seasons import file_sha256

COHORTS = ("fixed_matched", "aligned_matched")
BANDS = {"B3", "B4", "B5", "B8", "B11"}
WEIGHTS = {"confirmed": 1.0, "reported": 0.5}
CROPS = source_holdout.CROPS


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_shared_parcels(expected, observed):
    def index(rows):
        result = {row["partition_group"]: (row["crop"], row.get("test_cohort")) for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate parcel predictions")
        return result

    if index(expected) != index(observed):
        raise ValueError("Comparison requires identical parcel IDs, reference crops and campaigns")


def fit(model, rows, columns):
    weights = exploratory.normalized_training_weights(rows, WEIGHTS)
    return clone(model).fit(exploratory.feature_matrix(rows, columns),
                            [row["crop"] for row in rows], sample_weight=weights)


def candidate_bins(columns):
    bins = {bin_number: [name for name in columns if name.startswith(f"bin_{bin_number:02d}_")]
            for bin_number in range(1, 9)}
    if len(columns) != 72 or any(len(features) != 9 for features in bins.values()):
        raise ValueError("Expected the frozen eight-bin, 72-predictor recipe")
    return bins


def timing(bin_number, cohort, start_month):
    offset, half = divmod(bin_number - 1, 2)
    days = "days 1–15" if half == 0 else "day 16 through month end"
    return (f"calendar month {start_month + offset:02d}, {days}" if cohort == "fixed_matched"
            else f"aligned window month {offset + 1}, half {half + 1}, relative to selected green-up window")


def spatial_fold_rows(train, folds):
    all_ids = {row["partition_group"] for row in train}
    held_once = [parcel for _, held in folds for parcel in held]
    if len(held_once) != len(all_ids) or set(held_once) != all_ids:
        raise ValueError("Spatial folds must hold out every training parcel exactly once")
    fold_rows = []
    for fit_ids, held_ids in folds:
        if fit_ids != all_ids - held_ids:
            raise ValueError("Spatial fit and held-out parcels must be disjoint")
        fitting = [row for row in train if row["partition_group"] in fit_ids]
        held_out = [row for row in train if row["partition_group"] in held_ids]
        if {row["crop"] for row in fitting} != set(CROPS):
            raise ValueError("A spatial training fold lacks a target crop")
        fold_rows.append((fitting, held_out))

    return fold_rows


def spatial_scores(model, columns, fold_rows):
    predictions = []
    for fitting, held_out in fold_rows:
        model_fold = fit(model, fitting, columns)
        _, held_predictions = source_holdout.parcel_metrics(model_fold, held_out, columns)
        predictions.extend(held_predictions)
    return earth_engine.score_predictions(predictions, [r["prediction"] for r in predictions], CROPS)


def select_bin(model, train, bins, folds):
    fold_rows = spatial_fold_rows(train, folds)
    candidates = {number: spatial_scores(model, columns, fold_rows)["macro_f1"]
                  for number, columns in sorted(bins.items())}
    return max(candidates, key=lambda number: (candidates[number], -number)), candidates


def train_comparators(training_run, output):
    """Select bins and freeze fitted comparators, without opening external data."""
    training_run, output = training_run.resolve(), output.resolve()
    if output.is_relative_to(training_run):
        raise ValueError("Write Batch 2 outputs outside the frozen training run")
    models = training_run / "models"
    model_stage, _ = notebook_workflow._verify(models / "models_stage.json", "models_stage")
    features = training_run / "features_development.json"
    development = read_json(models / "development.json")
    if file_sha256(features) != development["feature_sha256"]:
        raise ValueError("Development features differ from the frozen models' training snapshot")
    data = read_json(features)
    source_holdout.require_matching(data["rows"])

    output.mkdir(parents=True, exist_ok=False)
    index = {}
    for cohort in COHORTS:
        rows = source_holdout.checked_rows(data["rows"][cohort], False)
        train = [row for row in rows if row["partition"] == "train"]
        validation = [row for row in rows if row["partition"] == "validation"]
        manifest_path = models / cohort / "model_manifest.json"
        manifest = read_json(manifest_path)
        split = {row["partition_group"]: row["partition"] for row in rows}
        if (split != manifest["development_split"]
                or {r["partition_group"] for r in train} & {r["partition_group"] for r in validation}):
            raise ValueError(f"{cohort}: development split differs from the frozen model")
        baseline, recipe = load_classifier(manifest_path)
        bins = candidate_bins(recipe["feature_columns"])
        folds = list(geographic_validation.spatial_folds(train))
        selected, candidates = select_bin(baseline, train, bins, folds)
        variants = {
            "single_composite": bins[selected],
            "spectral_bands_only": [name for name in recipe["feature_columns"] if name.split("_")[-1] in BANDS],
        }
        if len(variants["spectral_bands_only"]) != 40:
            raise ValueError("Expected five spectral bands across eight bins")
        print(f"{cohort}: selected bin {selected:02d} (training spatial CV macro-F1 {candidates[selected]:.3f})",
              flush=True)
        index[cohort] = {
            "baseline_manifest_sha256": file_sha256(manifest_path),
            "fold_held_ids": [sorted(held) for _, held in folds],
            "single_composite_candidates": candidates,
            "selection": "highest pooled spatial out-of-fold training macro-F1, then earliest bin",
            "development_split": split,
            "models": {},
        }
        for variant, columns in variants.items():
            model = fit(baseline, train, columns)
            metrics, predictions = source_holdout.parcel_metrics(model, validation, columns)
            folder = output / cohort / variant
            folder.mkdir(parents=True)
            package = folder / "model.joblib"
            joblib.dump(model, package)
            write_csv(folder / "validation_predictions.csv", predictions)
            selected_bin = selected if variant == "single_composite" else None
            recipe_path = folder / "recipe.json"
            save_json(recipe_path, {
                "cohort": cohort, "variant": variant, "feature_columns": columns,
                "composite_bin": selected_bin,
                "interval": timing(selected, cohort, data["fixed_start_month"]) if selected_bin else "all eight bins",
                "window_recipe": recipe["window_recipe"], "quality": recipe["quality"],
                "random_forest": model.get_params(), "label_weights": WEIGHTS,
                "model_sha256": file_sha256(package), "validation": metrics,
            })
            index[cohort]["models"][variant] = {
                "recipe": str(recipe_path.relative_to(output)), "sha256": file_sha256(recipe_path),
            }
    baseline_validation = {}
    for cohort in COHORTS:
        manifest_path = models / cohort / "model_manifest.json"
        baseline, recipe = load_classifier(manifest_path)
        rows = [row for row in data["rows"][cohort] if row["partition"] == "validation"]
        metrics, predictions = source_holdout.parcel_metrics(baseline, rows, recipe["feature_columns"])
        saved = read_json(manifest_path)["validation"]
        if metrics["confusion_matrix"] != saved["confusion_matrix"]:
            raise ValueError(f"{cohort}: frozen validation confusion matrix differs")
        write_csv(output / cohort / "interval_only_validation_predictions.csv", predictions)
        baseline_validation[cohort] = metrics
    receipt = output / "comparators.json"
    save_json(receipt, {
        "training_models_stage_sha256": file_sha256(model_stage), "development": development,
        "runtime": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "cohorts": index, "baseline_validation": baseline_validation,
    })
    return receipt


def freeze_shape_comparison(training_run, batch2_receipt):
    """Fit only shape-only and reuse the verified archived interval-plus-shape forests."""
    original = read_json(batch2_receipt)
    models = training_run / "models"
    stage, _ = notebook_workflow._verify(models / "models_stage.json", "models_stage")
    features = training_run / "features_development.json"
    if (file_sha256(stage) != original["training_models_stage_sha256"]
            or file_sha256(features) != original["development"]["feature_sha256"]):
        raise ValueError("Shape comparison must use the same frozen development inputs as Batch 2")
    data = read_json(features)
    source_holdout.require_matching(data["rows"])
    designs = source_holdout.experiments.feature_sets()
    archive = training_run / "shape_external_comparison"
    archived_models = read_json(archive / "shape_models.json")
    archived_results = archive / "external_comparison.json"
    output = batch2_receipt.parent / "shape_comparison"
    output.mkdir(exist_ok=False)

    index = {}
    for cohort in COHORTS:
        rows = source_holdout.checked_rows(data["rows"][cohort], False)
        train = [row for row in rows if row["partition"] == "train"]
        validation = [row for row in rows if row["partition"] == "validation"]
        split = {row["partition_group"]: row["partition"] for row in rows}
        baseline_path = models / cohort / "model_manifest.json"
        baseline_manifest = read_json(baseline_path)
        cohort_receipt = original["cohorts"][cohort]
        if (file_sha256(baseline_path) != cohort_receipt["baseline_manifest_sha256"]
                or split != baseline_manifest["development_split"]
                or split != cohort_receipt["development_split"]):
            raise ValueError(f"{cohort}: shape comparison development split differs")
        baseline, baseline_recipe = load_classifier(baseline_path)
        archive_manifest = archive / "frozen" / cohort / "model_manifest.json"
        if file_sha256(archive_manifest) != archived_models[cohort]["sha256"]:
            raise ValueError(f"{cohort}: archived interval-plus-shape manifest changed")
        archived_manifest = read_json(archive_manifest)
        combined, combined_recipe = load_classifier(archive_manifest)
        if (archived_manifest["development_split"] != split
                or archived_manifest["feature_handoff"] != baseline_manifest["feature_handoff"]
                or combined.get_params() != baseline.get_params()
                or combined_recipe["feature_columns"] != designs["interval_and_shape"]
                or any(combined_recipe[key] != baseline_recipe[key]
                       for key in ("feature_policy", "window_recipe", "quality"))):
            raise ValueError(f"{cohort}: archived interval-plus-shape recipe differs")
        index[cohort] = {"development_split": split,
                         "baseline_manifest_sha256": file_sha256(baseline_path), "models": {}}
        (output / cohort).mkdir()
        baseline_csv = Path(cohort) / "interval_only_validation_predictions.csv"
        copyfile(batch2_receipt.parent / baseline_csv, output / baseline_csv)
        shape_only = fit(baseline, train, designs["shape_only"])
        for variant, model in (("shape_only", shape_only), ("interval_and_shape", combined)):
            columns = designs[variant]
            metrics, predictions = source_holdout.parcel_metrics(model, validation, columns)
            reused = variant == "interval_and_shape"
            if reused and metrics != archived_manifest["validation"]:
                raise ValueError(f"{cohort}: archived interval-plus-shape validation scores differ")
            folder = output / cohort / variant
            folder.mkdir()
            model_path = folder / "model.joblib"
            joblib.dump(model, model_path)
            write_csv(folder / "validation_predictions.csv", predictions)
            recipe = folder / "recipe.json"
            save_json(recipe, {
                "cohort": cohort, "variant": variant, "feature_columns": columns,
                "window_recipe": baseline_recipe["window_recipe"], "quality": baseline_recipe["quality"],
                "random_forest": model.get_params(), "label_weights": WEIGHTS,
                "model_sha256": file_sha256(model_path), "validation": metrics,
                "archived_manifest_sha256": file_sha256(archive_manifest) if reused else None,
            })
            index[cohort]["models"][variant] = {
                "recipe": str(recipe.relative_to(output)), "sha256": file_sha256(recipe),
            }
    receipt = output / "comparators.json"
    save_json(receipt, {
        **original, "cohorts": index, "source_batch2_receipt_sha256": file_sha256(batch2_receipt),
        "archived_external_comparison": str(archived_results.resolve()),
        "archived_external_comparison_sha256": file_sha256(archived_results),
    })
    return receipt


def scoped_predictions(rows, population):
    return [row for row in rows if retained_reference(row["test_cohort"], row["crop"])
            and (population == "combined" or row["test_cohort"] == population)]


def metric_row(stage, population, cohort, variant, count, metrics):
    crop_f1 = {f"{crop}_f1": metrics["per_crop"][crop]["f1"]
               if metrics["per_crop"][crop]["support"] else None for crop in CROPS}
    return {
        "stage": stage, "population": population, "cohort": cohort, "model": variant,
        "predictors": count, "parcels": sum(metrics["per_crop"][crop]["support"] for crop in CROPS),
        **{f"{crop}_parcels": metrics["per_crop"][crop]["support"] for crop in CROPS},
        **crop_f1, "macro_f1": metrics.get("macro_f1"),
    }


def score_comparators(receipt, testing_run):
    """Verify frozen inputs and score the comparators on the revised Batch 1 population."""
    receipt = receipt.resolve()
    frozen = read_json(receipt)
    score_stage, test_receipt = notebook_workflow._verify(testing_run / "scores.json", "scores")
    if test_receipt["models_handoff_sha256"] != frozen["training_models_stage_sha256"]:
        raise ValueError("External test run uses a different frozen baseline")
    testing_run = score_stage.parent
    features = testing_run / "features_test.json"
    data = read_json(features)
    if any(data[key] != frozen["development"][key]
           for key in ("quality", "window_sha256", "fixed_start_month")):
        raise ValueError("External and development features use different preprocessing snapshots")
    source_holdout.require_matching(data["rows"])
    archive_path = frozen.get("archived_external_comparison")
    archive = Path(archive_path) if archive_path else None
    if archive and (file_sha256(archive) != frozen["archived_external_comparison_sha256"]
                    or read_json(archive)["test_sha256"] != file_sha256(features)):
        raise ValueError("Archived shape evaluation used a different test snapshot")
    tables, predictions = [], []
    all_baseline_rows = {}
    for cohort in COHORTS:
        rows = source_holdout.checked_rows(data["rows"][cohort], True)
        cohort_recipe = frozen["cohorts"][cohort]
        parcel_campaign = {}
        for row in rows:
            group, campaign = row["partition_group"], row["test_cohort"]
            if parcel_campaign.setdefault(group, campaign) != campaign:
                raise ValueError("A test parcel appears in multiple campaigns")
        if set(parcel_campaign) & cohort_recipe["development_split"].keys():
            raise ValueError("External test overlaps the development split")
        sealed_predictions = read_predictions(
            testing_run / "predictions" / "combined" / f"{cohort}_test_parcels.csv"
        )
        baseline_rows = [
            {"partition_group": group, "test_cohort": campaign, "crop": crop, "prediction": prediction}
            for group, (campaign, crop, prediction) in sorted(sealed_predictions.items())
        ]
        reference_rows = {
            row["partition_group"]: {key: row[key] for key in ("partition_group", "crop", "test_cohort")}
            for row in rows
        }
        require_shared_parcels(list(reference_rows.values()), baseline_rows)
        all_baseline_rows[cohort] = baseline_rows
        counts = {"interval_only": 72}
        validation = {"interval_only": frozen["baseline_validation"][cohort]}
        by_variant = {"interval_only": baseline_rows}

        def validation_ids(records):
            expected = {group for group, split in cohort_recipe["development_split"].items()
                        if split == "validation"}
            if len(records) != len(expected) or {row["partition_group"] for row in records} != expected:
                raise ValueError("Saved validation predictions differ from the locked validation IDs")

        baseline_validation_path = receipt.parent / cohort / "interval_only_validation_predictions.csv"
        with baseline_validation_path.open(newline="", encoding="utf-8") as stream:
            baseline_validation_rows = list(DictReader(stream))
        validation_ids(baseline_validation_rows)
        for variant, artifact in cohort_recipe["models"].items():
            recipe_path = receipt.parent / artifact["recipe"]
            if file_sha256(recipe_path) != artifact["sha256"]:
                raise ValueError(f"Comparator recipe changed: {recipe_path}")
            recipe = read_json(recipe_path)
            model_path = recipe_path.with_name("model.joblib")
            if file_sha256(model_path) != recipe["model_sha256"]:
                raise ValueError(f"Frozen comparator changed: {model_path}")
            model = joblib.load(model_path)
            columns = recipe["feature_columns"]
            _, predicted = source_holdout.parcel_metrics(model, rows, columns)
            for row in predicted:
                row["test_cohort"] = parcel_campaign[row["partition_group"]]
            require_shared_parcels(baseline_rows, predicted)
            if archive and variant == "interval_and_shape":
                archived = read_predictions(archive.parent / "combined" / "shape" / f"{cohort}_test_parcels.csv")
                archive_scores = read_json(archive)["scopes"]["combined"]["shape"][cohort]
                all_scores = earth_engine.score_predictions(predicted, [p["prediction"] for p in predicted], CROPS)
                if (archived != {p["partition_group"]: (p["test_cohort"], p["crop"], p["prediction"])
                                 for p in predicted}
                        or all_scores["confusion_matrix"] != archive_scores["confusion_matrix"]):
                    raise ValueError(f"{cohort}: archived interval-plus-shape predictions differ")
            with recipe_path.with_name("validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
                require_shared_parcels(baseline_validation_rows, list(DictReader(stream)))
            by_variant[variant] = predicted
            validation[variant] = recipe["validation"]
            counts[variant] = len(columns)
        for variant, metrics in validation.items():
            tables.append(metric_row("locked_validation", "development", cohort, variant, counts[variant], metrics))
        for population in REVISED_POPULATION_NAMES:
            for variant, records in by_variant.items():
                retained = scoped_predictions(records, population)
                metrics = earth_engine.score_predictions(retained, [row["prediction"] for row in retained], CROPS)
                if {row["crop"] for row in retained} != set(CROPS):
                    metrics.pop("macro_f1", None)
                tables.append(metric_row("external_test", population, cohort, variant, counts[variant], metrics))
                if population == "combined":
                    predictions.extend({"cohort": cohort, "model": variant,
                                        **{key: row[key] for key in baseline_rows[0]}}
                                       for row in retained)
    require_shared_parcels(*all_baseline_rows.values())
    output = receipt.parent
    write_csv(output / "comparison_metrics.csv", tables)
    write_csv(output / "external_predictions.csv", predictions)
    save_json(output / "external_evaluation.json", {
        "scores_stage_sha256": file_sha256(score_stage), "test_features_sha256": file_sha256(features),
        "comparators_sha256": file_sha256(receipt), "reported_populations": list(REVISED_POPULATION_NAMES),
        "shared_test_ids_and_labels_verified": True, "shared_validation_ids_and_labels_verified": True,
        "outputs": {name: file_sha256(output / name)
                    for name in ("comparison_metrics.csv", "external_predictions.csv")},
    })
    return tables


def render_report(receipt, tables, training_run):
    """Present the new comparisons and the saved development shape experiment."""
    frozen = read_json(receipt)
    lines = [
        "# Batch 2 model comparisons",
        "",
        "The matched comparisons use the same training weights, Random Forest settings, parcels, and pixel "
        "locations within each recipe. The 72-predictor frozen interval model is the reference. The "
        "single-composite model uses nine predictors from one half-month median composite. A composite can "
        "contain multiple acquisitions. The aligned recipe uses multi-date imagery to locate green-up. "
        "The bands-only model uses five spectral bands across eight bins (40 predictors).",
        "",
        "Bin selection used five spatial folds inside the training partition, pooled out-of-fold parcel "
        "macro-F1 and an earlier-bin tie-break. Locked validation was scored after selection.",
        "",
        "| Cohort | Selected bin | Timing | Training spatial CV macro-F1 |",
        "| --- | ---: | --- | ---: |",
    ]
    for cohort, entry in frozen["cohorts"].items():
        recipe = read_json(receipt.parent / entry["models"]["single_composite"]["recipe"])
        selected = recipe["composite_bin"]
        cv = entry["single_composite_candidates"][str(selected)]
        lines.append(f"| {cohort} | {selected} | {recipe['interval']} | {cv:.3f} |")
    lines += [
        "",
        "External scores describe an already examined benchmark and use the revised Batch 1 populations. "
        "JECAM is primary. The combined test includes JECAM and the three supplements. "
        "A dash indicates unavailable F1 for a missing reference crop or unavailable three-class macro-F1.",
    ]

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines.extend(comparison_sections(tables))

    prior = read_json(training_run / "matched_geographic_feature_comparison.json")
    if prior["feature_sha256"] != frozen["development"]["feature_sha256"]:
        raise ValueError("Saved shape comparison used a different development snapshot")
    lines += [
        "", "## Prior matched development feature comparison", "",
        "These are saved spatial out-of-fold results within development training.",
        "",
        "| Cohort | Feature set | Predictors | Carrot F1 | Cabbage F1 | Potato F1 | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in COHORTS:
        for variant, stats in prior["cohorts"][cohort].items():
            pooled = stats["pooled"]
            values = [cohort, variant, str(stats["predictors"]),
                      *(number(pooled["per_crop"][crop]["f1"]) for crop in CROPS),
                      number(pooled["macro_f1"])]
            lines.append("| " + " | ".join(values) + " |")
    report = receipt.parent / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def comparison_sections(tables):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines = []
    populations = {"development": "Locked development validation", **REVISED_POPULATION_NAMES}
    for population, label in populations.items():
        lines += [
            "", f"## {label}", "",
            "| Cohort | Model | Predictors | Parcels | Carrot F1 | Cabbage F1 | Potato F1 | Macro-F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in tables:
            if row["population"] == population:
                values = [row["cohort"], row["model"], str(row["predictors"]), str(row["parcels"]),
                          *(number(row[f"{crop}_f1"]) for crop in CROPS), number(row["macro_f1"])]
                lines.append("| " + " | ".join(values) + " |")

    return lines


def render_shape_report(receipt, tables):
    """Present the three feature-set comparisons on identical matched parcels."""
    lines = [
        "# Batch 2 three-way feature comparison",
        "",
        "The three models use 72 interval predictors, 48 temporal-shape predictors, or all 120. "
        "The interval-only and interval-plus-shape forests were already frozen. The two shape-only forests "
        "were fit on the locked development training parcels with the same pixel locations, label weights "
        "and Random Forest settings within each window recipe. The archived interval-plus-shape predictions "
        "reproduced on the saved test snapshot.",
        "",
        "External results use the revised Batch 1 populations and the same matched parcel IDs across "
        "feature sets and window recipes. This benchmark has already been examined. F1 is unavailable "
        "for absent reference crops and three-class macro-F1 requires all three reference crops.",
        *comparison_sections(tables),
    ]
    report = receipt.parent / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
