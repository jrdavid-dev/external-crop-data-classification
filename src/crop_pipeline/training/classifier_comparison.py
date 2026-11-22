"""Compare alternative classifiers with frozen matched Random Forests."""

import argparse
from importlib.metadata import version
from csv import DictReader
from pathlib import Path
from shutil import copyfile, copytree

import joblib
import numpy as np
import sklearn
from sklearn import config_context
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from crop_pipeline.common import save_json
from crop_pipeline.training import earth_engine, exploratory, geographic_validation
from crop_pipeline.training import model_comparisons as comparisons
from crop_pipeline.training import notebook_workflow, source_holdout
from crop_pipeline.training.external_report import DEFAULT_REPETITIONS, DEFAULT_SEED, REVISED_POPULATION_NAMES, write_csv
from crop_pipeline.training.model_followups import paired_macro_difference
from crop_pipeline.windows.seasons import file_sha256

CROPS = source_holdout.CROPS
CANDIDATES = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "rbf_svm": [{"C": c, "gamma": multiplier / 72}
                for c in (0.1, 1.0, 10.0) for multiplier in (0.3, 1.0, 3.0)],
}


XGBOOST_CANDIDATES = [
    {"max_depth": depth, "learning_rate": rate, "n_estimators": trees}
    for depth in (2, 4, 6)
    for rate, trees in ((0.05, 150), (0.05, 300), (0.1, 150))
]


class CropXGBoost:
    """Present XGBoost's integer classes as the pipeline's crop names."""

    def __init__(self, params):
        from xgboost import XGBClassifier

        self.encoder = LabelEncoder()
        self.estimator = XGBClassifier(
            objective="multi:softprob", tree_method="hist", random_state=42, n_jobs=4, **params
        )

    def fit(self, values, labels, sample_weight):
        encoded = self.encoder.fit_transform(labels)
        self.classes_ = self.encoder.classes_
        self.estimator.fit(values, encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, values):
        return self.estimator.predict_proba(values)


def parcel_calibration_folds(rows):
    """Stratify unique parcels, then map their split back to pixel indices."""
    labels = {}
    for row in rows:
        if labels.setdefault(row["partition_group"], row["crop"]) != row["crop"]:
            raise ValueError("Conflicting reference crops in calibration parcel")
    parcel_ids = sorted(labels)
    classes = [labels[parcel] for parcel in parcel_ids]
    if any(classes.count(crop) < 3 for crop in CROPS):
        raise ValueError("Need at least three training parcels of each crop for calibration")
    groups = np.asarray([row["partition_group"] for row in rows])
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = []
    for _, held in splitter.split(parcel_ids, classes):
        held_parcel_ids = [parcel_ids[i] for i in held]
        held_rows = np.isin(groups, held_parcel_ids)
        folds.append((np.flatnonzero(~held_rows), np.flatnonzero(held_rows)))
    return folds


def fit_classifier(kind, params, rows, columns, *, sample_weight=None):
    values = exploratory.feature_matrix(rows, columns)
    labels = [row["crop"] for row in rows]
    weights = (exploratory.normalized_training_weights(rows, comparisons.WEIGHTS)
               if sample_weight is None else sample_weight)
    if kind == "xgboost":
        return CropXGBoost(params).fit(values, labels, sample_weight=weights)
    # Route weights to the classifier and calibration, using an unweighted standard scaler.
    with config_context(enable_metadata_routing=True):
        scale = StandardScaler().set_fit_request(sample_weight=False)
        if kind == "logistic_regression":
            estimator = LogisticRegression(max_iter=1000, **params).set_fit_request(sample_weight=True)
            model = make_pipeline(scale, estimator)
        elif kind == "rbf_svm":
            estimator = SVC(kernel="rbf", **params).set_fit_request(sample_weight=True)
            model = CalibratedClassifierCV(make_pipeline(scale, estimator), method="sigmoid",
                                           cv=parcel_calibration_folds(rows), ensemble=False)
        else:
            raise ValueError(f"Unknown classifier {kind}")
        return model.fit(values, labels, sample_weight=weights)


def candidate_scores(kind, columns, fold_rows):
    results = []
    grid = XGBOOST_CANDIDATES if kind == "xgboost" else CANDIDATES[kind]
    for params in grid:
        if kind == "rbf_svm" and len(columns) != 72:
            params = {**params, "gamma": params["gamma"] * 72 / len(columns)}
        predictions = []
        for fit, held in fold_rows:
            model = fit_classifier(kind, params, fit, columns)
            _, scored = source_holdout.parcel_metrics(model, held, columns)
            predictions.extend(scored)
        metrics = earth_engine.score_predictions(predictions, [row["prediction"] for row in predictions], CROPS)
        results.append({"settings": params, "pooled_training_cv_scores": metrics})
    return results


def freeze_challengers(training, batch2_receipt, output, prior_receipt=None, shape_receipt=None):
    """Tune with development training only, score locked validation, and save fitted models."""
    training, output = training.resolve(), output.resolve()
    frozen_roots = [training, batch2_receipt.parent.resolve()]
    if prior_receipt is not None:
        frozen_roots.append(prior_receipt.parent.resolve())
    if shape_receipt is not None:
        frozen_roots.append(shape_receipt.parent.resolve())
    if any(output.is_relative_to(root) for root in frozen_roots):
        raise ValueError("Write the challenger models outside the existing frozen runs")
    stage, _ = notebook_workflow._verify(training / "models/models_stage.json", "models_stage")
    original = comparisons.read_json(batch2_receipt)
    features = training / "features_development.json"
    if (file_sha256(stage) != original["training_models_stage_sha256"]
            or file_sha256(features) != original["development"]["feature_sha256"]):
        raise ValueError("Challengers require the original frozen development features")
    prior = None
    if prior_receipt is not None:
        import xgboost

        prior = comparisons.read_json(prior_receipt)
        if prior["source_batch2_receipt_sha256"] != file_sha256(batch2_receipt):
            raise ValueError("Prior challengers use different development inputs")
    if shape_receipt is not None:
        shape = comparisons.read_json(shape_receipt)
        if prior is None or shape["source_batch2_receipt_sha256"] != file_sha256(batch2_receipt):
            raise ValueError("Shape comparison requires the corresponding frozen 72- and 120-feature runs")
    snapshot = comparisons.read_json(features)
    source_holdout.require_matching(snapshot["rows"])
    output.mkdir(parents=True, exist_ok=False)
    cohorts = {}
    for cohort in comparisons.COHORTS:
        rows = source_holdout.checked_rows(snapshot["rows"][cohort], False)
        train = [row for row in rows if row["partition"] == "train"]
        validation = [row for row in rows if row["partition"] == "validation"]
        split = {row["partition_group"]: row["partition"] for row in rows}
        previous = original["cohorts"][cohort]
        path = training / "models" / cohort / "model_manifest.json"
        if (file_sha256(path) != previous["baseline_manifest_sha256"] or split != previous["development_split"]
                or {r["partition_group"] for r in train} & {r["partition_group"] for r in validation}):
            raise ValueError("Frozen baseline or parcel split differs")
        forest_manifest = comparisons.read_json(path)
        columns = forest_manifest["identity"]["recipe"]["feature_columns"]
        comparisons.candidate_bins(columns)
        folds = list(geographic_validation.spatial_folds(train))
        held_ids = [sorted(held) for _, held in folds]
        if held_ids != previous["fold_held_ids"]:
            raise ValueError("Development geographic folds differ from Batch 2")
        fold_rows = comparisons.spatial_fold_rows(train, folds)
        cohorts[cohort] = {"development_split": split, "fold_held_ids": held_ids, "models": {}}
        baseline_csv = Path(cohort) / "interval_only_validation_predictions.csv"
        (output / cohort).mkdir()
        copyfile(batch2_receipt.parent / baseline_csv, output / baseline_csv)
        saved = []
        if prior is not None:
            earlier = prior["cohorts"][cohort]
            if earlier["development_split"] != split or earlier["fold_held_ids"] != held_ids:
                raise ValueError("Prior challengers use different development splits")
            variants = ("rbf_svm", "xgboost") if shape_receipt else tuple(CANDIDATES)
            saved.extend((prior_receipt, kind, earlier["models"][kind], columns) for kind in variants)
        kinds = ("xgboost",) if prior is not None else CANDIDATES
        if shape_receipt is not None:
            reference = shape["cohorts"][cohort]
            if reference["development_split"] != split:
                raise ValueError("Shape forest uses a different development split")
            combined = source_holdout.experiments.feature_sets()["interval_and_shape"]
            if combined[:72] != columns or len(combined) != 120:
                raise ValueError("Expected the 72 frozen inputs followed by 48 shape descriptors")
            saved.append((shape_receipt, "interval_and_shape", reference["models"]["interval_and_shape"], combined))
            columns = combined
            kinds = ("xgboost", "rbf_svm")
        for origin, kind, artifact, expected_columns in saved:
            recipe = origin.parent / artifact["recipe"]
            if (file_sha256(recipe) != artifact["sha256"]
                    or comparisons.read_json(recipe)["feature_columns"] != expected_columns):
                raise ValueError(f"Unexpected frozen inputs for {kind}")
            copytree(recipe.parent, output / Path(artifact["recipe"]).parent)
            cohorts[cohort]["models"][kind] = artifact
        for kind in kinds:
            variant = f"{kind}_shape" if shape_receipt else kind
            scores = candidate_scores(kind, columns, fold_rows)
            selected = max(scores, key=lambda row: row["pooled_training_cv_scores"]["macro_f1"])
            model = fit_classifier(kind, selected["settings"], train, columns)
            metrics, predictions = source_holdout.parcel_metrics(model, validation, columns)
            folder = output / cohort / variant
            folder.mkdir()
            package = folder / "model.joblib"
            joblib.dump(model, package)
            write_csv(folder / "validation_predictions.csv", predictions)
            recipe_path = folder / "recipe.json"
            save_json(recipe_path, {
                "classifier": kind, "feature_columns": columns,
                "window_recipe": forest_manifest["identity"]["recipe"]["window_recipe"],
                "selected_settings": selected["settings"], "development_candidates": scores,
                "model_sha256": file_sha256(package), "validation": metrics,
            })
            cohorts[cohort]["models"][variant] = {
                "recipe": str(recipe_path.relative_to(output)), "sha256": file_sha256(recipe_path),
            }
            print(f"{cohort} {variant}: {selected['settings']} · "
                  f"training spatial CV {selected['pooled_training_cv_scores']['macro_f1']:.3f} · "
                  f"locked validation {metrics['macro_f1']:.3f}", flush=True)
    receipt = output / "comparators.json"
    save_json(receipt, {
        "training_models_stage_sha256": file_sha256(stage), "development": original["development"],
        "source_batch2_receipt_sha256": file_sha256(batch2_receipt),
        "baseline_validation": original["baseline_validation"], "cohorts": cohorts,
        "runtime": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "selection": "highest pooled spatial out-of-fold training parcel macro-F1; ties keep candidate order",
        "standardization": "fit per model training split only",
        "svm_calibration": "sigmoid fitted to out-of-fold decision scores from three stratified parcel folds "
                           "within the current fitting set, seed 42",
        "label_weights": comparisons.WEIGHTS,
        **({"prior_comparison_receipt_sha256": file_sha256(prior_receipt),
            "xgboost_version": version("xgboost")} if prior is not None else {}),
        **({key: shape[key] for key in ("archived_external_comparison", "archived_external_comparison_sha256")}
           if shape_receipt else {}),
        **({"shape_receipt_sha256": file_sha256(shape_receipt)} if shape_receipt else {}),
    })
    return receipt


def paired_differences(output, baseline_model="interval_only", challengers=(*CANDIDATES, "xgboost"),
                       populations=REVISED_POPULATION_NAMES):
    with (output / "external_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(DictReader(stream))
    rng = np.random.default_rng(DEFAULT_SEED)
    results = []
    for scope in populations:
        for cohort in comparisons.COHORTS:
            records = [row for row in rows if row["cohort"] == cohort]
            baseline = comparisons.scoped_predictions([r for r in records if r["model"] == baseline_model], scope)
            for kind in challengers:
                if not any(r["model"] == kind for r in records):
                    continue
                candidate = comparisons.scoped_predictions([r for r in records if r["model"] == kind], scope)
                results.append({
                    "population": scope, "cohort": cohort, "classifier": kind, "parcels": len(candidate),
                    **paired_macro_difference(baseline, candidate, DEFAULT_REPETITIONS, rng),
                })
    return results


def compare_external(receipt, testing):
    """Score frozen alternatives and report paired differences on the original RF parcels."""
    output = receipt.parent.resolve()
    if output.is_relative_to(testing.resolve()):
        raise ValueError("Write challenger results outside the original testing run")
    rows = comparisons.score_comparators(receipt, testing)
    frozen = comparisons.read_json(receipt)
    has_shape = "shape_receipt_sha256" in frozen
    if has_shape:
        variants = ("interval_and_shape", "rbf_svm", "rbf_svm_shape", "xgboost", "xgboost_shape")
        paired = paired_differences(output, challengers=variants)
    else:
        paired = paired_differences(output)
    write_csv(output / "paired_macro_f1.csv", paired)
    feature_description = (
        "The 120-feature recipes append 48 vegetation-index shape descriptors to the 72 interval inputs. "
        if has_shape else "The Random Forest and challengers use the same 72 interval inputs. "
    )
    lines = [
        "# Matched classifier comparison", "",
        feature_description + "All runs share the same matched parcel and pixel locations and "
        "crop- and parcel-balanced training weights. Predictions average pixel "
        "class probabilities within parcels. The SVM uses sigmoid calibration fit using only the current "
        "training set, with three stratified folds that keep each parcel together. Standardization is "
        "fit inside each training or calibration split.",
        "",
        "Five geographic folds inside locked development training select settings by pooled out-of-fold "
        "parcel macro-F1. Locked validation checks the chosen settings. External testing uses frozen "
        "models on an already examined benchmark with the revised Batch 1 scope.",
        *(
            ["", "XGBoost uses histogram splits and multiclass probabilities with seed 42. "
             "The development grid varies tree depth, learning rate and tree count. "
             + ("The SVM gamma grid is scaled by predictor count. Both forests and the 72-feature "
                "XGBoost and SVM fits are reused." if has_shape else
                "The prior logistic regression and RBF SVM fits are reused.")]
            if "xgboost_version" in frozen else []
        ),
        "", "| Window | Classifier | Selected settings | Training spatial CV macro-F1 |",
        "| --- | --- | --- | ---: |",
    ]
    for cohort, group in frozen["cohorts"].items():
        for kind, artifact in group["models"].items():
            recipe = comparisons.read_json(output / artifact["recipe"])
            if "selected_settings" not in recipe:
                continue
            settings = recipe["selected_settings"]
            score = next(row["pooled_training_cv_scores"]["macro_f1"]
                         for row in recipe["development_candidates"] if row["settings"] == settings)
            options = ", ".join(f"{name}={value:.4g}" for name, value in settings.items())
            lines.append(f"| {cohort} | {kind} | {options} | {score:.3f} |")
    pretty = {
        "interval_only": "Random Forest", "logistic_regression": "Logistic regression",
        "rbf_svm": "RBF SVM", "xgboost": "XGBoost", "interval_and_shape": "Random Forest + shape",
        "rbf_svm_shape": "RBF SVM + shape", "xgboost_shape": "XGBoost + shape",
    }
    lines.extend(comparisons.comparison_sections([{**row, "model": pretty[row["model"]]} for row in rows]))
    lines += [
        "", "## Paired macro-F1 differences from Random Forest", "",
        f"Intervals use {DEFAULT_REPETITIONS:,} shared parcel resamples, seed {DEFAULT_SEED}, stratified by "
        "campaign and reference crop. Combined intervals condition on those observed counts. Three-class "
        "macro-F1 is unavailable for the single-crop supplements.",
        "", "| Population | Window | Challenger | Parcels | Difference [95% CI] |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for entry in paired:
        estimate = (f"{entry['estimate']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"
                    if entry["estimate"] is not None else "—")
        lines.append(f"| {REVISED_POPULATION_NAMES[entry['population']]} | {entry['cohort']} | "
                     f"{pretty[entry['classifier']]} | {entry['parcels']} | {estimate} |")
    if has_shape:
        scopes = ("jecam", "combined")
        contrasts = [
            ("interval_only", ("interval_and_shape",)),
            ("rbf_svm", ("rbf_svm_shape",)),
            ("xgboost", ("xgboost_shape",)),
            ("interval_and_shape", ("rbf_svm_shape", "xgboost_shape")),
        ]
        results = [
            {**entry, "baseline_classifier": baseline}
            for baseline, challengers in contrasts
            for entry in paired_differences(output, baseline, challengers, scopes)
        ]
        write_csv(output / "shape_contrasts.csv", results)
        lines += [
            "", "## Paired effects of shape and comparison with the 120-feature forest", "",
            "| Population | Window | Challenger | Reference | Parcels | Difference [95% CI] |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
        for entry in results:
            lines.append(
                f"| {REVISED_POPULATION_NAMES[entry['population']]} | {entry['cohort']} | "
                f"{pretty[entry['classifier']]} | {pretty[entry['baseline_classifier']]} | "
                f"{entry['parcels']} | {entry['estimate']:+.3f} "
                f"[{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}] |"
            )
    elif "xgboost_version" in frozen:
        head_to_head = paired_differences(output, "rbf_svm", ("xgboost",), ("jecam", "combined"))
        write_csv(output / "xgboost_vs_svm.csv", head_to_head)
        lines += [
            "", "## Paired XGBoost minus RBF SVM macro-F1", "",
            "| Population | Window | Parcels | Difference [95% CI] |",
            "| --- | --- | ---: | --- |",
        ]
        for entry in head_to_head:
            lines.append(
                f"| {REVISED_POPULATION_NAMES[entry['population']]} | {entry['cohort']} | "
                f"{entry['parcels']} | {entry['estimate']:+.3f} "
                f"[{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}] |"
            )
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(output / "comparison_methods.json", {
        "external_evaluation_sha256": file_sha256(output / "external_evaluation.json"),
        "paired_macro_f1_sha256": file_sha256(output / "paired_macro_f1.csv"),
        **({"shape_contrasts_sha256": file_sha256(output / "shape_contrasts.csv")} if has_shape else {}),
        "report_sha256": file_sha256(report), "bootstrap_repetitions": DEFAULT_REPETITIONS,
        "bootstrap_seed": DEFAULT_SEED, "bootstrap_method": "paired parcel percentile, stratified by campaign and crop",
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "testing", "batch2_receipt", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--add-xgboost-to", type=Path, metavar="PRIOR_RECEIPT",
                        help="fit XGBoost and reuse challengers from an existing comparators.json")
    parser.add_argument("--shape-forest", type=Path, metavar="SHAPE_RECEIPT",
                        help="fit 120-feature XGBoost and SVM alongside frozen comparisons")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.testing.resolve()):
        parser.error("Write the challenger run outside the original testing run")
    receipt = freeze_challengers(args.training, args.batch2_receipt, args.output, args.add_xgboost_to, args.shape_forest)
    print(compare_external(receipt, args.testing))


if __name__ == "__main__":
    # Serialize the model using its importable module name.
    from crop_pipeline.training.classifier_comparison import main as entry_point

    entry_point()
