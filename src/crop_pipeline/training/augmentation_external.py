"""Score the development-selected temporal augmentation on saved matched parcels."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.training import earth_engine, notebook_workflow, source_holdout
from crop_pipeline.training import model_comparisons as comparisons
from crop_pipeline.training import transfer_preprocessing as transfer
from crop_pipeline.training.external_report import DEFAULT_REPETITIONS, DEFAULT_SEED, read_predictions, write_csv
from crop_pipeline.training.model_followups import paired_macro_difference
from crop_pipeline.windows.seasons import file_sha256


def run(training, testing, development, output):
    if any(output.resolve().is_relative_to(root.resolve()) for root in (training, testing, development)):
        raise ValueError("Write results outside the frozen inputs")
    prior = comparisons.read_json(development / "run.json")
    features = training / "features_development.json"
    archive = training / "shape_external_comparison"
    manifest = archive / "frozen" / transfer.COHORT / "model_manifest.json"
    if (file_sha256(features) != prior["development_feature_sha256"]
            or file_sha256(manifest) != prior["baseline_manifest_sha256"]):
        raise ValueError("Development data or baseline differs from the augmentation experiment")
    forest, recipe = comparisons.load_classifier(manifest)
    if recipe["feature_columns"] != prior["feature_columns"] or recipe["feature_columns"] != transfer.COLUMNS:
        raise ValueError("Expected identical 120-feature recipes")
    snapshot = comparisons.read_json(features)
    policy = {key: snapshot[key] for key in ("quality", "window_sha256", "fixed_start_month")}
    rows = source_holdout.checked_rows(snapshot["rows"][transfer.COHORT], False)
    del snapshot
    split = {row["partition_group"]: row["partition"] for row in rows}
    if split != comparisons.read_json(manifest)["development_split"]:
        raise ValueError("Unexpected development partition")
    fitted = transfer.fit_augmented(forest, [row for row in rows if row["partition"] == "train"])
    del rows
    output.mkdir(parents=True, exist_ok=False)
    model = output / "augmented_forest.joblib"
    joblib.dump(fitted, model)
    save_json(output / "frozen_model.json", {
        "model_sha256": file_sha256(model), "development_receipt_sha256": file_sha256(development / "run.json"),
        "baseline_manifest_sha256": file_sha256(manifest), "feature_columns": transfer.COLUMNS,
        "augmentation": {"training_bin_shifts": [-1, 0, 1], "edge": "replicate"},
    })

    stage, _ = notebook_workflow._verify(testing / "scores.json", "scores")
    test_features = stage.parent / "features_test.json"
    archive_receipt = comparisons.read_json(archive / "external_comparison.json")
    test_digest = file_sha256(test_features)
    if test_digest != archive_receipt["test_sha256"]:
        raise ValueError("External snapshot differs from archived 120-feature baseline")
    snapshot = comparisons.read_json(test_features)
    source_holdout.require_matching(snapshot["rows"])
    if any(snapshot[key] != value for key, value in policy.items()):
        raise ValueError("External and development policies differ")
    held = source_holdout.checked_rows(snapshot["rows"][transfer.COHORT], True)
    campaigns = {row["partition_group"]: row["test_cohort"] for row in held}
    if campaigns.keys() & split.keys():
        raise ValueError("External parcel overlaps development")
    _, predictions = source_holdout.parcel_metrics(fitted, held, transfer.COLUMNS)
    for row in predictions:
        row["test_cohort"] = campaigns[row["partition_group"]]
    original = read_predictions(
        archive / "combined/shape" / f"{transfer.COHORT}_test_parcels.csv"
    )
    baseline = [{"partition_group": parcel, "test_cohort": campaign, "crop": crop, "prediction": p}
                for parcel, (campaign, crop, p) in original.items()]
    comparisons.require_shared_parcels(baseline, predictions)
    saved_metrics = earth_engine.score_predictions(baseline, [r["prediction"] for r in baseline], comparisons.CROPS)
    if saved_metrics["confusion_matrix"] != archive_receipt["scopes"]["combined"]["shape"][transfer.COHORT]["confusion_matrix"]:
        raise ValueError("Archived baseline predictions differ from the sealed score")

    scores, differences = [], []
    for population in ("jecam", "combined"):
        pair = [comparisons.scoped_predictions(records, population) for records in (baseline, predictions)]
        differences.append({"population": population, "parcels": len(pair[0]), **paired_macro_difference(
            *pair, DEFAULT_REPETITIONS, np.random.default_rng(DEFAULT_SEED)
        )})
        for kind, records in zip(("original", "augmented"), pair):
            metrics = earth_engine.score_predictions(records, [r["prediction"] for r in records], comparisons.CROPS)
            scores.append(comparisons.metric_row("external_test", population, transfer.COHORT, kind, 120, metrics))
    write_csv(output / "metrics.csv", scores)
    write_csv(output / "paired_macro_f1.csv", differences)
    write_csv(output / "predictions.csv", [
        {**{key: row[key] for key in ("partition_group", "test_cohort", "crop", "prediction")}, "model": kind}
        for kind, records in (("original", baseline), ("augmented", predictions))
        for row in comparisons.scoped_predictions(records, "combined")
    ])
    save_json(output / "evaluation.json", {"test_feature_sha256": test_digest, "bootstrap_seed": DEFAULT_SEED,
                                         "bootstrap_repetitions": DEFAULT_REPETITIONS})
    lines = [
        "# Temporal augmentation on matched external parcels", "",
        "Both fixed-window Random Forests use 120 features. The augmentation recipe shifts the training "
        "curves one bin earlier and later, repeats edge values, and recomputes the 48 shape features. "
        "The augmented forest was frozen before loading external feature rows. The original predictions "
        "come from the archived shape comparison. Both models use the same external parcel IDs.", "",
        "| Population | Model | Parcels | Carrot F1 | Cabbage F1 | Potato F1 | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores:
        f1 = " | ".join(f"{row[key]:.3f}" for key in ("carrot_f1", "cabbage_f1", "potato_f1", "macro_f1"))
        lines.append(f"| {row['population']} | {row['model']} | {row['parcels']} | {f1} |")
    lines += ["", "Paired 95% percentile intervals use 10,000 shared parcel draws within campaign and crop "
              "with seed 42. Combined intervals condition on the observed campaign and crop counts.", ""]
    for row in differences:
        lines.append(f"- {row['population']} macro-F1 change: {row['estimate']:+.3f} "
                     f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "testing", "development", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(run(args.training, args.testing, args.development, args.output))


if __name__ == "__main__":
    main()
