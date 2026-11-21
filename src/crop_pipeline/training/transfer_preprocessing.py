"""Evaluate temporal input changes on locked geographic development folds."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.training import earth_engine, geographic_validation, source_holdout
from crop_pipeline.training import model_comparisons as comparisons
from crop_pipeline.training.external_report import write_csv
from crop_pipeline.windows import seasons

COHORT = "fixed_matched"
COLUMNS = experiments.feature_sets()["interval_and_shape"]
CHANNELS = seasons.FEATURE_BANDS + experiments.INDEX_NAMES
SEED = 42
VARIANTS = (
    "baseline", "temporal_augmentation", "temporal_centering",
    "gap_nearest", "gap_interpolation",
)


def gap_bin(parcel):
    """Deterministically obscure the same interior bin for every pixel of a field."""
    digest = sha256(f"{SEED}:{parcel}".encode()).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 6


def transform(row, method, shift=0, positions=None):
    """Change one pixel's time series and regenerate the dependent shape features."""
    positions = experiments.calendar_positions(row) if positions is None else positions
    values = np.asarray([row[column] for column in COLUMNS[:72]], dtype=float).reshape(8, len(CHANNELS))
    if method == "shift":
        values = values[np.clip(np.arange(8) - shift, 0, 7)]
    elif method == "temporal_centering":
        values -= np.median(values, axis=0)
    elif method in ("gap_nearest", "gap_interpolation"):
        gap = gap_bin(row["partition_group"])
        left, right = gap - 1, gap + 1
        # Only use the intact neighboring bins in either reconstruction.
        fraction = (positions[gap] - positions[left]) / (positions[right] - positions[left])
        if method == "gap_nearest":
            values[gap] = values[left if fraction <= .5 else right]
        else:
            values[gap] = (1 - fraction) * values[left] + fraction * values[right]
    else:
        raise ValueError(f"Unknown temporal transform: {method}")
    result = {**row, **dict(zip(COLUMNS[:72], values.ravel()))}
    for index in experiments.INDEX_NAMES:
        result.update(experiments.temporal_shape(values[:, CHANNELS.index(index)], positions, index))
    return result


def fit_augmented(model, rows):
    copies = [transform(row, "shift", shift) for shift in (-1, 1) for row in rows]
    return comparisons.fit(model, rows + copies, COLUMNS)


def predict_split(model, fit, held):
    """Hold each simulated field gap identical between nearest and linear repair."""
    baseline = comparisons.fit(model, fit, COLUMNS)
    augmented = fit_augmented(model, fit)
    centered = comparisons.fit(model, [transform(row, "temporal_centering") for row in fit], COLUMNS)
    configurations = (
        (baseline, held),
        (augmented, held),
        (centered, [transform(row, "temporal_centering") for row in held]),
        (baseline, [transform(row, "gap_nearest") for row in held]),
        (baseline, [transform(row, "gap_interpolation") for row in held]),
    )
    result = {}
    for variant, (estimator, rows) in zip(VARIANTS, configurations):
        _, result[variant] = source_holdout.parcel_metrics(estimator, rows, COLUMNS)
        comparisons.require_shared_parcels(result["baseline"], result[variant])
    return result


def score_rows(predictions, stage, fold):
    scores = []
    for model, records in predictions.items():
        metrics = earth_engine.score_predictions(records, [r["prediction"] for r in records], comparisons.CROPS)
        if {r["crop"] for r in records} != set(comparisons.CROPS):
            metrics.pop("macro_f1", None)
        scores.append({"fold": fold, **comparisons.metric_row(stage, "development", COHORT, model, 120, metrics)})
    return scores


def report(output, scores):
    lines = [
        "# Fixed Random Forest geographic transfer experiments", "",
        "All variants use 120 predictors and the saved fixed-matched development snapshot. "
        "Each spatial fold holds out entire one-degree site groups, joining cells linked by a parcel. "
        "The secondary locked development validation uses the original parcel split. "
        "External test data play no role in these experiments.", "",
        "- Temporal augmentation adds training-only copies shifted one bin earlier and later with edge replication. "
        "The original crop and parcel weight shares are preserved. The forest keeps its bootstrap fraction, "
        "so it draws more rows per tree from the expanded training set.",
        "- Temporal normalization subtracts each pixel's eight-bin median separately for each band or index. "
        "This reduces additive level differences across fields and also removes absolute level information. "
        "Each pixel uses only its own window.",
        "- The gap experiment hides one deterministic interior bin per field. Linear interpolation and "
        "nearest-bin fill use the intact adjacent bins, with actual bin midpoint dates and earlier-bin tie-breaking. "
        "Both use the same forest fitted on complete windows. The intact baseline gives a reference for gap loss.",
        "- Every transform regenerates all 48 shape predictors. "
        "The saved features contain only complete windows, so this interpolation test measures simulated "
        "single-gap robustness on previously eligible fields. Real cloudy gaps, edge gaps, consecutive gaps, "
        "and coverage gains need separate evidence.", "",
        "| Evaluation | Variant | Parcels | Carrot F1 | Cabbage F1 | Potato F1 | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores:
        if row["fold"] != "all":
            continue
        values = [f"{row[key]:.3f}" if row[key] is not None else "—"
                  for key in ("carrot_f1", "cabbage_f1", "potato_f1", "macro_f1")]
        lines.append("| " + " | ".join([row["stage"], row["model"], str(row["parcels"]), *values]) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(training, batch2_receipt, output):
    if output.resolve().is_relative_to(training.resolve()):
        raise ValueError("Use a new directory outside the frozen training run")
    previous = comparisons.read_json(batch2_receipt)
    features = training / "features_development.json"
    digest = seasons.file_sha256(features)
    if digest != previous["development"]["feature_sha256"]:
        raise ValueError("Development snapshot differs from Batch 2")
    manifest = training / "shape_external_comparison/frozen" / COHORT / "model_manifest.json"
    model, recipe = comparisons.load_classifier(manifest)
    if recipe["feature_columns"] != COLUMNS:
        raise ValueError("Expected the frozen fixed temporal-shape forest")
    snapshot = comparisons.read_json(features)
    source_holdout.require_matching(snapshot["rows"])
    rows = source_holdout.checked_rows(snapshot["rows"][COHORT], False)
    del snapshot
    train = [r for r in rows if r["partition"] == "train"]
    validation = [r for r in rows if r["partition"] == "validation"]
    split = {r["partition_group"]: r["partition"] for r in rows}
    expected = previous["cohorts"][COHORT]
    if (split != expected["development_split"] or split != comparisons.read_json(manifest)["development_split"]
            or {r["partition_group"] for r in train} & {r["partition_group"] for r in validation}):
        raise ValueError("Development split differs from saved baseline")
    folds = list(geographic_validation.spatial_folds(train))
    if [sorted(held) for _, held in folds] != expected["fold_held_ids"]:
        raise ValueError("Development geographic folds differ from Batch 2")
    partitions = comparisons.spatial_fold_rows(train, folds)
    output.mkdir(parents=True, exist_ok=False)
    pooled = {variant: [] for variant in VARIANTS}
    scores, predictions = [], []
    for fold, (fit, held) in enumerate(partitions, 1):
        result = predict_split(model, fit, held)
        scores.extend(score_rows(result, "geographic_cv", fold))
        for variant, records in result.items():
            pooled[variant].extend(records)
            predictions.extend({**r, "stage": "geographic_cv", "fold": fold, "model": variant} for r in records)
        print(f"Completed geographic fold {fold}", flush=True)
    scores.extend(score_rows(pooled, "geographic_cv", "all"))
    result = predict_split(model, train, validation)
    scores.extend(score_rows(result, "locked_validation", "all"))
    for variant, records in result.items():
        predictions.extend({**r, "stage": "locked_validation", "fold": "all", "model": variant} for r in records)
    baseline = next(r for r in scores if r["stage"] == "locked_validation" and r["model"] == "baseline")
    if not np.isclose(baseline["macro_f1"], comparisons.read_json(manifest)["validation"]["macro_f1"]):
        raise ValueError("Baseline does not reproduce archived locked validation macro-F1")
    write_csv(output / "metrics.csv", scores)
    write_csv(output / "predictions.csv", predictions)
    save_json(output / "run.json", {
        "development_feature_sha256": digest, "baseline_manifest_sha256": seasons.file_sha256(manifest),
        "batch2_receipt_sha256": seasons.file_sha256(batch2_receipt), "fold_held_ids": expected["fold_held_ids"],
        "forest_settings": model.get_params(), "feature_columns": COLUMNS, "gap_seed": SEED,
    })
    report(output, scores)
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "batch2_receipt", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(run(args.training, args.batch2_receipt, args.output))


if __name__ == "__main__":
    main()
