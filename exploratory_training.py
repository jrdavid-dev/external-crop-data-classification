"""Held-out source-parcel evaluation of existing features, without imagery retrieval."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import earth_engine_training
import feature_experiments
import pilot_sampling
import season_workflow
from crop_workflow import save_json

CLASSES = ["carrot", "cabbage", "potato"]
SPLIT_POLICY = {
    "seed": 42,
    "validation_group_fraction": 0.2,
    "test_group_fraction": 0.2,
}


def existing_feature_tables(report_root, analysis_run_name=None):
    """Prefer the expanded pilot; fall back to saved earlier features without downloads."""
    analysis_folder = Path(report_root).parent / "analysis"
    run_names = (
        [analysis_run_name]
        if analysis_run_name
        else ["pilot_3000_usable_500", "pilot_3000", "2020"]
    )
    for run_folder in (analysis_folder / name for name in run_names):
        candidates = [
            run_folder / "seasons" / name / "complete_features.csv"
            for name in ("features", "estimated_features")
        ]
        available = [
            path
            for path in candidates
            if path.is_file() and season_workflow.read_table(path)
        ]
        if available:
            return available
    raise FileNotFoundError(
        "No saved complete feature tables found. Finish window feature preparation or select existing feature CSV paths explicitly."
    )


def validate_input_rows(rows):
    """Reject ambiguous identities and reserved evaluation rows before making splits."""
    if not rows or {row["crop"] for row in rows} != set(CLASSES):
        raise ValueError(
            "Exploratory evaluation requires complete rows for all three crops."
        )
    if any(row.get("partition") in {"train", "validation", "test"} for row in rows):
        raise ValueError(
            "Existing reserved partitions must not be reassigned by exploratory mode."
        )
    if any(not row.get("partition_group") for row in rows):
        raise ValueError(
            "Source parcel identity is required; pixels cannot be split randomly."
        )
    crops_by_group, groups_by_field, groups_by_location = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    for row in rows:
        group = row["partition_group"]
        crops_by_group[group].add(row["crop"])
        groups_by_location[pilot_sampling.pixel_location(row)].add(group)
        if row.get("field_id"):
            groups_by_field[row["field_id"]].add(group)
    if any(len(crops) != 1 for crops in crops_by_group.values()):
        raise ValueError("Conflicting crop labels within a source parcel group.")
    if any(
        len(groups) > 1
        for groups in list(groups_by_field.values()) + list(groups_by_location.values())
    ):
        raise ValueError(
            "Shared field identities or pixel locations span parcel groups; resolve aliases first."
        )


def parcel_partitions(rows, previous_assignments=None):
    """Keep old holdouts reserved and stratify only newly available parcel groups."""
    validate_input_rows(rows)
    previous_assignments = previous_assignments or {}
    if any(
        partition not in {"train", "validation", "test"}
        for partition in previous_assignments.values()
    ):
        raise ValueError("Unknown prior parcel partition.")
    assignments = {}
    for crop in CLASSES:
        groups = sorted(
            {row["partition_group"] for row in rows if row["crop"] == crop},
            key=lambda group: pilot_sampling.stable_rank(group, SPLIT_POLICY["seed"]),
        )
        if len(groups) < 5:
            raise ValueError(
                f"At least five source parcel groups are required for {crop}."
            )
        assignments.update(assign_crop_groups(groups, previous_assignments))
    return assignments


def assign_crop_groups(groups, previous_assignments):
    """Fill validation/test group targets without moving any previously assigned parcel."""
    assignments = {
        group: previous_assignments[group]
        for group in groups
        if group in previous_assignments
    }
    remaining = [group for group in groups if group not in assignments]
    counts = Counter(assignments.values())
    test_target = max(1, round(len(groups) * SPLIT_POLICY["test_group_fraction"]))
    validation_target = max(
        1, round(len(groups) * SPLIT_POLICY["validation_group_fraction"])
    )
    for group in remaining:
        partition = "train"
        if counts["test"] < test_target:
            partition = "test"
        elif counts["validation"] < validation_target:
            partition = "validation"
        assignments[group] = partition
        counts[partition] += 1
    return assignments


def prepare_existing_inputs(input_paths, output_folder, previous_split_path=None):
    """Lock the input hashes and split so reruns cannot quietly change the holdout."""
    rows = [row for path in input_paths for row in season_workflow.read_table(path)]
    previous_assignments = {}
    if previous_split_path is not None:
        prior_manifest = json.loads(Path(previous_split_path).read_text())
        if (
            prior_manifest.get("kind")
            != "exploratory_existing_source_parcel_evaluation"
        ):
            raise ValueError("Prior split must be an exploratory parcel manifest.")
        previous_assignments = prior_manifest["group_assignments"]
    assignments = parcel_partitions(rows, previous_assignments)
    prepared = [
        prepare_identity(row, assignments[row["partition_group"]]) for row in rows
    ]
    enhanced = feature_experiments.enhance_rows(
        prepared, feature_experiments.FEATURE_POLICY
    )
    enhanced = sorted(enhanced, key=lambda row: row["sample_id"])
    output_folder = Path(output_folder)
    manifest_path = output_folder / "exploratory_input.json"
    manifest = {
        "kind": "exploratory_existing_source_parcel_evaluation",
        "input_files": [
            {
                "path": str(Path(path).resolve()),
                "sha256": season_workflow.file_sha256(path),
            }
            for path in input_paths
        ],
        "split_policy": SPLIT_POLICY,
        "previous_split": (
            {
                "path": str(Path(previous_split_path).resolve()),
                "sha256": season_workflow.file_sha256(previous_split_path),
            }
            if previous_split_path is not None
            else None
        ),
        "group_assignments": assignments,
        "sample_fingerprint": season_workflow.record_fingerprint(
            [row["sample_id"] for row in enhanced]
        ),
        "feature_policy": feature_experiments.FEATURE_POLICY,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Exploratory inputs or splits changed. Use a new output folder; do not replace the existing test split."
        )
    save_json(manifest_path, manifest)
    season_workflow.write_records(output_folder / "partitioned_features.csv", enhanced)
    season_workflow.write_records(
        output_folder / "parcel_split.csv",
        [
            {"partition_group": group, "partition": partition}
            for group, partition in sorted(assignments.items())
        ],
    )
    configuration = {
        "purpose": manifest["kind"],
        "backend": "scikit_learn",
        "frozen": False,
        "study_classes": CLASSES,
        "feature_policy": dict(feature_experiments.FEATURE_POLICY),
        "window_policy": "exploratory_existing_estimates",
        "split_policy": SPLIT_POLICY,
        "forest_parameters": {
            "n_estimators": 1000,
            "random_state": 42,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "bootstrap": True,
            "max_samples": 0.5,
            "n_jobs": -1,
        },
        "confidence_weights": {"confirmed": 1.0, "reported": 0.5},
        "weighting": "source_parcel_confidence_sample_weight",
        "weight_normalization": "mean_one_preserves_relative_weights_and_half_sample_bootstrap",
        "imagery_policy": "saved_complete_feature_tables_resampling_not_inferred",
        "limitations": [
            "Held-out source parcel labels, not independent Philippine ground truth.",
            "Parcel groups are source identities, not independently verified field boundaries.",
            "Satellite-estimated windows do not verify crop-cycle relevance.",
            "Cabbage source/year differs from carrot and potato.",
            "Local sklearn bagging is not identical to Earth Engine smileRandomForest.",
        ],
    }
    save_json(output_folder / "configuration.json", configuration)
    return enhanced, configuration, split_summary(enhanced), manifest_path


def prepare_identity(row, partition):
    """Use a marked grouping proxy without promoting it to verified field truth."""
    return {
        **row,
        "original_field_id": row.get("field_id", ""),
        "field_id": row.get("field_id")
        or f"unverified_source_parcel|{row['partition_group']}",
        "field_identity_status": "source_parcel_proxy",
        "source": "external",
        "partition": partition,
    }


def split_summary(rows):
    return [
        {
            "crop": crop,
            "partition": partition,
            "pixels": sum(
                row["crop"] == crop and row["partition"] == partition for row in rows
            ),
            "parcel_groups": len(
                {
                    row["partition_group"]
                    for row in rows
                    if row["crop"] == crop and row["partition"] == partition
                }
            ),
        }
        for crop in CLASSES
        for partition in ("train", "validation", "test")
    ]


def feature_matrix(rows, columns):
    return np.asarray([[float(row[column]) for column in columns] for row in rows])


def evaluate_predictions(rows, predictions):
    """Report pixel metrics alongside equally weighted parcel accuracy."""
    metrics = earth_engine_training.score_predictions(rows, list(predictions), CLASSES)
    metrics["accuracy"] = float(
        accuracy_score([row["crop"] for row in rows], predictions)
    )
    correct, counts = Counter(), Counter()
    for row, prediction in zip(rows, predictions):
        group = row["partition_group"]
        counts[group] += 1
        correct[group] += int(row["crop"] == prediction)
    metrics["mean_parcel_accuracy"] = float(
        np.mean([correct[group] / count for group, count in counts.items()])
    )
    metrics["parcel_groups"] = len(counts)
    return metrics


def normalized_training_weights(rows, confidence_weights):
    """Keep relative parcel weights while preserving the intended bootstrap size.

    Recent sklearn versions compute fractional max_samples from the sum of
    sample weights. Mean-one weights prevent a small number of parcel budgets
    from reducing a tree's draw to only a handful of pixels.
    """
    weights = np.asarray(
        season_workflow.field_sample_weights(rows, confidence_weights), dtype=float
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Training confidence weights must be finite and positive.")
    return weights * (len(rows) / weights.sum())


def run_experiments(rows, configuration, output_folder):
    """Select on validation before evaluating any design on the same held-out parcels."""
    output_folder = Path(output_folder)
    partitions = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in ("train", "validation", "test")
    }
    assert_partition_separation(partitions)
    training_rows = partitions["train"]
    weights = normalized_training_weights(
        training_rows, configuration["confidence_weights"]
    )
    fitted, validation_metrics = {}, {}
    for design, columns in feature_experiments.feature_sets().items():
        classifier = RandomForestClassifier(**configuration["forest_parameters"])
        classifier.fit(
            feature_matrix(training_rows, columns),
            [row["crop"] for row in training_rows],
            sample_weight=weights,
        )
        predictions = classifier.predict(
            feature_matrix(partitions["validation"], columns)
        )
        validation_metrics[design] = evaluate_predictions(
            partitions["validation"], predictions
        )
        fitted[design] = (classifier, columns)
        destination = output_folder / design
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": classifier,
                "feature_columns": columns,
                "configuration": configuration,
            },
            destination / "model.joblib",
        )
        earth_engine_training.save_prediction_table(
            destination / "validation_predictions.csv",
            partitions["validation"],
            predictions,
        )
    selections = earth_engine_training.select_feature_designs(
        {"external_parcel_holdout": validation_metrics}
    )
    save_json(output_folder / "feature_selection.json", selections)
    test_metrics = evaluate_test_designs(fitted, partitions["test"], output_folder)
    results = {
        "purpose": configuration["purpose"],
        "backend": "scikit_learn",
        "split_summary": split_summary(rows),
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_selection": selections,
        "shared_test_sample_fingerprint": season_workflow.record_fingerprint(
            [row["sample_id"] for row in partitions["test"]]
        ),
        "limitations": configuration["limitations"],
    }
    save_json(output_folder / "experiment_metrics.json", results)
    return results, selections


def assert_partition_separation(partitions):
    group_sets = [
        set(row["partition_group"] for row in rows) for rows in partitions.values()
    ]
    if sum(len(groups) for groups in group_sets) != len(set.union(*group_sets)):
        raise ValueError("Parcel groups overlap between partitions.")
    for partition, rows in partitions.items():
        season_workflow.require_class_coverage(rows, CLASSES, partition)
    # Recheck identities across the complete input, regardless of imported partition labels.
    combined = [
        {key: value for key, value in row.items() if key != "partition"}
        for rows in partitions.values()
        for row in rows
    ]
    validate_input_rows(combined)


def evaluate_test_designs(fitted, test_rows, output_folder):
    metrics = {}
    for design, (classifier, columns) in fitted.items():
        predictions = classifier.predict(feature_matrix(test_rows, columns))
        metrics[design] = evaluate_predictions(test_rows, predictions)
        earth_engine_training.save_prediction_table(
            output_folder / design / "test_predictions.csv", test_rows, predictions
        )
    return metrics
