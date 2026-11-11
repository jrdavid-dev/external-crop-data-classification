"""Held-out source-parcel evaluation of existing features, without imagery retrieval."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from crop_pipeline.training import earth_engine as earth_engine_training
from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.sources import sampling as pilot_sampling
from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.imagery.quality import save_json

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
        "weighting": "equal_crop_then_equal_source_parcel_confidence_within_parcel_v1",
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
    """Give crops equal budgets, then parcels equal shares within each crop.

    Confidence only allocates a parcel's share among its pixels. Normalizing to
    mean one preserves the intended fractional bootstrap sample size.
    """
    if not rows:
        raise ValueError("Training rows must not be empty.")
    groups_by_crop = defaultdict(set)
    confidence_by_parcel = Counter()
    pixel_confidences = []
    for row in rows:
        group = row["partition_group"]
        crop = row["crop"]
        confidence = float(confidence_weights[row["label_status"]])
        if not np.isfinite(confidence) or confidence <= 0:
            raise ValueError("Training confidence weights must be finite and positive.")
        groups_by_crop[crop].add(group)
        confidence_by_parcel[(crop, group)] += confidence
        pixel_confidences.append(confidence)
    weights = np.asarray(
        [
            confidence
            / confidence_by_parcel[(row["crop"], row["partition_group"])]
            / len(groups_by_crop[row["crop"]])
            for row, confidence in zip(rows, pixel_confidences)
        ],
        dtype=float,
    )
    return weights * (len(rows) / weights.sum())


def run_experiments(rows, configuration, output_folder, reuse_completed=False):
    """Select on validation before evaluating any design on the same held-out parcels."""
    output_folder = Path(output_folder)
    partitions = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in ("train", "validation", "test")
    }
    assert_partition_separation(partitions)
    receipt_path = output_folder / "experiment_receipt.json"
    run_identity = {
        "samples": season_workflow.record_fingerprint(rows),
        "configuration": season_workflow.record_fingerprint(configuration),
    }
    if reuse_completed:
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("identity") != run_identity:
                raise ValueError("Saved experiment inputs differ; preserve this run and choose a new training_run_name")
            expected = {"experiment_metrics.json", "feature_selection.json", "training_weights.csv"}
            for design in feature_experiments.feature_sets():
                expected.update(f"{design}/{name}" for name in ("model.joblib", "validation_predictions.csv", "test_predictions.csv"))
            if set(receipt.get("files", {})) != expected:
                raise ValueError("Saved experiment receipt is incomplete; existing results were preserved")
            for name, record in receipt["files"].items():
                path = output_folder / name
                if not path.is_file() or season_workflow.file_sha256(path) != record["sha256"]:
                    raise ValueError("Saved experiment artifacts changed; existing results were preserved")
            metrics = json.loads((output_folder / "experiment_metrics.json").read_text())
            selections = json.loads((output_folder / "feature_selection.json").read_text())
            return metrics, selections
        if (output_folder / "experiment_metrics.json").exists() or any(
            (output_folder / design).exists() for design in feature_experiments.feature_sets()
        ):
            raise ValueError("Incomplete or unverified training results exist; preserve them and choose a new training_run_name")
    training_rows = partitions["train"]
    weights = normalized_training_weights(
        training_rows, configuration["confidence_weights"]
    )
    season_workflow.write_records(
        output_folder / "training_weights.csv",
        [
            {
                "sample_id": row["sample_id"],
                "crop": row["crop"],
                "partition_group": row["partition_group"],
                "weight": float(weight),
            }
            for row, weight in zip(training_rows, weights)
        ],
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
        "model_artifacts": {
            design: {
                "sha256": season_workflow.file_sha256(output_folder / design / "model.joblib"),
                "feature_columns": list(columns),
            }
            for design, (_, columns) in fitted.items()
        },
        "shared_test_sample_fingerprint": season_workflow.record_fingerprint(
            [row["sample_id"] for row in partitions["test"]]
        ),
        "limitations": configuration["limitations"],
    }
    save_json(output_folder / "experiment_metrics.json", results)
    if reuse_completed:
        names = ["experiment_metrics.json", "feature_selection.json", "training_weights.csv"]
        names.extend(f"{design}/{name}" for design in feature_experiments.feature_sets()
                     for name in ("model.joblib", "validation_predictions.csv", "test_predictions.csv"))
        save_json(receipt_path, {"identity": run_identity,
            "files": {name: {"sha256": season_workflow.file_sha256(output_folder / name)} for name in names}})
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


def prepare_feature_inputs(feature_handoff, output_root):
    """Consume notebook 05's exact enhanced rows; never recreate features or splits."""
    rows, configuration, coverage, feature_manifest = feature_experiments.load_pilot_feature_handoff(feature_handoff)
    partitions = {partition: [row for row in rows if row["partition"] == partition]
                  for partition in ("train", "validation", "test")}
    assert_partition_separation(partitions)
    for crop in CLASSES:
        if len({row["partition_group"] for row in rows if row["crop"] == crop}) < 5:
            raise ValueError(f"At least five source parcel groups are required for {crop}.")
    identity = {"feature_manifest_sha256": season_workflow.file_sha256(feature_handoff),
                "implementation_sha256": season_workflow.file_sha256(Path(__file__)),
                "forest_parameters": configuration["forest_parameters"],
                "weighting": "equal_crop_then_equal_source_parcel_confidence_within_parcel_v1"}
    folder = Path(output_root) / season_workflow.record_fingerprint(identity)[:16]
    manifest_path = folder / "exploratory_input.json"
    manifest = {
        "kind": "exploratory_existing_source_parcel_evaluation",
        "feature_handoff": str(Path(feature_handoff).resolve()),
        "identity": identity,
        "sample_fingerprint": feature_manifest["sample_fingerprint"],
        "group_assignments": {row["partition_group"]: row["partition"] for row in rows},
        "feature_policy": configuration["feature_policy"],
    }
    configuration = {
        **configuration,
        "purpose": "external_parcel_" + configuration["analysis_mode"],
        "backend": "scikit_learn", "frozen": False,
        "weighting": identity["weighting"],
        "weight_normalization": "mean_one_preserves_relative_weights_and_half_sample_bootstrap",
        "limitations": configuration.get("limitations", []) + [
            "Held-out external source labels, not independent Paoay ground truth.",
            "Satellite-estimated windows do not verify crop-cycle relevance.",
            "Parcel groups are source identities, not independently verified field boundaries.",
        ],
    }
    if folder.exists():
        if not manifest_path.is_file():
            raise ValueError("Training inputs changed; preserve the old run and use another output root.")
        saved_manifest = json.loads(manifest_path.read_text())
        if {key: value for key, value in saved_manifest.items() if key != "files"} != manifest:
            raise ValueError("Training inputs changed; preserve the old run and use another output root.")
        if set(saved_manifest.get("files", {})) != {"partitioned_features.csv", "parcel_split.csv", "configuration.json"}:
            raise ValueError("Existing training inputs are incomplete; preserve the run and use another output root.")
        for name, record in saved_manifest["files"].items():
            if not (folder / name).is_file() or season_workflow.file_sha256(folder / name) != record["sha256"]:
                raise ValueError("Saved training inputs changed; do not replace the locked run.")
        saved_rows = season_workflow.read_table(folder / "partitioned_features.csv")
        if saved_rows != rows or json.loads((folder / "configuration.json").read_text()) != configuration:
            raise ValueError("Saved training inputs changed; do not replace the locked run.")
        return rows, configuration, split_summary(rows), manifest_path
    folder.mkdir(parents=True)
    season_workflow.write_records(folder / "partitioned_features.csv", rows)
    season_workflow.write_records(folder / "parcel_split.csv", [
        {"partition_group": group, "partition": partition}
        for group, partition in sorted(manifest["group_assignments"].items())])
    save_json(folder / "configuration.json", configuration)
    manifest["files"] = {name: {"sha256": season_workflow.file_sha256(folder / name)}
                         for name in ("partitioned_features.csv", "parcel_split.csv", "configuration.json")}
    save_json(manifest_path, manifest)
    return rows, configuration, split_summary(rows), manifest_path


def inference_imagery_recipe():
    from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY

    return {"collection": "COPERNICUS/S2_SR_HARMONIZED", "request_version": REQUEST_VERSION,
                    "resampling": RESAMPLING_POLICY, "spectral_scale": 0.0001,
                    "bands": list(season_workflow.FEATURE_BANDS),
                    "feature_bands": list(season_workflow.FEATURE_BANDS),
                    "collection_bands": ["B3", "B4", "B5", "B8", "B11", "B2"],
                    "valid_data_rule": "all six collection bands must be valid before compositing",
                    "clear_counts_scaled": False,
                    "output_resolution_metres": 10,
                    "grid": "saved sampling-grid CRS and affine transform",
                    "input_nodata_value": -9999,
                    "nodata_handling": "mask nodata before reflectance scaling",
                    "cloud_probability_threshold": 40,
                    "cloud_probability_rule": "probability < threshold",
                    "excluded_scl_classes": [0, 1, 3, 8, 9, 10, 11],
                    "cloud_probability_collection": "COPERNICUS/S2_CLOUD_PROBABILITY",
                    "missing_cloud_probability": "exclude",
                    "scene_edges": "valid_B8A_and_B9",
                    "compositing": "same_date_mosaics_then_calendar_half_month_medians"}


def export_selected_model(output_folder, feature_handoff):
    """Package the validation-selected fitted model for later local inference, without refitting."""
    import shutil

    output_folder = Path(output_folder)
    rows, configuration, coverage, feature_manifest = feature_experiments.load_pilot_feature_handoff(feature_handoff)
    training_manifest_path = output_folder / "exploratory_input.json"
    training_manifest = json.loads(training_manifest_path.read_text())
    if training_manifest["identity"]["feature_manifest_sha256"] != season_workflow.file_sha256(feature_handoff):
        raise ValueError("Selected model's training inputs differ from the feature handoff.")
    if set(training_manifest.get("files", {})) != {"partitioned_features.csv", "parcel_split.csv", "configuration.json"}:
        raise ValueError("Training input manifest is incomplete; do not package an unverified model run.")
    for name, record in training_manifest["files"].items():
        if season_workflow.file_sha256(output_folder / name) != record["sha256"]:
            raise ValueError("Saved training inputs changed; do not package a different recipe with the model.")
    selection_path = output_folder / "feature_selection.json"
    selections = json.loads(selection_path.read_text())
    selection = selections.get("external_parcel_holdout", {})
    design = selection.get("selected_design")
    if selection.get("status") != "selected_on_validation" or design not in feature_experiments.feature_sets():
        raise ValueError("A validation-selected feature design is required before model export.")
    # Re-derive selection from validation only; held-out test metrics never choose a model.
    metrics_path = output_folder / "experiment_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    expected = earth_engine_training.select_feature_designs({"external_parcel_holdout": metrics["validation"]})
    if selections != expected:
        raise ValueError("Model selection differs from the saved validation-only selection.")
    model_path = output_folder / design / "model.joblib"
    artifact = metrics.get("model_artifacts", {}).get(design, {})
    if (artifact.get("feature_columns") != feature_experiments.feature_sets()[design]
            or artifact.get("sha256") != season_workflow.file_sha256(model_path)):
        raise ValueError("Selected model artifact differs from the model evaluated in the saved metrics.")
    recipe = {
        "status": "unvalidated_external_parcel_prototype", "analysis_mode": configuration["analysis_mode"],
        "selected_design": design, "selection": selection,
        "feature_columns": feature_experiments.feature_sets()[design],
        "feature_policy": configuration["feature_policy"],
        "study_classes": list(CLASSES), "class_codes": {crop: index + 1 for index, crop in enumerate(CLASSES)},
        "nodata_code": 0,
        "probability_columns": "classifier.classes_; map class labels through class_codes",
        "imagery": inference_imagery_recipe(),
        "window_recipe": configuration.get("window_recipe", {}),
        "window_convention": "four_calendar_months_eight_half_month_bins_end_exclusive",
        "window_policy": configuration["window_policy"],
        "quality": {"minimum_valid_fraction": coverage.get("window_recipe", {}).get("minimum_valid_fraction", coverage.get("minimum_valid_fraction", 0.8)),
                    "finite_features_required": True, "missing_values": "exclude_no_imputation"},
        "training_configuration": json.loads((output_folder / "configuration.json").read_text()),
        "source_feature_manifest_sha256": season_workflow.file_sha256(feature_handoff),
        "training_input_manifest_sha256": season_workflow.file_sha256(training_manifest_path),
        "sample_fingerprint": feature_manifest["sample_fingerprint"],
        "limitations": configuration.get("limitations", []) + [
            "Freeze model and preprocessing before inspecting Paoay ground-truth scores.",
            "Class codes describe known target crops only; other land cover needs a separate mask or strategy.",
        ],
    }
    identity = {"model_sha256": season_workflow.file_sha256(model_path),
                "recipe": recipe, "selection_sha256": season_workflow.file_sha256(selection_path)}
    folder = output_folder / "selected_model" / season_workflow.record_fingerprint(identity)[:16]
    manifest_path = folder / "model_manifest.json"
    if folder.exists():
        if not manifest_path.is_file():
            raise ValueError("Incomplete selected-model package; preserve it before choosing a new destination.")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("kind") != "paoay_inference_model_package"
                or set(manifest.get("files", {})) != {"model.joblib", "recipe.json"}
                or manifest.get("identity") != identity) or any(
            season_workflow.file_sha256(folder / name) != record["sha256"]
            for name, record in manifest["files"].items()):
            raise ValueError("Selected-model package changed; existing files were preserved.")
        return manifest_path
    folder.mkdir(parents=True)
    shutil.copyfile(model_path, folder / "model.joblib")
    save_json(folder / "recipe.json", recipe)
    manifest = {"format_version": 1, "kind": "paoay_inference_model_package",
                "status": recipe["status"], "identity": identity,
                "files": {name: {"sha256": season_workflow.file_sha256(folder / name)}
                          for name in ("model.joblib", "recipe.json")},
                "feature_handoff": str(Path(feature_handoff).resolve()),
                "training_metrics_sha256": season_workflow.file_sha256(metrics_path)}
    save_json(manifest_path, manifest)
    return manifest_path
