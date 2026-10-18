"""Train reproducible Earth Engine forests on the same three feature designs."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.imagery.quality import save_json

DEFAULT_FOREST = dict(season_workflow.EARTH_ENGINE_FOREST)
MAXIMUM_REQUEST_BYTES = 9_000_000


def forest_settings(configuration, feature_count):
    settings = {**DEFAULT_FOREST, **configuration.get("earth_engine_forest", {})}
    if settings["number_of_trees"] != 1000 or settings["seed"] != 42:
        raise ValueError("The revised experiment fixes 1,000 trees and seed 42.")
    expected_weighting = "field_confidence_bootstrap"
    if (
        settings["variables_per_split"] != "square_root"
        or settings["weighting"] != expected_weighting
    ):
        raise ValueError(
            "Use the frozen square-root and field-confidence bootstrap policies."
        )
    leaf_population = float(settings["minimum_leaf_population"])
    if (
        not leaf_population.is_integer()
        or leaf_population < 1
        or not 0 < float(settings["bag_fraction"]) <= 1
    ):
        raise ValueError("Minimum leaf population and bag fraction must be positive.")
    return {
        "numberOfTrees": 1000,
        "seed": 42,
        "variablesPerSplit": max(1, math.floor(math.sqrt(feature_count))),
        "minLeafPopulation": int(settings["minimum_leaf_population"]),
        "bagFraction": float(settings["bag_fraction"]),
    }


def validate_experiment(rows, configuration):
    season_workflow.validate_configuration(configuration)
    if (
        configuration.get("feature_policy_frozen") is not True
        or not configuration.get("feature_development_evidence", "").strip()
    ):
        raise ValueError(
            "Freeze the feature recipe with development evidence before fitting."
        )
    feature_experiments.validate_policy(configuration["feature_policy"])
    season_workflow.validate_training_window_policy(rows, configuration)
    registry = {
        row["partition_group"]: {
            column: row[column] for column in season_workflow.FIELD_COLUMNS
        }
        for row in rows
    }
    validated_rows = season_workflow.assign_partitions(rows, list(registry.values()))
    if validated_rows != rows:
        raise ValueError("Inconsistent field metadata within a partition group.")
    if any(
        row.get("label_status") not in configuration["confidence_weights"]
        for row in rows
    ):
        raise ValueError("Training rows require confirmed or reported crop labels.")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample identities in experiment inputs.")
    for row in rows:
        for column in feature_experiments.feature_sets()["interval_and_shape"]:
            feature_experiments.finite_value(row, column)
    return split_experiment_rows(rows, configuration["study_classes"])


def split_experiment_rows(rows, classes):
    test_rows = [row for row in rows if row["partition"] == "test"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    external = [
        row
        for row in rows
        if row["source"] == "external" and row["partition"] == "train"
    ]
    local = [
        row for row in rows if row["source"] == "local" and row["partition"] == "train"
    ]
    training_sets = {
        "external_only": external,
        "local_only": local,
        "combined": external + local,
    }
    for name, training_rows in training_sets.items():
        season_workflow.require_class_coverage(training_rows, classes, name)
    season_workflow.require_class_coverage(
        test_rows, classes, "Independent local test set"
    )
    if validation_rows:
        season_workflow.require_class_coverage(
            validation_rows, classes, "Reserved validation set"
        )
    return training_sets, validation_rows, test_rows


def confidence_bootstrap(rows, configuration):
    """GEE has no sample_weight argument: resample explicitly, keeping class counts.

    This is an approximation to the previous weighted sklearn fit, not an exact
    equivalent. The same draw is reused across all three feature designs.
    """
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    generator = np.random.default_rng(42)
    weights_by_sample = dict(
        zip(
            [row["sample_id"] for row in ordered],
            season_workflow.field_sample_weights(
                ordered, configuration["confidence_weights"]
            ),
        )
    )
    sampled = []
    for crop in configuration["study_classes"]:
        crop_rows = [row for row in ordered if row["crop"] == crop]
        weights = np.asarray(
            [weights_by_sample[row["sample_id"]] for row in crop_rows],
            dtype=float,
        )
        indices = generator.choice(
            len(crop_rows), size=len(crop_rows), replace=True, p=weights / weights.sum()
        )
        sampled.extend(crop_rows[int(index)] for index in indices)
    return sampled


def packed_collection(earth_engine, rows, columns, classes, prediction=False):
    """Pack shared column names once to avoid repeating them in every request row."""
    class_codes = {crop: index for index, crop in enumerate(classes)}
    names = list(columns) + (["sample_index"] if prediction else ["crop_code"])
    values = [
        [float(row[column]) for column in columns]
        + [index if prediction else class_codes[row["crop"]]]
        for index, row in enumerate(rows)
    ]
    return earth_engine.FeatureCollection(
        earth_engine.List(values).map(
            lambda record: earth_engine.Feature(
                None, earth_engine.Dictionary.fromLists(names, record)
            )
        )
    )


def guard_request_size(expression):
    if len(expression.serialize().encode("utf-8")) > MAXIMUM_REQUEST_BYTES:
        raise ValueError(
            "Earth Engine inline request exceeds the safe payload budget. Use a smaller development budget or a separately prepared EE table asset; do not silently drop rows."
        )


def fit_forest(earth_engine, rows, configuration, columns):
    collection = packed_collection(
        earth_engine, rows, columns, configuration["study_classes"]
    )
    classifier = earth_engine.Classifier.smileRandomForest(
        **forest_settings(configuration, len(columns))
    ).train(features=collection, classProperty="crop_code", inputProperties=columns)
    guard_request_size(classifier)
    return classifier


def predict_rows(earth_engine, classifier, rows, columns, classes, batch_size=200):
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection = packed_collection(
            earth_engine, batch, columns, classes, prediction=True
        )
        result = collection.classify(classifier, "prediction").select(
            ["sample_index", "prediction"]
        )
        guard_request_size(result)
        payload = result.getInfo()
        predictions.extend(decode_predictions(payload, len(batch), classes))
    return predictions


def decode_predictions(payload, expected_count, classes):
    records = {
        int(feature["properties"]["sample_index"]): feature["properties"]["prediction"]
        for feature in payload["features"]
    }
    if len(payload["features"]) != expected_count or set(records) != set(
        range(expected_count)
    ):
        raise ValueError(
            "Earth Engine predictions lost or duplicated sample identities."
        )
    codes = [float(records[index]) for index in range(expected_count)]
    if any(not code.is_integer() or not 0 <= code < len(classes) for code in codes):
        raise ValueError("Earth Engine returned an unknown class code.")
    return [classes[int(code)] for code in codes]


def score_predictions(rows, predictions, classes):
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    if len(rows) != len(predictions):
        raise ValueError("Prediction count differs from the shared evaluation table.")
    if not rows:
        return None
    labels = [row["crop"] for row in rows]
    precision, recall, scores, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
    )
    return {
        "macro_f1": float(scores.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classes
        ).tolist(),
        "per_crop": {
            crop: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(scores[index]),
                "support": int(support[index]),
            }
            for index, crop in enumerate(classes)
        },
    }


def save_model(classifier, path, configuration, columns, training_rows, bootstrap_rows):
    explanation = classifier.explain().getInfo()
    trees = explanation.get("trees")
    if not trees or not all(isinstance(tree, str) for tree in trees):
        raise ValueError(
            "Earth Engine did not return fitted trees; do not save a training expression as a fitted model."
        )
    save_json(
        path,
        {
            "model_kind": "earth_engine_fitted_tree_ensemble",
            "trees": trees,
            "classes": configuration["study_classes"],
            "feature_columns": columns,
            "feature_policy": configuration["feature_policy"],
            "configuration": configuration,
            "forest_settings": forest_settings(configuration, len(columns)),
            "training_sample_fingerprint": season_workflow.record_fingerprint(
                [row["sample_id"] for row in training_rows]
            ),
            "bootstrap_sample_fingerprint": season_workflow.record_fingerprint(
                [row["sample_id"] for row in bootstrap_rows]
            ),
        },
    )


def load_model(earth_engine, path):
    record = json.loads(Path(path).read_text())
    if record.get("model_kind") != "earth_engine_fitted_tree_ensemble":
        raise ValueError("Unsupported fitted Earth Engine model file.")
    return earth_engine.Classifier.decisionTreeEnsemble(record["trees"]), record


def select_feature_designs(validation_results):
    """No validation rows means no selection; test scores never enter this function."""
    selections = {}
    for source, results in validation_results.items():
        eligible = [
            (design, metrics)
            for design, metrics in results.items()
            if metrics is not None
        ]
        if not eligible:
            selections[source] = {
                "status": "pending_validation",
                "selected_design": None,
            }
            continue
        design, metrics = min(
            eligible,
            key=lambda item: (
                -item[1]["macro_f1"],
                len(feature_experiments.feature_sets()[item[0]]),
                item[0],
            ),
        )
        selections[source] = {
            "status": "selected_on_validation",
            "selected_design": design,
            "validation_macro_f1": metrics["macro_f1"],
            "selection_rule": "highest_validation_macro_f1_then_fewest_features_then_name",
        }
    return selections


def train_feature_experiments(rows, configuration, output_folder, earth_engine):
    training_sets, validation_rows, test_rows = validate_experiment(rows, configuration)
    output_folder = Path(output_folder)
    fitted, validation_results = [], {}
    for source_name, training_rows in training_sets.items():
        records, results = fit_source_designs(
            earth_engine,
            source_name,
            training_rows,
            validation_rows,
            configuration,
            output_folder,
        )
        fitted.extend(records)
        validation_results[source_name] = results
    selections = select_feature_designs(validation_results)
    save_json(output_folder / "feature_selection.json", selections)
    metrics = evaluate_fitted_designs(earth_engine, fitted, test_rows, configuration)
    save_json(output_folder / "experiment_metrics.json", metrics)
    return metrics, selections


def fit_source_designs(
    earth_engine,
    source_name,
    training_rows,
    validation_rows,
    configuration,
    output_folder,
):
    bootstrap_rows = confidence_bootstrap(training_rows, configuration)
    multiplicities = Counter(row["sample_id"] for row in bootstrap_rows)
    source_folder = output_folder / source_name
    season_workflow.write_records(
        source_folder / "training_draw.csv",
        [
            {
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "crop": row["crop"],
                "multiplicity": multiplicities[row["sample_id"]],
            }
            for row in training_rows
        ],
    )
    results, fitted = {}, []
    for design, columns in feature_experiments.feature_sets().items():
        destination = source_folder / design
        classifier = fit_forest(earth_engine, bootstrap_rows, configuration, columns)
        predictions = predict_rows(
            earth_engine,
            classifier,
            validation_rows,
            columns,
            configuration["study_classes"],
        )
        results[design] = score_predictions(
            validation_rows, predictions, configuration["study_classes"]
        )
        save_model(
            classifier,
            destination / "model.json",
            configuration,
            columns,
            training_rows,
            bootstrap_rows,
        )
        save_prediction_table(
            destination / "validation_predictions.csv", validation_rows, predictions
        )
        fitted.append(
            {
                "source": source_name,
                "design": design,
                "classifier": classifier,
                "columns": columns,
                "folder": destination,
                "training_pixels": len(training_rows),
                "training_fields": len({row["field_id"] for row in training_rows}),
                "validation": results[design],
            }
        )
    return fitted, results


def save_prediction_table(path, rows, predictions):
    season_workflow.write_records(
        path,
        [
            {
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "crop": row["crop"],
                "prediction": prediction,
            }
            for row, prediction in zip(rows, predictions)
        ],
    )


def evaluate_fitted_designs(earth_engine, fitted, test_rows, configuration):
    metrics = {}
    for record in fitted:
        predictions = predict_rows(
            earth_engine,
            record["classifier"],
            test_rows,
            record["columns"],
            configuration["study_classes"],
        )
        result = {
            "feature_count": len(record["columns"]),
            "training_pixels": record["training_pixels"],
            "training_fields": record["training_fields"],
            "test_pixels": len(test_rows),
            "test_fields": len({row["field_id"] for row in test_rows}),
            "test_sample_fingerprint": season_workflow.record_fingerprint(
                [row["sample_id"] for row in test_rows]
            ),
            "validation": record["validation"],
            "test": score_predictions(
                test_rows, predictions, configuration["study_classes"]
            ),
        }
        metrics.setdefault(record["source"], {})[record["design"]] = result
        save_json(record["folder"] / "metrics.json", result)
        save_prediction_table(
            record["folder"] / "test_predictions.csv", test_rows, predictions
        )
    return metrics
