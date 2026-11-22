"""Compare matched aligned training cohorts on identical saved validation inputs."""

from collections import Counter
import json

import joblib
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline import aligned_month_check
from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.sources import adaptive, test_cohort
from crop_pipeline.training import model_comparisons, notebook_workflow, source_holdout
from crop_pipeline.training.external_report import write_csv
from crop_pipeline.windows import external, json_rows


VERSIONS = {"threshold": 20, "gap_25_days": 25}


def locked_validation(run, split):
    groups = {group for group in split["matched_parcel_groups"]
              if split["assignments"][group] == "validation"}
    rows = [row for cohort, row, span in json_rows.read_rows(
        run / "features_development.json", {"aligned_matched", "fixed_matched", "aligned_full",
                                            "fixed_full"}, container="rows"
    ) if span is not None and cohort == "aligned_matched" and row["partition"] == "validation"]
    source_holdout.checked_rows(rows, False)
    if {row["partition_group"] for row in rows} != groups or not groups:
        raise ValueError("Saved aligned validation features do not match the locked matched split")
    if {row["crop"] for row in rows} != set(source_holdout.CROPS):
        raise ValueError("Locked validation must contain all study crops")
    return rows


def locked_training(rows, assignments, new_groups, collection_assignments, validation):
    held_locations = {external.pixel_key(row)[2:] for row in validation}
    result = []
    for row in rows:
        group = row["partition_group"]
        partition = assignments.get(group)
        if partition is None and group in new_groups:
            partition = collection_assignments.get(group)
        if partition == "train" and external.pixel_key(row)[2:] not in held_locations:
            result.append({**row, "partition": "train"})
    if {row["crop"] for row in result} != set(source_holdout.CROPS):
        raise ValueError("Training version is missing a study crop after applying the locked split")
    source_holdout.checked_rows(result, False)
    return result


def add_savi(row):
    result = dict(row)
    for index in range(1, 9):
        prefix = f"bin_{index:02d}"
        result[f"{prefix}_SAVI"] = experiments.savi_value(
            row[f"{prefix}_B8"], row[f"{prefix}_B4"],
            experiments.FEATURE_POLICY["savi_soil_adjustment"],
        )
    return result


def prepare_training(repository):
    """Rebuild the two matched training inputs and their common locked validation."""
    root, report, _ = aligned_month_check.paths(repository)
    columns = experiments.feature_sets()[source_holdout.DESIGN]
    raw_columns = [column for column in columns if not column.endswith("_SAVI")]
    features, selected = {}, {}

    def record_feature(key, row):
        segment = None if row["window_status"] == "reference_supported" else row["window_start"]
        features.setdefault((key, segment), {
            **{name: row[name] for name in raw_columns},
            "partition_group": key[0], "region_id": key[1],
            "source_crs": key[2], "x": key[3], "y": key[4],
            "crop": row["crop"], "source_dataset": test_cohort.source_role(row["source_dataset"]),
            "label_status": row.get("label_status", "reported"),
            "window_start": row["window_start"], "window_end": row["window_end"],
        })

    def record_matched(label, keys, starts, references):
        selected[label] = [
            features[key, None if key in references else starts[key[0]]] for key in keys
        ]

    run, batches, new_groups, *_ = aligned_month_check.compare(
        root, record_feature=record_feature, record_matched=record_matched, include_jecam=False
    )
    features.clear()
    timing = json.loads((run / "seasons/automatic_timing_policy.json").read_text())
    if timing["quality_policy"]["maximum_onset_gap_days"] != 20:
        raise ValueError("The saved preprocessing run must use a 20-day baseline onset gap")
    split = json.loads((run / "development_split.json").read_text())
    validation = locked_validation(run, split)
    assignments = adaptive.assignments(report)
    fitting = {version: locked_training(rows, split["assignments"], new_groups, assignments, validation)
               for version, rows in selected.items()}
    return run, batches, new_groups, columns, fitting, validation


def train(repository):
    run, batches, new_groups, columns, training, validation = prepare_training(repository)
    output = notebook_workflow._new_run(repository, "training")
    metrics_rows, prediction_tables, details, training_membership = [], [], {}, []
    for version, gap in VERSIONS.items():
        fitting = training[version]
        groups = {row["partition_group"]: row["crop"] for row in fitting}
        class_counts = Counter(groups.values())
        new_counts = Counter(crop for group, crop in groups.items() if group in new_groups)
        print(f"Fitting {gap}-day model on {len(groups)} fields and {len(fitting)} pixels.", flush=True)
        recipe = RandomForestClassifier(
            n_estimators=1000, random_state=42, max_features="sqrt", min_samples_leaf=2,
            max_samples=.5, n_jobs=-1,
        )
        training_features = [add_savi(row) for row in fitting]
        model = model_comparisons.fit(recipe, training_features, columns)
        metrics, predictions = source_holdout.parcel_metrics(model, validation, columns)
        prediction_tables.append(predictions)
        joblib.dump({"classifier": model, "feature_columns": columns}, output / f"aligned_{gap}d.joblib")
        training_membership.extend({"gap_days": gap, "partition_group": group, "crop": crop,
                                    "new_download": group in new_groups}
                                   for group, crop in sorted(groups.items()))
        write_csv(output / f"validation_predictions_{gap}d.csv", predictions)
        details[gap] = metrics
        metrics_rows.append({
            "gap_days": gap, "training_fields": len(groups), "training_pixels": len(fitting),
            "validation_fields": len(predictions), "macro_f1": metrics["macro_f1"],
            **{f"{crop}_f1": metrics["per_crop"][crop]["f1"] for crop in source_holdout.CROPS},
            **{f"training_{crop}_fields": class_counts[crop] for crop in source_holdout.CROPS},
            **{f"new_{crop}_training_fields": new_counts[crop] for crop in source_holdout.CROPS},
        })
    model_comparisons.require_shared_parcels(*prediction_tables)
    write_csv(output / "validation_metrics.csv", metrics_rows)
    write_csv(output / "training_fields.csv", training_membership)
    save_json(output / "aligned_gap_validation.json", {
        "preprocessing_run": str(run), "completed_batches_added": batches,
        "validation_features": "Identical original aligned-matched validation pixels for both models",
        "validation_crop_counts": dict(Counter(row["crop"] for row in prediction_tables[0])),
        "validation_metrics": details, "feature_columns": columns,
        "forest_parameters": recipe.get_params(),
    })
    return output, metrics_rows
