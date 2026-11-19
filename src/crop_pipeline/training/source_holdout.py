"""Freeze external Random Forests and score reserved external test parcels."""

from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.sources import test_cohort
from crop_pipeline.training import earth_engine, exploratory
from crop_pipeline.windows import seasons

CROPS = exploratory.CLASSES
DESIGN = "interval_only"
STATUS = "unvalidated_external_parcel_prototype"


def checked_rows(rows, test):
    sources = {test_cohort.TEST} if test else test_cohort.DEVELOPMENT
    splits = {"test"} if test else {"train", "validation"}
    if any(row["partition"] not in splits
           or (row["source_dataset"] not in sources
               and not (test and row.get("test_cohort") not in (None, "", "jecam")))
           or (not test and row.get("test_cohort"))
           for row in rows):
        raise ValueError("Unexpected source or partition in external experiment inputs")
    return rows


def require_matching(rows):
    from crop_pipeline.windows.external import pixel_key

    fixed = [pixel_key(row) for row in rows["fixed_matched"]]
    aligned = [pixel_key(row) for row in rows["aligned_matched"]]
    if len(fixed) != len(set(fixed)) or set(fixed) != set(aligned) or len(fixed) != len(aligned):
        raise ValueError("Matched cohorts must use the same unique pixel locations")


def parcel_metrics(model, rows, columns):
    grouped = defaultdict(list)
    labels = {}
    sites = {}
    for row, probabilities in zip(rows, model.predict_proba(exploratory.feature_matrix(rows, columns))):
        group = row["partition_group"]
        if group in labels and labels[group] != row["crop"]:
            raise ValueError(f"Conflicting labels for {group}")
        labels[group] = row["crop"]
        sites.setdefault(group, row.get("site_cell", "unavailable"))
        grouped[group].append(probabilities)
    predictions = [
        {"partition_group": group, "crop": labels[group], "site_cell": sites[group],
         "prediction": str(model.classes_[np.argmax(np.mean(values, axis=0))])}
        for group, values in sorted(grouped.items())
    ]
    metrics = earth_engine.score_predictions(
        predictions, [row["prediction"] for row in predictions], CROPS
    )
    if set(labels.values()) != set(CROPS):
        metrics.pop("macro_f1")
        metrics.pop("balanced_accuracy")
    pixels = Counter(row["crop"] for row in rows)
    counts = Counter(labels.values())
    metrics["coverage"] = {crop: {"parcels": counts[crop], "pixels": pixels[crop],
                                 "target_met": counts[crop] >= 20 and pixels[crop] >= 400}
                           for crop in CROPS}
    metrics["counts_by_site_cell"] = {
        site: dict(Counter(row["crop"] for row in predictions if row["site_cell"] == site))
        for site in sorted(set(sites.values()))
    }
    metrics["status"] = ("incomplete_class_coverage" if len(counts) < len(CROPS)
                         else "targets_met" if all(item["target_met"] for item in metrics["coverage"].values())
                         else "pilot_below_targets")
    return metrics, predictions


def freeze(development_path, *, destination=None):
    path = Path(development_path)
    data = json.loads(path.read_text())
    require_matching(data["rows"])
    fixed_start_month = data.get("fixed_start_month", 1)
    if type(fixed_start_month) is not int or not 1 <= fixed_start_month <= 9:
        raise ValueError("Fixed four-month window must start in January through September")
    columns = experiments.feature_sets()[DESIGN]
    fingerprint = seasons.file_sha256(path)
    destination = Path(destination) if destination is not None else path.parent / ("models_" + fingerprint[:12])
    destination.mkdir(exist_ok=False)
    save_json(destination / "development.json", {
        "feature_sha256": fingerprint, "quality": data["quality"],
        "window_sha256": data["window_sha256"],
        "fixed_start_month": fixed_start_month,
        "implementation_sha256": seasons.file_sha256(inspect.getfile(experiments)),
    })
    manifests = {}
    validation_results = {}
    for cohort in ("full", "matched"):
        pair = {name: checked_rows(data["rows"][f"{name}_{cohort}"], False)
                for name in ("fixed", "aligned")}
        for name, rows in pair.items():
            if any({row["crop"] for row in rows if row["partition"] == split} != set(CROPS)
                   for split in ("train", "validation")):
                print(f"{name}_{cohort}: insufficient development coverage")
                continue
            train = [row for row in rows if row["partition"] == "train"]
            validation = [row for row in rows if row["partition"] == "validation"]
            exploratory.validate_input_rows([{k: v for k, v in row.items() if k != "partition"} for row in rows])
            if {r["partition_group"] for r in train} & {r["partition_group"] for r in validation}:
                raise ValueError("Development parcels cross the training/validation split")
            weights = exploratory.normalized_training_weights(train, {"confirmed": 1., "reported": .5})
            choices = []
            candidate_metrics = []
            for leaf in ((2,) if cohort == "matched" else (2, 4)):
                model = RandomForestClassifier(n_estimators=1000, random_state=42, max_features="sqrt",
                                               min_samples_leaf=leaf, max_samples=.5, n_jobs=-1)
                model.fit(exploratory.feature_matrix(train, columns), [row["crop"] for row in train],
                          sample_weight=weights)
                metrics, _ = parcel_metrics(model, validation, columns)
                candidate_metrics.append({"min_samples_leaf": leaf, "metrics": metrics})
                choices.append((metrics["macro_f1"], model, metrics))
            _, model, metrics = max(choices, key=lambda choice: choice[0])
            validation_results[f"{name}_{cohort}"] = {
                "candidates": candidate_metrics, "selected_min_samples_leaf": model.min_samples_leaf,
                "selection_metric": "validation_parcel_macro_f1",
            }
            folder = destination / f"{name}_{cohort}"
            folder.mkdir()
            configuration = {"feature_policy": experiments.FEATURE_POLICY, "study_classes": CROPS}
            recipe = {
                **configuration, "status": STATUS, "selected_design": DESIGN,
                "selection": {"status": "selected_on_validation", "selected_design": DESIGN},
                "feature_columns": columns, "class_codes": {crop: i + 1 for i, crop in enumerate(CROPS)},
                "nodata_code": 0, "imagery": exploratory.inference_imagery_recipe(),
                "window_convention": "four_calendar_months_eight_half_month_bins_end_exclusive",
                "window_recipe": {"type": name, "start_month": fixed_start_month if name == "fixed" else None},
                "quality": {"minimum_valid_fraction": .8, "finite_features_required": True,
                            "missing_values": "exclude_no_imputation"},
                "source_feature_manifest_sha256": seasons.file_sha256(destination / "development.json"),
            }
            joblib.dump({"classifier": model, "feature_columns": columns, "configuration": configuration},
                        folder / "model.joblib")
            save_json(folder / "recipe.json", recipe)
            files = {f: {"sha256": seasons.file_sha256(folder / f)} for f in ("model.joblib", "recipe.json")}
            manifest = folder / "model_manifest.json"
            save_json(manifest, {
                "format_version": 1, "kind": "paoay_inference_model_package", "status": STATUS,
                "identity": {"recipe": recipe, "model_sha256": files["model.joblib"]["sha256"]},
                "files": files, "feature_handoff": str(destination / "development.json"),
                "validation": metrics, "development_groups": sorted({r["partition_group"] for r in rows}),
                "development_split": {r["partition_group"]: r["partition"] for r in rows},
                "model_id": files["model.joblib"]["sha256"],
                "experiment_cohort": cohort,
                "training_majority_class": Counter({r["partition_group"]: r["crop"] for r in train}.values()).most_common(1)[0][0],
            })
            manifests[f"{name}_{cohort}"] = str(manifest)
    if not manifests:
        raise ValueError("No window cohort has all three development crops in both splits")
    save_json(destination / "validation_candidates.json", validation_results)
    index = destination / "models.json"
    save_json(index, {name: {"path": value, "sha256": seasons.file_sha256(value)}
                      for name, value in manifests.items()})
    return index


def score(test_path, model_index, *, destination=None, predictions_directory=None, scope="jecam"):
    from crop_pipeline.paoay.predictions import load_classifier

    data = json.loads(Path(test_path).read_text())
    require_matching(data["rows"])
    index = json.loads(Path(model_index).read_text())
    destination = Path(destination) if destination is not None else Path(model_index).parent
    predictions_directory = Path(predictions_directory) if predictions_directory is not None else destination
    scores = {}
    for name, entry in index.items():
        path = Path(entry["path"])
        if seasons.file_sha256(path) != entry["sha256"]:
            raise ValueError("Frozen model manifest changed")
        manifest = json.loads(path.read_text())
        provenance = json.loads(Path(manifest["feature_handoff"]).read_text())
        if (any(provenance[key] != data[key] for key in ("quality", "window_sha256"))
                or provenance.get("fixed_start_month", 1) != data.get("fixed_start_month", 1)):
            raise ValueError("Test features come from a different experiment snapshot")
        all_rows = checked_rows(data["rows"][name], True)
        if set(manifest["development_groups"]) & {row["partition_group"] for row in all_rows}:
            raise ValueError("Test parcel overlaps a development group")
        rows = [row for row in all_rows if scope == "combined"
                or row.get("test_cohort", "jecam") == scope]
        if not rows:
            scores[name] = {"status": "no_usable_test_rows"}
            continue
        model, recipe = load_classifier(path)
        scores[name], predictions = parcel_metrics(model, rows, recipe["feature_columns"])
        cohorts = {row["partition_group"]: row.get("test_cohort", "jecam") for row in rows}
        for row in predictions:
            row["test_cohort"] = cohorts[row["partition_group"]]
        seasons.write_records(predictions_directory / f"{name}_test_parcels.csv", predictions)
    save_json(destination / "test_scores.json", scores)
    return scores
