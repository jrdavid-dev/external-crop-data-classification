"""Pooled CropHarvest and Sen4AgriNet/FranceCrops experiments from Earth Engine tables.

Paoay is deliberately excluded: its labels are calendar-inferred and come from six
fields, so it cannot yet serve as training or test truth.
"""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.imagery.quality import save_json
from crop_pipeline.sources import sampling
from crop_pipeline.training import exploratory
from crop_pipeline.windows import seasons as season_workflow

CLASSES = exploratory.CLASSES
SEED = 42
TEST_GROUP_FRACTION = 0.2
CV_FOLDS = 5
SAME_PIXEL_METERS = 10.0
SPATIAL_BLOCK_KM = 10.0
EARTH_RADIUS_M = 6_371_000.0
ASSETS = {
    "cropharvest": "cropharvest/cropharvest_features",
    "francecrops_s4a": "francecrops_s4a/partitioned_features",
}
METADATA = {
    "cropharvest": ["crop", "partition_group", "country", "year"],
    "francecrops_s4a": ["crop", "partition_group", "source_dataset", "source_year", "label_status"],
}
BASE_COLUMNS = season_workflow.feature_columns()
# CropHarvest saved no window dates, so every source uses nominal half-month midpoints.
NOMINAL_POSITIONS = np.arange(8) * (365.25 / 24)
CONFIDENCE_WEIGHTS = {"confirmed": 1.0, "reported": 0.5}
FOREST_PARAMETERS = {
    "n_estimators": 1000,
    "random_state": SEED,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "bootstrap": True,
    "max_samples": 0.5,
    "n_jobs": -1,
}
MODEL_NAMES = ("random_forest", "gradient_boosting", "logistic_regression")


def fetch_table(earth_engine, project, table):
    """Download one feature table with point coordinates; no imagery is read."""
    asset_id = f"projects/{project}/assets/{ASSETS[table]}"
    collection = earth_engine.FeatureCollection(asset_id).select(BASE_COLUMNS + METADATA[table])
    records = []
    for feature in collection.getInfo()["features"]:
        longitude, latitude = feature["geometry"]["coordinates"]
        records.append(
            {**feature["properties"], "asset_index": feature["id"], "lon": longitude, "lat": latitude}
        )
    return pd.DataFrame(records), asset_id


def cached_tables(earth_engine, project, input_folder, refresh=False):
    """Reuse saved downloads so reruns score identical rows unless a refresh is requested."""
    input_folder = Path(input_folder)
    input_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = input_folder / "input_manifest.json"
    tables, manifest = {}, {"kind": "external_combined_earth_engine_inputs", "tables": {}}
    for table in ASSETS:
        path = input_folder / f"{table}.csv"
        if refresh or not path.is_file():
            if earth_engine is None:
                raise FileNotFoundError(f"{path} is missing; sign in to Earth Engine to download it.")
            frame, asset_id = fetch_table(earth_engine, project, table)
            frame.to_csv(path, index=False)
        else:
            asset_id = f"projects/{project}/assets/{ASSETS[table]}"
        tables[table] = pd.read_csv(path, dtype={"partition_group": str, "asset_index": str})
        manifest["tables"][table] = {
            "asset_id": asset_id,
            "path": str(path.resolve()),
            "sha256": season_workflow.file_sha256(path),
            "rows": len(tables[table]),
        }
    save_json(manifest_path, manifest)
    return tables, manifest


def harmonize(tables):
    """Give both sources one schema, with source-prefixed parcel identities."""
    cropharvest, france = tables["cropharvest"], tables["francecrops_s4a"]
    frames = [
        pd.DataFrame(
            {
                "domain": "cropharvest",
                "region": cropharvest["country"].astype(str),
                "year": cropharvest["year"].astype(int),
                "label_status": "reported",
                "asset_index": cropharvest["asset_index"].astype(str),
                "partition_group": "cropharvest|" + cropharvest["partition_group"].astype(str),
            }
        ),
        pd.DataFrame(
            {
                "domain": france["source_dataset"].astype(str),
                "region": "France",
                "year": france["source_year"].astype(int),
                "label_status": france["label_status"].astype(str),
                "asset_index": france["asset_index"].astype(str),
                "partition_group": france["source_dataset"].astype(str)
                + "|"
                + france["partition_group"].astype(str),
            }
        ),
    ]
    for frame, source in zip(frames, (cropharvest, france)):
        frame["crop"] = source["crop"].astype(str).to_numpy()
        frame["lon"] = source["lon"].to_numpy(dtype=float)
        frame["lat"] = source["lat"].to_numpy(dtype=float)
        for column in BASE_COLUMNS:
            frame[column] = source[column].to_numpy(dtype=float)
    data = pd.concat(frames, ignore_index=True)
    data.insert(0, "sample_id", data["domain"] + "|" + data["asset_index"])
    if set(data["crop"]) != set(CLASSES):
        raise ValueError(f"Expected crops {CLASSES}; found {sorted(set(data['crop']))}.")
    if data["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample identities across the combined tables.")
    if not np.isfinite(data[BASE_COLUMNS].to_numpy()).all():
        raise ValueError("Nonfinite base features; export complete windows rather than filling gaps.")
    if set(data["label_status"]) - set(CONFIDENCE_WEIGHTS):
        raise ValueError("Unknown label status in the combined tables.")
    return data.sort_values("sample_id").reset_index(drop=True)


def add_derived_features(data, adjustment=0.5):
    """Recompute SAVI and temporal shapes identically for every source."""
    derived = data.copy()
    for bin_number in range(1, 9):
        prefix = f"bin_{bin_number:02d}"
        near_infrared, red = derived[f"{prefix}_B8"], derived[f"{prefix}_B4"]
        derived[f"{prefix}_SAVI"] = (1 + adjustment) * (near_infrared - red) / (
            near_infrared + red + adjustment
        )
    shape_columns = feature_experiments.shape_columns()
    shapes = []
    for _, row in derived.iterrows():
        features = {}
        for index_name in feature_experiments.INDEX_NAMES:
            values = [row[f"bin_{bin_number:02d}_{index_name}"] for bin_number in range(1, 9)]
            features.update(
                feature_experiments.temporal_shape(values, NOMINAL_POSITIONS, index_name)
            )
        shapes.append({column: float(features[column]) for column in shape_columns})
    derived = pd.concat([derived, pd.DataFrame(shapes, index=derived.index)], axis=1)
    columns = feature_experiments.feature_sets()["interval_and_shape"]
    if not np.isfinite(derived[columns].to_numpy()).all():
        raise ValueError("Derived features are nonfinite; inspect reflectance values.")
    return derived


def link_split_units(data, radius_m=SAME_PIXEL_METERS):
    """Join parcel groups that share a pixel location so they never straddle a split."""
    groups = data["partition_group"].to_numpy()
    if (data.groupby("partition_group")["crop"].nunique() > 1).any():
        raise ValueError("Conflicting crop labels within a source parcel group.")
    parent = {group: group for group in set(groups)}

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    coordinates = np.radians(data[["lat", "lon"]].to_numpy())
    neighbours = BallTree(coordinates, metric="haversine").query_radius(
        coordinates, r=radius_m / EARTH_RADIUS_M
    )
    linked_pairs = set()
    for index, matches in enumerate(neighbours):
        for match in matches:
            if groups[index] != groups[match]:
                linked_pairs.add(tuple(sorted((groups[index], groups[match]))))
                parent[root(groups[index])] = root(groups[match])
    units = data.copy()
    units["split_unit"] = [root(group) for group in groups]
    audit = pd.DataFrame(sorted(linked_pairs), columns=["group_a", "group_b"])
    if len(audit):
        crops = data.groupby("partition_group")["crop"].first()
        audit["crop_a"] = audit["group_a"].map(crops)
        audit["crop_b"] = audit["group_b"].map(crops)
    return units, audit


def add_spatial_blocks(data, block_km=SPATIAL_BLOCK_KM):
    """Assign each split unit to the grid cell holding its centroid."""
    centroids = data.groupby("split_unit")[["lat", "lon"]].mean()
    degrees = block_km / 111.32
    cells = (
        np.floor(centroids["lat"] / degrees).astype(int).astype(str)
        + "_"
        + np.floor(centroids["lon"] * np.cos(np.radians(centroids["lat"])) / degrees)
        .astype(int)
        .astype(str)
    )
    blocked = data.copy()
    blocked["spatial_block"] = blocked["split_unit"].map(cells)
    return blocked


def lock_test_split(data, split_path, fraction=TEST_GROUP_FRACTION):
    """Hold out a fixed share of split units in every crop-by-source stratum."""
    units = data.groupby("split_unit").agg(
        crop=("crop", lambda values: Counter(values).most_common(1)[0][0]),
        domain=("domain", lambda values: Counter(values).most_common(1)[0][0]),
    )
    test_units = set()
    for _, stratum in units.groupby(["crop", "domain"]):
        ranked = sorted(stratum.index, key=lambda unit: sampling.stable_rank(unit, SEED))
        count = max(1, round(len(ranked) * fraction)) if len(ranked) >= 3 else 0
        test_units.update(ranked[:count])
    split = pd.DataFrame(
        {
            "split_unit": units.index,
            "partition": ["test" if unit in test_units else "development" for unit in units.index],
        }
    )
    split_path = Path(split_path)
    if split_path.is_file():
        saved = pd.read_csv(split_path, dtype=str)
        if not saved.sort_values("split_unit").reset_index(drop=True).equals(
            split.sort_values("split_unit").reset_index(drop=True).astype(str)
        ):
            raise ValueError(
                "The locked test split differs from the current inputs. Use a new output folder."
            )
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
    partitioned = data.copy()
    partitioned["partition"] = partitioned["split_unit"].map(
        dict(zip(split["split_unit"], split["partition"]))
    )
    for partition, rows in partitioned.groupby("partition"):
        if set(rows["crop"]) != set(CLASSES):
            raise ValueError(f"The {partition} partition lacks a crop.")
    return partitioned


def split_counts(data):
    return (
        data.groupby(["partition", "crop", "domain"])
        .agg(pixels=("sample_id", "size"), parcels=("partition_group", "nunique"))
        .reset_index()
    )


def training_weights(rows, weighted=True):
    if not weighted:
        return np.ones(len(rows))
    records = rows[["crop", "partition_group", "label_status"]].to_dict("records")
    return exploratory.normalized_training_weights(records, CONFIDENCE_WEIGHTS)


def make_model(name):
    if name == "random_forest":
        return RandomForestClassifier(**FOREST_PARAMETERS)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED
        )
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    raise ValueError(f"Unknown model {name}.")


def fit_model(name, rows, columns, weighted=True):
    model = make_model(name)
    weights = training_weights(rows, weighted)
    features, labels = rows[columns].to_numpy(), rows["crop"].to_numpy()
    if name == "logistic_regression":
        model.fit(features, labels, logisticregression__sample_weight=weights)
    else:
        model.fit(features, labels, sample_weight=weights)
    return model


def class_probabilities(model, rows, columns):
    probabilities = model.predict_proba(rows[columns].to_numpy())
    order = [list(model.classes_).index(crop) for crop in CLASSES]
    return probabilities[:, order]


def evaluate(rows, probabilities):
    """Score pixels, and parcels by their mean predicted probability."""
    truth = rows["crop"].to_numpy()
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    precision, recall, scores, support = precision_recall_fscore_support(
        truth, predictions, labels=CLASSES, zero_division=0
    )
    parcel_probabilities = (
        pd.DataFrame(probabilities, columns=CLASSES)
        .assign(partition_group=rows["partition_group"].to_numpy(), crop=truth)
        .groupby("partition_group")
        .agg({**{crop: "mean" for crop in CLASSES}, "crop": "first"})
    )
    parcel_predictions = parcel_probabilities[CLASSES].idxmax(axis=1)
    return {
        "pixels": int(len(rows)),
        "parcels": int(len(parcel_probabilities)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(scores.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "parcel_accuracy": float(accuracy_score(parcel_probabilities["crop"], parcel_predictions)),
        "parcel_macro_f1": float(
            f1_score(
                parcel_probabilities["crop"], parcel_predictions, labels=CLASSES,
                average="macro", zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CLASSES).tolist(),
        "per_crop": {
            crop: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(scores[index]),
                "support": int(support[index]),
            }
            for index, crop in enumerate(CLASSES)
        },
    }


def cross_validate(development, designs, models, group_column="split_unit", weighted=True,
                   folds=CV_FOLDS):
    """Grouped, crop-stratified folds; whole parcels (or blocks) stay in one fold."""
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    records = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(development, development["crop"], development[group_column])
    ):
        train, validation = development.iloc[train_index], development.iloc[validation_index]
        for design in designs:
            columns = feature_experiments.feature_sets()[design]
            for name in models:
                model = fit_model(name, train, columns, weighted)
                metrics = evaluate(validation, class_probabilities(model, validation, columns))
                records.append(
                    {
                        "fold": fold,
                        "design": design,
                        "model": name,
                        "grouping": group_column,
                        "weighted": weighted,
                        **{
                            key: metrics[key]
                            for key in ("accuracy", "macro_f1", "balanced_accuracy",
                                        "parcel_accuracy", "parcel_macro_f1", "parcels")
                        },
                        **{f"recall_{crop}": metrics["per_crop"][crop]["recall"] for crop in CLASSES},
                    }
                )
    return pd.DataFrame(records)


def summarize_cv(results):
    metrics = ["macro_f1", "parcel_macro_f1", "accuracy", "balanced_accuracy"] + [
        f"recall_{crop}" for crop in CLASSES
    ]
    summary = results.groupby(["grouping", "weighted", "design", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index().sort_values("macro_f1_mean", ascending=False)


def select_configuration(summary):
    """Highest mean CV macro-F1, then fewer features; test scores never enter."""
    eligible = summary[(summary["grouping"] == "split_unit") & summary["weighted"]].copy()
    eligible["feature_count"] = eligible["design"].map(
        lambda design: len(feature_experiments.feature_sets()[design])
    )
    best = eligible.sort_values(
        ["macro_f1_mean", "feature_count", "design", "model"], ascending=[False, True, True, True]
    ).iloc[0]
    return {
        "design": best["design"],
        "model": best["model"],
        "cv_macro_f1_mean": float(best["macro_f1_mean"]),
        "cv_macro_f1_std": float(best["macro_f1_std"]),
        "selection_rule": "highest_grouped_cv_macro_f1_then_fewest_features",
    }


def bootstrap_interval(rows, probabilities, repetitions=1000, seed=SEED):
    """Resample whole parcels within each crop to bound test macro-F1."""
    generator = np.random.default_rng(seed)
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    frame = pd.DataFrame(
        {"group": rows["partition_group"].to_numpy(), "crop": rows["crop"].to_numpy(),
         "prediction": predictions}
    )
    members = {group: part for group, part in frame.groupby("group")}
    groups_by_crop = frame.groupby("crop")["group"].unique()
    scores = []
    for _ in range(repetitions):
        sample = pd.concat(
            [
                members[group]
                for crop in CLASSES
                for group in generator.choice(groups_by_crop[crop], len(groups_by_crop[crop]))
            ]
        )
        scores.append(
            f1_score(sample["crop"], sample["prediction"], labels=CLASSES, average="macro",
                     zero_division=0)
        )
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"macro_f1_low": float(low), "macro_f1_high": float(high), "repetitions": repetitions}


def final_test(data, selection, output_folder):
    """Fit the selected configuration on all development parcels and score the test once."""
    columns = feature_experiments.feature_sets()[selection["design"]]
    development = data[data["partition"] == "development"]
    test = data[data["partition"] == "test"]
    model = fit_model(selection["model"], development, columns)
    probabilities = class_probabilities(model, test, columns)
    metrics = evaluate(test, probabilities)
    metrics["bootstrap"] = bootstrap_interval(test, probabilities)
    metrics["by_domain"] = {
        domain: evaluate(test[mask], probabilities[mask])
        for domain in sorted(test["domain"].unique())
        for mask in [(test["domain"] == domain).to_numpy()]
    }
    output_folder = Path(output_folder)
    joblib.dump(
        {"model": model, "feature_columns": columns, "classes": CLASSES, "selection": selection},
        output_folder / "selected_model.joblib",
    )
    predictions = test[["sample_id", "domain", "region", "partition_group", "crop"]].copy()
    predictions["prediction"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    for index, crop in enumerate(CLASSES):
        predictions[f"probability_{crop}"] = probabilities[:, index]
    predictions.to_csv(output_folder / "test_predictions.csv", index=False)
    importance = permutation_importance(
        model, test[columns].to_numpy(), test["crop"].to_numpy(), scoring="f1_macro",
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
    importances = pd.DataFrame(
        {"feature": columns, "importance_mean": importance.importances_mean,
         "importance_std": importance.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importances.to_csv(output_folder / "permutation_importance.csv", index=False)
    return metrics, importances


def transfer_experiments(data, selection):
    """Train on one source family, test on the other; no rows are shared."""
    columns = feature_experiments.feature_sets()[selection["design"]]
    france = data["domain"].isin(["sen4agrinet", "francecrops"])
    directions = {
        "cropharvest_to_france": (~france, france),
        "france_to_cropharvest": (france, ~france),
    }
    results = {}
    for name, (train_mask, test_mask) in directions.items():
        train, test = data[train_mask], data[test_mask]
        model = fit_model(selection["model"], train, columns)
        probabilities = class_probabilities(model, test, columns)
        metrics = evaluate(test, probabilities)
        metrics["by_region"] = {
            region: evaluate(test[mask], probabilities[mask])
            for region in sorted(test["region"].unique())
            for mask in [(test["region"] == region).to_numpy()]
        }
        results[name] = metrics
    return results


def save_results(output_folder, **results):
    output_folder = Path(output_folder)
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_folder / f"{name}.csv", index=False)
        else:
            (output_folder / f"{name}.json").write_text(json.dumps(value, indent=2, default=str))
