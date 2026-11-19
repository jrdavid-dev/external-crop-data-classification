"""Run spatial cross-validation inside the locked development training split."""

import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from crop_pipeline.training import earth_engine, exploratory, source_holdout
from crop_pipeline.windows.seasons import file_sha256


def spatial_folds(rows):
    """Assign whole parcels and their site cells together across five folds."""
    parcels = {}
    parents = {}

    def root(cell):
        while parents[cell] != cell:
            cell = parents[cell]
        return cell
    for row in rows:
        group = row["partition_group"]
        site = row["site_cell"]
        lat, lon = map(int, site.split(","))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Invalid site cell: {site}")
        cell = (lat, lon)
        parents.setdefault(cell, cell)
        crop, first_cell = parcels.setdefault(group, (row["crop"], cell))
        if crop != row["crop"]:
            raise ValueError(f"Inconsistent crop for {group}")
        # Merge cells linked by a parcel split across archive fragments.
        parents[root(cell)] = root(first_cell)
    groups = sorted(parcels)
    crops = [parcels[group][0] for group in groups]
    components = [root(parcels[group][1]) for group in groups]
    cell_ids = {cell: index for index, cell in enumerate(sorted(set(components)))}
    spatial_groups = [cell_ids[cell] for cell in components]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train, held in splitter.split(groups, crops, spatial_groups):
        yield {groups[i] for i in train}, {groups[i] for i in held}


def evaluate(features, model_dir):
    data = json.loads(features.read_text())
    source_holdout.require_matching(data["rows"])
    columns = source_holdout.experiments.feature_sets()[source_holdout.DESIGN]
    classes = set(source_holdout.CROPS)
    results = {}
    selection = json.loads((model_dir / "validation_candidates.json").read_text())
    for cohort, rows in data["rows"].items():
        rows = source_holdout.checked_rows(rows, False)
        locked_validation = {r["partition_group"] for r in rows if r["partition"] == "validation"}
        rows = [r for r in rows if r["partition"] == "train"]
        if locked_validation.intersection(r["partition_group"] for r in rows):
            raise ValueError("A parcel crosses the locked development partitions")
        manifest = json.loads((model_dir / cohort / "model_manifest.json").read_text())
        # Read hyperparameters from the saved selection receipt, without loading joblib.
        leaf = selection[cohort]["selected_min_samples_leaf"]
        predictions = []
        folds = []
        for fit_ids, held_ids in spatial_folds(rows):
            fit = [r for r in rows if r["partition_group"] in fit_ids]
            held = [r for r in rows if r["partition_group"] in held_ids]
            if {r["crop"] for r in fit} != classes:
                raise ValueError(f"{cohort}: a spatial training fold lacks a crop")
            weights = exploratory.normalized_training_weights(fit, {"confirmed": 1., "reported": .5})
            model = RandomForestClassifier(n_estimators=1000, random_state=42, max_features="sqrt",
                                           min_samples_leaf=leaf, max_samples=.5, n_jobs=-1)
            model.fit(exploratory.feature_matrix(fit, columns), [r["crop"] for r in fit],
                      sample_weight=weights)
            metrics, fold_predictions = source_holdout.parcel_metrics(model, held, columns)
            predictions.extend(fold_predictions)
            folds.append({"metrics": metrics, "predictions": fold_predictions,
                          "training_parcels": len(fit_ids)})
        results[cohort] = {
            "min_samples_leaf": leaf,
            "locked_validation_macro_f1": manifest["validation"]["macro_f1"],
            "pooled": earth_engine.score_predictions(
                predictions, [r["prediction"] for r in predictions], source_holdout.CROPS
            ),
            "folds": folds,
        }
        print(f"{cohort}: spatial macro-F1 {results[cohort]['pooled']['macro_f1']:.3f}", flush=True)
    return {"feature_sha256": file_sha256(features), "predictor_count": len(columns),
            "block": "existing one-degree site_cell", "seed": 42, "cohorts": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    digest = json.loads((args.validation_dir / "development.json").read_text())["feature_sha256"]
    if file_sha256(args.features) != digest:
        parser.error("Feature snapshot does not match the original validation run")
    report = evaluate(args.features, args.validation_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
