"""Rebuild ``matched_geographic_feature_comparison.json`` for a training run.

Testing 03 reports a "prior matched development feature comparison": spatial out-of-fold
results within development training for the 72 interval, 48 shape and 120 combined
predictors. The original file was produced on another machine and its script is not in the
repository. This tool recomputes it for a local training run with the same procedure as
``crop_pipeline.training.geographic_validation``: five site-cell folds within the locked
training split, the saved Random Forest settings and weights, and parcel-level scoring.

    .venv/bin/python tools/rebuild_matched_feature_comparison.py pipeline_outputs/training/run_<id>
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from crop_pipeline.training import earth_engine, exploratory, source_holdout  # noqa: E402
from crop_pipeline.training.geographic_validation import spatial_folds  # noqa: E402
from crop_pipeline.windows.seasons import file_sha256  # noqa: E402

COHORTS = ("fixed_matched", "aligned_matched")
VARIANTS = ("interval_only", "shape_only", "interval_and_shape")


def cross_validate(rows, columns, leaf):
    classes = set(source_holdout.CROPS)
    predictions = []
    for fit_ids, held_ids in spatial_folds(rows):
        fit = [r for r in rows if r["partition_group"] in fit_ids]
        held = [r for r in rows if r["partition_group"] in held_ids]
        if {r["crop"] for r in fit} != classes:
            raise ValueError("A spatial training fold lacks a crop")
        weights = exploratory.normalized_training_weights(fit, {"confirmed": 1., "reported": .5})
        model = RandomForestClassifier(n_estimators=1000, random_state=42, max_features="sqrt",
                                       min_samples_leaf=leaf, max_samples=.5, n_jobs=-1)
        model.fit(exploratory.feature_matrix(fit, columns), [r["crop"] for r in fit], sample_weight=weights)
        _, fold_predictions = source_holdout.parcel_metrics(model, held, columns)
        predictions.extend(fold_predictions)
    return earth_engine.score_predictions(predictions, [r["prediction"] for r in predictions], source_holdout.CROPS)


def rebuild(training_run):
    training_run = Path(training_run)
    features = training_run / "features_development.json"
    development = json.loads((training_run / "validation/development.json").read_text())
    if file_sha256(features) != development["feature_sha256"]:
        raise ValueError("Development features do not match this training run")
    selection = json.loads((training_run / "validation/validation_candidates.json").read_text())
    data = json.loads(features.read_text())
    source_holdout.require_matching(data["rows"])
    sets = source_holdout.experiments.feature_sets()
    cohorts = {}
    for cohort in COHORTS:
        rows = source_holdout.checked_rows(data["rows"][cohort], False)
        validation = {r["partition_group"] for r in rows if r["partition"] == "validation"}
        rows = [r for r in rows if r["partition"] == "train"]
        if validation.intersection(r["partition_group"] for r in rows):
            raise ValueError("A parcel crosses the locked development partitions")
        leaf = selection[cohort]["selected_min_samples_leaf"]
        cohorts[cohort] = {}
        for variant in VARIANTS:
            pooled = cross_validate(rows, sets[variant], leaf)
            cohorts[cohort][variant] = {"predictors": len(sets[variant]), "min_samples_leaf": leaf, "pooled": pooled}
            print(f"{cohort} {variant}: spatial macro-F1 {pooled['macro_f1']:.3f}", flush=True)
    return {"feature_sha256": development["feature_sha256"], "block": "existing one-degree site_cell",
            "seed": 42, "rebuilt_by": "tools/rebuild_matched_feature_comparison.py", "cohorts": cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_run", type=Path)
    args = parser.parse_args()
    report = rebuild(args.training_run)
    with (args.training_run / "matched_geographic_feature_comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
