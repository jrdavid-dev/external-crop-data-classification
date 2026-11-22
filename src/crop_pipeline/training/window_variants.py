"""Development-only re-check of the window-model follow-ups from the external study.

For each window cohort of a frozen training run, scores the original 72 interval
predictors and the follow-up variants on the shared development views
(``development_views``): interval-plus-shape and shape-only predictors, parcel-median
rows, one-setting Random Forest changes, and (matched cohorts) the 50/50 fixed and
aligned probability ensemble. Only the run's development snapshot is read, so the
evidence for adopting a variant never comes from test parcels.

    .venv/bin/python -m crop_pipeline.training.window_variants \
        pipeline_outputs/training/run_<id> pipeline_outputs/training/window_variants_<label>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.training import development_views, exploratory, notebook_workflow, source_holdout
from crop_pipeline.training.model_followups import TUNING_CANDIDATES
from crop_pipeline.training.transfer_followups import parcel_medians
from crop_pipeline.windows.seasons import file_sha256

COHORTS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")


def variants():
    """``{name: (feature columns, forest settings, parcel medians)}``; ``original`` comes first."""
    sets = experiments.feature_sets()
    interval = sets["interval_only"]
    result = {"original": (interval, {}, False),
              "interval_and_shape": (sets["interval_and_shape"], {}, False),
              "shape_only": (sets["shape_only"], {}, False),
              "parcel_median": (interval, {}, True)}
    for name, params in TUNING_CANDIDATES.items():
        if name != "original":
            result[f"rf_{name}"] = (interval, params, False)
    return result


def cohort_views(rows, columns, params, medians, seed=42):
    rows = parcel_medians(rows, columns) if medians else rows
    X = exploratory.feature_matrix(rows, columns)
    if not np.isfinite(X).all():
        raise ValueError("Development predictors must be finite")
    return development_views.view_probabilities(X, rows, {**development_views.SELECTION_FOREST, **params}, seed)


def evaluate(training_run, output, cohorts=COHORTS, chosen=None, progress=print):
    training_run, output = Path(training_run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(training_run):
        raise ValueError("Use a new output directory outside the frozen training run")
    receipt, _ = notebook_workflow._verify(training_run / "training.json", "training")
    features = training_run / "features_development.json"
    data = json.loads(features.read_text())
    source_holdout.require_matching(data["rows"])
    specs = {name: spec for name, spec in variants().items() if chosen is None or name in chosen}
    results, matched = [], {}
    for cohort in cohorts:
        rows = source_holdout.checked_rows(data["rows"][cohort], False)
        missing = {c for columns, _, _ in specs.values() for c in columns} - set(rows[0])
        if missing:
            raise ValueError(f"{cohort}: development rows lack {len(missing)} predictor columns")
        for name, (columns, params, medians) in specs.items():
            views = cohort_views(rows, columns, params, medians)
            if name == "original" and cohort.endswith("_matched"):
                matched[cohort] = views
            results.append({"cohort": cohort, "variant": name, "predictors": len(columns),
                            "parcel_medians": medians, "forest_changes": params,
                            **development_views.summarize(views)})
            if progress:
                progress(f"{cohort} {name}: selection {results[-1]['selection_score']:.3f}")
    if len(matched) == 2:
        views = development_views.average(matched["fixed_matched"], matched["aligned_matched"])
        results.append({"cohort": "matched_ensemble", "variant": "fixed_aligned_50_50", "predictors": 144,
                        "parcel_medians": False, "forest_changes": {}, **development_views.summarize(views)})
    output.mkdir(parents=True)
    save_json(output / "window_variants.json", {
        "training_run": training_run.name, "training_receipt_sha256": file_sha256(receipt),
        "development_features_sha256": file_sha256(features), "test_rows_read": False,
        "views": list(development_views.VIEWS), "forest": development_views.SELECTION_FOREST,
        "selection_score": "mean of spatial CV macro-F1 and dataset-holdout macro-F1", "results": results})
    return output / "window_variants.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(evaluate(args.training_run, args.output))


if __name__ == "__main__":
    main()
