"""Rebuild ``shape_external_comparison/`` for a local training and testing run.

Testing 03's three-way comparison reuses archived 120-predictor (interval plus shape)
forests, fitted on another machine and stored beside the training run, and checks its own
scores against their archived external predictions. That archive and its script are not in
the repository. This tool refits those forests for a local run, with the frozen matched
baseline's Random Forest settings, weights and locked split, saves them in the same package
layout, and scores them once on the run's external test features.

Because the archive is rebuilt here, Testing 03's archive checks then confirm only that the
rebuild is internally consistent, not agreement with the original archive.

    .venv/bin/python tools/rebuild_shape_archive.py pipeline_outputs/training/run_<id> pipeline_outputs/testing/run_<id>
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from crop_pipeline.common import save_json  # noqa: E402
from crop_pipeline.training import earth_engine, model_comparisons, notebook_workflow, source_holdout  # noqa: E402
from crop_pipeline.windows.seasons import file_sha256  # noqa: E402

COHORTS = model_comparisons.COHORTS
DESIGN = "interval_and_shape"


def package(folder, baseline_dir, classifier, columns, validation):
    """Save a model package in the Training 02 layout with the combined design."""
    folder.mkdir(parents=True)
    artifact = joblib.load(baseline_dir / "model.joblib")
    joblib.dump({"classifier": classifier, "feature_columns": columns,
                 "configuration": artifact["configuration"]}, folder / "model.joblib")
    recipe = json.loads((baseline_dir / "recipe.json").read_text())
    recipe.update(feature_columns=columns, selected_design=DESIGN,
                  selection={"status": "selected_on_validation", "selected_design": DESIGN})
    save_json(folder / "recipe.json", recipe)
    manifest = json.loads((baseline_dir / "model_manifest.json").read_text())
    model_sha = file_sha256(folder / "model.joblib")
    manifest.update(files={"model.joblib": {"sha256": model_sha}, "recipe.json": {"sha256": file_sha256(folder / "recipe.json")}},
                    identity={"recipe": recipe, "model_sha256": model_sha}, validation=validation, model_id=model_sha,
                    rebuilt_by="tools/rebuild_shape_archive.py")
    save_json(folder / "model_manifest.json", manifest)
    return folder / "model_manifest.json"


def rebuild(training_run, testing_run):
    training_run, testing_run = Path(training_run), Path(testing_run)
    archive = training_run / "shape_external_comparison"
    if archive.exists():
        raise FileExistsError(f"{archive} already exists; it is never overwritten")
    notebook_workflow._verify(training_run / "models/models_stage.json", "models_stage")
    score_stage, _ = notebook_workflow._verify(testing_run / "scores.json", "scores")
    development = json.loads((training_run / "features_development.json").read_text())
    test_path = score_stage.parent / "features_test.json"
    test = json.loads(test_path.read_text())
    source_holdout.require_matching(development["rows"])
    source_holdout.require_matching(test["rows"])
    columns = source_holdout.experiments.feature_sets()[DESIGN]
    models, scopes = {}, {}
    for cohort in COHORTS:
        baseline_dir = training_run / "models" / cohort
        baseline, _ = model_comparisons.load_classifier(baseline_dir / "model_manifest.json")
        rows = source_holdout.checked_rows(development["rows"][cohort], False)
        train = [r for r in rows if r["partition"] == "train"]
        validation = [r for r in rows if r["partition"] == "validation"]
        model = model_comparisons.fit(baseline, train, columns)
        metrics, _ = source_holdout.parcel_metrics(model, validation, columns)
        manifest = package(archive / "frozen" / cohort, baseline_dir, model, columns, metrics)
        models[cohort] = {"sha256": file_sha256(manifest)}
        test_rows = source_holdout.checked_rows(test["rows"][cohort], True)
        campaign = {r["partition_group"]: r["test_cohort"] for r in test_rows}
        _, predicted = source_holdout.parcel_metrics(model, test_rows, columns)
        for row in predicted:
            row["test_cohort"] = campaign[row["partition_group"]]
        scopes[cohort] = earth_engine.score_predictions(predicted, [p["prediction"] for p in predicted],
                                                        model_comparisons.CROPS)
        out = archive / "combined" / "shape" / f"{cohort}_test_parcels.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["partition_group", "test_cohort", "crop", "prediction"])
            writer.writeheader()
            writer.writerows({k: p[k] for k in writer.fieldnames} for p in predicted)
        print(f"{cohort}: validation macro-F1 {metrics['macro_f1']:.3f}, "
              f"external macro-F1 {scopes[cohort].get('macro_f1')}", flush=True)
    save_json(archive / "shape_models.json", models)
    save_json(archive / "external_comparison.json", {
        "test_sha256": file_sha256(test_path), "design": DESIGN, "rebuilt_by": "tools/rebuild_shape_archive.py",
        "scopes": {"combined": {"shape": scopes}}})
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("testing_run", type=Path)
    args = parser.parse_args()
    print("Archive:", rebuild(args.training_run, args.testing_run))


if __name__ == "__main__":
    main()
