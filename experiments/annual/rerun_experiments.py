"""Rerun the annual-family experiments on the current inputs: select, freeze, test and map each.

Every experiment uses the same annual inputs (current ``pipeline_outputs/annual/current_inputs.json``)
and the annual workflow's own stages, so each run has sealed receipts:

- recipes: the annual recipe (A, Sentinel-2 + Sentinel-1), Sentinel-2 only, the curve-shape
  recipes (B, B+radar, C), the green-up recipes and the short-cycle radar recipes;
- classifiers on recipe A: Random Forest tuning, weighted logistic regression and gradient
  boosting, each tuned by spatial CV in the training split (the external study's rule).

Development scores use the pixels usable by every recipe. Each model is then frozen on the
locked training split, scored once on the external tests, and mapped over Paoay and the Atok
boundary. The annual stage pointers are saved first and restored afterwards, so the annual
recipe stays the current annual model.

    .venv/bin/python -m experiments.annual.rerun_experiments pipeline_outputs/experiments/<new_folder>
"""

import argparse
import json
from pathlib import Path

from crop_pipeline.common import save_json
from experiments.annual import development_views, maps, workflow as aw
from experiments.classifiers import classifiers
from experiments.curve_shape import shape  # noqa: F401  (registers the curve-shape recipes)
from experiments.greenup import greenup  # noqa: F401  (registers the green-up recipes)
from experiments.short_cycle import short_cycle

RECIPES = {
    "A annual S2+S1": "annual_s2_s1",
    "S2 only": "annual_s2",
    "curve shape B": "annual_shape",
    "curve shape B+radar": "annual_shape_s1",
    "curve shape C": "annual_s2_s1_shape",
    "green-up timing": "annual_s2_s1_greenup_timing",
    "green-up referenced": "annual_s2_s1_greenup_referenced",
    "short-cycle radar": "s1_cycle",
    "short-cycle radar + S2": "s1_cycle_plus_s2_harmonic",
}
CLASSIFIER_FAMILIES = {"random_forest": "RF tuned", "logistic_regression": "logistic regression",
                       "gradient_boosting": "gradient boosting"}
POINTERS = ("selection", "models", "test", "paoay", "atok")


def register():
    aw.register_recipes({k: v for k, v in short_cycle.RECIPES.items() if k not in aw.RECIPES})


def external_scores(test_run):
    scores = {}
    for scope in json.loads((test_run / "reports/test_scopes.json").read_text()):
        folder = test_run / "reports" / ("" if scope == "jecam" else scope)
        scores[scope] = json.loads((folder / "test_scores.json").read_text())["annual"]
    return scores


def run_one(root, inputs, label, recipe, development, classifier=None, estimator=None, progress=print):
    selection = aw._new_run(root, "selection")
    save_json(selection / "selection.json", {
        "selected": recipe, "classifier": classifier or aw.DEFAULT_CLASSIFIER,
        "classifier_rule": "default" if classifier is None else "spatial CV in the training split",
        "selection_score": development["selection_score"], "rule": f"experiment rerun: {label}",
        "forest": development_views.SELECTION_FOREST, "candidates": {recipe: development},
        "test_rows_used": False})
    receipt = aw.seal(selection, "selection", inputs_handoff=str(inputs))
    models = aw.freeze(root, receipt, estimator=estimator)
    test_run = aw.test(root, models).parent
    mapped = {}
    for area in ("paoay", "atok"):
        map_run = maps.map_area(root, area, models_handoff=models).parent
        mapped[area] = {"run": str(map_run), **json.loads((map_run / "map_summary.json").read_text())}
    model_info = json.loads((models.parent / "model.json").read_text())
    if progress:
        progress(f"{label}: frozen {models.parent.name}, tested {test_run.name}, mapped")
    return {"recipe": recipe, "classifier": classifier or aw.DEFAULT_CLASSIFIER, "development": development,
            "models_run": str(models.parent), "locked_validation": model_info["validation"],
            "test_run": str(test_run), "external": external_scores(test_run), "maps": mapped}


def rerun(root, output, progress=print):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    register()
    annual_root = root / "pipeline_outputs" / aw.AREA
    saved = {stage: (annual_root / f"current_{stage}.json").read_bytes() for stage in POINTERS}
    try:
        inputs, _ = aw._input(root, None, "inputs")
        rows, arrays = aw.load_inputs(inputs.parent)
        built = {recipe: aw.recipe_matrix(recipe, arrays) for recipe in RECIPES.values()}
        development = aw.development_candidates(rows, arrays, built=built, progress=progress)
        index = aw.development_index(rows, built)
        results = {"inputs_handoff": str(inputs), "development_pixels": int(len(index)), "experiments": {}}
        for label, recipe in RECIPES.items():
            results["experiments"][label] = run_one(root, inputs, label, recipe, development[recipe], progress=progress)
            save_json(output / "experiments.json", results)
        X = built["annual_s2_s1"][0]
        tuning, compared = classifiers.compare(X[index], [rows[i] for i in index], progress=progress)
        results["classifier_tuning"] = tuning
        for family, label in CLASSIFIER_FAMILIES.items():
            setting = compared[family]["setting"]
            results["experiments"][f"{label} ({setting})"] = run_one(
                root, inputs, label, "annual_s2_s1", compared[family],
                classifier={"family": family, "setting": setting},
                estimator=classifiers.estimator(family, setting, final=True), progress=progress)
            save_json(output / "experiments.json", results)
    finally:
        for stage, content in saved.items():
            (annual_root / f"current_{stage}.json").write_bytes(content)
    return output / "experiments.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print("Summary:", rerun(root, args.output))


if __name__ == "__main__":
    main()
