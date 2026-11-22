"""Experiments A, B, B+radar and C: does the external study's curve-shape description help the annual forest?

Only the feature representation changes; the Random Forest, its settings, training weights,
pixels and parcels are the same for every recipe:

- A ``annual_s2_s1`` — standard annual harmonic Sentinel-2 + Sentinel-1 (the primary model)
- B ``annual_shape`` — the study's curve-shape descriptors only
- B+radar ``annual_shape_s1`` — curve shape plus the same Sentinel-1 terms as A
- C ``annual_s2_s1_shape`` — A plus curve shape

Each recipe is scored on the development views (pixels usable by every recipe), then frozen on
the training split and tested once through the annual workflow's own stages, so every run has
receipts. Paired comparisons are against A on the same parcels.
"""

import json
from pathlib import Path

from crop_pipeline.common import save_json
from crop_pipeline.training import annual_workflow as aw, development_views, paired_comparison
from experiments.curve_shape import shape  # noqa: F401  (registers the curve-shape recipes)

RECIPES = {"A": "annual_s2_s1", "B": "annual_shape", "B+radar": "annual_shape_s1", "C": "annual_s2_s1_shape"}


def run(root, output, inputs_handoff=None, progress=print):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a new output folder")
    inputs, _ = aw._input(root, inputs_handoff, "inputs")
    rows, arrays = aw.load_inputs(inputs.parent)
    built = {name: aw.recipe_matrix(name, arrays) for name in RECIPES.values()}
    development = aw.development_candidates(rows, arrays, built=built, progress=progress)
    output.mkdir(parents=True)
    tests = {}
    for label, recipe in RECIPES.items():
        selection = aw._new_run(root, "selection")
        save_json(selection / "selection.json", {
            "selected": recipe, "classifier": {"family": "random_forest", "setting": "original"},
            "classifier_rule": "default", "selection_score": development[recipe]["selection_score"],
            "rule": f"experiment {label}: recipe fixed in advance, default Random Forest",
            "forest": development_views.SELECTION_FOREST, "candidates": {recipe: development[recipe]},
            "test_rows_used": False})
        receipt = aw.seal(selection, "selection", inputs_handoff=str(inputs))
        models = aw.freeze(root, receipt)
        tests[label] = aw.test(root, models).parent
        if progress:
            progress(f"{label} ({recipe}) frozen and tested: {tests[label].name}")
    for label in ("B", "B+radar", "C"):
        for revised in (False, True):
            paired_comparison.report(tests[label], "annual", tests["A"],
                                     output / f"paired_{label.replace('+', '_')}_vs_A{'_revised' if revised else ''}",
                                     ["annual"], revised=revised)
    summary = {"inputs_handoff": str(inputs), "recipes": RECIPES, "test_rows_used_for_selection": False,
               "development": development, "test_runs": {k: str(v) for k, v in tests.items()}, "external": {}}
    for label, run_folder in tests.items():
        scores = {}
        for scope in json.loads((run_folder / "reports/test_scopes.json").read_text()):
            folder = run_folder / "reports" / ("" if scope == "jecam" else scope)
            scores[scope] = json.loads((folder / "test_scores.json").read_text())["annual"]
        summary["external"][label] = scores
    save_json(output / "shape_experiment.json", summary)
    return output / "shape_experiment.json"
