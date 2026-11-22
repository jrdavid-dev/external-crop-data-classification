"""Score Jose's window models, his comparison variants and the annual experiment on shared parcels.

Reads saved parcel predictions only: the Testing 01 run's frozen window models, the Testing 03
and Training 03 comparison folders beside it (``<run>_batch2/``), and an annual test run. Every
model is scored on the parcels that all of them predicted, within the revised reporting scope,
with the parcel bootstrap of the paired comparison (resampling within campaign and reference
crop). Differences are each model minus the 72-predictor fixed matched baseline.

The external parcels were already examined for the window models, so this is retrospective.

    .venv/bin/python -m experiments.annual.compare_models pipeline_outputs/testing/run_<id> \\
        pipeline_outputs/annual/test/run_<id> pipeline_outputs/reports/<new_folder>
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.training.external_report import read_predictions
from experiments.annual import paired_comparison as pc

BASELINE = "fixed_matched / interval_only (72)"
COMPARISONS = {  # folder beside the testing run -> models it holds
    "": ("interval_only", "single_composite", "spectral_bands_only"),
    "shape_comparison": ("shape_only", "interval_and_shape"),
    "indices_only_comparison": ("indices_only",),
    "tuned_comparison": ("tuned",),
}
PREDICTORS = {"interval_only": 72, "single_composite": 9, "spectral_bands_only": 40, "shape_only": 48,
              "interval_and_shape": 120, "indices_only": 32, "tuned": 72}
POPULATIONS = ("jecam", "combined")


def load(testing_run, annual_run):
    """``{model name: {parcel: (campaign, crop, prediction)}}`` from saved prediction files."""
    testing_run, annual_run = Path(testing_run), Path(annual_run)
    models = {}
    for cohort in ("fixed_full", "aligned_full"):
        models[f"{cohort} (72)"] = read_predictions(testing_run / "predictions/combined" / f"{cohort}_test_parcels.csv")
    batch = testing_run.parent / f"{testing_run.name}_batch2"
    for folder, variants in COMPARISONS.items():
        with (batch / folder / "external_predictions.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["model"] in variants:
                    name = f"{row['cohort']} / {row['model']} ({PREDICTORS[row['model']]})"
                    models.setdefault(name, {})[row["partition_group"]] = (row["test_cohort"], row["crop"], row["prediction"])
    models["annual experiment (S2 + S1)"] = read_predictions(annual_run / "predictions/combined/annual_test_parcels.csv")
    return models


def shared(models, population):
    """Parcels every model predicted in the revised scope; the references must agree."""
    selected = {name: pc.select(rows, population, True) for name, rows in models.items()}
    keys = set.intersection(*(set(rows) for rows in selected.values()))
    for key in keys:
        if len({rows[key][:2] for rows in selected.values()}) != 1:
            raise ValueError(f"{key}: models disagree on the reference campaign or crop")
    first = next(iter(selected.values()))
    return {key: first[key] for key in keys}, {name: {k: rows[k][2] for k in keys} for name, rows in selected.items()}


def compare(models, repetitions=10_000, seed=42):
    rows = []
    for population in POPULATIONS:
        parcels, predictions = shared(models, population)
        points, samples, support = pc.paired(predictions, parcels, repetitions, np.random.default_rng(seed))
        for name in predictions:
            estimate = pc.interval(points[name][("macro_f1", "")], samples[name][("macro_f1", "")])
            gain = pc.difference(points, samples, name, BASELINE) if name != BASELINE else {}
            rows.append({"population": population, "model": name, "parcels": len(parcels),
                         **{f"{crop}_parcels": support[crop] for crop in pc.CROPS},
                         "macro_f1": estimate["estimate"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
                         **{f"{crop}_f1": points[name][("f1", crop)] for crop in pc.CROPS},
                         "minus_baseline": gain.get("estimate"), "minus_baseline_ci_low": gain.get("ci_low"),
                         "minus_baseline_ci_high": gain.get("ci_high"),
                         "share_of_resamples_improved": gain.get("share_of_resamples_improved")})
    return rows


def render(rows):
    def f(value):
        return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.3f}"

    lines = ["# Window models, comparison variants and the annual experiment", "",
             pc.RETROSPECTIVE_NOTE, "",
             f"Every model is scored on the same parcels. Differences are against {BASELINE}; "
             "95% parcel-bootstrap intervals (10,000 resamples, seed 42, within campaign and crop)."]
    for population in POPULATIONS:
        subset = sorted((r for r in rows if r["population"] == population), key=lambda r: -r["macro_f1"])
        lines += ["", f"## {'JECAM (primary test)' if population == 'jecam' else 'Combined external pool'} "
                      f"({subset[0]['parcels']} parcels)", "",
                  "| Model | Macro-F1 | Carrot F1 | Cabbage F1 | Potato F1 | Minus baseline | Resamples improved |",
                  "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
        for r in subset:
            gain = "—" if r["minus_baseline"] is None else (
                f"{r['minus_baseline']:+.3f} [{r['minus_baseline_ci_low']:+.3f}, {r['minus_baseline_ci_high']:+.3f}]")
            share = "—" if r["share_of_resamples_improved"] is None else f"{100 * r['share_of_resamples_improved']:.1f}%"
            lines.append(f"| {r['model']} | {f(r['macro_f1'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] | "
                         f"{f(r['carrot_f1'])} | {f(r['cabbage_f1'])} | {f(r['potato_f1'])} | {gain} | {share} |")
    return "\n".join(lines) + "\n"


def report(testing_run, annual_run, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = compare(load(testing_run, annual_run))
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(render(rows), encoding="utf-8")
    save_json(output / "inputs.json", {"testing_run": str(Path(testing_run).resolve()),
                                       "annual_run": str(Path(annual_run).resolve()), "baseline": BASELINE})
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("testing_run", type=Path)
    parser.add_argument("annual_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(report(args.testing_run, args.annual_run, args.output).read_text())


if __name__ == "__main__":
    main()
