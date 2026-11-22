"""Paired external comparison of a challenger model with the frozen baseline models.

Both runs use the main testing layout (``reports/test_scopes.json``, per-scope
``test_scores.json`` and ``majority_baseline.json``, and
``predictions/<scope>/<model>_test_parcels.csv``). For each baseline model the challenger
is scored on the parcels that baseline model scored, with the parcel resampling of
``external_report``: seed, campaign-by-reference-crop strata, 95% percentile intervals.
Saved confusion matrices are checked against the prediction files before scoring.

    .venv/bin/python -m crop_pipeline.training.paired_comparison \
        pipeline_outputs/annual/test/run_<id> annual pipeline_outputs/testing/run_<id> \
        pipeline_outputs/reports/<new_folder> [--revised-scope]

The external parcels were scored for earlier models before this comparison was run, so
it is reported as retrospective evidence, not a fresh held-out test.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from crop_pipeline.training.external_report import (
    CROPS, DEFAULT_REPETITIONS, DEFAULT_SEED, F1_FLOOR, POPULATION_NAMES, REVISED_POPULATION_NAMES, REVISION_NOTE,
    confusion, interval, metric_values, parcel_draws, read_json, read_predictions, retained_reference,
    verify_saved, write_csv,
)

RETROSPECTIVE_NOTE = (
    "Retrospective comparison: these external parcels had already been scored for earlier models, "
    "so this is not a fresh held-out test."
)


def load_run(run, models=None):
    """Campaign predictions per model, checked against saved scores; returns (parcels, majority, campaigns)."""
    run = Path(run)
    reports, predictions = run / "reports", run / "predictions"
    scopes = read_json(reports / "test_scopes.json")
    if not scopes or scopes[0] != "jecam" or (len(scopes) > 1 and scopes[-1] != "combined"):
        raise ValueError("Expected JECAM first and the combined pool last")
    campaigns = scopes[:-1] if len(scopes) > 1 else scopes
    names = models or list(read_json(reports / "test_scores.json"))
    parcels, majority = {name: {} for name in names}, {}
    for scope in campaigns:
        folder = "" if scope == "jecam" else scope
        saved = read_json(reports / folder / "test_scores.json")
        baselines = read_json(reports / folder / "majority_baseline.json")
        for name in names:
            path = predictions / folder / f"{name}_test_parcels.csv"
            if not path.exists():
                continue
            rows = read_predictions(path, scope)
            if parcels[name].keys() & rows.keys():
                raise ValueError(f"Duplicate parcel across campaigns for {name}")
            matrix = confusion(np.array([CROPS.index(r[1]) for r in rows.values()], dtype=int),
                               np.array([CROPS.index(r[2]) for r in rows.values()], dtype=int))
            verify_saved(saved[name], matrix, metric_values(matrix), f"{run.name}/{scope}/{name}")
            parcels[name].update(rows)
            crop = baselines[name]["training_majority_class"]
            if majority.setdefault(name, crop) != crop:
                raise ValueError(f"Majority crop differs across campaigns for {name}")
    if len(scopes) > 1:
        for name in names:
            path = predictions / "combined" / f"{name}_test_parcels.csv"
            if path.exists() and read_predictions(path) != parcels[name]:
                raise ValueError(f"Saved combined predictions differ from the campaign pool for {name}")
    return parcels, majority, campaigns


def select(parcels, scope, revised):
    rows = {key: row for key, row in parcels.items() if scope == "combined" or row[0] == scope}
    if revised:
        rows = {key: row for key, row in rows.items() if retained_reference(*row[:2])}
    return rows


def paired(predictors, parcels, repetitions, rng):
    keys = sorted(parcels)
    rows = [parcels[key] for key in keys]
    actual = np.array([CROPS.index(row[1]) for row in rows], dtype=int)
    draws = parcel_draws(rows, repetitions, rng)
    points, samples = {}, {}
    for name, labels in predictors.items():
        labels = np.array([CROPS.index(labels[key]) for key in keys], dtype=int)
        points[name] = metric_values(confusion(actual, labels))
        samples[name] = metric_values(confusion(actual[draws], labels[draws]))
    return points, samples, {crop: int((actual == i).sum()) for i, crop in enumerate(CROPS)}


def difference(points, samples, first, second, metric=("macro_f1", "")):
    gains = samples[first][metric] - samples[second][metric]
    limits = interval(points[first][metric] - points[second][metric], gains)
    finite = gains[np.isfinite(gains)]
    limits["share_of_resamples_improved"] = float((finite > 0).mean()) if len(finite) else None
    return limits


def compare(challenger, challenger_majority, baseline, baseline_majority, campaigns,
            seed=DEFAULT_SEED, repetitions=DEFAULT_REPETITIONS, revised=False):
    """Paired macro-F1 rows and per-crop F1 rows for every population and baseline model."""
    rng = np.random.default_rng(seed)
    names = REVISED_POPULATION_NAMES if revised else POPULATION_NAMES
    scopes = [s for s in (*campaigns, *(["combined"] if len(campaigns) > 1 else [])) if s in names]
    summary, crops = [], []

    def record(scope, comparison, points, samples, support, predictors, base):
        for predictor in predictors:
            for crop in CROPS:
                if support[crop]:
                    limits = interval(points[predictor]["f1", crop], samples[predictor]["f1", crop])
                    crops.append({"population": scope, "comparison": comparison, "predictor": predictor,
                                  "crop": crop, "support": support[crop], **limits,
                                  "share_of_resamples_improved": None, "f1_below_floor": limits["estimate"] < F1_FLOOR})
        if base != predictors[0]:
            for crop in CROPS:
                if support[crop]:
                    crops.append({"population": scope, "comparison": comparison,
                                  "predictor": f"challenger minus {base}", "crop": crop, "support": support[crop],
                                  **difference(points, samples, predictors[0], base, ("f1", crop)),
                                  "f1_below_floor": None})

    for scope in scopes:
        own = select(challenger, scope, revised)
        if not own:
            continue
        labels = {"challenger": {k: r[2] for k, r in own.items()},
                  "challenger_majority": dict.fromkeys(own, challenger_majority)}
        points, samples, support = paired(labels, own, repetitions, rng)
        summary.append({"population": scope, "comparison": "challenger_own_coverage", "parcels": len(own),
                        "challenger_only_parcels": None,
                        "challenger_macro_f1": interval(points["challenger"]["macro_f1", ""],
                                                        samples["challenger"]["macro_f1", ""]),
                        "baseline_macro_f1": None, "difference": None,
                        "challenger_minus_majority": difference(points, samples, "challenger", "challenger_majority"),
                        "baseline_minus_majority": None})
        record(scope, "challenger_own_coverage", points, samples, support, ["challenger"], "challenger")
        for model, rows in baseline.items():
            theirs = select(rows, scope, revised)
            if not theirs:
                continue
            missing = theirs.keys() - own.keys()
            if missing:
                raise ValueError(f"Challenger lacks {len(missing)} {model} parcels in {scope}")
            if any(own[key][:2] != row[:2] for key, row in theirs.items()):
                raise ValueError(f"Campaign or reference crop differs for shared {model} parcels in {scope}")
            labels = {"challenger": {k: own[k][2] for k in theirs}, model: {k: r[2] for k, r in theirs.items()},
                      "challenger_majority": dict.fromkeys(theirs, challenger_majority),
                      "baseline_majority": dict.fromkeys(theirs, baseline_majority[model])}
            points, samples, support = paired(labels, theirs, repetitions, rng)
            summary.append({
                "population": scope, "comparison": f"versus_{model}", "parcels": len(theirs),
                "challenger_only_parcels": len(own) - len(theirs),
                "challenger_macro_f1": interval(points["challenger"]["macro_f1", ""], samples["challenger"]["macro_f1", ""]),
                "baseline_macro_f1": interval(points[model]["macro_f1", ""], samples[model]["macro_f1", ""]),
                "difference": difference(points, samples, "challenger", model),
                "challenger_minus_majority": difference(points, samples, "challenger", "challenger_majority"),
                "baseline_minus_majority": difference(points, samples, model, "baseline_majority"),
            })
            record(scope, f"versus_{model}", points, samples, support, ["challenger", model], model)
    return summary, crops


INTERVALS = {"challenger_macro_f1": False, "baseline_macro_f1": False, "difference": True,
             "challenger_minus_majority": True, "baseline_minus_majority": True}


def flatten(rows):
    flat = []
    for row in rows:
        out = {key: value for key, value in row.items() if key not in INTERVALS}
        for key, paired_difference in INTERVALS.items():
            value = row[key] or {}
            for part in ("estimate", "ci_low", "ci_high", "valid_replicates") + (
                    ("share_of_resamples_improved",) if paired_difference else ()):
                out[f"{key}_{part}"] = value.get(part)
        flat.append(out)
    return flat


def render(metadata, summary, crops):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    def ci(value):
        if not value or value.get("estimate") is None:
            return "—"
        return f"{number(value['estimate'])} [{number(value['ci_low'])}, {number(value['ci_high'])}]"

    def signed(value):
        if not value or value.get("estimate") is None:
            return "—"
        return f"{value['estimate']:+.3f} [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]"

    names = REVISED_POPULATION_NAMES if metadata["revised_scope"] else POPULATION_NAMES
    challenger = metadata["challenger_model"]
    lines = [f"# {challenger} versus the baseline models", "",
             f"Challenger `{challenger}` from `{metadata['challenger_run']}`; baseline models from "
             f"`{metadata['baseline_run']}`. Each comparison uses only the parcels that baseline model scored. "
             "Both predictions and both training-majority baselines use the same resampled parcels: "
             f"{metadata['repetitions']:,} repetitions, seed {metadata['seed']}, sampling with replacement within "
             "campaign and reference crop, 95% percentile intervals.", "", RETROSPECTIVE_NOTE, "",
             "A positive difference means the challenger scores higher. Three-class macro-F1 is unavailable for "
             "single-crop campaigns; their crop F1 is listed below and in `per_crop_f1.csv`."]
    if metadata["revised_scope"]:
        lines += ["", REVISION_NOTE]
    for scope in dict.fromkeys(row["population"] for row in summary):
        rows = [row for row in summary if row["population"] == scope]
        if all(row["challenger_macro_f1"]["estimate"] is None for row in rows):
            continue
        lines += ["", f"## {names.get(scope, scope)}", "",
                  "| Comparison | Parcels | Challenger macro-F1 | Baseline macro-F1 | Challenger − baseline | "
                  "Share of resamples improved | Challenger − its majority |",
                  "| --- | ---: | --- | --- | --- | ---: | --- |"]
        for row in rows:
            share = (row["difference"] or {}).get("share_of_resamples_improved")
            lines.append("| " + " | ".join([
                row["comparison"].replace("versus_", "vs ").replace("challenger_", ""), str(row["parcels"]),
                ci(row["challenger_macro_f1"]), ci(row["baseline_macro_f1"]), signed(row["difference"]),
                "—" if share is None else f"{share:.1%}", signed(row["challenger_minus_majority"])]) + " |")
    index = {(r["population"], r["comparison"], r["predictor"], r["crop"]): r for r in crops}
    single = [r for r in crops if r["predictor"] == "challenger" and r["population"] not in ("jecam", "combined")
              and r["population"] in names]
    if single:
        lines += ["", "## Single-crop campaigns (crop F1)", "",
                  "| Campaign | Comparison | Crop | Parcels | Challenger F1 | Challenger − baseline F1 |",
                  "| --- | --- | --- | ---: | --- | --- |"]
        for row in single:
            model = row["comparison"].replace("versus_", "")
            diff = index.get((row["population"], row["comparison"], f"challenger minus {model}", row["crop"]))
            lines.append("| " + " | ".join([names[row["population"]],
                                              row["comparison"].replace("versus_", "vs ").replace("challenger_", ""),
                                              row["crop"], str(row["support"]), ci(row), signed(diff)]) + " |")
    return "\n".join(lines) + "\n"


def report(challenger_run, challenger_model, baseline_run, output, baseline_models=None,
           seed=DEFAULT_SEED, repetitions=DEFAULT_REPETITIONS, revised=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    if repetitions < 2:
        raise ValueError("Use at least two bootstrap repetitions")
    challenger, challenger_majority, _ = load_run(challenger_run, [challenger_model])
    baseline, baseline_majority, campaigns = load_run(baseline_run, baseline_models)
    if Path(challenger_run).resolve() == Path(baseline_run).resolve():
        baseline.pop(challenger_model, None)  # never compare a model with itself
    summary, crops = compare(challenger[challenger_model], challenger_majority[challenger_model], baseline,
                             baseline_majority, campaigns, seed=seed, repetitions=repetitions, revised=revised)
    metadata = {"challenger_run": Path(challenger_run).resolve().name, "challenger_model": challenger_model,
                "challenger_majority_crop": challenger_majority[challenger_model],
                "baseline_run": Path(baseline_run).resolve().name, "baseline_majority_crops": baseline_majority,
                "seed": seed, "repetitions": repetitions, "sampling_unit": "partition_group",
                "sampling": "With replacement within campaign and reference-crop strata",
                "interval_method": "Percentile, 2.5th and 97.5th quantiles of finite replicates",
                "revised_scope": revised, "retrospective": RETROSPECTIVE_NOTE, "numpy_version": np.__version__}
    output.mkdir(parents=True)
    write_csv(output / "paired_macro_f1.csv", flatten(summary))
    write_csv(output / "per_crop_f1.csv", crops)
    (output / "methods.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(metadata, summary, crops), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("challenger_run", type=Path)
    parser.add_argument("challenger_model")
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-models", nargs="+")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--revised-scope", action="store_true", help="Use the revised reporting scope")
    args = parser.parse_args()
    print(report(args.challenger_run, args.challenger_model, args.baseline_run, args.output, args.baseline_models,
                 args.seed, args.repetitions, args.revised_scope))


if __name__ == "__main__":
    main()
