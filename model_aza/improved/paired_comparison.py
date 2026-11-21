"""Paired external comparison: improved model versus the original frozen models.

Scores the improved pipeline's saved test parcels against each original model from a
main-pipeline testing run on the parcels both cover, with the same parcel resampling
as ``crop_pipeline.training.external_report`` (seed, strata, percentile intervals).
Each original model is compared on its own coverage; the improved model's own-coverage
scores are reported alongside. Read-only: nothing in either run is changed.

The external test parcels were already examined before this comparison was designed,
so the results are retrospective, not a fresh held-out test.

    .venv/bin/python -m model_aza.improved.paired_comparison \
        pipeline_outputs/improved/test/run_<id>/test_receipt.json \
        pipeline_outputs/testing/run_<id> pipeline_outputs/reports/<new_folder> [--revised-scope]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from crop_pipeline.training.external_report import (
    CROPS, DEFAULT_REPETITIONS, DEFAULT_SEED, F1_FLOOR, MODELS, POPULATION_NAMES, REVISED_POPULATION_NAMES,
    REVISION_NOTE, confusion, interval, metric_values, parcel_draws, read_json, read_predictions,
    retained_reference, verify_saved, write_csv,
)
from model_aza.improved import workflow

RETROSPECTIVE_NOTE = (
    "Retrospective comparison: the external test parcels had already been scored and inspected "
    "for both workflows before this comparison was defined, so it is not a fresh held-out test."
)


def load_improved(test_receipt):
    """Verified improved predictions ``{parcel: (campaign, reference, prediction)}`` and its metadata."""
    receipt, _ = workflow.verify(Path(test_receipt), "test")
    scores = read_json(receipt.parent / "test_scores.json")
    parcels = read_predictions(receipt.parent / "test_parcels.csv")
    return parcels, {"candidate": scores["candidate"], "majority": scores["training_majority_class"],
                     "run": receipt.parent.name}


def load_original(run):
    """Per-model campaign predictions from a main-pipeline testing run, checked against its saved scores."""
    run = Path(run)
    reports, predictions = run / "reports", run / "predictions"
    scopes = read_json(reports / "test_scopes.json")
    if not scopes or scopes[0] != "jecam" or scopes[-1] != "combined":
        raise ValueError("Expected JECAM first and the combined pool last")
    parcels, majority = {model: {} for model in MODELS}, {}
    for scope in scopes[:-1]:
        folder = "" if scope == "jecam" else scope
        saved = read_json(reports / folder / "test_scores.json")
        baselines = read_json(reports / folder / "majority_baseline.json")
        for model in MODELS:
            rows = read_predictions(predictions / folder / f"{model}_test_parcels.csv", scope)
            if parcels[model].keys() & rows.keys():
                raise ValueError(f"Duplicate parcel across campaigns for {model}")
            matrix = confusion(np.array([CROPS.index(r[1]) for r in rows.values()], dtype=int),
                               np.array([CROPS.index(r[2]) for r in rows.values()], dtype=int))
            verify_saved(saved[model], matrix, metric_values(matrix), f"{scope}/{model}")
            parcels[model].update(rows)
            crop = baselines[model]["training_majority_class"]
            if majority.setdefault(model, crop) != crop:
                raise ValueError(f"Majority crop differs across campaigns for {model}")
    combined = {model: read_predictions(predictions / "combined" / f"{model}_test_parcels.csv")
                for model in MODELS}
    if combined != parcels:
        raise ValueError("Saved combined predictions differ from the campaign pool")
    return parcels, majority, scopes[:-1]


def select(parcels, scope, revised):
    rows = {key: row for key, row in parcels.items() if scope == "combined" or row[0] == scope}
    if revised:
        rows = {key: row for key, row in rows.items() if retained_reference(*row[:2])}
    return rows


def paired(predictors, parcels, repetitions, rng):
    """Point metrics and bootstrap samples for several predictors on the same parcels and draws."""
    keys = sorted(parcels)
    rows = [parcels[key] for key in keys]
    actual = np.array([CROPS.index(row[1]) for row in rows], dtype=int)
    draws = parcel_draws(rows, repetitions, rng)
    points, samples = {}, {}
    for name, labels in predictors.items():
        labels = np.array([CROPS.index(labels[key]) for key in keys], dtype=int)
        points[name] = metric_values(confusion(actual, labels))
        samples[name] = metric_values(confusion(actual[draws], labels[draws]))
    support = {crop: int((actual == i).sum()) for i, crop in enumerate(CROPS)}
    return points, samples, support


def difference(points, samples, first, second, metric=("macro_f1", "")):
    limits = interval(points[first][metric] - points[second][metric],
                      samples[first][metric] - samples[second][metric])
    gains = samples[first][metric] - samples[second][metric]
    gains = gains[np.isfinite(gains)]
    limits["share_of_resamples_improved"] = float((gains > 0).mean()) if len(gains) else None
    return limits


def compare(improved, improved_majority, original, original_majority, campaigns,
            seed=DEFAULT_SEED, repetitions=DEFAULT_REPETITIONS, revised=False):
    """Return paired macro-F1 rows and per-crop F1 rows for every population and original model."""
    rng = np.random.default_rng(seed)
    names = REVISED_POPULATION_NAMES if revised else POPULATION_NAMES
    scopes = [scope for scope in (*campaigns, "combined") if scope in names]
    summary, crops = [], []

    def record(scope, comparison, points, samples, support, predictors, base):
        for predictor in predictors:
            for crop in CROPS:
                if support[crop]:
                    limits = interval(points[predictor]["f1", crop], samples[predictor]["f1", crop])
                    crops.append({"population": scope, "comparison": comparison, "predictor": predictor,
                                  "crop": crop, "support": support[crop], **limits,
                                  "share_of_resamples_improved": None,
                                  "f1_below_floor": limits["estimate"] < F1_FLOOR})
        for crop in CROPS:
            if support[crop] and base != predictors[0]:
                limits = difference(points, samples, predictors[0], base, ("f1", crop))
                crops.append({"population": scope, "comparison": comparison,
                              "predictor": f"{predictors[0]} minus {base}", "crop": crop,
                              "support": support[crop], **limits, "f1_below_floor": None})

    for scope in scopes:
        own = select(improved, scope, revised)
        if not own:
            continue
        labels = {"improved": {k: r[2] for k, r in own.items()},
                  "improved_majority": dict.fromkeys(own, improved_majority)}
        points, samples, support = paired(labels, own, repetitions, rng)
        vs_majority = difference(points, samples, "improved", "improved_majority")
        summary.append({"population": scope, "comparison": "improved_own_coverage", "parcels": len(own),
                        "improved_macro_f1": interval(points["improved"]["macro_f1", ""],
                                                      samples["improved"]["macro_f1", ""]),
                        "original_macro_f1": None, "difference": None, "improved_minus_majority": vs_majority,
                        "original_minus_majority": None, "improved_only_parcels": None})
        record(scope, "improved_own_coverage", points, samples, support, ["improved"], "improved")
        for model in MODELS:
            theirs = select(original[model], scope, revised)
            if not theirs:
                continue
            missing = theirs.keys() - own.keys()
            if missing:
                raise ValueError(f"Improved predictions lack {len(missing)} {model} parcels in {scope}")
            if any(own[key][:2] != row[:2] for key, row in theirs.items()):
                raise ValueError(f"Campaign or reference crop differs for shared {model} parcels in {scope}")
            labels = {"improved": {k: own[k][2] for k in theirs}, model: {k: r[2] for k, r in theirs.items()},
                      "improved_majority": dict.fromkeys(theirs, improved_majority),
                      "original_majority": dict.fromkeys(theirs, original_majority[model])}
            points, samples, support = paired(labels, theirs, repetitions, rng)
            summary.append({
                "population": scope, "comparison": f"versus_{model}", "parcels": len(theirs),
                "improved_macro_f1": interval(points["improved"]["macro_f1", ""], samples["improved"]["macro_f1", ""]),
                "original_macro_f1": interval(points[model]["macro_f1", ""], samples[model]["macro_f1", ""]),
                "difference": difference(points, samples, "improved", model),
                "improved_minus_majority": difference(points, samples, "improved", "improved_majority"),
                "original_minus_majority": difference(points, samples, model, "original_majority"),
                "improved_only_parcels": len(own) - len(theirs),
            })
            record(scope, f"versus_{model}", points, samples, support, ["improved", model], model)
    return summary, crops


INTERVALS = {"improved_macro_f1": False, "original_macro_f1": False, "difference": True,
             "improved_minus_majority": True, "original_minus_majority": True}


def flatten(rows):
    """One CSV column per interval part, with the same columns on every row."""
    flat = []
    for row in rows:
        out = {key: value for key, value in row.items() if key not in INTERVALS}
        for key, paired_difference in INTERVALS.items():
            value = row[key] or {}
            parts = ("estimate", "ci_low", "ci_high", "valid_replicates")
            for part in parts + (("share_of_resamples_improved",) if paired_difference else ()):
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
    lines = [
        "# Improved model versus original frozen models",
        "",
        f"Improved model `{metadata['improved_candidate']}` (`{metadata['improved_run']}`) against the four original "
        f"models in `{metadata['original_run']}`. Each comparison uses only the parcels that original model scored; "
        "the improved model covers all of them. Both predictions and both training-majority baselines use the same "
        f"resampled parcels: {metadata['repetitions']:,} repetitions, seed {metadata['seed']}, sampling with "
        "replacement within campaign and reference crop, 95% percentile intervals.",
        "",
        RETROSPECTIVE_NOTE,
        "",
        "A difference is positive when the improved model scores higher. Three-class macro-F1 is unavailable "
        "for single-crop campaigns; their per-crop F1 is in `per_crop_f1.csv`.",
    ]
    if metadata["revised_scope"]:
        lines += ["", REVISION_NOTE]
    for scope in dict.fromkeys(row["population"] for row in summary):
        rows = [row for row in summary if row["population"] == scope]
        if all(row["improved_macro_f1"]["estimate"] is None for row in rows):
            continue
        lines += ["", f"## {names.get(scope, scope)}", "",
                  "| Comparison | Parcels | Improved macro-F1 | Original macro-F1 | Improved − original | "
                  "Share of resamples improved | Improved − its majority |",
                  "| --- | ---: | --- | --- | --- | ---: | --- |"]
        for row in rows:
            share = (row["difference"] or {}).get("share_of_resamples_improved")
            lines.append("| " + " | ".join([
                row["comparison"].replace("versus_", "vs "), str(row["parcels"]), ci(row["improved_macro_f1"]),
                ci(row["original_macro_f1"]), signed(row["difference"]),
                "—" if share is None else f"{share:.1%}", signed(row["improved_minus_majority"]),
            ]) + " |")
    single = [row for row in crops if row["predictor"] in ("improved",) or " minus " in row["predictor"]]
    single = [row for row in single if names.get(row["population"]) and row["population"] not in ("jecam", "combined")]
    if single:
        lines += ["", "## Single-crop campaigns (crop F1)", "",
                  "| Campaign | Comparison | Crop | Parcels | Improved F1 | Improved − original F1 |",
                  "| --- | --- | --- | ---: | --- | --- |"]
        index = {(r["population"], r["comparison"], r["predictor"], r["crop"]): r for r in crops}
        for row in single:
            if row["predictor"] != "improved":
                continue
            model = row["comparison"].replace("versus_", "")
            diff = index.get((row["population"], row["comparison"], f"improved minus {model}", row["crop"]))
            lines.append("| " + " | ".join([names[row["population"]], row["comparison"].replace("versus_", "vs "),
                                              row["crop"], str(row["support"]), ci(row), signed(diff)]) + " |")
    return "\n".join(lines) + "\n"


def report(test_receipt, original_run, output, seed=DEFAULT_SEED, repetitions=DEFAULT_REPETITIONS, revised=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    if repetitions < 2:
        raise ValueError("Use at least two bootstrap repetitions")
    improved, info = load_improved(test_receipt)
    original, majority, campaigns = load_original(original_run)
    summary, crops = compare(improved, info["majority"], original, majority, campaigns,
                             seed=seed, repetitions=repetitions, revised=revised)
    metadata = {
        "improved_run": info["run"], "improved_candidate": info["candidate"],
        "improved_majority_crop": info["majority"], "original_run": Path(original_run).resolve().name,
        "original_majority_crops": majority, "seed": seed, "repetitions": repetitions,
        "sampling_unit": "partition_group", "sampling": "With replacement within campaign and reference-crop strata",
        "interval_method": "Percentile, 2.5th and 97.5th quantiles of finite replicates",
        "revised_scope": revised, "retrospective": RETROSPECTIVE_NOTE, "numpy_version": np.__version__,
    }
    output.mkdir(parents=True)
    write_csv(output / "paired_macro_f1.csv", flatten(summary))
    write_csv(output / "per_crop_f1.csv", crops)
    (output / "methods.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(metadata, summary, crops), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("improved_test_receipt", type=Path)
    parser.add_argument("original_run", type=Path, help="Main-pipeline testing run with predictions/ and reports/")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--revised-scope", action="store_true", help="Use Jose's revised reporting scope")
    args = parser.parse_args()
    print(report(args.improved_test_receipt, args.original_run, args.output,
                 args.seed, args.repetitions, args.revised_scope))


if __name__ == "__main__":
    main()
