"""Evaluate three follow-up transfer hypotheses with the saved matched cohorts."""

import argparse
from collections import defaultdict
from csv import DictReader, field_size_limit
from pathlib import Path

import joblib
import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.features.experiments import enhance_window
from crop_pipeline.paoay.predictions import load_classifier
from crop_pipeline.sources.sampling import pixel_location
from crop_pipeline.training import exploratory, model_comparisons as comparisons, notebook_workflow, source_holdout
from crop_pipeline.training.external_report import (
    DEFAULT_REPETITIONS, DEFAULT_SEED, REVISED_POPULATION_NAMES, confusion, metric_values,
    read_predictions, write_csv,
)
from crop_pipeline.training.model_followups import paired_macro_difference
from crop_pipeline.windows.seasons import feature_columns, file_sha256

CROPS = source_holdout.CROPS
COHORTS = comparisons.COHORTS
PHASES = ("locked_validation", "external_test")


def parcels(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["partition_group"]].append(row)
    return groups


def parcel_medians(rows, columns):
    result = []
    for members in parcels(rows).values():
        first = members[0]
        if any(row["crop"] != first["crop"] or row["partition"] != first["partition"] for row in members):
            raise ValueError("Conflicting labels or splits for a parcel median")
        medians = np.median(exploratory.feature_matrix(members, columns), axis=0)
        result.append({**first, **dict(zip(columns, medians, strict=True))})
    return result


def pixel_key(row):
    return (row["region_id"], *pixel_location(row))


def ndvi_amplitude(rows):
    return float(np.ptp(np.median([[row[f"bin_{i:02d}_NDVI"] for i in range(1, 9)]
                                    for row in rows], axis=0)))


def strongest_windows(rows_by_phase, archive, policy):
    """Switch only to a higher-amplitude archived window covering every original pixel."""
    original = {
        group: (phase, members)
        for phase, rows in rows_by_phase.items()
        for group, members in parcels(rows).items()
    }
    if len(original) != sum(len(parcels(rows)) for rows in rows_by_phase.values()):
        raise ValueError("A parcel appears in multiple phases")
    keys = {group: {pixel_key(row): row for row in members} for group, (_, members) in original.items()}
    options = defaultdict(lambda: defaultdict(dict))
    archived_ids = set()
    columns = feature_columns()
    old_limit = field_size_limit()
    try:
        # Some legacy source-revision fields contain embedded inventories.
        field_size_limit(max(old_limit, 32 * 1024 * 1024))
        with archive.open(newline="", encoding="utf-8") as stream:
            for row in DictReader(stream):
                group = row["partition_group"]
                if group not in keys:
                    continue
                archived_ids.add(group)
                pixel = pixel_key(row)
                previous = keys[group].get(pixel)
                if previous is None:
                    continue
                window = row["window_start"], row["window_end"]
                previous_window = previous["window_start"], previous["window_end"]
                if window == previous_window:
                    continue
                candidate = {column: float(row[column]) for column in columns}
                existing = options[group][window].setdefault(pixel, candidate)
                if existing != candidate:
                    raise ValueError(f"Conflicting archived features for {group} {window}")
    finally:
        field_size_limit(old_limit)
    changes, replacements = [], {}
    complete = set()
    for group, (phase, members) in original.items():
        choices = [(window, values) for window, values in sorted(options[group].items())
                   if values.keys() == keys[group].keys()]
        if not choices:
            continue
        complete.add(group)
        window, values = max(choices, key=lambda choice: ndvi_amplitude(choice[1].values()))
        old_amplitude, new_amplitude = ndvi_amplitude(members), ndvi_amplitude(values.values())
        if new_amplitude <= old_amplitude:
            continue
        changes.append({
            "partition_group": group, "phase": phase, "crop": members[0]["crop"],
            "old_window_start": members[0]["window_start"], "old_window_end": members[0]["window_end"],
            "new_window_start": window[0], "new_window_end": window[1],
            "old_ndvi_amplitude": old_amplitude, "new_ndvi_amplitude": new_amplitude,
        })
        replacements[group] = {key: enhance_window(
            {**keys[group][key], **candidate, "window_start": window[0], "window_end": window[1]}, policy)
            for key, candidate in values.items()}
    diagnostics = {
        phase: {"parcels": len(groups), "parcels_in_candidate_archive": len(groups & archived_ids),
                "parcels_with_complete_alternative": len(groups & complete),
                "parcels_switched": len(groups & replacements.keys())}
        for phase, rows in rows_by_phase.items()
        for groups in (set(parcels(rows)),)
    }
    updated = {phase: [replacements[row["partition_group"]][pixel_key(row)]
                       if row["partition_group"] in replacements else row for row in rows]
               for phase, rows in rows_by_phase.items()}
    return updated, changes, diagnostics


def parcel_scores(model, rows, columns):
    groups = parcels(rows)
    probabilities = model.predict_proba(exploratory.feature_matrix(rows, columns))
    by_group = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        by_group[row["partition_group"]].append(probability)
    result = []
    for group, members in groups.items():
        labels = {(row["crop"], row.get("test_cohort", "development")) for row in members}
        if len(labels) != 1:
            raise ValueError(f"Conflicting parcel metadata: {group}")
        crop, campaign = next(iter(labels))
        mean = np.mean(by_group[group], axis=0)
        result.append({"partition_group": group, "crop": crop, "test_cohort": campaign,
                       "usable_pixels": len(members), "prediction": model.classes_[mean.argmax()],
                       **{f"prob_{name}": float(p) for name, p in zip(model.classes_, mean, strict=True)}})
    return result


def average_windows(fixed, aligned):
    comparisons.require_shared_parcels(fixed, aligned)
    other = {row["partition_group"]: row for row in aligned}
    result = []
    for row in fixed:
        pair = other[row["partition_group"]]
        if row["usable_pixels"] != pair["usable_pixels"]:
            raise ValueError("Ensemble requires shared matched pixel support")
        probabilities = {crop: (row[f"prob_{crop}"] + pair[f"prob_{crop}"]) / 2 for crop in sorted(CROPS)}
        result.append({**row, **{f"prob_{crop}": value for crop, value in probabilities.items()},
                       "prediction": max(probabilities, key=probabilities.get)})
    return result


def measurements(rows):
    encode = CROPS.index
    actual = np.array([encode(row["crop"]) for row in rows], dtype=int)
    predicted = np.array([encode(row["prediction"]) for row in rows], dtype=int)
    values = metric_values(confusion(actual, predicted))
    macro = values["macro_f1", ""]
    return {"parcels": len(rows), "correct": int((actual == predicted).sum()),
            "macro_f1": float(macro) if np.isfinite(macro) else None,
            **{f"{crop}_f1": float(values["f1", crop]) if i in actual else None
               for i, crop in enumerate(CROPS)}}


def run(training, testing, archive, output):
    output = output.resolve()
    if any(output.is_relative_to(root.resolve()) for root in (training, testing, archive.parent)):
        raise ValueError("Use a new output directory outside the frozen inputs")
    stage, _ = notebook_workflow._verify(training / "models/models_stage.json", "models_stage")
    test_stage, test_receipt = notebook_workflow._verify(testing / "scores.json", "scores")
    development = training / "features_development.json"
    frozen = comparisons.read_json(training / "models/development.json")
    if (test_receipt["models_handoff_sha256"] != file_sha256(stage)
            or frozen["feature_sha256"] != file_sha256(development)):
        raise ValueError("Input snapshots differ from the frozen run")
    dev = comparisons.read_json(development)
    test = comparisons.read_json(testing / "features_test.json")
    for snapshot in (dev, test):
        source_holdout.require_matching(snapshot["rows"])
        if any(snapshot[key] != frozen[key] for key in ("quality", "window_sha256", "fixed_start_month")):
            raise ValueError("Feature provenance differs from the frozen run")
    output.mkdir(parents=True, exist_ok=False)
    predictions = {phase: {} for phase in PHASES}
    aligned = {}
    for cohort in COHORTS:
        dev_rows = source_holdout.checked_rows(dev["rows"][cohort], False)
        train = [row for row in dev_rows if row["partition"] == "train"]
        validation = [row for row in dev_rows if row["partition"] == "validation"]
        external = source_holdout.checked_rows(test["rows"][cohort], True)
        manifest_path = training / "models" / cohort / "model_manifest.json"
        manifest = comparisons.read_json(manifest_path)
        if ({row["partition_group"]: row["partition"] for row in dev_rows} != manifest["development_split"]
                or {row["partition_group"] for row in external} & manifest["development_split"].keys()
                or set(parcels(train)) & set(parcels(validation))):
            raise ValueError("Follow-up changes the frozen parcel partitions")
        forest, recipe = load_classifier(manifest_path)
        columns = recipe["feature_columns"]
        median = comparisons.fit(forest, parcel_medians(train, columns), columns)
        package = output / f"{cohort}_parcel_median.joblib"
        joblib.dump(median, package)
        save_json(package.with_suffix(".json"), {"feature_columns": columns, "settings": median.get_params(),
                  "training_parcel_ids": sorted(parcels(train)), "model_sha256": file_sha256(package)})

        baseline = f"{cohort}_original"
        for phase, rows in zip(PHASES, (validation, external), strict=True):
            original = parcel_scores(forest, rows, columns)
            median_rows = parcel_scores(median, parcel_medians(rows, columns), columns)
            support = {row["partition_group"]: row["usable_pixels"] for row in original}
            for row in median_rows:
                row["usable_pixels"] = support[row["partition_group"]]
            comparisons.require_shared_parcels(original, median_rows)
            predictions[phase][baseline] = original
            predictions[phase][f"{cohort}_median"] = median_rows
        saved = read_predictions(testing / "predictions/combined" / f"{cohort}_test_parcels.csv")
        if saved != {r["partition_group"]: (r["test_cohort"], r["crop"], r["prediction"])
                     for r in predictions["external_test"][baseline]}:
            raise ValueError(f"{cohort}: archived probability predictions did not reproduce")
        expected = manifest["validation"]
        actual = measurements(predictions["locked_validation"][baseline])
        if not np.isclose(actual["macro_f1"], expected["macro_f1"]):
            raise ValueError(f"{cohort}: archived validation macro-F1 did not reproduce")
        if cohort == "aligned_matched":
            aligned = {"forest": forest, "recipe": recipe,
                       "rows": {"train": train, "locked_validation": validation, "external_test": external}}

    for phase in PHASES:
        predictions[phase]["ensemble_50_50"] = average_windows(
            predictions[phase]["fixed_matched_original"], predictions[phase]["aligned_matched_original"])
    print("Ensemble and median models scored; checking archived cycle candidates.", flush=True)
    updated, changes, coverage = strongest_windows(aligned["rows"], archive, aligned["recipe"]["feature_policy"])
    columns = aligned["recipe"]["feature_columns"]
    forest = comparisons.fit(aligned["forest"], updated["train"], columns)
    package = output / "aligned_matched_strongest_cycle.joblib"
    joblib.dump(forest, package)
    save_json(package.with_suffix(".json"), {"feature_columns": columns, "settings": forest.get_params(),
              "training_parcel_ids": sorted(parcels(updated["train"])), "model_sha256": file_sha256(package),
              "timing_rule": "highest NDVI amplitude among archived timing-qualified windows covering every "
                             "original pixel; retain current window on ties"})
    for phase in PHASES:
        rows = parcel_scores(forest, updated[phase], columns)
        comparisons.require_shared_parcels(predictions[phase]["aligned_matched_original"], rows)
        predictions[phase]["aligned_matched_strongest_cycle"] = rows
    return export_results(output, predictions, changes, coverage, stage, test_stage, archive, training)


def export_results(output, predictions, changes, coverage, stage, test_stage, archive, training):
    rows, scores, paired = [], [], []
    rng = np.random.default_rng(DEFAULT_SEED)
    pairs = {
        "ensemble_50_50": ("fixed_matched_original", "aligned_matched_original"),
        "fixed_matched_median": ("fixed_matched_original",),
        "aligned_matched_median": ("aligned_matched_original",),
        "aligned_matched_strongest_cycle": ("aligned_matched_original",),
    }
    for phase, by_model in predictions.items():
        scopes = ("development",) if phase == "locked_validation" else REVISED_POPULATION_NAMES
        for scope in scopes:
            scoped = {model: (records if scope == "development" else comparisons.scoped_predictions(records, scope))
                      for model, records in by_model.items()}
            for model, records in scoped.items():
                for segment in ("all", "one_pixel", "multi_pixel"):
                    selected = records if segment == "all" else [
                        row for row in records if (row["usable_pixels"] == 1) == (segment == "one_pixel")]
                    if selected:
                        scores.append({"phase": phase, "population": scope, "model": model,
                                       "support_segment": segment, **measurements(selected)})
                if scope in ("combined", "development"):
                    rows.extend({"phase": phase, "model": model, **row} for row in records)
            if scope in ("jecam", "combined"):
                for variant, baselines in pairs.items():
                    for baseline in baselines:
                        paired.append({
                            "population": scope, "model": variant, "reference_model": baseline,
                            **paired_macro_difference(scoped[baseline], scoped[variant], DEFAULT_REPETITIONS, rng),
                        })
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "metrics.csv", scores)
    write_csv(output / "paired_macro_f1.csv", paired)
    if changes:
        write_csv(output / "cycle_changes.csv", changes)
    prior = comparisons.read_json(training / "matched_geographic_strongest_cycle.json")
    source_digest = comparisons.read_json(training / "models/development.json")["feature_sha256"]
    if prior["feature_sha256"] != source_digest:
        raise ValueError("Saved timing cross-validation used a different development snapshot")
    save_json(output / "methods.json", {
        "training_stage_sha256": file_sha256(stage), "test_stage_sha256": file_sha256(test_stage),
        "cycle_archive_sha256": file_sha256(archive), "alternate_window_coverage": coverage,
        "prior_spatial_training_macro_f1": {"original": prior["baseline"]["macro_f1"],
                                            "strongest_cycle": prior["metrics"]["macro_f1"]},
        "bootstrap": {"seed": DEFAULT_SEED, "repetitions": DEFAULT_REPETITIONS,
                      "method": "paired percentile, stratified by campaign and reference crop"},
    })
    report = output / "report.md"
    report.write_text(render_report(scores, paired, coverage, prior), encoding="utf-8")
    return report


def render_report(scores, paired, coverage, prior):
    lines = [
        "# Matched transfer follow-up experiments", "",
        "The pre-specified 50/50 probability average reuses the two frozen forests. Parcel-median forests "
        "use one median feature vector per field, with the original Random Forest settings and crop-balanced "
        "parcel weights. The timing experiment selects the highest NDVI amplitude from archived timing-qualified "
        "windows that cover every original pixel, keeping the current window on ties. It fits only on development "
        "training. The revised external benchmark has already been examined. All comparisons share the original "
        "matched parcels; pixel support for the median comparison refers to the source pixels.",
        "", "## Scores and paired differences", "",
        "| Population | Model | Parcels | Carrot F1 | Cabbage F1 | Potato F1 | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    def number(x):
        return "—" if x is None else f"{x:.3f}"

    for row in scores:
        if row["support_segment"] == "all":
            scope = REVISED_POPULATION_NAMES.get(row["population"], "Locked development validation")
            values = [scope, row["model"], str(row["parcels"]),
                      *(number(row[f"{crop}_f1"]) for crop in CROPS), number(row["macro_f1"])]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Paired 95% percentile intervals resample the same parcels for both models within campaign "
              "and reference crop. The combined interval conditions on the observed counts.",
              "", "| Population | Candidate | Reference | Macro-F1 difference [95% CI] |",
              "| --- | --- | --- | --- |"]
    for row in paired:
        lines.append(f"| {row['population']} | {row['model']} | {row['reference_model']} | "
                     f"{row['estimate']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |")
    lines += ["", "## Timing rule coverage", "",
              "| Phase | Parcels | In candidate archive | Complete alternatives | Switched |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for phase, counts in coverage.items():
        lines.append("| " + " | ".join([phase, *map(str, counts.values())]) + " |")
    lines += ["", f"Saved development spatial cross-validation macro-F1 was "
              f"{prior['baseline']['macro_f1']:.3f} for the original timing rule and "
              f"{prior['metrics']['macro_f1']:.3f} for strongest cycle. NDVI amplitude and date overlap "
              "cannot independently establish crop occupancy.", "",
              "The metrics table also separates one-pixel and multi-pixel fields.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "testing", "archive", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(run(args.training, args.testing, args.archive, args.output))


if __name__ == "__main__":
    main()
