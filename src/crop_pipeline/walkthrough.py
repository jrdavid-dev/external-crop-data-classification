"""Read-only summaries of the latest completed crop-classification run for the walkthrough notebook.

Only small saved outputs are read (CSV tables, score files, model manifests, figures).
Nothing here selects parcels, submits jobs, trains models or writes pipeline state.
"""

import csv
import json
from pathlib import Path

CROPS = ("carrot", "cabbage", "potato")
COHORTS = ("fixed_full", "aligned_full", "fixed_matched", "aligned_matched")
PARTITIONS = ("train", "validation", "test")
F1_TARGET = 0.5
COHORT_LABELS = {
    "fixed_full": "Fixed calendar · full",
    "aligned_full": "Aligned green-up · full",
    "fixed_matched": "Fixed calendar · matched",
    "aligned_matched": "Aligned green-up · matched",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pointer_target(path):
    """Manifest named by a stage pointer. Hashes are not recomputed: this view never feeds a later stage."""
    record = _read_json(path)
    if record.get("status") != "prepared":
        raise RuntimeError(f"{Path(path).name} is {record.get('status')}: {record.get('reason', 'finish that notebook first')}")
    return Path(record["path"])


def current_runs(root):
    """Run folders the latest completed notebooks point to."""
    outputs = Path(root) / "pipeline_outputs"
    features = _pointer_target(outputs / "preprocessing" / "current_features.json")
    models = _pointer_target(outputs / "training" / "current_models.json")
    scores = _pointer_target(outputs / "testing" / "current_scores.json")
    return {"preprocessing": features.parent, "training": models.parent.parent, "testing": scores.parent}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reference_table(preprocessing):
    """Reference parcels collected per source, crop and partition, before window and quality checks."""
    totals = {}
    for row in _read_csv(Path(preprocessing) / "reference_profile.csv"):
        key = (row["source"], row["partition"])
        totals.setdefault(key, {crop: 0 for crop in CROPS})[row["crop"]] += int(row["parcels"])
    return [{"Source": source, "Partition": partition, **{crop.title(): counts[crop] for crop in CROPS},
             "Total": sum(counts.values())}
            for (source, partition), counts in sorted(totals.items(), key=lambda item: (
                PARTITIONS.index(item[0][1]), item[0][0]))]


def quality_summary(preprocessing):
    rows = _read_csv(Path(preprocessing) / "quality_counts.csv")
    audited = sum(int(row["audited_tile_intervals"]) for row in rows)
    passed = sum(int(row["quality_pass_tile_intervals"]) for row in rows)
    return {"audited": audited, "passed": passed, "pass_rate": passed / audited if audited else None}


def sample_status(preprocessing):
    """Usable parcels and pixels against targets, one row per cohort, partition and crop."""
    rows = []
    for row in _read_csv(Path(preprocessing) / "usable_samples.csv"):
        parcels, pixels = int(row["usable_parcel_groups"]), int(row["usable_pixels"])
        target_parcels, target_pixels = int(row["target_parcels"]), int(row["target_pixels"])
        rows.append({"Recipe": COHORT_LABELS.get(row["cohort"], row["cohort"]), "Partition": row["partition"],
                     "Crop": row["crop"], "Parcels / target": f"{parcels:,} / {target_parcels:,}",
                     "Pixels / target": f"{pixels:,} / {target_pixels:,}",
                     "Met": parcels >= target_parcels and pixels >= target_pixels})
    return rows


def _round(value):
    return None if value is None else round(float(value), 2)


def _f1_row(label, metrics):
    per_crop = metrics.get("per_crop", {})
    f1 = {crop: per_crop.get(crop, {}).get("f1") for crop in CROPS}
    known = [value for value in f1.values() if value is not None]
    return {"Recipe": label, "Macro-F1": _round(metrics.get("macro_f1")),
            **{f"{crop.title()} F1": _round(f1[crop]) for crop in CROPS},
            "Crops ≥ 0.50": f"{sum(value >= F1_TARGET for value in known)} / {len(CROPS)}"}


def validation_scores(training):
    """Validation metrics stored in each frozen model's manifest."""
    index = _read_json(Path(training) / "validation" / "models.json")
    return [_f1_row(COHORT_LABELS.get(cohort, cohort), _read_json(index[cohort]["path"])["validation"])
            for cohort in COHORTS if cohort in index]


def test_scores(testing):
    """External-test metrics with the training-majority baseline beside each recipe."""
    reports = Path(testing) / "reports"
    scores = _read_json(reports / "test_scores.json")
    baselines = _read_json(reports / "majority_baseline.json")
    rows = []
    for cohort in COHORTS:
        if cohort not in scores:
            continue
        row = _f1_row(COHORT_LABELS.get(cohort, cohort), scores[cohort])
        baseline = baselines.get(cohort, {}).get("metrics", {}).get("macro_f1")
        macro = scores[cohort].get("macro_f1")
        row["Baseline macro-F1"] = _round(baseline)
        row["Above baseline"] = None if macro is None or baseline is None else macro > baseline
        rows.append(row)
    return rows


def collection_status(root):
    """Saved batch states and selected parcel groups per source and partition (no cloud calls)."""
    from crop_pipeline.collection_summary import inspect_collection

    summary = inspect_collection(root)
    batches = {}
    for batch in summary["batches"]:
        counts = batches.setdefault(batch["source"], {})
        counts[batch["status"]] = counts.get(batch["status"], 0) + 1
    selected = {}
    for row in summary["profile"]:
        entry = selected.setdefault((row["source"], row["partition"]), {crop: 0 for crop in CROPS})
        entry[row["crop"]] += int(row["selected_parcel_groups"])
    return {
        "roles": summary["source_roles"],
        "batches": [{"Source": source, **counts, "Total": sum(counts.values())}
                    for source, counts in sorted(batches.items())],
        "selected": [{"Source": source, "Partition": partition, **{crop.title(): counts[crop] for crop in CROPS},
                      "Total": sum(counts.values())}
                     for (source, partition), counts in sorted(selected.items(), key=lambda item: (
                         item[0][0], PARTITIONS.index(item[0][1]) if item[0][1] in PARTITIONS else 9))],
    }


def sample_funnel(preprocessing):
    """Parcels per partition and crop at each step: collected, complete window, usable after the cap."""
    folder = Path(preprocessing)
    collected = {}
    for row in _read_csv(folder / "reference_profile.csv"):
        key = (row["partition"], row["crop"])
        collected[key] = collected.get(key, 0) + int(row["parcels"])
    windows = {(row["window"], row["partition"], row["crop"]): int(row["parcel_groups"])
               for row in _read_csv(folder / "window_counts.csv")}
    samples = _read_csv(folder / "usable_samples.csv")
    usable = {(row["cohort"], row["partition"], row["crop"]): int(row["usable_parcel_groups"]) for row in samples}
    targets = {row["partition"]: int(row["target_parcels"]) for row in samples}
    return [{"Partition": partition, "Crop": crop, "Collected": collected.get((partition, crop), 0),
             "Complete fixed window": windows.get(("fixed", partition, crop), 0),
             "Complete green-up window": windows.get(("aligned", partition, crop), 0),
             "Usable fixed (full)": usable.get(("fixed_full", partition, crop), 0),
             "Usable aligned (full)": usable.get(("aligned_full", partition, crop), 0),
             "Usable matched": usable.get(("fixed_matched", partition, crop), 0),
             "Target parcels": targets.get(partition)}
            for partition in PARTITIONS for crop in CROPS]


def greenup_outcomes(preprocessing):
    """Green-up detector outcomes and the onset month of each accepted candidate."""
    rows = _read_csv(Path(preprocessing) / "greenup" / "candidate_greenup.csv")
    outcomes = {}
    months = {month: 0 for month in range(1, 13)}
    for row in rows:
        outcomes[(row["status"], row["reason"])] = outcomes.get((row["status"], row["reason"]), 0) + 1
        if row["status"] == "candidate" and row["onset_interval_start"]:
            months[int(row["onset_interval_start"][5:7])] += 1
    return {
        "outcomes": [{"Outcome": status, "Reason": reason.replace("_", " "), "Parcel-years": count}
                     for (status, reason), count in sorted(outcomes.items(), key=lambda item: -item[1])],
        "onset_months": months,
        "settings": _read_json(Path(preprocessing) / "seasons" / "automatic_timing_policy.json"),
    }


def leaf_selection(training):
    """Validation macro-F1 of every minimum-leaf candidate, and which one was kept."""
    candidates = _read_json(Path(training) / "validation" / "validation_candidates.json")
    return [{"Recipe": COHORT_LABELS.get(cohort, cohort),
             **{f"Leaf {item['min_samples_leaf']}": _round(item["metrics"].get("macro_f1"))
                for item in entry["candidates"]},
             "Kept leaf size": entry["selected_min_samples_leaf"]}
            for cohort, entry in candidates.items()]


def test_geography(testing, cohort="aligned_matched"):
    """External-test parcels per country and crop for one recipe.

    Aligned rows do not carry the country field, so it is taken from fixed rows in the same 1° site cell.
    """
    data = _read_json(Path(testing) / "features_test.json")
    countries_by_cell = {row["site_cell"]: row["country"] for rows in data["rows"].values()
                         for row in rows if row.get("country") and row.get("site_cell")}
    parcels = {}
    for row in data["rows"].get(cohort, []):
        country = row.get("country") or countries_by_cell.get(row.get("site_cell"), "unrecorded")
        parcels.setdefault((country, row["crop"]), set()).add(row["partition_group"])
    countries = sorted({country for country, _ in parcels})
    return [{"Country": country, **{crop.title(): len(parcels.get((country, crop), ())) for crop in CROPS}}
            for country in countries]


def forest_details(training, cohort="fixed_full"):
    """Settings of a frozen Random Forest and its impurity importance per variable and per half-month bin.

    Loads the project's own frozen model package; only use trusted local files.
    """
    import joblib

    package = joblib.load(Path(training) / "models" / cohort / "model.joblib")
    model, columns = package["classifier"], package["feature_columns"]
    parameters = model.get_params()
    settings = {"Trees": parameters["n_estimators"], "Features tried per split": parameters["max_features"],
                "Bootstrap share per tree": parameters["max_samples"],
                "Minimum leaf size": parameters["min_samples_leaf"], "Random seed": parameters["random_state"],
                "Predictors": len(columns), "Classes": ", ".join(map(str, model.classes_))}
    by_variable, by_bin = {}, {}
    for column, importance in zip(columns, model.feature_importances_):
        _, number, variable = column.split("_", 2)
        by_variable[variable] = by_variable.get(variable, 0.0) + float(importance)
        by_bin[int(number)] = by_bin.get(int(number), 0.0) + float(importance)
    return {"settings": settings, "by_variable": by_variable, "by_bin": dict(sorted(by_bin.items()))}


def figures(runs):
    """Saved figures from each stage, keyed by a short name; missing files are omitted."""
    candidates = {
        "reference_parcels": runs["preprocessing"] / "figures" / "reference_parcels.png",
        "quality_pass_rates": runs["preprocessing"] / "figures" / "quality_pass_rates.png",
        "cloud_shadow_by_half_month": runs["preprocessing"] / "figures" / "cloud_shadow_by_half_month.png",
        "window_counts": runs["preprocessing"] / "figures" / "window_counts.png",
        **{f"usable_{cohort}": runs["preprocessing"] / "figures" / f"usable_vs_targets_{cohort}.png" for cohort in COHORTS},
        "validation_macro_f1": runs["training"] / "figures" / "validation_macro_f1.png",
        "test_macro_f1": runs["testing"] / "reports" / "figures" / "test_macro_f1.png",
        "test_per_crop_f1": runs["testing"] / "reports" / "figures" / "test_per_crop_f1.png",
        "test_confusion_matrices": runs["testing"] / "reports" / "figures" / "test_confusion_matrices.png",
    }
    for path in sorted((runs["preprocessing"] / "figures").glob("parcel_*.png")):
        candidates[path.stem] = path
    return {name: path for name, path in candidates.items() if path.is_file()}
