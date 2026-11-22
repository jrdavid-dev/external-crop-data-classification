"""Verify the frozen defense baseline and extract the evidence for an offline demo."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np

from crop_pipeline.common import interval_bins
from crop_pipeline.features.experiments import interval_columns
from crop_pipeline.sources.cropharvest import parse_date, resolve_observation_year
from crop_pipeline.sources.sampling import pixel_location
from crop_pipeline.training import external_report, study_manifest
from crop_pipeline.training.exploratory import feature_matrix
from crop_pipeline.training.source_holdout import require_matching
from crop_pipeline.windows.seasons import file_sha256, window_intervals

ROOT = Path(__file__).resolve().parents[1]
PREP = Path("pipeline_outputs/preprocessing/external_xxu0iz4o")
TRAIN = Path("pipeline_outputs/training/run_fa40f5d6bb31")
TEST = Path("pipeline_outputs/testing/run_81a0d0a2812e")
ORIGINAL = TEST.with_name(TEST.name + "_batch1")
REVISED = TEST.with_name(TEST.name + "_batch1_revised")
DIAGNOSTICS = TEST.with_name(TEST.name + "_batch3")
STUDY = TEST.with_name(TEST.name + "_batch4") / "study_manifest.json"
FOLLOWUPS = TEST.with_name(TEST.name + "_transfer_followups") / "predictions.csv"
MODEL = TRAIN / "models/fixed_matched"
MODEL_SHA = "e2c79a54445e4b806cdc3de7ce73578750409466f3a021b021748c84a387cee7"
ORIGINAL_NOTEBOOK = Path("notebooks/SakAI_crop_classification_walkthrough.ipynb")
EXAMPLE_ID = "cropharvest|jecam|2019|jecam:7450"
CROPS = external_report.CROPS
FITTED_CLASSES = ["cabbage", "carrot", "potato"]
SUPPORT = {
    "train": [(72, 1223), (50, 961), (77, 1127)],
    "validation": [(17, 316), (13, 260), (39, 552)],
    "jecam": [(37, 115), (11, 93), (35, 137)],
}
REMOVED = {
    "cropharvest|ile-de-france|2019|ile-de-france:4045",
    "cropharvest|ile-de-france|2019|ile-de-france:6059",
    "cropharvest|ile-de-france|2019|ile-de-france:6074",
    "cropharvest|ile-de-france|2019|ile-de-france:6075",
    "cropharvest|martinique-france|2019|martinique-france:2333",
    "cropharvest|reunion-france|2019|reunion-france:1858",
}


def require(condition, path, details):
    if not condition:
        raise ValueError(f"{path}: {details}")


def keyed(rows, key, path):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, path, f"duplicate key {identity!r}")
        result[identity] = row
    return result


class Inputs:
    def __init__(self):
        self.sources = {}

    def record(self, path, selection, key, count):
        name = path.as_posix()
        digest = self.sources[name]["sha256"] if name in self.sources else file_sha256(ROOT / path)
        entry = self.sources.setdefault(name, {"sha256": digest, "extracts": []})
        entry["extracts"].append({"filter": selection, "row_key": key, "row_count": count})
        return digest

    def json(self, path):
        return study_manifest.read_json(ROOT / path)

    def csv(self, path):
        with (ROOT / path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


def matched_features(inputs, path, receipt, test=False):
    payload = inputs.json(path)
    rows = {name: payload["rows"][name] for name in ("fixed_matched", "aligned_matched")}
    del payload
    require_matching(rows)
    if test:
        rows = {name: [r for r in records if r["test_cohort"] == "jecam"]
                for name, records in rows.items()}
    require_matching(rows)
    for name, records in rows.items():
        digest = inputs.record(path, "rows." + name + ("; test_cohort=jecam" if test else ""),
                               "partition_group, region_id, CRS, x, y", len(records))
        require(digest == receipt[path.name], path, "SHA-256 differs from feature-stage receipt")
    return rows


def group_features(rows, columns, path):
    groups = defaultdict(list)
    for row in rows:
        groups[row["partition_group"]].append(row)
        require(row["missing_bins"] == "" and
                all(row[f"bin_{i:02d}_clear_count"] > 0 for i in range(1, 9)) and
                all(np.isfinite(row[c]) for c in columns),
                path, f"incomplete features for {row['partition_group']}")
    for key, pixels in groups.items():
        require(1 <= len(pixels) <= 20 and len({r["crop"] for r in pixels}) == 1,
                path, f"expected one label and 1–20 rows for {key}, got {len(pixels)} rows")
    return groups


def replay(classifier, columns, rows, saved, path):
    probabilities = classifier.predict_proba(feature_matrix(rows, columns))
    by_parcel = defaultdict(list)
    crop = {}
    for row, scores in zip(rows, probabilities, strict=True):
        group = row["partition_group"]
        by_parcel[group].append(scores)
        crop[group] = row["crop"]
    require(set(by_parcel) == set(saved), path,
            f"prediction ID difference: {len(set(by_parcel) ^ set(saved))}")
    result = {}
    for group, scores in sorted(by_parcel.items()):
        mean = np.mean(scores, axis=0)
        named = dict(zip(FITTED_CLASSES, mean.tolist(), strict=True))
        decision = FITTED_CLASSES[int(mean.argmax())]
        expected = saved[group]
        require(expected["crop"] == crop[group] and expected["prediction"] == decision
                and int(expected["usable_pixels"]) == len(scores)
                and all(abs(value - float(expected["prob_" + name])) < 1e-10
                        for name, value in named.items()),
                path, f"replay differs at {group}")
        # Retain the verified archive values for stable serialization across thread schedules.
        result[group] = {"crop": crop[group], "prediction": decision, "pixels": len(scores),
                         "scores": {name: float(expected["prob_" + name]) for name in named}}
    return result


def audit_dates(inputs, fixed, aligned, sites):
    raw_path = Path("data/external/cropharvest/labels.geojson")
    keys = {parcel.rsplit("|", 1)[-1] for parcel in fixed}
    properties = (f["properties"] for f in inputs.json(raw_path)["features"])
    source_key = lambda r: f"{r['dataset']}:{r['index']}"
    raw = keyed((r for r in properties if source_key(r) in keys), source_key, raw_path)
    require(set(raw) == keys and len(keys) == len(fixed), raw_path,
            f"expected {len(fixed)} unique matches, found {len(raw)}")
    inputs.record(raw_path, "matched primary JECAM IDs only", "dataset:index", len(raw))
    audit_path = Path("pipeline_outputs/test_audit_2026-11-19/parcel_audit.json")
    audit = keyed((r for r in inputs.json(audit_path)
                   if r["cohort"] == "fixed_matched" and r["scope"] == "jecam"),
                  lambda r: r["group"], audit_path)
    require(set(audit) == set(fixed), audit_path, "primary matched ID mismatch")
    inputs.record(audit_path, "fixed_matched; jecam; reference fields only",
                  "group", len(audit))
    result = {}
    for group, pixels in sorted(fixed.items()):
        row = pixels[0]
        label = raw[group.rsplit("|", 1)[-1]]
        observed = parse_date(label["collection_date"])
        year = resolve_observation_year(label["collection_date"], label["export_end_date"])["year"]
        require(label["label"] == row["crop"] and year == row["year"] == int(group.split("|")[2])
                and label["is_test"] is False and label["harvest_date"] is None
                and audit[group]["raw_matches"] == [label] and audit[group]["source_year"] == year,
                raw_path, f"label/date mismatch at {group}")
        windows = {}
        for name, cohort in (("fixed", fixed), ("aligned", aligned)):
            dates = {(r["window_start"], r["window_end"]) for r in cohort[group]}
            require(len(dates) == 1, raw_path, f"inconsistent {name} window at {group}")
            start, end = dates.pop()
            windows[name] = {"start": start, "end": end,
                             "in_window": date.fromisoformat(start) <= observed < date.fromisoformat(end)}
        require((windows["fixed"]["start"], windows["fixed"]["end"])
                == (f"{year}-05-01", f"{year}-09-01"), raw_path, f"wrong fixed dates at {group}")
        result[group] = {"crop": row["crop"], "site": sites[group]["source_site"],
                         "collection_date": observed.isoformat(),
                         "planting_date": parse_date(label["planting_date"]).isoformat(),
                         "export_end": parse_date(label["export_end_date"]).isoformat(),
                         "harvest_date": None, **windows}
    aligned_hits = Counter(r["crop"] for r in result.values() if r["aligned"]["in_window"])
    require(not any(r["fixed"]["in_window"] for r in result.values()) and
            aligned_hits == {"carrot": 24, "cabbage": 6, "potato": 32}, raw_path,
            f"fixed or aligned timing totals differ (aligned {dict(aligned_hits)})")
    expected_sites = {
        "Madagascar": (75, "2019-02-18", "2019-04-10", "2018-10-01", "2020-02-01"),
        "Senegal": (8, "2021-11-01", "2021-11-01", "2021-07-01", "2022-02-01"),
    }
    require({r["site"] for r in result.values()} == set(expected_sites), raw_path, "unexpected site set")
    for site, (count, first, last, planting, export_end) in expected_sites.items():
        site_rows = [r for r in result.values() if r["site"] == site]
        dates = [r["collection_date"] for r in site_rows]
        require(len(dates) == count and (min(dates), max(dates)) == (first, last)
                and {r["planting_date"] for r in site_rows} == {planting}
                and {r["export_end"] for r in site_rows} == {export_end},
                raw_path, f"unexpected date coverage for {site}: {len(dates)} rows")
    return result


def audit_quality(inputs, support):
    path = PREP / "quality/quality.json"
    data = inputs.json(path)
    regions = keyed((r["properties"] for r in data["regions"]
                     if r["properties"]["partition"] == "test"
                     and r["properties"].get("test_cohort", "jecam") == "jecam"),
                    lambda r: r["fragment_id"], path)
    candidates = defaultdict(set)
    expected = set()
    for region, row in regions.items():
        candidates[row["crop"]].add(row["partition_group"])
        expected.update((region, b["name"]) for b in interval_bins(int(row["year"])))
    rows = keyed((r for r in data["quality_rows"] if r["region_id"] in regions),
                 lambda r: (r["region_id"], r["name"]), path)
    del data
    require(rows.keys() <= expected, path, f"{len(rows.keys() - expected)} unexpected region-bins")
    statuses = Counter()
    supported = rejected = 0
    by_crop = defaultdict(Counter)
    for row in rows.values():
        outcome = ("passed" if row["status"] == "quality_pass"
                   else "below" if row["valid_pixels"] > 0 else "zero")
        statuses[outcome] += 1
        summary = by_crop[row["crop"]]
        summary["saved"] += 1
        summary["passed"] += outcome == "passed"
        if row["pre_mask_count"] is not None and row["cloud_shadow_count"] is not None:
            supported += row["pre_mask_count"]
            rejected += row["cloud_shadow_count"]
            summary["known"] += 1
            summary["rejected"] += row["cloud_shadow_count"]
    counts = {crop: len(groups) for crop, groups in candidates.items()}
    missing = len(expected - rows.keys())
    require((len(regions), len(expected), missing, dict(statuses), supported, rejected, counts)
            == (227, 5448, 0, {"passed": 4849, "below": 39, "zero": 560}, 179266, 54806,
                {"carrot": 73, "cabbage": 56, "potato": 92}), path,
            f"annual count mismatch: {len(regions)} regions, {statuses}, {counts}, {supported}/{rejected}")
    coverage = DIAGNOSTICS / "imagery_coverage.csv"
    masks = DIAGNOSTICS / "archive_mask_coverage.csv"
    mask_rows = [r for r in inputs.csv(masks) if r["population"] == "jecam"]
    require(sum(int(r["observed_cloud_or_shadow_rejections"]) for r in mask_rows) == rejected,
            masks, "cloud and shadow rejection total differs")
    for row in inputs.csv(coverage):
        if (row["population"], row["model"]) != ("jecam", "fixed_matched"):
            continue
        crop = row["crop"]
        value = by_crop[crop]
        fields = ("candidate_parcels_in_quality_snapshot", "candidate_region_bins_saved",
                  "candidate_region_bins_quality_pass", "region_bins_with_pre_mask_counts",
                  "observed_cloud_or_shadow_rejections", "usable_parcels", "usable_pixels")
        require(tuple(int(row[k]) for k in fields) ==
                (counts[crop], value["saved"], value["passed"], value["known"], value["rejected"],
                 *support[crop]), coverage, f"coverage differs for {crop}")
    inputs.record(coverage, "fixed_matched; jecam", "crop", 3)
    inputs.record(masks, "jecam", "partition, population, source, crop", len(mask_rows))
    quality_receipt = PREP / "quality/quality_stage_receipt.json"
    historical = {Path(p).name: h for p, h in inputs.json(quality_receipt)["files"].items()}
    inputs.record(quality_receipt, "quality snapshot hash", "quality.json", 1)
    require(inputs.record(path, "test; jecam", "region ID and expected annual bin name", len(rows))
            == historical[path.name], path, "SHA-256 differs from quality-stage receipt")
    return {"regions": len(regions), "expected": len(expected), "missing": missing,
            "statuses": dict(statuses), "supported": supported, "rejected": rejected, "candidates": counts}


def local_evidence(inputs):
    base = Path("data/pipeline_work/paoay")
    ref = "eff4797788de4f7c3d98"
    geometry = base / f"references/bundles/{ref}/reference_areas.geojson"
    ledger = base / f"field_review/records/{ref}/observations.csv"
    report = base / "field_review/bundles/929cda2b0290abe9b68e/report.md"
    areas = keyed((r["properties"] for r in inputs.json(geometry)["features"]),
                  lambda r: r["reference_id"], geometry)
    observations = inputs.csv(ledger)
    dated = [r for r in observations if r["observation_date"] and r["observed_crop"]]
    require(len(areas) == len(observations) == 7 and not dated and
            {r["reference_id"] for r in observations} == set(areas) and
            "Recorded observations: 0. Correction proposals: 0." in (ROOT / report).read_text(),
            ledger, f"unexpected local evidence: {len(areas)} areas and {len(dated)} dated labels")
    inputs.record(geometry, "all reference areas", "reference_id", len(areas))
    inputs.record(ledger, "all template rows; count nonblank date and crop", "reference_id", len(observations))
    inputs.record(report, "Recorded observations / Correction proposals", "field review summary", 1)
    return {"areas": sorted(r["name"] for r in areas.values()), "dated": len(dated)}


def prepare():
    inputs = Inputs()
    study = inputs.json(STUDY)
    verified = study_manifest.verify(ROOT, ROOT / STUDY)
    inputs.record(STUDY, "confirmed study artifact hashes", "artifact_sha256 path", verified)
    cohort = study["original_cohorts"]["fixed_matched"]
    require(study["paoay"]["designated_application_model"] is None, STUDY, "unexpected local model")
    inputs.record(ORIGINAL_NOTEBOOK, "preparation checksum only", "whole notebook", 1)
    recipe_path, manifest_path = MODEL / "recipe.json", MODEL / "model_manifest.json"
    recipe, manifest = inputs.json(recipe_path), inputs.json(manifest_path)
    columns = recipe["feature_columns"]
    require(columns == interval_columns() and len(set(columns)) == 72 and
            recipe["selected_design"] == "interval_only" and recipe["window_recipe"]["start_month"] == 5,
            recipe_path, "wrong fixed-calendar interval recipe")
    require(inputs.record(recipe_path, "ordered interval predictors", "feature column", 72)
            == manifest["files"]["recipe.json"]["sha256"], recipe_path, "recipe checksum mismatch")
    inputs.record(manifest_path, "frozen train/validation membership", "partition_group", 268)

    model_path = MODEL / "model.joblib"
    require(inputs.record(model_path, "original frozen fitted forest", "model file", 1)
            == manifest["model_id"] == manifest["files"]["model.joblib"]["sha256"] == MODEL_SHA,
            model_path, "model SHA-256 differs from the frozen baseline")
    runtime = {name: version(name) for name in ("numpy", "scikit-learn", "joblib")}
    runtime["python"] = platform.python_version()
    require(all(study["runtime_used_to_package"][k] == v for k, v in runtime.items()),
            STUDY, f"baseline replay runtime differs: {runtime}")
    bundle = joblib.load(ROOT / model_path)
    classifier = bundle["classifier"]
    require(bundle["feature_columns"] == columns and classifier.n_features_in_ == 72
            and classifier.classes_.tolist() == FITTED_CLASSES, model_path, "feature or class order differs")
    params = {k: classifier.get_params()[k] for k in ("n_estimators", "random_state", "criterion",
              "max_features", "bootstrap", "max_samples", "min_samples_leaf", "max_depth")}
    require(params == dict(n_estimators=1000, random_state=42, criterion="gini", max_features="sqrt",
                          bootstrap=True, max_samples=0.5, min_samples_leaf=2, max_depth=None),
            model_path, f"unexpected fitted forest configuration: {params}")
    receipt_path = PREP / "features/features_stage_receipt.json"
    receipt = {Path(p).name: h for p, h in inputs.json(receipt_path)["files"].items()}
    inputs.record(receipt_path, "historical feature snapshot hashes", "features file basename", 2)
    window_receipt = PREP / "windows/windows_stage_receipt.json"
    inputs.record(window_receipt, "saved preprocessing rules", "stage receipt", 1)
    source_inputs = PREP / "quality/source_inputs.json"
    inputs.record(source_inputs, "saved source provenance", "source input ledger", 1)
    dev_path, test_path = TRAIN / "features_development.json", TEST / "features_test.json"
    development = matched_features(inputs, dev_path, receipt)
    test = matched_features(inputs, test_path, receipt, test=True)
    split = manifest["development_split"]
    groups = group_features(development["fixed_matched"], columns, dev_path)
    aligned_development = group_features(development["aligned_matched"], columns, dev_path)
    require(all({(r["crop"], r["partition"]) for r in rows} ==
                {(r["crop"], r["partition"]) for r in aligned_development[g]}
                for g, rows in groups.items()), dev_path, "matched development labels or roles differ")
    require(set(groups) == set(split) and set(groups) == set(manifest["development_groups"])
            and all(all(r["partition"] == split[g] for r in rows) for g, rows in groups.items())
            and all({g for g, p in split.items() if p == role} == set(cohort[name])
                    for role, name in (("train", "development_train_parcels"),
                                       ("validation", "development_validation_parcels"))),
            dev_path, "development membership differs from the frozen split")
    primary = group_features(test["fixed_matched"], columns, test_path)
    aligned_primary = group_features(test["aligned_matched"], columns, test_path)
    require(not set(primary) & set(groups), test_path, "development and primary parcels overlap")
    require(all(r[0]["crop"] == aligned_primary[g][0]["crop"] for g, r in primary.items())
            and sum(len(pixels) == 1 for pixels in primary.values()) == 24,
            test_path, "matched test labels or one-pixel support differ")
    populations = {role: {g: rows for g, rows in groups.items() if split[g] == role}
                   for role in ("train", "validation")}
    populations["jecam"] = primary
    support, locations = {}, {}
    for role, parcels in populations.items():
        support[role] = {crop: [sum(rows[0]["crop"] == crop for rows in parcels.values()),
                                sum(len(rows) for rows in parcels.values() if rows[0]["crop"] == crop)]
                         for crop in CROPS}
        require([tuple(support[role][c]) for c in CROPS] == SUPPORT[role], dev_path,
                f"{role} counts differ: {support[role]}")
        centres = defaultdict(list)
        for group, rows in parcels.items():
            for row in rows:
                centres[pixel_location(row)].append(group)
        shared = sum(len(set(ids)) > 1 for ids in centres.values())
        require(shared == 0, dev_path, f"{role}: {shared} locations shared by different parcels")
        locations[role] = set(centres)
        if role == "train":
            require(len({ids[0] for ids in centres.values() if len(ids) > 1}) == 2,
                    dev_path, "expected repeats within two training parcels")
    distinct = {role: len(coords) for role, coords in locations.items()}
    require(distinct == {"train": 3308, "validation": 1128, "jecam": 345} and
            sum(map(len, locations.values())) == len(set.union(*locations.values())),
            dev_path, f"duplicate or cross-split locations: {distinct}")

    for filename, key in (("jecam_overlap.json", "excluded_parcels"),
                          ("test_overlap_exclusions.json", "jecam")):
        path = PREP / "quality" / filename
        require(inputs.json(path)[key] == [], path, "unexpected JECAM footprint exclusions")
        inputs.record(path, key, "excluded parcel", 0)
    exclusions = PREP / "quality/development_overlap_exclusions.json"
    inputs.record(exclusions, "source receipt for shared-location exclusions", "window recipe",
                  len(inputs.json(exclusions)))

    validation = [r for rows in populations["validation"].values() for r in rows]
    followups = [r for r in inputs.csv(FOLLOWUPS) if r["model"] == "fixed_matched_original"]
    checked = {}
    for role, phase, rows in (("validation", "locked_validation", validation),
                               ("jecam", "external_test", test["fixed_matched"])):
        saved = keyed((r for r in followups if r["phase"] == phase
                       and r["test_cohort"] == ("development" if role == "validation" else "jecam")),
                      lambda r: r["partition_group"], FOLLOWUPS)
        inputs.record(FOLLOWUPS, f"fixed_matched_original; {phase}; {role}", "partition_group", len(saved))
        checked[role] = replay(classifier, columns, rows, saved, FOLLOWUPS)
    pred = TEST / "predictions/fixed_matched_test_parcels.csv"
    primary_saved = external_report.read_predictions(ROOT / pred, campaign="jecam")
    primary_replayed = {g: ("jecam", r["crop"], r["prediction"]) for g, r in checked["jecam"].items()}
    require(primary_saved == primary_replayed, pred, "primary predictions differ by parcel ID")
    inputs.record(pred, "primary JECAM fixed_matched", "partition_group", len(primary_saved))
    combined_path = TEST / "predictions/combined/fixed_matched_test_parcels.csv"
    combined = external_report.read_predictions(ROOT / combined_path)
    revised = {g: r for g, r in combined.items() if external_report.retained_reference(*r[:2])}
    require(set(combined) == set(cohort["original_external_parcels"])
            and set(revised) == set(cohort["reported_external_parcels"])
            and combined.keys() - revised.keys() == REMOVED
            and {g: r for g, r in revised.items() if r[0] == "jecam"} == primary_saved,
            combined_path, "scope revision or unchanged primary set differs")
    inputs.record(combined_path, "original combined; verify post-review removals", "partition_group", len(combined))

    dev_sites = Path("pipeline_outputs/development_error_review_2026-11-19/parcel_diagnostics.json")
    dev_records = keyed((r for r in inputs.json(dev_sites) if r["cohort"] == "fixed_matched"),
                        lambda r: r["partition_group"], dev_sites)
    require(set(dev_records) == set(groups) and
            all(r["crop"] == groups[g][0]["crop"] and r["stage"] ==
                ("geographic_cv" if split[g] == "train" else "locked_validation")
                for g, r in dev_records.items()), dev_sites, "site join disagrees with frozen split")
    inputs.record(dev_sites, "fixed_matched; site metadata only", "partition_group", len(dev_records))
    external_sites = DIAGNOSTICS / "parcel_diagnostics.csv"
    sites = keyed((r for r in inputs.csv(external_sites) if r["model"] == "fixed_matched"
                   and r["population"] == "jecam"), lambda r: r["partition_group"], external_sites)
    require({g: ("jecam", r["reference_crop"], r["prediction"]) for g, r in sites.items()}
            == primary_saved, external_sites, "site join disagrees with frozen primary predictions")
    inputs.record(external_sites, "fixed_matched; jecam", "partition_group", len(sites))
    records = [
        {"split": split[g], "source": r["source"], "country": g.split("|")[1],
         "year": r["year"], "crop": r["crop"], "cells": sorted(set(r["site_cell"].split(";")))}
        for g, r in sorted(dev_records.items())
    ] + [
        {"split": "jecam", "source": "cropharvest", "country": r["source_site"],
         "year": int(g.split("|")[2]), "crop": r["reference_crop"],
         "cells": sorted(set(r["site_cell"].split(";")))}
        for g, r in sorted(sites.items())
    ]
    spatial = {role: {cell for r in records if r["split"] == role for cell in r["cells"]}
               for role in ("train", "validation")}
    require((len(spatial["train"]), len(spatial["validation"]),
             len(spatial["train"] & spatial["validation"])) == (42, 30, 22),
            dev_sites, "unexpected shared site cells")

    dates = audit_dates(inputs, primary, aligned_primary, sites)
    pixel, = primary[EXAMPLE_ID]
    require(pixel_location(pixel) == ("EPSG:32738", 712705, 7812555), test_path, "example location differs")
    ndvi = [pixel[f"bin_{i:02d}_NDVI"] for i in range(1, 9)]
    clear = [pixel[f"bin_{i:02d}_clear_count"] for i in range(1, 9)]
    require([round(x, 4) for x in ndvi] == [0.7523, 0.1804, 0.4640, 0.7633, 0.8716, 0.8747, 0.8402, 0.7674]
            and clear == [3, 4, 1, 3, 4, 5, 5, 5], test_path, "example NDVI or clear-date counts differ")
    basemap = Path("notebooks/assets/classification_basemap.geojson")
    inputs.record(basemap, "local display outlines", "feature position",
                  len(inputs.json(basemap)["features"]))
    support_path = PREP / "features/usable_samples.csv"
    saved_support = {(r["partition"], r["crop"]):
                     [int(r["usable_parcel_groups"]), int(r["usable_pixels"])]
                     for r in inputs.csv(support_path) if r["cohort"] == "fixed_matched"
                     and r["partition"] in ("train", "validation")}
    require(saved_support == {(role, crop): support[role][crop]
                              for role in ("train", "validation") for crop in CROPS},
            support_path, "development support differs from the feature snapshot")
    inputs.record(support_path, "fixed_matched; train and validation", "split and crop",
                  len(saved_support))
    for folder, names in ((ORIGINAL, ("confusion_matrices.csv", "paired_macro_f1.csv", "parcel_metrics.csv")),
                          (REVISED, ("parcel_metrics.csv",))):
        for filename in names:
            path = folder / filename
            rows = [r for r in inputs.csv(path) if r["model"] == "fixed_matched"
                    and r["population"] in ("jecam", "combined")]
            inputs.record(path, "fixed_matched; jecam and combined", "saved report CSV row", len(rows))
        path = folder / "methods.json"
        method = inputs.json(path)
        require(method["repetitions"] == 10000 and method["seed"] == 42
                and method["numpy_version"] == runtime["numpy"],
                path, "bootstrap runtime or recipe differs")
        inputs.record(path, "archived bootstrap configuration", "methods", 1)
    return {
        "sources": inputs.sources, "runtime": runtime, "class_order": FITTED_CLASSES,
        "feature_count": len(columns), "model_parameters": params, "verified_artifacts": verified,
        "support": support, "distinct_centres": distinct, "sites": records, "dates": dates,
        "example": {"id": EXAMPLE_ID, "location": pixel_location(pixel), "bins": window_intervals(pixel),
                    "ndvi": ndvi, "clear": clear, "red": pixel["bin_01_B4"], "nir": pixel["bin_01_B8"]},
        "quality": audit_quality(inputs, support["jecam"]), "paoay": local_evidence(inputs),
        "replay": checked, "removed_combined_ids": sorted(REMOVED),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pipeline_outputs/defense_fixed_matched/evidence.json")
    args = parser.parse_args()
    evidence = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Verified frozen replay: 69/69 locked validation, 83/83 primary JECAM.")
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
