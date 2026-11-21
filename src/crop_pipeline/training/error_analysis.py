"""Batch 3 joins and reports for the original frozen classification experiment."""

from collections import Counter, defaultdict
from csv import DictReader, field_size_limit
from datetime import date, timedelta
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union

from crop_pipeline.common import save_json
from crop_pipeline.paoay.predictions import load_classifier
from crop_pipeline.sources.test_cohort import source_footprints
from crop_pipeline.training import forest_diagnostics, notebook_workflow, source_holdout
from crop_pipeline.training.external_report import (
    MODELS, REVISED_POPULATION_NAMES, read_predictions, retained_reference, write_csv,
)
from crop_pipeline.windows.seasons import file_sha256, window_intervals

CROPS = source_holdout.CROPS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_rows(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def observation_date(audit, year):
    dates = set()
    for item in audit.get("raw_matches", ()):
        if item.get("collection_date"):
            day = date.fromisoformat(item["collection_date"][:10])
            # January 1 in these references is annual metadata.
            if (day.month, day.day) != (1, 1):
                dates.add(day)
        elif item.get("MAIN_CROP_DATE"):
            day = date(year, 1, 1) + timedelta(days=int(item["MAIN_CROP_DATE"]) - 1)
            if day.year == year:
                dates.add(day)
    return next(iter(dates)) if len(dates) == 1 else None


@lru_cache
def grid_transform(crs):
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform


def boundary_distance(rows, geometry):
    if geometry is None:
        return None
    distances = []
    for crs, pixels in group_rows(rows, "grid_crs").items():
        projected = transform(grid_transform(crs), geometry)
        distances.extend(projected.boundary.distance(Point(row["x"], row["y"])) for row in pixels)
    return min(distances)


def parcel_evidence(cohort, rows, sealed, votes, audits, quality, footprints, region_grids):
    by_parcel = group_rows(rows, "partition_group")
    if by_parcel.keys() != sealed.keys() or by_parcel.keys() != votes.keys():
        raise ValueError(f"{cohort}: external features and prediction IDs differ")
    result = []
    for parcel, pixels in sorted(by_parcel.items()):
        campaign, actual, predicted = sealed[parcel]
        if {(row["test_cohort"], row["crop"]) for row in pixels} != {(campaign, actual)}:
            raise ValueError(f"{cohort}: frozen reference label or campaign differs for {parcel}")
        hard = votes[parcel]
        if hard["probability_prediction"] != predicted:
            raise ValueError(f"{cohort}: forest probability prediction differs for {parcel}")
        if not retained_reference(campaign, actual):
            continue
        audit = audits[cohort, parcel]
        if audit["pixels"] != len(pixels) or audit["crop"] != actual:
            raise ValueError(f"{cohort}: saved reference audit differs for {parcel}")

        windows = sorted({(row["region_id"], row["window_start"], row["window_end"]) for row in pixels})
        records = [quality.get((region, item["name"])) for region, start, end in windows
                   for item in window_intervals({"window_start": start, "window_end": end})]
        clear = np.array([[row[f"bin_{i:02d}_clear_count"] for i in range(1, 9)] for row in pixels])
        ndvi = np.median([[row[f"bin_{i:02d}_NDVI"] for i in range(1, 9)] for row in pixels], axis=0)
        ref = observation_date(audit, int(pixels[0]["source_year"]))
        known = [record for record in records if record and record.get("pre_mask_count") is not None]
        shadow_counts = [record["cloud_shadow_count"] for record in known
                         if record.get("cloud_shadow_count") is not None]
        distance = boundary_distance(
            [dict(row, grid_crs=region_grids[row["region_id"]]) for row in pixels], footprints.get(parcel))
        sites = sorted({row["site_cell"] for row in pixels})
        source_site = parcel.split("|")[1]
        if campaign == "jecam":
            jecam_sites = {"-20,46": "Madagascar", "-20,47": "Madagascar",
                           "13,-15": "Senegal", "15,-17": "Senegal"}
            source_site = ";".join(sorted({jecam_sites.get(site, site) for site in sites}))
        result.append({
            "model": cohort, "population": campaign, "partition_group": parcel,
            "source": pixels[0]["source_dataset"], "source_site": source_site,
            "site_cell": ";".join(sites),
            "reference_crop": actual, "prediction": predicted, "correct": actual == predicted,
            "usable_pixels": len(pixels), "window_starts": ";".join(sorted({start for _, start, _ in windows})),
            "window_ends_exclusive": ";".join(sorted({end for _, _, end in windows})),
            "reference_date": ref.isoformat() if ref else None,
            "reference_date_in_window": any(start <= ref.isoformat() < end for _, start, end in windows)
            if ref else None,
            "min_clear_observations_in_a_pixel_bin": int(clear.min()),
            "mean_clear_observations_per_pixel_bin": float(clear.mean()),
            "one_clear_observation_fraction": float((clear == 1).mean()),
            "window_region_bins": len(records), "missing_quality_bins": sum(r is None for r in records),
            "pre_mask_counts_available_bins": len(known),
            "cloud_or_shadow_rejections_observed": sum(shadow_counts) if shadow_counts else None,
            "min_pixel_center_to_original_boundary_m": distance,
            "ndvi_peak_bin": int(ndvi.argmax() + 1),
            "ndvi_peak_at_window_edge": int(ndvi.argmax()) in (0, 7),
            "mixed_field_status": "unknown", "terrain_shadow_status": "unknown",
            "crop_transition_status": "unknown", "error_cause": "unknown" if actual != predicted else None,
            **hard,
        })
    return result


def campaign_summary(parcels):
    summary = []
    for model in MODELS:
        cohort = [row for row in parcels if row["model"] == model]
        for scope in REVISED_POPULATION_NAMES:
            for crop in CROPS:
                rows = [row for row in cohort if row["reference_crop"] == crop
                        and (scope == "combined" or row["population"] == scope)]
                if not rows:
                    continue
                pred = Counter(row["prediction"] for row in rows)
                dated = [row for row in rows if row["reference_date"]]
                summary.append({
                    "model": model, "population": scope, "crop": crop, "parcels": len(rows),
                    "usable_pixels": sum(row["usable_pixels"] for row in rows),
                    "errors": sum(not row["correct"] for row in rows),
                    **{f"predicted_{name}": pred[name] for name in CROPS},
                    "one_pixel_parcels": sum(row["usable_pixels"] == 1 for row in rows),
                    "mean_one_clear_observation_fraction": float(np.mean(
                        [row["one_clear_observation_fraction"] for row in rows])),
                    "reference_dates_available": len(dated),
                    "reference_dates_in_window": sum(row["reference_date_in_window"] for row in dated),
                    "hard_vote_winner_differs": sum(row["hard_vote_winner"] != row["prediction"] for row in rows),
                })
    return summary


def matched_outcomes(parcels):
    by_model = {model: {row["partition_group"]: row for row in parcels if row["model"] == model}
                for model in ("fixed_matched", "aligned_matched")}
    fixed, aligned = by_model.values()
    if fixed.keys() != aligned.keys():
        raise ValueError("Matched parcel IDs differ")
    result = []
    for group, a in fixed.items():
        b = aligned[group]
        if any(a[key] != b[key] for key in ("reference_crop", "population", "usable_pixels")):
            raise ValueError("Matched reference metadata or pixel support differs")
        status = ("both_correct" if a["correct"] and b["correct"] else
                  "fixed_only_correct" if a["correct"] else "aligned_only_correct" if b["correct"] else "both_wrong")
        result.append({
            "partition_group": group, "population": a["population"], "reference_crop": a["reference_crop"],
            "usable_pixels": a["usable_pixels"], "fixed_prediction": a["prediction"],
            "aligned_prediction": b["prediction"], "outcome": status,
        })
    return result


def cabbage_trace(development, cycle_rows):
    train = {
        key: {row["partition_group"] for row in development[key]
              if row["partition"] == "train" and row["crop"] == "cabbage"}
        for key in ("fixed_full", "aligned_full")
    }
    if {row["source_dataset"] for rows in development.values() for row in rows
            if row["partition"] == "train" and row["crop"] == "cabbage"} != {"francecrops"}:
        raise ValueError("Unexpected cabbage training source")
    cycle = group_rows([row for row in cycle_rows if row["crop"] == "cabbage"], "partition_group")
    fixed, aligned = train.values()
    results = []
    for group in sorted(fixed | aligned):
        records = cycle.get(group, ())
        status = ("both_full_windows" if group in fixed and group in aligned else
                  "aligned_full_only" if group in aligned else
                  "cycle_audit_no_usable_aligned_window" if records else "no_cycle_selection_record")
        results.append({
            "partition_group": group, "source": "francecrops", "outcome": status,
            "cycle_audit_records": len(records),
            "cycle_selection_statuses": ";".join(sorted({r["cycle_selection_status"] for r in records})),
            "cycle_selection_reasons": ";".join(sorted({r["cycle_selection_reason"] for r in records})),
        })
    return results


def imagery_coverage(regions, quality_rows, summary):
    """Summarize archived candidate support and missing mask counts by campaign/crop."""
    locations = defaultdict(list)
    for feature in regions:
        row = feature["properties"]
        campaign = row.get("test_cohort", "jecam")
        if row["partition"] == "test" and retained_reference(campaign, row["crop"]):
            locations[campaign, row["crop"]].append(row)
    lookup = group_rows(quality_rows, "region_id")
    result = []
    for scope in REVISED_POPULATION_NAMES:
        if scope == "combined":
            continue
        for crop in CROPS:
            rows = locations.get((scope, crop))
            if not rows:
                continue
            intervals = [item for row in rows for item in lookup.get(row["fragment_id"], ())]
            mask_counts = [item for item in intervals if item.get("pre_mask_count") is not None]
            archived = {
                "population": scope, "crop": crop,
                "candidate_parcels_in_quality_snapshot": len({row["partition_group"] for row in rows}),
                "candidate_fragment_pixels": sum(row["selected_pixels"] for row in rows),
                "candidate_region_bins_saved": len(intervals),
                "candidate_region_bins_quality_pass": sum(item["status"] == "quality_pass" for item in intervals),
                "region_bins_with_pre_mask_counts": len(mask_counts),
                "region_bins_without_pre_mask_counts": len(intervals) - len(mask_counts),
                "observed_cloud_or_shadow_rejections": sum(item["cloud_shadow_count"] for item in mask_counts
                    if item.get("cloud_shadow_count") is not None)
                if any(item.get("cloud_shadow_count") is not None for item in mask_counts) else None,
            }
            for cohort in MODELS:
                support = next((row for row in summary if (row["model"], row["population"], row["crop"])
                                == (cohort, scope, crop)), None)
                usable = support["parcels"] if support else 0
                result.append({**archived, "model": cohort, "usable_parcels": usable,
                               "usable_pixels": support["usable_pixels"] if support else 0,
                               "candidate_parcels_absent_from_cohort":
                               archived["candidate_parcels_in_quality_snapshot"] - usable})
    return result


def archive_mask_coverage(regions, quality_rows):
    """Keep partial counts and missing legacy counts visible by source and split."""
    properties = {row["properties"]["fragment_id"]: row["properties"] for row in regions}
    groups = defaultdict(list)
    for row in quality_rows:
        info = properties[row["region_id"]]
        partition = info["partition"]
        campaign = info.get("test_cohort", "jecam") if partition == "test" else ""
        if partition == "test" and not retained_reference(campaign, info["crop"]):
            continue
        groups[partition, campaign, info["source_dataset"], info["crop"]].append(row)
    results = []
    for (partition, campaign, source, crop), rows in sorted(groups.items()):
        available = [row for row in rows if row.get("pre_mask_count") is not None]
        rejections = [row["cloud_shadow_count"] for row in available
                      if row.get("cloud_shadow_count") is not None]
        results.append({
            "partition": partition, "population": campaign, "source": source, "crop": crop,
            "saved_region_bins": len(rows), "region_bins_with_pre_mask_counts": len(available),
            "region_bins_without_pre_mask_counts": len(rows) - len(available),
            "region_bins_with_cloud_or_shadow_counts": len(rejections),
            "observed_cloud_or_shadow_rejections": sum(rejections) if rejections else None,
        })
    return results


def inspect_batch3(training_run, testing_run, preprocessing, audit_path, output):
    """Read frozen inputs and write a new Batch 3 report and traceable tables."""
    training_run, testing_run, output = (path.resolve() for path in (training_run, testing_run, output))
    if any(output.is_relative_to(home) for home in (training_run, testing_run, preprocessing.resolve())):
        raise ValueError("Use a new output directory outside the frozen inputs")
    models = training_run / "models"
    model_stage, _ = notebook_workflow._verify(models / "models_stage.json", "models_stage")
    test_stage, receipt = notebook_workflow._verify(testing_run / "scores.json", "scores")
    dev_file = training_run / "features_development.json"
    test_file = testing_run / "features_test.json"
    spine = read(models / "development.json")
    if (receipt["models_handoff_sha256"] != file_sha256(model_stage)
            or spine["feature_sha256"] != file_sha256(dev_file)):
        raise ValueError("Frozen forests or their development snapshot changed")
    development = read(dev_file)
    test = read(test_file)
    if any(test[key] != spine[key] or development[key] != spine[key]
           for key in ("quality", "window_sha256", "fixed_start_month")):
        raise ValueError("Development and external snapshots have different preprocessing provenance")
    source_holdout.require_matching(development["rows"])
    source_holdout.require_matching(test["rows"])
    quality_file = preprocessing / "quality/quality.json"
    if quality_file.resolve() != Path(test["quality"]).resolve():
        raise ValueError("Quality evidence does not belong to the frozen feature snapshot")
    quality = read(quality_file)
    region_grids = {row["properties"]["fragment_id"]: row["properties"]["sampling_grid"]["crs"]
                    for row in quality["regions"] if row["properties"].get("sampling_grid")}
    quality_by_bin = {(row["region_id"], row["name"]): row for row in quality["quality_rows"]}
    audits = {(row["cohort"], row["group"]): row for row in read(audit_path)}
    archive_geometries = defaultdict(list)
    for feature in source_footprints(quality["sources"]):
        group = feature["properties"]["partition_group"]
        if any((cohort, group) in audits for cohort in MODELS):
            archive_geometries[group].append(shape(feature["geometry"]))
    footprints = {group: unary_union(regions) for group, regions in archive_geometries.items()}
    cycle_path = preprocessing / "windows/seasons/cycle_selection_audit.csv"
    previous_limit = field_size_limit()
    try:
        # Older audit rows embed a complete source-revision record in one field.
        field_size_limit(max(previous_limit, cycle_path.stat().st_size))
        with cycle_path.open(newline="", encoding="utf-8") as stream:
            cabbage = cabbage_trace(development["rows"], DictReader(stream))
    finally:
        field_size_limit(previous_limit)

    output.mkdir(parents=True, exist_ok=False)
    parcel_records, pixels, oob, trees, oob_parcels, importance, grouped = ([] for _ in range(7))
    for cohort in MODELS:
        dev_rows = source_holdout.checked_rows(development["rows"][cohort], False)
        train = [row for row in dev_rows if row["partition"] == "train"]
        path = models / cohort / "model_manifest.json"
        manifest = read(path)
        if {row["partition_group"]: row["partition"] for row in dev_rows} != manifest["development_split"]:
            raise ValueError("Frozen parcel split differs from model manifest")
        rows = source_holdout.checked_rows(test["rows"][cohort], True)
        if {row["partition_group"] for row in rows} & manifest["development_split"].keys():
            raise ValueError("External rows overlap development")
        model, recipe = load_classifier(path)
        columns = recipe["feature_columns"]
        oob_summary, tree_rows, oob_rows = forest_diagnostics.oob_statistics(model, train, columns)
        votes, pixel_votes = forest_diagnostics.hard_vote_statistics(model, rows, columns)
        sealed = read_predictions(testing_run / "predictions/combined" / f"{cohort}_test_parcels.csv")
        parcel_records.extend(parcel_evidence(cohort, rows, sealed, votes, audits, quality_by_bin, footprints, region_grids))
        pixels.extend({"model": cohort, "population": sealed[row["partition_group"]][0], **row}
                      for row in pixel_votes if retained_reference(sealed[row["partition_group"]][0], row["crop"]))
        detail, totals = forest_diagnostics.impurity_importance(model, columns)
        for destination, records in ((oob, oob_summary), (trees, tree_rows), (oob_parcels, oob_rows),
                                     (importance, detail), (grouped, totals)):
            destination.extend({"model": cohort, **row} for row in records)
        print(f"{cohort}: reproduced {len(sealed)} saved predictions and inspected {len(model.estimators_)} trees",
              flush=True)

    summary = campaign_summary(parcel_records)
    matched = matched_outcomes(parcel_records)
    losses = imagery_coverage(quality["regions"], quality["quality_rows"], summary)
    archive_masks = archive_mask_coverage(quality["regions"], quality["quality_rows"])
    if any(row["candidate_parcels_absent_from_cohort"] < 0 for row in losses):
        raise ValueError("Eligible parcel count exceeds saved candidate inventory")
    # Select traceable examples from both outcomes and prioritize low pixel support.
    examples = []
    for rows in group_rows(parcel_records, "model").values():
        for crop in CROPS:
            for correct in (False, True):
                pool = [row for row in rows if row["reference_crop"] == crop and row["correct"] == correct]
                examples.extend(sorted(pool, key=lambda r: (r["usable_pixels"], r["partition_group"]))[:2])

    for name, records in {
        "parcel_diagnostics": parcel_records, "campaign_crop_errors": summary,
        "matched_window_outcomes": matched, "imagery_coverage": losses, "archive_mask_coverage": archive_masks,
        "representative_parcels": examples,
        "training_cabbage_trace": cabbage, "oob_summary": oob, "oob_tree_draws": trees,
        "oob_parcels": oob_parcels, "feature_importance": importance, "grouped_importance": grouped,
        "pixel_hard_votes": pixels,
    }.items():
        write_csv(output / f"{name}.csv", records)
    report = output / "report.md"
    report.write_text(render_report(summary, matched, losses, archive_masks, cabbage, oob, grouped, examples), encoding="utf-8")
    save_json(output / "inputs.json", {
        "frozen_models_stage_sha256": file_sha256(model_stage), "testing_stage_sha256": file_sha256(test_stage),
        "development_features_sha256": file_sha256(dev_file), "test_features_sha256": file_sha256(test_file),
        "quality_sha256": file_sha256(quality_file), "reference_audit_sha256": file_sha256(audit_path),
        "cycle_audit_sha256": file_sha256(cycle_path), "populations": list(REVISED_POPULATION_NAMES),
    })
    return report


def render_report(summary, matched, coverage, archive_masks, cabbage, oob, grouped, examples):
    lines = [
        "# Batch 3 — Error analysis and forest diagnostics", "",
        "The four original frozen forests were inspected on the revised Batch 1 reporting scope. "
        "All original external probability-based predictions reproduced. JECAM is the primary test. "
        "The combined result pools JECAM and its three reported supplements.",
        "", "## External errors and imagery support", "",
        "| Population | Model | Crop | Parcels | Usable pixels | Errors | One-pixel parcels | "
        "Mean fraction of pixel-bins with one clear date | Reference date inside / available |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in REVISED_POPULATION_NAMES:
        for row in summary:
            if row["population"] == scope:
                values = [REVISED_POPULATION_NAMES[scope], row["model"], row["crop"],
                          row["parcels"], row["usable_pixels"], row["errors"], row["one_pixel_parcels"],
                          f"{row['mean_one_clear_observation_fraction']:.1%}",
                          f"{row['reference_dates_in_window']} / {row['reference_dates_available']}"]
                lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += [
        "", "A dated observation falling inside a window establishes timing overlap. "
        "The date alone cannot establish crop occupancy throughout the window. January 1 annual "
        "placeholders and absent dates are treated as unavailable.",
        "", "## Matched-window outcomes", "",
        "| Population | Both correct | Fixed only correct | Aligned only correct | Both wrong |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    outcomes = ("both_correct", "fixed_only_correct", "aligned_only_correct", "both_wrong")
    for scope, name in REVISED_POPULATION_NAMES.items():
        counts = Counter(row["outcome"] for row in matched if scope == "combined" or row["population"] == scope)
        lines.append("| " + " | ".join([name, *(str(counts[key]) for key in outcomes)]) + " |")
    lines += [
        "", "## Representative errors", "",
        "Examples prioritize low usable pixel counts. The parcel table also includes correct examples, crop "
        "predictions, distances from usable pixel centres to original parcel boundaries, window dates, and NDVI "
        "peak bins. Boundary proximity can flag possible mixed pixels. Window-edge NDVI peaks can flag unresolved "
        "cycle timing. Mixed occupancy, terrain shadow, and crop transitions remain unknown in these examples. "
        "Archived mask counts, where available, combine cloud and shadow and do not isolate terrain shadow.",
        "", "| Model | Parcel ID | Reference → predicted | Pixels | Minimum boundary distance | NDVI peak bin |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        row = next((item for item in examples if item["model"] == model and not item["correct"]), None)
        if row:
            parcel = row["partition_group"].replace("|", "\\|")
            distance = row["min_pixel_center_to_original_boundary_m"]
            offset = f"{distance:.1f} m" if distance is not None else "unavailable"
            lines.append(f"| {model} | `{parcel}` | {row['reference_crop']} → {row['prediction']} | "
                         f"{row['usable_pixels']} | {offset} | {row['ndvi_peak_bin']} |")
    archive = [row for row in coverage if row["model"] == MODELS[0]]
    available = sum(row["region_bins_with_pre_mask_counts"] for row in archive)
    missing = sum(row["region_bins_without_pre_mask_counts"] for row in archive)
    missing_development = sum(row["region_bins_without_pre_mask_counts"] for row in archive_masks
                              if row["partition"] != "test")
    lines += [
        "", "## Imagery losses and training cabbage", "",
        f"Across the retained external campaigns, {missing:,} candidate region-bins have unavailable pre-mask "
        f"counts and {available:,} have recorded counts. The coverage table distinguishes candidate support, "
        "quality pass, and final model support. Candidate parcels absent from a cohort can reflect multiple "
        f"screening stages. Across development, {missing_development:,} saved region-bins lack pre-mask counts. "
        "The archive mask table reports available counts by source, split, and crop. "
        "Missing legacy counts are unavailable.",
        "", "All training cabbage in these forests came from FranceCrops. The retired Dutch cabbage campaign "
        "was a separate test campaign.",
        "", "| Training cabbage outcome | Parcels |", "| --- | ---: |",
    ]
    for outcome, count in sorted(Counter(row["outcome"] for row in cabbage).items()):
        lines.append(f"| {outcome.replace('_', ' ')} | {count} |")
    lines += [
        "", "## Internal out-of-bag error", "",
        "Per-tree samples were reconstructed with the saved tree seeds, ordered training rows, and weighted "
        "half-size bootstrap. Pixel OOB averages probabilities from trees that omitted a pixel; its parcel "
        "summary averages those pixel probabilities. Either can benefit from neighboring pixels of the same parcel "
        "being in a tree's training draw. Whole-parcel OOB uses only trees that omitted every pixel of that parcel. "
        "It remains an internal diagnostic with uneven tree support and spatial dependence across parcels. "
        "These results are separate from the Batch 1 external confidence intervals.",
        "", "| Model | Diagnostic | Scored / eligible units | Error | Mean contributors |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in oob:
        error = f"{row['error_rate']:.1%}" if row["error_rate"] is not None else "unavailable"
        contributors = (f"{row['mean_contributors']:.1f} {row['contributor_unit']}"
                        if row["mean_contributors"] is not None else "unavailable")
        lines.append(f"| {row['model']} | {row['method']} | {row['scored_units']} / {row['eligible_units']} | "
                     f"{error} | {contributors} |")
    lines += [
        "", "## Grouped importance and hard votes", "",
        "Grouped importance sums impurity importance by spectral variable and by half-month bin. "
        "Correlated predictors can share importance. The table gives the largest group in each view.",
        "", "| Model | Top band or index | Share | Top bin | Share | Vote winner differs / external parcels |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        band, bin_ = [max((row for row in grouped if row["model"] == model and row["grouping"] == key),
                          key=lambda row: row["importance"])
                      for key in ("band_or_index", "half_month_bin")]
        counts = [row for row in summary if row["model"] == model and row["population"] == "combined"]
        lines.append(f"| {model} | {band['group']} | {band['importance']:.1%} | {bin_['group']} | "
                     f"{bin_['importance']:.1%} | {sum(r['hard_vote_winner_differs'] for r in counts)} / "
                     f"{sum(r['parcels'] for r in counts)} |")
    lines += [
        "", "Each hard pixel vote uses a tree's highest-probability crop. Each hard parcel vote first averages "
        "that tree's probabilities across the parcel pixels. Ties follow frozen class order. The parcel table "
        "records each crop's vote fraction, the largest fraction, and the hard vote fraction for the original "
        "probability-based prediction. External accuracy uses the original predictions.", "",
    ]
    return "\n".join(lines)
