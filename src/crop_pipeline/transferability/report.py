"""Text tables for sections 3, 8 and 9: dataset specification, exclusion funnel, limitations and acceptance criteria."""

import numpy as np
import pandas as pd

from .atok import FIELDS, LABEL_ROWS
from .config import (ATOK_ELEVATION_M, CLASSES, LAPSE_C_PER_M, MIN_CLEAR, MIN_PARCELS_PER_CROP, OOD_QUANTILE, PROTOTYPE,
                     TARGET_SELECTIVE_ACCURACY)
from .external import MATCH_RADIUS_M, STARCH_POTATO
from .maps import MAP_WINDOWS

# (item, specification, status); {placeholders} are filled from config
DATASET_SPECIFICATION = [
    ("Classes", "carrot, cabbage, potato", "implemented"),
    ("Classes: others", "built-up + non-crop as one class from NAMRIA ELUM CY-2023 non-crop polygons; drone imagery to check",
     "decided; not trained (no external rows)"),
    ("Classes: other crops", "lettuce, radish, seedbeds (B2, Area 5) need an other_crop class", "not modelled"),
    ("Unknown (maps)", "calibrated top-class probability below the threshold for {target:.0%} selective accuracy (4.6b, per model), "
     "or out-of-distribution (kNN distance above the {ood:.0%} quantile of the external test set); minimum coverage requirement 50%",
     "implemented"),
    ("Predictors", "interval: 8 half-months x (B3, B4, B5, B8, B11, NDVI, NDRE, NDMI, SAVI) = 72; shape: 48 temporal-shape features; "
     "both classifiers run on both sets", "implemented"),
    ("Prototype", "RF on interval predictors (1,000 trees, sqrt, min leaf 2, 50% bootstrap): prespecified; tuned RF and LR reported alongside",
     "implemented"),
    ("Temporal matching: external", "Sen4AgriNet/FranceCrops: 8 half-months from detected onset (notebook 04); CropHarvest: window dates not recorded",
     "as exported"),
    ("Temporal matching: Atok", "8 half-months from the calendar planting month: Mar (potato, B1 carrot), Jul (cabbage), Nov (carrot)",
     "implemented"),
    ("Clear-data rule", "external: complete windows only; Atok: >= {min_clear} of 8 clear, gaps interpolated (sensitivity 5-8 in 6.2; "
     "8 = no interpolation)", "implemented"),
    ("Climate experiment", "primary: external target regions (verified labels), source sets with >= {min_parcels} parcels of each target crop, "
     "per-target Spearman and drop-one-source robustness; Atok: exploratory only", "implemented"),
    ("Split: external", "parcels within 10 m linked; 20% of units per crop x source locked as test (v1 file reused); "
     "grouped 5-fold CV with the same folds everywhere", "implemented"),
    ("Split: external transfer", "leave-one-region-out and single-region train x test matrix", "implemented"),
    ("Split: Atok", "field-independent holdout needs >= 2 fields per crop; not possible (potato: 1 field), so the local benchmark is exploratory",
     "blocked by data"),
    ("Weighting", "equal total weight per crop, split across parcels; reported labels 0.5; source-balanced variant tested in 4.3", "implemented"),
    ("Cropland mask", "NAMRIA ELUM CY-2023 'Crops' polygons within the ROI", "implemented"),
    ("Exclusion rules", "see the funnel and record-level table below", "implemented"),
]


def dataset_specification():
    values = {"target": TARGET_SELECTIVE_ACCURACY, "ood": OOD_QUANTILE, "min_clear": MIN_CLEAR, "min_parcels": MIN_PARCELS_PER_CROP}
    return pd.DataFrame([(item, spec.format(**values), status) for item, spec, status in DATASET_SPECIFICATION],
                        columns=["item", "specification", "status"])


def exclusion_funnel(catalogue, ch_match, ch_rows, audits, data, tables, atok, atok_windows):
    """Counts at each selection step per source and unit, with the share retained."""
    exported = set(ch_match.loc[ch_match["distance_m"] <= MATCH_RADIUS_M, "dataset"])
    starch = catalogue["label"].astype(str).str.lower().str.contains(STARCH_POTATO)
    in_export = catalogue["dataset"].isin(exported)
    timing, usable = audits["timing"], audits["usable"]
    steps = [  # source, step, unit, count, rule
        ("CropHarvest", "target-crop labels in the catalogue", "labels", len(catalogue),
         "carrot/cabbage/potato names; sweet potato, Chinese cabbage, cauliflower excluded"),
        ("CropHarvest", "in an exported dataset", "labels", in_export.sum(), "canada, croplands, kenya not exported"),
        ("CropHarvest", "table potatoes only", "labels", (in_export & ~starch).sum(), "starch potato dropped"),
        ("CropHarvest", "parcels in the table", "parcels", ch_rows["partition_group"].nunique(), ""),
        ("CropHarvest", "pixels in the table", "pixels", len(ch_rows), "complete windows only"),
        ("Sen4AgriNet + FranceCrops", "onset candidates", "parcel-seasons", len(timing), ""),
        ("Sen4AgriNet + FranceCrops", "passed timing checks", "parcel-seasons", (timing["automatic_timing_status"] == "passed").sum(),
         "automatic onset timing checks"),
        ("Sen4AgriNet + FranceCrops", "window selected", "parcel-seasons", len(audits["windows"]), "earliest complete window per parcel"),
        ("Sen4AgriNet + FranceCrops", "parcels in the table", "parcels", data.loc[data["domain"] != "cropharvest", "partition_group"].nunique(), ""),
        ("Sen4AgriNet + FranceCrops", "usable pixels available", "pixels", usable["available_usable_pixels"].sum(), "complete windows"),
        ("Sen4AgriNet + FranceCrops", "pixels in the table", "pixels", usable["selected_usable_pixels"].sum(), "cap: 500 pixels per crop"),
        ("Atok (provisional)", "interior pixels x labelled windows", "pixel-windows", atok_windows["pixel_windows"].sum(),
         "10 m inward buffer; 7 field-seasons x 2019-2025"),
        ("Atok (provisional)", f">= {MIN_CLEAR} of 8 clear composites", "pixel-windows", len(atok), "clear-data rule"),
        ("This notebook", "external rows in", "pixels", len(tables["cropharvest"]) + len(tables["francecrops_s4a"]), ""),
        ("This notebook", "after harmonisation checks", "pixels", len(data),
         "non-finite, duplicate, unknown crop or label status: stop, never drop"),
    ]
    funnel = pd.DataFrame([{"source": s, "step": n, "unit": u, "count": int(c), "rule": r} for s, n, u, c, r in steps])
    first = funnel.groupby(["source", "unit"])["count"].transform("first")
    funnel["dropped"] = -funnel.groupby(["source", "unit"])["count"].diff().fillna(0).astype(int)
    funnel["retained_%"] = (100 * funnel["count"] / first).round(1)
    return funnel


def cropharvest_record_status(catalogue, ch_match):
    """Why each CropHarvest target-crop label is or is not in the training table."""
    exported = set(ch_match.loc[ch_match["distance_m"] <= MATCH_RADIUS_M, "dataset"])
    matched_index = set(ch_match.loc[ch_match["distance_m"] <= MATCH_RADIUS_M, "index"])
    starch = catalogue["label"].astype(str).str.lower().str.contains(STARCH_POTATO)
    records = catalogue[["index", "dataset", "label", "target_crop", "lat", "lon"]].copy()
    records["status"] = np.select(
        [~records["dataset"].isin(exported), starch.to_numpy(), records["index"].isin(matched_index)],
        ["dropped: dataset not exported", "dropped: starch potato", "kept: matched to a training-table parcel"],
        "dropped: not in table (screening or sampling; reason not logged at export)")
    return records


def crop_distribution_before_after(records, ch_rows):
    return pd.DataFrame({"catalogue labels": records["target_crop"].value_counts(),
                         "table parcels": ch_rows.groupby("crop")["partition_group"].nunique()}).reindex(CLASSES)


def best_atok_model(atok_results):
    """(results indexed by model, majority-baseline name, best model by balanced accuracy)."""
    table = atok_results.set_index("model")
    majority = next(m for m in table.index if m.startswith("Baseline: always"))
    best = table.drop(index=[m for m in table.index if m.startswith("Baseline")])["balanced_acc"].idxmax()
    return table, majority, best


def limitations(r):
    """Limitation, evidence computed in this notebook, requirement and priority. `r` holds the notebook's results."""
    atok, atok_units, grid = r["atok"], r["atok_units"], r["grid"]
    table, majority, best = best_atok_model(r["atok_results"])
    month_clear, mappable, best_year, worst_year = r["month_clear"], r["mappable"], r["best_year"], r["worst_year"]
    may_aug = np.mean([month_clear[m] for m in (5, 6, 7, 8)])
    other_months = np.mean([month_clear[m] for m in (1, 2, 3, 4, 9, 10, 11, 12)])
    prototype_test = r["test_results"].set_index("model").loc[PROTOTYPE]
    grid_table, calibration_fit, transfer = r["grid_table"], r["calibration"]["table"], r["threshold_transfer"]
    causes, tested = r["rejection_causes"], r["target_stats"].dropna(subset=["rho_climate"])
    rejected = causes["out of distribution only"] + causes["both"]
    seasonal = [(f, c, m) for f, c, m, basis in LABEL_ROWS if basis.startswith("rotation")]
    outside_visit = [f"{f} {c}" for f, c, m in seasonal if m not in (11, 12, 1, 2)]
    labelled_area = [grid.field_area_ha[f] for f in FIELDS]
    baguio_shift = LAPSE_C_PER_M * (ATOK_ELEVATION_M - r["pagasa_elevation_m"])
    agreement = r["agreement"]["agree_share_of_known"]
    rows = [
        ("Atok labels are not ground truth", "undated advisor rotations repeated 2019-2025; 0 verified fields",
         "Dated, field-checked crop observations per plot and season, kept separate from calendar labels", "Critical"),
        ("Too few independent Atok fields",
         "; ".join(f"{c}: {int(atok_units.loc[c, 'independent fields'])}" for c in CLASSES) + f" fields ({len(atok):,} pixel-windows, not independent)",
         "At least 5 verified fields per crop; report fields and field-seasons as the sample size", "Critical"),
        ("No field-independent Atok holdout", "potato in 1 field; local benchmark shares fields between train and test",
         "Field-independent holdout with >= 2 test fields per crop", "Critical"),
        ("Majority baseline beats the models on Atok OA",
         f"always-{majority.split()[-1]} OA {table.loc[majority, 'OA']:.2f}; best model by balanced accuracy ({best}) "
         f"OA {table.loc[best, 'OA']:.2f}, balanced acc {table.loc[best, 'balanced_acc']:.2f}",
         "Report balanced accuracy, per-crop recall and baselines; judge models by the acceptance criteria (9)", "High"),
        ("Source confound", f"France -> CropHarvest: CropHarvest France rows score {r['france_accuracy']:.3f} accuracy (v1), "
         "same country and climate, different source",
         "Cross-source coverage for every crop; test source harmonisation separately from climate", "High"),
        ("Overfitting / source shift", f"prototype train-validation macro-F1 gap {r['prototype_gap']:.2f} in grouped CV; "
         f"best tuned gap {grid_table.iloc[0]['gap']:.2f}",
         "More parcels per crop and source; stricter (region-level) validation; source-balanced training (4.3)", "High"),
        ("Small external test", f"prototype test macro-F1 {prototype_test['macro_F1']:.3f} (95% parcel CI {prototype_test['macro_F1_95CI']}), "
         f"{r['test_parcels']} test parcels", "More parcels; repeated grouped splits", "High"),
        ("Domain shift to Atok", f"{atok['ood'].mean():.0%} of Atok pixel-windows out of distribution (vs {1 - OOD_QUANTILE:.0%} of external test)",
         "Training data from comparable highland systems; local labelled data", "High"),
        ("May-Aug cloud gap", f"{may_aug:.0%} of ROI half-months clear in May-Aug vs {other_months:.0%} otherwise; "
         f"{mappable.loc[best_year].iloc[1]:.0%} of cropland mappable Jul-Oct even in {best_year}, "
         f"{mappable.loc[worst_year].iloc[1]:.0%} in {worst_year}", "Add Sentinel-1 SAR (VV/VH)", "High"),
        ("Map instability", f"pooled vs climate-set RF agree on {agreement.min():.0%}-{agreement.max():.0%} of known pixels ({best_year})",
         "Verified labels to decide between models; report disagreement maps with every map", "High"),
        ("Calibration and threshold do not establish Atok reliability",
         f"external test ECE raw {calibration_fit.iloc[-2]['ECE']:.3f} -> {r['calibration']['method']} {calibration_fit.iloc[-1]['ECE']:.3f}; "
         f"threshold set on other regions reaches {TARGET_SELECTIVE_ACCURACY:.0%} kept accuracy in {int(transfer['meets target'].sum())} "
         f"of {len(transfer)} unseen regions; Atok not assessable",
         "Calibrate and choose the threshold on verified Atok data", "High"),
        ("Rejection leaves little classified area",
         f"{best_year}: classified share of mappable cropland {causes['classified'].min():.0%}-{causes['classified'].max():.0%}; "
         f"out-of-distribution alone or with uncertainty {rejected.min():.0%}-{rejected.max():.0%}",
         "Training data closer to Atok (reduces OOD) and verified labels to set the threshold; minimum coverage 50%", "High"),
        ("Climate effect not shown on verified targets",
         f"{int((tested['rho_climate'] < 0).sum())} of {len(tested)} external targets have negative rho; "
         f"negative after dropping any single source in {int(tested['negative_after_every_drop'].sum())}",
         "More regions per crop so climate can be separated from source", "High"),
        ("Field visit (Jan-Feb) misses the main seasons",
         f"{len(outside_visit)} of {len(seasonal)} seasonal field-crops not in the ground in Jan-Feb: {', '.join(outside_visit)}",
         "Jul-Oct observation round; record planting/harvest dates", "High"),
        ("No others / other-crop class", f"{1 - r['cropland'].sum() / grid.roi_mask.sum():.0%} of the ROI outside the ELUM Crops mask; "
         "other crops inside it are forced into 3 crops or 'unknown'",
         "Digitise others (ELUM CY-2023, drone) and label other crops", "Medium"),
        ("Suitability: SoilGrids 250 m", f"one cell = 6.25 ha; labelled fields {min(labelled_area):.1f}-{max(labelled_area):.1f} ha",
         "Field-scale soil data", "Medium"),
        ("Suitability: single Baguio station",
         f"Baguio {r['pagasa_elevation_m']:.0f} m vs Atok {ATOK_ELEVATION_M} m: lapse correction {baguio_shift:.1f} °C",
         "Station or loggers at Atok elevation to verify the lapse rate", "Medium"),
    ]
    return pd.DataFrame(rows, columns=["limitation", "evidence", "requirement", "priority"])


def acceptance_criteria(r):
    """Proposed pilot criteria against the best provisional model; fixed before the verified pilot."""
    table, majority, best = best_atok_model(r["atok_results"])
    best_row, atok_units = table.loc[best], r["atok_units"]
    min_recall = min(best_row[f"PA_{c}"] for c in CLASSES)
    min_precision = np.nanmin([best_row[f"UA_{c}"] for c in CLASSES])
    provisional_fields = ", ".join(f"{c} {int(atok_units.loc[c, 'independent fields'])}" for c in CLASSES)
    field_split, transfer, causes = r["field_split"], r["threshold_transfer"], r["rejection_causes"]
    worst_window = r["mappable"].loc[r["best_year"], list(MAP_WINDOWS.values())].min()
    majority_oa = table.loc[majority, "OA"]
    ood_share = r["atok"]["ood"].mean()
    rows = [
        ("Verified independent fields per crop", ">= 5",
         f"{int(atok_units['verified (dated, field-checked) fields'].min())} verified (provisional fields: {provisional_fields})", False),
        ("Field-independent holdout", "yes, >= 2 test fields per crop", "no" if not field_split else "yes", field_split),
        ("Per-crop recall (producer's accuracy)", ">= 0.60 for every crop", f"min {min_recall:.2f}", min_recall >= 0.60),
        ("Per-crop precision (user's accuracy)", ">= 0.60 for every crop", f"min {min_precision:.2f}", min_precision >= 0.60),
        ("Macro-F1", ">= 0.60", f"{best_row['macro_F1']:.2f}", best_row["macro_F1"] >= 0.60),
        ("Balanced accuracy", ">= 0.53 (chance + 0.20)", f"{best_row['balanced_acc']:.2f}", best_row["balanced_acc"] >= 0.53),
        ("Overall accuracy above the majority baseline", f"> {majority_oa:.2f} + 0.10", f"{best_row['OA']:.2f}", best_row["OA"] > majority_oa + 0.10),
        ("Classified share of mappable cropland", ">= 50% in every window, at a threshold validated on verified Atok data",
         "not assessable (no verified labels); at external-set thresholds, worst window: "
         + ", ".join(f"{m.split(' (')[0]} {v:.0%}" for m, v in causes.groupby("model")["classified"].min().items()), False),
        ("Abstention threshold transfers to unseen regions", f">= {TARGET_SELECTIVE_ACCURACY:.0%} kept accuracy in >= 80% of held-out regions",
         f"{int(transfer['meets target'].sum())} of {len(transfer)} regions", transfer["meets target"].mean() >= 0.8),
        ("Mappable cropland per season", ">= 70% in every window", f"min {worst_window:.0%} ({r['best_year']})", worst_window >= 0.70),
        ("In-distribution inputs", "<= 20% out of distribution", f"{ood_share:.0%}", ood_share <= 0.20),
        ("Calibration on verified Atok data", "ECE <= 0.10", "not assessable (no verified labels)", False),
    ]
    return pd.DataFrame(rows, columns=["criterion", "target", f"current (best provisional model: {best})", "met"])


def summary(r):
    """Headline numbers saved to summary.json."""
    test_results, criteria = r["test_results"], r["criteria"]
    transfer, controls = r["threshold_transfer"], r["control_scores"]
    return {
        "atok_target": r["target_note"],
        "climate_set": r["climate_set"], "aligned_set": r["aligned_set"], "eligible_region_sets": r["eligible_sets"],
        "climate_vs_atok_spearman": r["correlations"].set_index("distance")["spearman_rho"].round(3).to_dict(),
        "climate_set_vs_size_matched_controls": {"climate_set": round(r["climate_score"], 3),
                                                 "controls_mean": round(float(np.mean(controls)), 3),
                                                 "controls_sd": round(float(np.std(controls, ddof=1)), 3)},
        "external_test": test_results.set_index("model")[["OA", "balanced_acc", "macro_F1", "macro_F1_95CI"]].round(3).to_dict("index"),
        "tuned_params": r["tuned_params"], "prototype_cv_gap": round(float(r["prototype_gap"]), 3),
        "abstention_raw": r["abstention"], "calibration_fitted": r["calibration"]["table"].round(3).to_dict("records"),
        "calibrated_thresholds": {k: v[2] for k, v in r["map_models"].items()},
        "threshold_transfer_regions_meeting_target": f"{int(transfer['meets target'].sum())} of {len(transfer)}",
        "climate_external_targets": r["target_stats"].round(3).to_dict("records"),
        "rejection_causes": r["rejection_causes"].round(3).to_dict("records"), "ood_threshold": r["ood_threshold"],
        "atok_ood_share": round(float(r["atok"]["ood"].mean()), 3),
        "atok": r["atok_results"].set_index("model")[["OA", "balanced_acc", "macro_F1"]].round(3).to_dict("index"),
        "best_year": r["best_year"], "worst_year": r["worst_year"],
        "map_agreement": r["agreement"].set_index("window")["agree_share_of_known"].round(3).to_dict(),
        "criteria_met": f"{int(criteria['met'].sum())} of {len(criteria)}",
    }
