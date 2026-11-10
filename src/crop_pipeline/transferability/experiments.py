"""Model diagnostics, transfer and climate-matching experiments (sections 1.5 and 4-6)."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from crop_pipeline.training import external_combined as ext

from .config import (CLASSES, INTERVAL, LR_SHAPE, MIN_PARCELS_PER_CROP, N_TREES_SEARCH, PROTOTYPE, SEED,
                     SHAPE, TARGET_SELECTIVE_ACCURACY, VARS)
from .external import window_means
from .models import (THRESHOLDS, aligned_proba, apply_calibrator, baseline_rows, choose_threshold, diagnostics,
                     expected_calibration_error, fit_calibrator, fit_rf, fit_spec, grouped_oof, jeffries_matusita, predicted, risk_coverage,
                     selective, source_balanced_weights)

ATOK = "Atok (provisional)"


# 1.5 Domain shift and separability

def atok_shift(data, atok, regions):
    """Wasserstein distance, in external SD units, between Atok and each region for the crops they share."""
    ext_means = window_means(data, VARS).assign(region=data["source_region"].to_numpy(), crop=data["crop"].to_numpy())
    atok_means = window_means(atok, VARS).assign(region=ATOK, crop=atok["crop"].to_numpy())
    scale = ext_means[VARS].std()
    rows = []
    for region in regions:
        shared = sorted(set(ext_means.loc[ext_means["region"] == region, "crop"]) & set(atok_means["crop"]))
        row = {"region": region, "shared_crops": len(shared)}
        for variable in VARS:
            row[variable] = float(np.mean([wasserstein_distance(
                atok_means.loc[atok_means["crop"] == c, variable] / scale[variable],
                ext_means.loc[(ext_means["region"] == region) & (ext_means["crop"] == c), variable] / scale[variable])
                for c in shared]))
        rows.append(row)
    shift = pd.DataFrame(rows).set_index("region")
    shift["mean"] = shift[VARS].mean(axis=1)
    return pd.concat([ext_means, atok_means], ignore_index=True), shift


def separability_groups(data, atok):
    return {
        "External, verified (pooled)": data,
        "Sen4AgriNet only (carrot, potato)": data[data["source"] == "Sen4AgriNet"],
        "CropHarvest only": data[data["source"] == "CropHarvest"],
        "Atok (provisional labels)": atok,
    }


def crop_separability(groups, min_samples=30):
    """Jeffries-Matusita distance between each crop pair at each half-month, on all 9 variables."""
    rows = []
    for group, frame in groups.items():
        crops = [c for c in CLASSES if (frame["crop"] == c).sum() >= min_samples]
        for k, first in enumerate(crops):
            for second in crops[k + 1:]:
                for b in range(1, 9):
                    columns = [f"bin_{b:02d}_{v}" for v in VARS]
                    rows.append({"group": group, "pair": f"{first} vs {second}", "bin": b,
                                 "JM": jeffries_matusita(frame.loc[frame["crop"] == first, columns].to_numpy(),
                                                         frame.loc[frame["crop"] == second, columns].to_numpy())})
    return pd.DataFrame(rows)


# 4.3 RF tuning and overfitting

def cv_rf(development, folds, params, weighting="crop and parcel", seeds=(SEED, SEED + 1), n_estimators=N_TREES_SEARCH):
    """Grouped-CV training and validation macro-F1 of an interval RF."""
    scores = []
    for seed in seeds:
        for train_index, validation_index in folds:
            train, validation = development.iloc[train_index], development.iloc[validation_index]
            weights = ext.training_weights(train) if weighting == "crop and parcel" else source_balanced_weights(train)
            model = fit_rf(train, INTERVAL, weights, n_estimators=n_estimators, random_state=seed, **params)
            scores.append((diagnostics(validation["crop"], aligned_proba(model, validation, INTERVAL))["macro_F1"],
                           diagnostics(train["crop"], aligned_proba(model, train, INTERVAL))["macro_F1"]))
    scores = np.array(scores)
    return {"validation_macro_F1": scores[:, 0].mean(), "validation_sd": scores[:, 0].std(ddof=1),
            "train_macro_F1": scores[:, 1].mean(), "gap": scores[:, 1].mean() - scores[:, 0].mean()}


TUNING_GRID = [{"min_samples_leaf": leaf, "max_features": features, "max_depth": depth}
               for leaf in (1, 2, 5, 10, 25) for features in ("sqrt", 0.5) for depth in (None, 8)]


def tuning_grid(development, folds, prototype_params):
    grid = pd.DataFrame([{**params, **cv_rf(development, folds, params)} for params in TUNING_GRID])
    grid["prototype"] = [p == prototype_params for p in TUNING_GRID]
    return grid.sort_values(["validation_macro_F1", "min_samples_leaf"], ascending=[False, False]).reset_index(drop=True)


def best_params(grid):
    """The top grid row as RandomForestClassifier keyword arguments."""
    params = {k: (None if pd.isna(v) else v) for k, v in grid.iloc[0][["min_samples_leaf", "max_features", "max_depth"]].items()}
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    if params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    if params["max_features"] not in ("sqrt",):
        params["max_features"] = float(params["max_features"])
    return params


def weighting_comparison(development, folds, configurations):
    return pd.DataFrame([{"configuration": label, "weighting": w, **cv_rf(development, folds, params, w)}
                         for label, params in configurations.items()
                         for w in ("crop and parcel", "source-balanced")])


# 4.4 Learning curves

def learning_curve(development, folds, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Validation scores when training on a share of each fold's training parcels (stratified by crop)."""
    rows_out = []
    for fold, (train_index, validation_index) in enumerate(folds):
        train_fold, validation = development.iloc[train_index], development.iloc[validation_index]
        units = train_fold.groupby("split_unit")["crop"].first()
        for fraction in fractions:
            for repeat in range(3 if fraction < 1 else 1):
                rng = np.random.default_rng(SEED + 100 * repeat + fold)
                chosen = [u for _, group in units.groupby(units) for u in
                          rng.choice(group.index, max(2, int(round(fraction * len(group)))), replace=False)]
                rows = train_fold[train_fold["split_unit"].isin(chosen)]
                model = fit_rf(rows, INTERVAL, random_state=SEED + repeat)
                val = diagnostics(validation["crop"], aligned_proba(model, validation, INTERVAL))
                rows_out.append({"fold": fold, "fraction": fraction, "train_parcels": rows["partition_group"].nunique(),
                                 "validation_macro_F1": val["macro_F1"],
                                 "train_macro_F1": diagnostics(rows["crop"], aligned_proba(model, rows, INTERVAL))["macro_F1"],
                                 **{f"recall_{c}": val[f"PA_{c}"] for c in CLASSES}})
    return pd.DataFrame(rows_out)


# 4.5 Importance and ablation

VARIABLE_GROUPS = {v: [c for c in INTERVAL if c.endswith(f"_{v}")] for v in VARS}
BIN_GROUPS = {f"bin {b}": [c for c in INTERVAL if c.startswith(f"bin_{b:02d}_")] for b in range(1, 9)}


def importance_and_ablation(development, folds, repeats=5):
    """Grouped permutation importance (by variable and by half-month) and refit-without-variable ablation."""
    importance_rows, ablation_rows = [], []
    for fold, (train_index, validation_index) in enumerate(folds):
        train, validation = development.iloc[train_index], development.iloc[validation_index]
        model = fit_rf(train, INTERVAL)
        base = diagnostics(validation["crop"], aligned_proba(model, validation, INTERVAL))["macro_F1"]
        rng = np.random.default_rng(SEED + fold)
        for kind, groups in (("variable", VARIABLE_GROUPS), ("bin", BIN_GROUPS)):
            for name, columns in groups.items():
                for _ in range(repeats):
                    shuffled = validation.copy()
                    shuffled[columns] = validation[columns].to_numpy()[rng.permutation(len(validation))]
                    drop = base - diagnostics(validation["crop"], aligned_proba(model, shuffled, INTERVAL))["macro_F1"]
                    importance_rows.append({"fold": fold, "kind": kind, "group": name, "drop": drop})
        for name, columns in VARIABLE_GROUPS.items():
            kept = [c for c in INTERVAL if c not in columns]
            reduced = fit_rf(train, kept)
            ablation_rows.append({"fold": fold, "removed": name,
                                  "change": diagnostics(validation["crop"], aligned_proba(reduced, validation, kept))["macro_F1"] - base})
    return pd.DataFrame(importance_rows), pd.DataFrame(ablation_rows)


# 4.6 Calibration, abstention and out-of-distribution

CALIBRATION_MODELS = {PROTOTYPE: ("rf", INTERVAL), LR_SHAPE: ("lr", SHAPE)}


def out_of_fold(development, folds, specs=CALIBRATION_MODELS):
    oof = {name: np.zeros((len(development), len(CLASSES))) for name in specs}
    for train_index, validation_index in folds:
        train, validation = development.iloc[train_index], development.iloc[validation_index]
        for name, (kind, columns) in specs.items():
            oof[name][validation_index] = aligned_proba(fit_spec(kind, train, columns), validation, columns)
    return oof


def reliability(truth, proba, min_count=15):
    """Reliability-diagram points (bins with >= min_count samples) and the selective-prediction curve."""
    confidence = proba.max(1)
    correct = predicted(proba) == np.asarray(truth)
    edges = np.linspace(0, 1, 11)
    centres, observed = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.sum() >= min_count:
            centres.append(confidence[m].mean())
            observed.append(correct[m].mean())
    curve = [(float((confidence >= t).mean()), float(correct[confidence >= t].mean()) if (confidence >= t).any() else np.nan)
             for t in THRESHOLDS]
    return {"centres": centres, "observed": observed, "coverage": [c for c, _ in curve], "accuracy": [a for _, a in curve],
            "ECE": expected_calibration_error(truth, proba), "accuracy_all": correct.mean(), "mean_confidence": confidence.mean()}


def calibration_check(development, test, pooled, oof):
    """Reliability of the raw probabilities, out of fold and on the locked test."""
    curves, rows = {}, []
    for name in oof:
        for label, truth, proba in (("out-of-fold", development["crop"], oof[name]),
                                    ("test", test["crop"], aligned_proba(pooled[name][0], test, pooled[name][1]))):
            curve = reliability(truth, proba)
            curves[(name, label)] = curve
            rows.append({"model": name, "data": label, "ECE": curve["ECE"], "accuracy": curve["accuracy_all"],
                         "mean_confidence": curve["mean_confidence"]})
    return curves, pd.DataFrame(rows)


def raw_abstention(development, test, pooled, oof_proba):
    """Threshold on the prototype's raw out-of-fold probabilities, and its coverage/accuracy on the locked test."""
    oof_conf = oof_proba.max(1)
    oof_correct = predicted(oof_proba) == development["crop"].to_numpy()
    reaching = [t for t in THRESHOLDS if (oof_conf >= t).any() and oof_correct[oof_conf >= t].mean() >= TARGET_SELECTIVE_ACCURACY]
    threshold = float(min(reaching)) if reaching else float(THRESHOLDS.max())
    test_proba = aligned_proba(pooled[PROTOTYPE][0], test, INTERVAL)
    keep = test_proba.max(1) >= threshold
    correct = predicted(test_proba) == test["crop"].to_numpy()
    return {"threshold": threshold, "target_accuracy": TARGET_SELECTIVE_ACCURACY,
            "oof_coverage": float((oof_conf >= threshold).mean()), "oof_accuracy_kept": float(oof_correct[oof_conf >= threshold].mean()),
            "test_coverage": float(keep.mean()), "test_accuracy_kept": float(correct[keep].mean()) if keep.any() else np.nan,
            "reached": bool(reaching)}


# 4.6b Fitted calibration and threshold transfer

def fit_calibration(development, test, pooled, folds, raw_oof):
    """Cross-fit temperature and matrix scaling; keep whichever (or none) gives the lowest out-of-fold ECE.

    Returns a dict with the chosen method, the comparison table, the pooled calibrator, the calibrated
    test probabilities and the calibrated abstention threshold.
    """
    fold_of = np.empty(len(development), int)
    for k, (_, validation_index) in enumerate(folds):
        fold_of[validation_index] = k
    truth = development["crop"].to_numpy()
    test_raw = aligned_proba(pooled[PROTOTYPE][0], test, INTERVAL)
    rows, cross_fitted = [], {"none": raw_oof}
    rows.append({"data": "development, out of fold", "probabilities": "none (raw)",
                 "ECE": expected_calibration_error(truth, raw_oof), "accuracy": diagnostics(truth, raw_oof)["OA"]})
    for method in ("temperature", "matrix"):
        calibrated = np.zeros_like(raw_oof)
        for k in range(len(folds)):
            inside = fold_of == k
            calibrated[inside] = apply_calibrator(fit_calibrator(raw_oof[~inside], truth[~inside], method), raw_oof[inside])
        cross_fitted[method] = calibrated
        rows.append({"data": "development, out of fold", "probabilities": f"{method} (cross-fitted)",
                     "ECE": expected_calibration_error(truth, calibrated), "accuracy": diagnostics(truth, calibrated)["OA"]})
    method = min(cross_fitted, key=lambda m: expected_calibration_error(truth, cross_fitted[m]))  # development data only
    calibrator = fit_calibrator(raw_oof, truth, method)
    test_cal = apply_calibrator(calibrator, test_raw)
    threshold, reached = choose_threshold(cross_fitted[method], truth)
    rows += [{"data": "locked test", "probabilities": "raw", "ECE": expected_calibration_error(test["crop"], test_raw),
              "accuracy": diagnostics(test["crop"], test_raw)["OA"]},
             {"data": "locked test", "probabilities": f"{method} (chosen)", "ECE": expected_calibration_error(test["crop"], test_cal),
              "accuracy": diagnostics(test["crop"], test_cal)["OA"]}]
    coverage, accuracy_kept = selective(test_cal, test["crop"], threshold)
    return {"method": method, "table": pd.DataFrame(rows), "calibrator": calibrator, "test_calibrated": test_cal,
            "threshold": threshold, "reached": reached, "test_coverage": coverage, "test_accuracy_kept": accuracy_kept}


def threshold_transfer(data, regions, method):
    """Calibrate and choose the threshold on all other regions, then apply both to the held-out region."""
    rows, curves = [], {}
    for region in regions:
        train_rows = data[data["source_region"] != region].reset_index(drop=True)
        held = data[data["source_region"] == region]
        region_oof = grouped_oof(train_rows)
        calibrator = fit_calibrator(region_oof, train_rows["crop"], method)
        threshold, _ = choose_threshold(apply_calibrator(calibrator, region_oof), train_rows["crop"])
        model = fit_rf(train_rows, INTERVAL, n_estimators=N_TREES_SEARCH)
        held_raw = aligned_proba(model, held, INTERVAL)
        held_cal = apply_calibrator(calibrator, held_raw)
        coverage, accuracy_kept = selective(held_cal, held["crop"], threshold)
        curves[region] = risk_coverage(held_cal, held["crop"])
        rows.append({"held_out_region": region, "n": len(held), "threshold (from other regions)": threshold,
                     "ECE raw": expected_calibration_error(held["crop"], held_raw),
                     "ECE calibrated": expected_calibration_error(held["crop"], held_cal),
                     "accuracy (all)": diagnostics(held["crop"], held_cal)["OA"], "coverage at threshold": coverage,
                     "accuracy of kept": accuracy_kept,
                     "meets target": bool(np.isfinite(accuracy_kept) and accuracy_kept >= TARGET_SELECTIVE_ACCURACY)})
    return pd.DataFrame(rows), curves


# 4.7 Transfer to unseen regions

def leave_one_region_out(data, regions):
    rows = []
    for region in regions:
        held = data[data["source_region"] == region]
        train = data[data["source_region"] != region]
        result = diagnostics(held["crop"], aligned_proba(fit_rf(train, INTERVAL), held, INTERVAL))
        majority = baseline_rows(held["crop"])[0]
        rows.append({"held_out_region": region, "parcels": held["partition_group"].nunique(),
                     "crops": ", ".join(sorted(held["crop"].unique())),
                     "crops missing from training": ", ".join(sorted(set(held["crop"]) - set(train["crop"]))) or "-",
                     "majority_OA": majority["OA"], **{k: result[k] for k in ("n", "OA", "balanced_acc", "macro_F1")},
                     **{f"PA_{c}": result[f"PA_{c}"] for c in CLASSES}})
    return pd.DataFrame(rows)


def target_rows(data, atok, target):
    return atok if target.startswith("Atok") else data[data["source_region"] == target]


def transfer_matrix(data, atok, regions):
    """Train on one region (with >= 2 crops), test on every other region and Atok over the crops they share."""
    train_regions = [r for r in regions if data.loc[data["source_region"] == r, "crop"].nunique() >= 2]
    targets = list(regions) + [ATOK]
    matrix = pd.DataFrame(np.nan, index=train_regions, columns=targets)
    shared_count = pd.DataFrame(0, index=train_regions, columns=targets)
    models = {}
    for region in train_regions:
        train = data[data["source_region"] == region]
        model = fit_rf(train, INTERVAL)
        models[region] = model
        for target in targets:
            if target == region:
                continue
            rows = target_rows(data, atok, target)
            shared = sorted(set(train["crop"]) & set(rows["crop"]))
            rows = rows[rows["crop"].isin(shared)]
            if not shared:
                continue
            matrix.loc[region, target] = diagnostics(rows["crop"], aligned_proba(model, rows, INTERVAL))["balanced_acc"]
            shared_count.loc[region, target] = len(shared)
    return {"train_regions": train_regions, "targets": targets, "matrix": matrix, "shared_count": shared_count, "models": models}


def one_vs_rest_transfer(data, atok, transfer, similarity):
    """ROC AUC per crop for each single-region model, where both regions have the crop and another crop."""
    rows = []
    for region in transfer["train_regions"]:
        train_crops = set(data.loc[data["source_region"] == region, "crop"])
        for target in transfer["targets"]:
            if target == region:
                continue
            frame = target_rows(data, atok, target)
            target_crops = set(frame["crop"])
            proba = aligned_proba(transfer["models"][region], frame, INTERVAL)
            for crop in CLASSES:
                if crop in train_crops and crop in target_crops and len(train_crops) >= 2 and len(target_crops) >= 2:
                    positive = (frame["crop"] == crop).to_numpy()
                    rows.append({"train": region, "test": target, "crop": crop, "positives": int(positive.sum()),
                                 "negatives": int((~positive).sum()), "AUC": roc_auc_score(positive, proba[:, CLASSES.index(crop)])})
    ovr = pd.DataFrame(rows)
    ovr["climate_distance"] = [similarity.climate(a, b) for a, b in zip(ovr["train"], ovr["test"])]
    ovr["geographic_km"] = [similarity.km(a, b) for a, b in zip(ovr["train"], ovr["test"])]
    return ovr


def auc_summary(ovr):
    target_kind = ovr["test"].str.startswith("Atok").map({True: ATOK, False: "external"})
    return ovr.groupby(["crop", target_kind])["AUC"].agg(["count", "mean", "min", "max"]).round(3)


def similarity_correlations(ovr):
    """Spearman correlation of one-vs-rest AUC with climate and geographic distance (pairs are not independent)."""
    external = ovr[~ovr["test"].str.startswith("Atok")]
    subsets = [("all points", ovr), ("external targets only", external)] + [
        (f"external, {c} only", external[external["crop"] == c]) for c in CLASSES]
    rows = []
    for name, subset in subsets:
        for measure in ("climate_distance", "geographic_km"):
            rho, p = spearmanr(subset[measure], subset["AUC"]) if len(subset) >= 5 else (np.nan, np.nan)
            rows.append({"points": name, "n": len(subset), "measure": measure, "spearman_rho": rho, "p (not independent)": p})
    return pd.DataFrame(rows)


# 5. Climate matching

def climate_external_targets(data, regions, similarity):
    """Every eligible source-region set scored on each external target region (verified labels)."""
    targets = [r for r in regions if data.loc[data["source_region"] == r, "crop"].nunique() >= 2]
    all_parcels = (data.groupby(["source_region", "crop"])["partition_group"].nunique().unstack(fill_value=0)
                   .reindex(columns=CLASSES, fill_value=0))
    rows_out = []
    for target in targets:
        target_frame = data[data["source_region"] == target]
        target_crops = sorted(target_frame["crop"].unique())
        sources = [r for r in regions if r != target]
        for k in range(1, len(sources) + 1):
            for combo in combinations(sources, k):
                if not (all_parcels.loc[list(combo), target_crops].sum() >= MIN_PARCELS_PER_CROP).all():
                    continue
                rows = data[data["source_region"].isin(combo)]
                parcels = rows.groupby("source_region")["partition_group"].nunique()
                model = fit_rf(rows, INTERVAL, n_estimators=N_TREES_SEARCH)
                result = diagnostics(target_frame["crop"], aligned_proba(model, target_frame, INTERVAL))
                rows_out.append({"target": target, "sources": " + ".join(combo), "n_sources": k,
                                 "train_parcels": int(parcels.sum()),
                                 "climate_distance": float(np.average([similarity.climate(r, target) for r in parcels.index], weights=parcels)),
                                 "geographic_km": float(np.average([similarity.km(r, target) for r in parcels.index], weights=parcels)),
                                 "balanced_acc": result["balanced_acc"], "macro_F1": result["macro_F1"]})
    return targets, pd.DataFrame(rows_out)


def climate_target_stats(data, external_sets, regions):
    """Per target: Spearman rho of climate distance vs balanced accuracy, size-adjusted slope and drop-one-source range."""
    rows = []
    for target, part in external_sets.groupby("target", sort=False):
        row = {"target": target, "eligible_sets": len(part), "crops": data.loc[data["source_region"] == target, "crop"].nunique()}
        if len(part) >= 5:
            row["rho_climate"], row["p_climate"] = spearmanr(part["climate_distance"], part["balanced_acc"])
            row["rho_geographic"], _ = spearmanr(part["geographic_km"], part["balanced_acc"])
            design = np.column_stack([np.ones(len(part)), part["climate_distance"], np.log(part["train_parcels"])])
            row["slope_climate_adj_size"] = np.linalg.lstsq(design, part["balanced_acc"], rcond=None)[0][1]
            drop_rhos = []
            for region in regions:
                rest = part[~part["sources"].str.contains(region, regex=False)]
                if len(rest) >= 5 and rest["climate_distance"].nunique() > 2:
                    drop_rhos.append(spearmanr(rest["climate_distance"], rest["balanced_acc"])[0])
            row["rho_range_dropping_one_source"] = f"{np.nanmin(drop_rhos):+.2f} to {np.nanmax(drop_rhos):+.2f}" if drop_rhos else "-"
            row["negative_after_every_drop"] = bool(drop_rhos) and bool(np.nanmax(drop_rhos) < 0)
            near = part["climate_distance"] <= part["climate_distance"].quantile(1 / 3)
            far = part["climate_distance"] >= part["climate_distance"].quantile(2 / 3)
            row["best_vs_worst_climate_tercile"] = part.loc[near, "balanced_acc"].mean() - part.loc[far, "balanced_acc"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def development_parcels(development, regions):
    return (development.groupby(["source_region", "crop"])["partition_group"].nunique().unstack(fill_value=0)
            .reindex(index=regions, columns=CLASSES, fill_value=0))


def eligible_region_sets(dev_parcels):
    regions = list(dev_parcels.index)
    return [list(combo) for k in range(1, len(regions) + 1) for combo in combinations(regions, k)
            if (dev_parcels.loc[list(combo)].sum() >= MIN_PARCELS_PER_CROP).all()]


def ranked_prefix(dev_parcels, order):
    """The shortest prefix of a ranked region list with enough parcels of every crop."""
    for k in range(1, len(order) + 1):
        if (dev_parcels.loc[order[:k]].sum() >= MIN_PARCELS_PER_CROP).all():
            return list(order[:k])


def atok_region_sets(development, atok, eligible_sets, dev_parcels, distances, climate_set, aligned_set):
    """Every eligible development region set scored on the provisional Atok labels (exploratory)."""
    rows_out = []
    for regions in eligible_sets:
        rows = development[development["source_region"].isin(regions)]
        weights = rows.groupby("source_region")["partition_group"].nunique()
        model = fit_rf(rows, INTERVAL, n_estimators=N_TREES_SEARCH)
        result = diagnostics(atok["crop"], aligned_proba(model, atok, INTERVAL))
        rows_out.append({"regions": " + ".join(regions), "n_regions": len(regions), "train_parcels": int(weights.sum()),
                         "calendar_distance": float(np.average(distances.loc[weights.index, "climate_distance"], weights=weights)),
                         "aligned_distance": float(np.average(distances.loc[weights.index, "aligned_distance"], weights=weights)),
                         **{f"parcels_{c}": int(dev_parcels.loc[regions, c].sum()) for c in CLASSES},
                         "atok_balanced_acc": result["balanced_acc"], "atok_macro_F1": result["macro_F1"], "atok_OA": result["OA"],
                         "is_climate_set": set(regions) == set(climate_set), "is_aligned_set": set(regions) == set(aligned_set)})
    return pd.DataFrame(rows_out).sort_values("atok_balanced_acc", ascending=False).reset_index(drop=True)


def region_set_correlations(region_sets):
    rows = []
    for distance_name in ("calendar_distance", "aligned_distance"):
        rho, p = spearmanr(region_sets[distance_name], region_sets["atok_balanced_acc"])
        design = np.column_stack([np.ones(len(region_sets)), region_sets[distance_name], np.log(region_sets["train_parcels"])])
        coefficients, *_ = np.linalg.lstsq(design, region_sets["atok_balanced_acc"], rcond=None)
        rows.append({"distance": distance_name, "spearman_rho": rho, "p_value": p,
                     "slope_per_distance_unit (adjusted for log parcels)": coefficients[1],
                     "slope_per_log_parcels": coefficients[2]})
    rho_size, p_size = spearmanr(region_sets["train_parcels"], region_sets["atok_balanced_acc"])
    rows.append({"distance": "training parcels (no climate)", "spearman_rho": rho_size, "p_value": p_size})
    return pd.DataFrame(rows)


def size_matched_controls(development, atok, climate_rows, draws=20):
    """Atok balanced accuracy of random parcel sets with the climate set's parcels per crop."""
    per_crop = climate_rows.groupby("crop")["partition_group"].unique()
    scores = []
    for draw in range(draws):
        rng = np.random.default_rng(SEED + draw)
        chosen = [g for crop in CLASSES for g in rng.choice(development.loc[development["crop"] == crop, "partition_group"].unique(),
                                                            len(per_crop[crop]), replace=False)]
        rows = development[development["partition_group"].isin(chosen)]
        model = fit_rf(rows, INTERVAL, n_estimators=N_TREES_SEARCH)
        scores.append(diagnostics(atok["crop"], aligned_proba(model, atok, INTERVAL))["balanced_acc"])
    return scores


def drop_one_region(region_sets, regions):
    rows = []
    for region in regions:
        has = region_sets["regions"].str.contains(region, regex=False)
        rest = region_sets[~has]
        rho, p = spearmanr(rest["aligned_distance"], rest["atok_balanced_acc"]) if len(rest) >= 5 else (np.nan, np.nan)
        rows.append({"region": region, "sets_with": int(has.sum()), "mean_atok_balanced_acc_with": region_sets.loc[has, "atok_balanced_acc"].mean(),
                     "mean_without": rest["atok_balanced_acc"].mean(), "sets_without": len(rest),
                     "spearman_rho_without (aligned)": rho, "p_without": p})
    return pd.DataFrame(rows)


# 6. Atok evaluation (provisional labels)

def atok_evaluation(atok, models):
    rows, predictions = baseline_rows(atok["crop"]), {}
    for name, (model, columns) in models.items():
        proba = aligned_proba(model, atok, columns)
        predictions[name] = proba
        rows.append({"model": name, **diagnostics(atok["crop"], proba)})
    return pd.DataFrame(rows), predictions


def clear_sensitivity(atok_all, models, names, minimums=(5, 6, 7, 8)):
    rows_out = []
    for minimum in minimums:
        rows = atok_all[atok_all["n_clear"] >= minimum]
        for name in names:
            model, columns = models[name]
            result = diagnostics(rows["crop"], aligned_proba(model, rows, columns))
            rows_out.append({"min_clear": minimum, "model": name, "n": len(rows),
                             "field_season_years": rows["partition_group"].nunique(),
                             **{k: result[k] for k in ("OA", "balanced_acc", "macro_F1")},
                             **{f"PA_{c}": result[f"PA_{c}"] for c in CLASSES}})
    return pd.DataFrame(rows_out)


def sensitivity_table(sensitivity):
    return sensitivity.pivot(index="min_clear", columns="model", values="balanced_acc").join(
        sensitivity.groupby("min_clear")[["n", "field_season_years"]].first())


def failure_table(atok, predictions, failure_models):
    """Per pixel-window correctness of each model, with clear counts and OOD flags."""
    failure = atok[["field", "crop", "start_month", "year", "n_clear", "ood"]].copy()
    for label, name in failure_models.items():
        failure[f"correct_{label}"] = predicted(predictions[name]) == atok["crop"].to_numpy()
    by_window = failure.groupby(["field", "crop", "start_month"]).agg(
        years=("year", "nunique"), pixel_windows=("year", "size"), median_clear=("n_clear", "median"), ood_share=("ood", "mean"),
        **{f"correct_{s}": (f"correct_{s}", "mean") for s in failure_models})
    return failure, by_window


def local_benchmark(atok, transferred_models):
    """70/30 Atok split by field if every crop has >= 2 fields, else by field-season-year (fitting diagnostic only)."""
    fields_per_crop = atok.groupby("crop")["field"].nunique().reindex(CLASSES, fill_value=0)
    field_split = bool((fields_per_crop >= 2).all())
    group_column = "field" if field_split else "partition_group"
    groups = atok.groupby(group_column)["crop"].agg(lambda s: s.mode().iloc[0])
    rng = np.random.default_rng(SEED)
    test_groups = set()
    for _, members in groups.groupby(groups):
        order = rng.permutation(members.index.to_numpy())
        test_groups.update(order[:max(1, int(round(0.3 * len(order))))])
    local_test = atok[atok[group_column].isin(test_groups)].reset_index(drop=True)
    local_train = atok[~atok[group_column].isin(test_groups)].reset_index(drop=True)
    weights = local_train.groupby(["crop", group_column])["sample_id"].transform("size")
    weights = 1.0 / (weights * local_train.groupby("crop")[group_column].transform("nunique"))
    local_model = RandomForestClassifier(**ext.FOREST_PARAMETERS).fit(
        local_train[INTERVAL].to_numpy(), local_train["crop"].to_numpy(), sample_weight=(weights / weights.mean()).to_numpy())
    rows = baseline_rows(local_test["crop"])
    rows.append({"model": "Local RF: weak-label fitting diagnostic (shares fields; not accuracy)",
                 **diagnostics(local_test["crop"], aligned_proba(local_model, local_test, INTERVAL))})
    for name, (model, columns) in transferred_models.items():
        rows.append({"model": f"{name} (transferred)", **diagnostics(local_test["crop"], aligned_proba(model, local_test, columns))})
    return {"fields_per_crop": fields_per_crop, "field_split": field_split, "test_groups": test_groups,
            "test_crops": local_test["crop"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
            "table": pd.DataFrame(rows)}
