"""Compare overfitting remedies on the external crop data with grouped and region-held-out validation.

Every remedy is fitted inside the training part of each split, so feature selection, PCA
and augmentation never see validation rows. Only development parcels are used here; the
locked test set is scored separately, once, for the prototype and the CV-best remedy.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.training import external_combined as ext

SEED = ext.SEED
CLASSES = list(ext.CLASSES)
BANDS = ["B3", "B4", "B5", "B8", "B11"]
INDICES = ["NDVI", "NDRE", "NDMI", "SAVI"]
BINS = 8
SEARCH_TREES = 500  # as in notebook 07 v2's tuning grid; final models use 1,000


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    model: str = "rf"                      # "rf" or "lr"
    params: dict = field(default_factory=dict)
    reduce: str | None = None              # None, "importance:<k>" or "pca:<k>"
    augment: tuple = ()                    # subset of ("noise", "shift")


PROTOTYPE = Strategy("RF prototype (leaf 2, sqrt features)", "baseline", params={"min_samples_leaf": 2})
STRATEGIES = [
    PROTOTYPE,
    Strategy("RF, leaf 25, 50% features (v2 CV-tuned)", "reduce complexity",
             params={"min_samples_leaf": 25, "max_features": 0.5}),
    Strategy("RF, depth 4, leaf 10", "reduce complexity", params={"max_depth": 4, "min_samples_leaf": 10}),
    Strategy("LR, L2, C = 1", "regularization", "lr", {"C": 1.0}),
    Strategy("LR, L2, C = 0.1", "regularization", "lr", {"C": 0.1}),
    Strategy("LR, L2, C = 0.01", "regularization", "lr", {"C": 0.01}),
    Strategy("LR, L1, C = 0.1", "regularization", "lr", {"C": 0.1, "l1_ratio": 1.0}),
    Strategy("RF, top 20 features by importance", "feature selection", reduce="importance:20"),
    Strategy("RF, 10 PCA components", "feature selection", reduce="pca:10"),
    Strategy("LR (C = 0.1), 10 PCA components", "feature selection", "lr", {"C": 0.1}, "pca:10"),
    Strategy("RF + reflectance noise", "data augmentation", augment=("noise",)),
    Strategy("RF + timing shift (±1 half-month)", "data augmentation", augment=("shift",)),
    Strategy("RF + noise + timing shift", "data augmentation", augment=("noise", "shift")),
    Strategy("RF, leaf 25, 50% features + noise + shift", "combined",
             params={"min_samples_leaf": 25, "max_features": 0.5}, augment=("noise", "shift")),
]


def interval_columns():
    return feature_experiments.feature_sets()["interval_only"]


def _band_cube(rows):
    """(pixels, 8 bins, 5 bands) reflectance array."""
    return np.stack([rows[[f"bin_{b:02d}_{band}" for b in range(1, BINS + 1)]].to_numpy(float) for band in BANDS], -1)


def _with_bands(rows, cube):
    """Copy of ``rows`` with new band values and indices recomputed exactly as in the tables."""
    out = rows.copy()
    b4, b5, b8, b11 = (cube[..., BANDS.index(name)] for name in ("B4", "B5", "B8", "B11"))
    values = {**{band: cube[..., i] for i, band in enumerate(BANDS)},
              "NDVI": (b8 - b4) / (b8 + b4), "NDRE": (b8 - b5) / (b8 + b5),
              "NDMI": (b8 - b11) / (b8 + b11), "SAVI": 1.5 * (b8 - b4) / (b8 + b4 + 0.5)}
    for name, array in values.items():
        for b in range(BINS):
            out[f"bin_{b + 1:02d}_{name}"] = array[:, b]
    return out


def augment(rows, kinds, rng, copies=2, gain_sd=0.05, noise_sd=0.02):
    """Original rows plus ``copies`` perturbed copies that keep crop, parcel and weights.

    noise: a per-pixel, per-band gain (sd ``gain_sd``, as from illumination or atmosphere)
    times per-observation noise (sd ``noise_sd``). shift: the 8 half-months moved by one
    step earlier or later, repeating the edge value, as from onset-timing error.
    """
    if not kinds:
        return rows
    cube = _band_cube(rows)
    pieces = [rows]
    for _ in range(copies):
        new = cube.copy()
        if "shift" in kinds:
            step = rng.choice([-1, 1], size=len(new))
            index = np.clip(np.arange(BINS)[None, :] + step[:, None], 0, BINS - 1)
            new = np.take_along_axis(new, index[:, :, None], axis=1)
        if "noise" in kinds:
            gain = rng.normal(1, gain_sd, size=(len(new), 1, len(BANDS)))
            new = np.clip(new * gain * rng.normal(1, noise_sd, size=new.shape), 1e-4, None)
        pieces.append(_with_bands(rows, new))
    return pd.concat(pieces, ignore_index=True)


def make_estimator(strategy, seed=SEED, trees=SEARCH_TREES):
    steps = []
    if strategy.model == "lr" or (strategy.reduce or "").startswith("pca"):
        steps.append(("scale", StandardScaler()))
    if strategy.reduce:
        kind, k = strategy.reduce.split(":")
        if kind == "pca":
            steps.append(("reduce", PCA(n_components=int(k), random_state=seed)))
        elif kind == "importance":
            steps.append(("reduce", SelectFromModel(
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                max_features=int(k), threshold=-np.inf)))
        else:
            raise ValueError(f"Unknown reduction {strategy.reduce}.")
    if strategy.model == "rf":
        params = {**ext.FOREST_PARAMETERS, "n_estimators": trees, "random_state": seed, **strategy.params}
        steps.append(("model", RandomForestClassifier(**params)))
    elif strategy.model == "lr":
        solver = "saga" if strategy.params.get("l1_ratio", 0) > 0 else "lbfgs"  # L2 is the default penalty
        steps.append(("model", LogisticRegression(max_iter=5000, solver=solver, **strategy.params)))
    else:
        raise ValueError(f"Unknown model {strategy.model}.")
    return Pipeline(steps)


def fit(strategy, rows, seed=SEED, trees=SEARCH_TREES):
    """Fit on ``rows`` (augmented first if the strategy says so) with crop-and-parcel weights."""
    rows = augment(rows, strategy.augment, np.random.default_rng(seed))
    estimator = make_estimator(strategy, seed, trees)
    weights = ext.training_weights(rows)
    fit_params = {"model__sample_weight": weights}
    if (strategy.reduce or "").startswith("importance"):
        fit_params["reduce__sample_weight"] = weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # LR convergence on tiny folds
        estimator.fit(rows[interval_columns()].to_numpy(), rows["crop"].to_numpy(), **fit_params)
    return estimator


def predict(estimator, rows):
    return estimator.predict(rows[interval_columns()].to_numpy())


def macro_f1(truth, prediction):
    present = [crop for crop in CLASSES if (np.asarray(truth) == crop).any()]
    return f1_score(truth, prediction, labels=present, average="macro", zero_division=0)


def balanced_accuracy(truth, prediction):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # predicted crops absent from the held-out region
        return balanced_accuracy_score(truth, prediction)


def grouped_folds(development, folds=5, seed=SEED):
    """The same parcel-grouped folds as notebook 07 v1/v2."""
    return list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
        development, development["crop"], development["split_unit"]))


def holdout_regions(development, min_crops=2):
    """Regions that keep at least ``min_crops`` crops, so balanced accuracy is meaningful."""
    crops = development.groupby("source_region")["crop"].nunique()
    return sorted(crops[crops >= min_crops].index)


def evaluate(strategy, development, folds, regions, seed=SEED):
    """Grouped-CV train/validation macro-F1 and leave-one-region-out balanced accuracy."""
    cv = []
    for train_index, validation_index in folds:
        train, validation = development.iloc[train_index], development.iloc[validation_index]
        estimator = fit(strategy, train, seed)
        cv.append((macro_f1(train["crop"], predict(estimator, train)),
                   macro_f1(validation["crop"], predict(estimator, validation))))
    cv = np.array(cv)
    region_scores = {}
    for region in regions:
        held = development["source_region"] == region
        estimator = fit(strategy, development[~held], seed)
        region_scores[region] = balanced_accuracy(development.loc[held, "crop"],
                                                  predict(estimator, development[held]))
    return {"strategy": strategy.name, "family": strategy.family,
            "cv_train_macro_f1": cv[:, 0].mean(), "cv_validation_macro_f1": cv[:, 1].mean(),
            "cv_validation_sd": cv[:, 1].std(ddof=1), "gap": cv[:, 0].mean() - cv[:, 1].mean(),
            "region_balanced_accuracy": float(np.mean(list(region_scores.values()))),
            **{f"region: {region}": score for region, score in region_scores.items()}}


def compare(development, strategies=None, seed=SEED, progress=None):
    strategies = STRATEGIES if strategies is None else strategies
    folds, regions = grouped_folds(development, seed=seed), holdout_regions(development)
    rows = []
    for strategy in strategies:
        if progress:
            progress(strategy.name)
        rows.append(evaluate(strategy, development, folds, regions, seed))
    return pd.DataFrame(rows).set_index("strategy")


def score_on_test(strategy, data, seed=SEED, trees=ext.FOREST_PARAMETERS["n_estimators"]):
    """Fit on all development parcels and score the locked test set once."""
    development = data[data["partition"] == "development"]
    test = data[data["partition"] == "test"].reset_index(drop=True)
    prediction = predict(fit(strategy, development, seed, trees), test)
    return {"strategy": strategy.name, "test_macro_f1": macro_f1(test["crop"], prediction),
            "test_balanced_accuracy": balanced_accuracy(test["crop"], prediction)}


def cached_comparison(output_folder, data, refresh=False, progress=None):
    """Run the comparison once and reuse the saved tables afterwards.

    The CV-best strategy (highest mean validation macro-F1) is chosen before the test set
    is scored, and only it and the prototype are scored there.
    """
    output_folder = Path(output_folder)
    comparison_path, test_path = output_folder / "strategy_comparison.csv", output_folder / "strategy_test.csv"
    if not refresh and comparison_path.is_file() and test_path.is_file():
        return pd.read_csv(comparison_path, index_col="strategy"), pd.read_csv(test_path, index_col="strategy")
    development = data[data["partition"] == "development"].reset_index(drop=True)
    comparison = compare(development, progress=progress)
    best = comparison["cv_validation_macro_f1"].idxmax()
    by_name = {strategy.name: strategy for strategy in STRATEGIES}
    chosen = [PROTOTYPE.name] + ([best] if best != PROTOTYPE.name else [])
    test = pd.DataFrame([score_on_test(by_name[name], data) for name in chosen]).set_index("strategy")
    test["role"] = ["prespecified prototype", "CV-best remedy"][: len(test)]
    output_folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path)
    test.to_csv(test_path)
    return comparison, test


def plot_comparison(comparison):
    """Training vs validation macro-F1 per strategy, and region-held-out balanced accuracy."""
    import matplotlib.pyplot as plt

    families = list(dict.fromkeys(comparison["family"]))
    colors = dict(zip(families, ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]))
    order = comparison.index[::-1]
    y = np.arange(len(order))
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 0.42 * len(order) + 1.6), sharey=True)
    for i, name in enumerate(order):
        row = comparison.loc[name]
        left.plot([row["cv_validation_macro_f1"], row["cv_train_macro_f1"]], [i, i], color="#BBBBBB", lw=2, zorder=1)
        left.scatter(row["cv_train_macro_f1"], i, color="white", edgecolor=colors[row["family"]], s=40, zorder=2)
        left.errorbar(row["cv_validation_macro_f1"], i, xerr=row["cv_validation_sd"], fmt="o",
                      color=colors[row["family"]], ms=6, capsize=2, zorder=3)
    base = comparison.loc[PROTOTYPE.name]
    left.axvline(base["cv_validation_macro_f1"], color="black", lw=0.6, ls="--")
    left.set_yticks(y, order)
    left.set_xlim(0, 1.02)
    left.set_xlabel("Macro-F1: open = training, filled = validation (± sd over folds)")
    left.set_title("Grouped CV: does the remedy close the gap?", fontsize=10)
    right.barh(y, comparison.loc[order, "region_balanced_accuracy"],
               color=[colors[f] for f in comparison.loc[order, "family"]])
    right.axvline(base["region_balanced_accuracy"], color="black", lw=0.6, ls="--")
    right.set_xlim(0, 1)
    right.set_xlabel("Mean balanced accuracy on held-out regions")
    right.set_title("Leave-one-region-out: does it transfer better?", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[f], label=f) for f in families]
    right.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    return figure
