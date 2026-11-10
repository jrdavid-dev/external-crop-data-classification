"""Model fitting, metrics, calibration, abstention and out-of-distribution checks."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from crop_pipeline.training import external_combined as ext

from .config import CLASSES, INTERVAL, LR_SHAPE, N_TREES_SEARCH, PROTOTYPE, SEED, SHAPE, TARGET_SELECTIVE_ACCURACY

MODEL_SPECS = {  # name -> (kind, predictors)
    PROTOTYPE: ("rf", INTERVAL),
    "LR, interval": ("lr", INTERVAL),
    "RF, shape": ("rf", SHAPE),
    LR_SHAPE: ("lr", SHAPE),
}
PROTOTYPE_PARAMS = {"min_samples_leaf": 2, "max_features": "sqrt", "max_depth": None}
THRESHOLDS = np.round(np.arange(0.34, 0.96, 0.02), 2)
CAL_EPS = 1e-3


def grouped_folds(rows, seed=SEED):
    """The same 5 parcel-grouped folds as v1's CV."""
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(rows, rows["crop"], rows["split_unit"]))


def aligned_proba(model, rows, columns):
    """Class probabilities in CLASSES order, whatever classes the model saw."""
    raw = model.predict_proba(rows[columns].to_numpy())
    out = np.zeros((len(rows), len(CLASSES)))
    for i, crop in enumerate(model.classes_):
        out[:, CLASSES.index(crop)] = raw[:, i]
    return out


def predicted(proba):
    return np.asarray(CLASSES)[proba.argmax(1)]


def diagnostics(truth, proba):
    """Overall, balanced and macro-F1 scores plus producer's (PA) and user's (UA) accuracy per crop."""
    truth = np.asarray(truth)
    prediction = predicted(proba)
    present = [c for c in CLASSES if (truth == c).any()]
    precision, recall, _, support = precision_recall_fscore_support(truth, prediction, labels=CLASSES, zero_division=0)
    out = {"n": len(truth), "OA": accuracy_score(truth, prediction),
           "balanced_acc": float(np.mean([recall[CLASSES.index(c)] for c in present])),
           "macro_F1": f1_score(truth, prediction, labels=present, average="macro", zero_division=0)}
    for i, crop in enumerate(CLASSES):
        out[f"PA_{crop}"] = recall[i] if support[i] else np.nan
        out[f"UA_{crop}"] = precision[i] if (prediction == crop).any() else np.nan
    return out


def baseline_rows(truth):
    """Always-majority and guess-with-class-shares baselines."""
    truth = np.asarray(truth)
    shares = pd.Series(truth).value_counts(normalize=True)
    majority = shares.idxmax()
    one_hot = np.zeros((len(truth), len(CLASSES)))
    one_hot[:, CLASSES.index(majority)] = 1
    expected = {"model": "Baseline: guess with class shares (expected)", "n": len(truth), "OA": float((shares ** 2).sum()),
                "balanced_acc": 1 / len(shares), "macro_F1": 1 / len(shares)}
    for crop in CLASSES:
        expected[f"PA_{crop}"] = expected[f"UA_{crop}"] = float(shares.get(crop, np.nan))
    return [{"model": f"Baseline: always {majority}", **diagnostics(truth, one_hot)}, expected]


def fit_rf(rows, columns=None, weights=None, **params):
    columns = INTERVAL if columns is None else columns
    model = RandomForestClassifier(**{**ext.FOREST_PARAMETERS, **params})
    model.fit(rows[columns].to_numpy(), rows["crop"].to_numpy(),
              sample_weight=ext.training_weights(rows) if weights is None else weights)
    return model


def fit_lr(rows, columns):
    return ext.fit_model("logistic_regression", rows, columns)


def fit_spec(kind, rows, columns):
    return fit_rf(rows, columns) if kind == "rf" else fit_lr(rows, columns)


def source_balanced_weights(rows):
    """Equal weight per crop, then per source within the crop, then per parcel; reported labels 0.5."""
    status = rows["label_status"].map(ext.CONFIDENCE_WEIGHTS).to_numpy(float)
    weights = np.zeros(len(rows))
    position = pd.Series(np.arange(len(rows)), index=rows.index)
    for _, crop_rows in rows.groupby("crop"):
        sources = crop_rows["source"].nunique()
        for _, source_rows in crop_rows.groupby("source"):
            parcels = source_rows["partition_group"].nunique()
            for _, parcel_rows in source_rows.groupby("partition_group"):
                index = position[parcel_rows.index].to_numpy()
                weights[index] = status[index] / status[index].sum() / parcels / sources
    return weights * len(weights) / weights.sum()


def test_row(name, model, test, columns):
    """One locked-test result row with the parcel-bootstrap 95% interval of macro-F1."""
    proba = aligned_proba(model, test, columns)
    interval = ext.bootstrap_interval(test, proba)
    return {"model": name, **diagnostics(test["crop"], proba),
            "macro_F1_95CI": f"{interval['macro_f1_low']:.3f}-{interval['macro_f1_high']:.3f}"}


def fit_pooled_models(development, test):
    """The four prespecified model/predictor pairs trained on all development parcels, scored on the locked test."""
    pooled = {}
    rows = [{"model": r.pop("model"), **r, "macro_F1_95CI": ""} for r in baseline_rows(test["crop"])]
    for name, (kind, columns) in MODEL_SPECS.items():
        model = fit_spec(kind, development, columns)
        pooled[name] = (model, columns)
        rows.append(test_row(name, model, test, columns))
    return pooled, pd.DataFrame(rows)


def append_test_row(results, row):
    return pd.concat([results, pd.DataFrame([row])], ignore_index=True)


# Calibration and abstention

def expected_calibration_error(truth, proba, bins=10):
    confidence = proba.max(1)
    correct = predicted(proba) == np.asarray(truth)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs(correct[m].mean() - confidence[m].mean()) * m.mean()
                     for lo, hi in zip(edges[:-1], edges[1:]) for m in [(confidence > lo) & (confidence <= hi)] if m.any()))


class TemperatureCalibrator:
    """One parameter: p = softmax(log p_raw / T). Never changes the predicted crop."""

    def fit(self, log_proba, truth):
        index = np.array([CLASSES.index(c) for c in np.asarray(truth)])

        def nll(log_t):
            scaled = log_proba / np.exp(log_t)
            scaled = scaled - scaled.max(1, keepdims=True)
            return -(scaled[np.arange(len(index)), index] - np.log(np.exp(scaled).sum(1))).mean()

        self.temperature = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        self.classes_ = np.asarray(CLASSES)
        return self

    def predict_proba(self, log_proba):
        scaled = log_proba / self.temperature
        scaled = np.exp(scaled - scaled.max(1, keepdims=True))
        return scaled / scaled.sum(1, keepdims=True)


def fit_calibrator(proba, truth, method):
    """method: 'none' (identity), 'temperature' or 'matrix' (multinomial LR on log probabilities)."""
    features = np.log(np.clip(proba, CAL_EPS, 1))
    if method == "none":
        identity = TemperatureCalibrator()
        identity.temperature, identity.classes_ = 1.0, np.asarray(CLASSES)
        return identity
    if method == "temperature":
        return TemperatureCalibrator().fit(features, truth)
    return LogisticRegression(max_iter=5000, C=10.0).fit(features, np.asarray(truth))


def apply_calibrator(calibrator, proba):
    raw = calibrator.predict_proba(np.log(np.clip(proba, CAL_EPS, 1)))
    out = np.zeros((len(proba), len(CLASSES)))
    for i, crop in enumerate(calibrator.classes_):
        out[:, CLASSES.index(crop)] = raw[:, i]
    return out


class Calibrated:
    """A fitted model plus its calibrator, usable wherever aligned_proba() is used."""

    def __init__(self, model, calibrator):
        self.model, self.calibrator, self.classes_ = model, calibrator, np.asarray(CLASSES)

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)
        aligned = np.zeros((len(X), len(CLASSES)))
        for i, crop in enumerate(self.model.classes_):
            aligned[:, CLASSES.index(crop)] = raw[:, i]
        return apply_calibrator(self.calibrator, aligned)


def choose_threshold(proba, truth, target=TARGET_SELECTIVE_ACCURACY):
    """Lowest threshold whose kept predictions reach `target` accuracy; (threshold, reached)."""
    confidence = proba.max(1)
    correct = predicted(proba) == np.asarray(truth)
    reaching = [t for t in THRESHOLDS if (confidence >= t).any() and correct[confidence >= t].mean() >= target]
    return (float(min(reaching)), True) if reaching else (float(THRESHOLDS.max()), False)


def selective(proba, truth, threshold):
    """(coverage, accuracy of kept predictions) at a threshold."""
    keep = proba.max(1) >= threshold
    correct = predicted(proba) == np.asarray(truth)
    return float(keep.mean()), (float(correct[keep].mean()) if keep.any() else np.nan)


def risk_coverage(proba, truth):
    order = np.argsort(-proba.max(1))
    correct = (predicted(proba) == np.asarray(truth))[order]
    kept = np.arange(1, len(order) + 1)
    return kept / len(order), np.cumsum(correct) / kept


def grouped_oof(rows, columns=None, n_estimators=N_TREES_SEARCH):
    """Out-of-fold RF probabilities over 5 parcel-grouped folds of `rows`."""
    rows = rows.reset_index(drop=True)
    out = np.zeros((len(rows), len(CLASSES)))
    for train_index, validation_index in grouped_folds(rows):
        model = fit_rf(rows.iloc[train_index], columns, n_estimators=n_estimators)
        out[validation_index] = aligned_proba(model, rows.iloc[validation_index], INTERVAL if columns is None else columns)
    return out


def calibrated_model(rows, method):
    """Final RF on `rows`, a calibrator fitted on its grouped out-of-fold probabilities, and its threshold."""
    model = fit_rf(rows, INTERVAL)
    oof = grouped_oof(rows)
    calibrator = fit_calibrator(oof, rows["crop"], method)
    threshold, _ = choose_threshold(apply_calibrator(calibrator, oof), rows["crop"])
    return model, calibrator, threshold


class DistanceToTraining:
    """Mean distance to the 5 nearest development samples in standardised interval features."""

    def __init__(self, development):
        self.scaler = StandardScaler().fit(development[INTERVAL])
        self.neighbours = NearestNeighbors(n_neighbors=5).fit(self.scaler.transform(development[INTERVAL]))

    def __call__(self, rows):
        return self.neighbours.kneighbors(self.scaler.transform(rows[INTERVAL]))[0].mean(1)


def jeffries_matusita(a, b, ridge=1e-6):
    """Jeffries-Matusita distance between two Gaussian classes (0 = identical, 2 = fully separable)."""
    pooled_sd = np.vstack([a, b]).std(0) + 1e-12
    a, b = a / pooled_sd, b / pooled_sd
    m = a.mean(0) - b.mean(0)
    s1, s2 = np.cov(a, rowvar=False) + ridge * np.eye(a.shape[1]), np.cov(b, rowvar=False) + ridge * np.eye(a.shape[1])
    s = (s1 + s2) / 2
    bhattacharyya = m @ np.linalg.solve(s, m) / 8 + 0.5 * (np.linalg.slogdet(s)[1]
                                                           - 0.5 * (np.linalg.slogdet(s1)[1] + np.linalg.slogdet(s2)[1]))
    return float(2 * (1 - np.exp(-bhattacharyya)))
