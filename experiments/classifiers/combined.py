"""Combine the external study's model choices with the annual recipe, on development data only.

Three families of changes from the window-model study are tried on the annual
Sentinel-2 + Sentinel-1 predictors:

- Random Forest tuning: the study's one-setting changes (``model_followups.TUNING_CANDIDATES``)
  plus Training 01's leaf size 4.
- Other classifiers with the study's grids (``classifier_comparison``): weighted logistic
  regression, and gradient-boosted trees with the XGBoost grid run through scikit-learn's
  ``HistGradientBoostingClassifier`` (XGBoost needs the OpenMP system library, which is not
  installed). The RBF SVM is left out: calibrated kernel SVMs do not scale to ~70,000 pixels.
- Green-up timing from the study's detector (``imagery.quality.detect_greenup`` with the
  aligned-map settings) run on each pixel's annual NDVI series: (a) timing features of the
  strongest detected cycle added to the annual predictors, and (b) every harmonic phase measured
  from that green-up onset instead of from 1 January. Pixels without a detected cycle use the
  NDVI harmonic trough, so coverage stays complete.

Within each family, settings are chosen by spatial CV in the training split (the study's tuning
rule). Families are then compared on the development selection score.
"""

from datetime import date

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.annual import features as annual
from crop_pipeline.imagery import quality
from crop_pipeline.paoay import aligned
from experiments.annual import development_views
from crop_pipeline.training.classifier_comparison import CANDIDATES as CLASSIFIER_GRIDS
from crop_pipeline.training.classifier_comparison import XGBOOST_CANDIDATES
from crop_pipeline.training.model_followups import TUNING_CANDIDATES

REFERENCE_YEAR = 2021  # bins are dated in one non-leap year; only relative timing is used
DETECTOR = aligned.DETECTOR
FOREST = development_views.SELECTION_FOREST


# Green-up timing

def bin_dates(year=REFERENCE_YEAR):
    rows = []
    for interval in aligned.interval_bins(year):
        start, end = date.fromisoformat(interval["start"]), date.fromisoformat(interval["end"])
        rows.append((interval["start"], interval["end"], (start + (end - start) // 2).isoformat()))
    return rows


def detector_rows(ndvi, clear):
    """Observation rows for the study's detector: one per pixel and clear-or-not half-month."""
    dates = bin_dates()
    rows = []
    for i in range(len(ndvi)):
        for b, (start, end, middle) in enumerate(dates):
            ok = bool(clear[i, b]) and np.isfinite(ndvi[i, b])
            rows.append({"region_id": i, "date": middle, "start": start, "end": end, "source_kind": "observed_daily",
                         "valid_fraction": 1.0 if ok else 0.0, "NDVI": float(ndvi[i, b]) if ok else None,
                         "NDRE": None, "NDMI": None})
    return rows


def fraction_of_year(day):
    origin = date(REFERENCE_YEAR, 1, 1)
    return ((date.fromisoformat(day) - origin).days % 365) / 365


def greenup_timing(s2, s2_valid, chunk=5000):
    """Per pixel: detected cycles, cycles passing the detector, and the strongest cycle's timing.

    Returns a dict of arrays: ``cycles``, ``accepted``, ``amplitude``, ``baseline``,
    ``onset_to_peak_days``, ``onset`` (fraction of year) and ``detected``.
    """
    ndvi = annual.indices(s2)[..., 0]
    n = len(ndvi)
    out = {"cycles": np.zeros(n), "accepted": np.zeros(n), "amplitude": np.zeros(n), "baseline": np.full(n, np.nan),
           "onset_to_peak_days": np.zeros(n), "onset": np.full(n, np.nan), "detected": np.zeros(n)}
    best = np.full(n, -np.inf)
    for first in range(0, n, chunk):
        part = np.arange(first, min(n, first + chunk))
        rows = detector_rows(ndvi[part], s2_valid[part])
        for result in quality.detect_greenup(rows, DETECTOR):
            i = part[int(result["region_id"])]
            if result.get("onset_interval_start") is None or result.get("amplitude") is None:
                continue
            out["cycles"][i] += 1
            out["accepted"][i] += result["status"] in ("candidate", "provisional", "ambiguous")
            if result["amplitude"] > best[i]:
                best[i] = result["amplitude"]
                onset = date.fromisoformat(result["onset_interval_start"])
                out["onset"][i] = fraction_of_year(result["onset_interval_start"])
                out["amplitude"][i] = result["amplitude"]
                out["baseline"][i] = result["baseline"]
                peak = date.fromisoformat(result["peak_date"]) if result.get("peak_date") else onset
                out["onset_to_peak_days"][i] = (peak - onset).days % 365
                out["detected"][i] = 1
    return out


def harmonic_reference(s2, s2_valid, timing):
    """Reference time per pixel: detected onset, else the NDVI first-harmonic trough."""
    cube = np.concatenate([s2, annual.indices(s2)], axis=-1)
    coefficients, _ = annual.fit_harmonics(cube[:, :, len(annual.BANDS)], s2_valid)  # NDVI
    peak = (np.arctan2(coefficients[:, 2], coefficients[:, 1]) / (2 * np.pi)) % 1
    trough = (peak + .5) % 1
    return np.where(np.isfinite(timing["onset"]), timing["onset"], trough)


def rereference(X, columns, reference):
    """Rotate every (cos_phaseK, sin_phaseK) pair by -2 pi K reference, so timing is measured from green-up."""
    X = X.copy()
    for j, column in enumerate(columns):
        if column.endswith("_cos_phase1") or column.endswith("_cos_phase2"):
            k = int(column[-1])
            s = columns.index(column.replace("_cos_", "_sin_"))
            angle = np.arctan2(X[:, s], X[:, j]) - 2 * np.pi * k * reference
            X[:, j], X[:, s] = np.cos(angle), np.sin(angle)
    return X


TIMING_COLUMNS = ("greenup_cycles", "greenup_accepted_cycles", "greenup_amplitude", "greenup_baseline",
                  "greenup_onset_to_peak_days", "greenup_detected")


def timing_matrix(timing):
    baseline = np.where(np.isfinite(timing["baseline"]), timing["baseline"], 0.0)
    return np.column_stack([timing["cycles"], timing["accepted"], timing["amplitude"], baseline,
                            timing["onset_to_peak_days"], timing["detected"]])


def greenup_recipes(s2, s2_valid, s1, timing=None):
    """``{name: (X, ok, columns)}`` for the two green-up recipes, sharing one detector run."""
    timing = greenup_timing(s2, s2_valid) if timing is None else timing
    X, ok, columns = annual.RECIPES["annual_s2_s1"](s2, s2_valid, s1)
    extra = timing_matrix(timing)
    with_timing = (np.hstack([X, extra]), ok, columns + list(TIMING_COLUMNS))
    referenced = rereference(X, columns, harmonic_reference(s2, s2_valid, timing))
    return {"annual_s2_s1_greenup_timing": with_timing,
            "annual_s2_s1_greenup_referenced": (np.hstack([referenced, extra]), ok,
                                                [f"{c}_from_greenup" if "_phase" in c else c for c in columns]
                                                + list(TIMING_COLUMNS))}


# Classifiers

class WeightedLogistic(BaseEstimator, ClassifierMixin):
    """Unweighted standardization, then weighted logistic regression (as in the classifier comparison)."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LogisticRegression(C=self.C, max_iter=1000).fit(self.scaler_.transform(X), y,
                                                                       sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(X))


def boosted(max_depth, learning_rate, n_estimators):
    """The XGBoost grid point as scikit-learn gradient-boosted trees (no early stopping)."""
    return HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate, max_iter=n_estimators,
                                          early_stopping=False, random_state=42)


def forest(**changes):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=42, n_jobs=-1, **{**FOREST, **changes})


def families():
    """``{family: {setting label: estimator}}``; the first setting of the forest family is the current default."""
    tuning = {"original": forest(), "leaf_4_training01": forest(min_samples_leaf=4)}
    tuning.update({label: forest(**params) for label, params in TUNING_CANDIDATES.items() if label != "original"})
    return {
        "random_forest": tuning,
        "logistic_regression": {f"C={p['C']:g}": WeightedLogistic(**p) for p in CLASSIFIER_GRIDS["logistic_regression"]},
        "gradient_boosting": {f"depth={p['max_depth']},rate={p['learning_rate']:g},trees={p['n_estimators']}": boosted(**p)
                              for p in XGBOOST_CANDIDATES},
    }


def tune(X, rows, settings, progress=None):
    """Spatial-CV macro-F1 per setting in the training split; ties go to the first listed setting."""
    scores = {}
    for label, estimator in settings.items():
        views = development_views.view_probabilities(X, rows, estimator=estimator, views=("spatial_cv",))
        pooled = {}
        for parcels in views["spatial_cv"].values():
            pooled.update(parcels)
        scores[label] = development_views.scores(pooled)["macro_f1"]
        if progress:
            progress(f"  {label}: spatial CV {scores[label]:.3f}")
    return max(scores, key=lambda label: (round(scores[label], 6), -list(settings).index(label))), scores


def evaluate(X, rows, estimator):
    return development_views.summarize(development_views.view_probabilities(X, rows, estimator=estimator))
