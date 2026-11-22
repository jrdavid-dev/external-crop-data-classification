"""Classifier families and settings for development-only selection.

The grids are the external study's: one-setting Random Forest changes
(``model_followups.TUNING_CANDIDATES``) plus Training 01's leaf size 4, weighted logistic
regression and gradient-boosted trees (``classifier_comparison``). Every estimator accepts the
pipeline's ``sample_weight``. Gradient boosting uses XGBoost when it loads; otherwise
scikit-learn's ``HistGradientBoostingClassifier`` with the same depth, learning rate and number
of trees, and ``gradient_boosting_backend`` records which. The study's RBF SVM is not included:
calibrated kernel SVMs do not scale to the annual recipe's pixel counts.

On macOS, XGBoost needs the OpenMP runtime (``brew install libomp``); scikit-learn's bundled copy
also works when ``DYLD_FALLBACK_LIBRARY_PATH`` points to ``sklearn/.dylibs`` before Python starts.
"""

from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from crop_pipeline.training import development_views
from crop_pipeline.training.classifier_comparison import CANDIDATES, XGBOOST_CANDIDATES
from crop_pipeline.training.model_followups import TUNING_CANDIDATES

DEFAULT = ("random_forest", "original")


@lru_cache(maxsize=1)
def xgboost_available():
    try:
        import xgboost  # noqa: F401
    except Exception:  # missing package or an OpenMP runtime that cannot be loaded
        return False
    return True


def gradient_boosting_backend():
    return "xgboost" if xgboost_available() else "sklearn_hist_gradient_boosting"


class WeightedLogistic(BaseEstimator, ClassifierMixin):
    """Unweighted standardization, then weighted logistic regression."""

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


class GradientBoosting(BaseEstimator, ClassifierMixin):
    """Gradient-boosted trees over crop names: XGBoost when available, else scikit-learn's histogram version."""

    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=150, backend=None):
        self.max_depth, self.learning_rate, self.n_estimators, self.backend = max_depth, learning_rate, n_estimators, backend

    def fit(self, X, y, sample_weight=None):
        backend = self.backend or gradient_boosting_backend()
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        if backend == "xgboost":
            from xgboost import XGBClassifier
            # No fixed objective: a dataset-holdout fit can see only two crops, where XGBoost needs a binary objective.
            self.model_ = XGBClassifier(tree_method="hist", random_state=42, n_jobs=4,
                                        max_depth=self.max_depth, learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators)
        else:
            self.model_ = HistGradientBoostingClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                                         max_iter=self.n_estimators, early_stopping=False,
                                                         random_state=42)
        self.backend_ = backend
        self.model_.fit(X, self.encoder_.transform(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X))


def forest(final=False, **changes):
    base = development_views.FOREST if final else development_views.SELECTION_FOREST
    return RandomForestClassifier(random_state=42, n_jobs=-1, **{**base, **changes})


def families(final=False):
    """``{family: {setting: estimator}}``; listing order breaks ties, so the current default comes first.

    ``final`` gives the freezing forests 1,000 trees instead of the selection forests' 300.
    """
    tuning = {"original": forest(final), "leaf_4_training01": forest(final, min_samples_leaf=4)}
    tuning.update({label: forest(final, **params) for label, params in TUNING_CANDIDATES.items() if label != "original"})
    return {
        "random_forest": tuning,
        "logistic_regression": {f"C={p['C']:g}": WeightedLogistic(**p) for p in CANDIDATES["logistic_regression"]},
        "gradient_boosting": {f"depth={p['max_depth']},rate={p['learning_rate']:g},trees={p['n_estimators']}":
                              GradientBoosting(**p) for p in XGBOOST_CANDIDATES},
    }


def estimator(family, setting, final=True):
    return families(final)[family][setting]


# Classifier comparison on development views (moved from the annual workflow; recorded, never used to choose)

def tune_family(X, rows, settings, progress=None):
    """The study's tuning rule: highest pooled spatial-CV macro-F1 in the training split; ties to the first listed."""
    scores = {}
    for label, model in settings.items():
        views = development_views.view_probabilities(X, rows, estimator=model, views=("spatial_cv",))
        pooled = {g: p for folds in views["spatial_cv"].values() for g, p in folds.items()}
        scores[label] = development_views.scores(pooled)["macro_f1"]
        if progress:
            progress(f"  {label}: spatial CV {scores[label]:.3f}")
    order = list(settings)
    return max(scores, key=lambda label: (round(scores[label], 6), -order.index(label))), scores


def compare(X, rows, families_=None, progress=print):
    """Tune each family, then score each family's chosen setting on every development view."""
    families_ = families_ or families()
    tuning, results = {}, {}
    for family, settings in families_.items():
        if progress:
            progress(f"tuning {family} ({len(settings)} settings)")
        chosen, scores = tune_family(X, rows, settings, progress)
        tuning[family] = {"chosen": chosen, "spatial_cv_macro_f1": scores}
        summary = development_views.summarize(development_views.view_probabilities(X, rows,
                                                                                   estimator=settings[chosen]))
        results[family] = {"setting": chosen, **summary}
        if progress:
            progress(f"{family} {chosen}: selection score {summary['selection_score']:.3f}")
    return tuning, results


def best_family(results):
    """Highest selection score (rounded to 1e-6); ties go to the earlier family (Random Forest first)."""
    order = list(results)
    return max(results, key=lambda f: (round(results[f]["selection_score"], 6), -order.index(f)))


def compare_on_selection(repository, selection_handoff=None, families_=None, progress=print):
    """Run the comparison on a pipeline selection's recipe and development pixels; returns (tuning, results)."""
    import json
    from crop_pipeline.training import annual_workflow as aw

    selection, record = aw._input(repository, selection_handoff, "selection")
    recipe = json.loads((selection.parent / "selection.json").read_text())["selected"]
    inputs, _ = aw.verify(record["inputs_handoff"], "inputs")
    rows, arrays = aw.load_inputs(inputs.parent)
    built = {recipe: aw.recipe_matrix(recipe, arrays)}
    index = aw.development_index(rows, built)
    return compare(built[recipe][0][index], [rows[i] for i in index], families_, progress)
