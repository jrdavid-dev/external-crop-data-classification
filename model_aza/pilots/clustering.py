"""Gaussian mixture clustering with BIC model choice and a stability check."""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def select_gmm(X, components=range(2, 11), covariance="full", seed=0, sample=20000):
    """Fit GMMs on standardized features; returns (best model, scaler, BIC table)."""
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    fit_rows = Z if len(Z) <= sample else Z[rng.choice(len(Z), sample, replace=False)]
    table, best = [], None
    for k in components:
        model = GaussianMixture(k, covariance_type=covariance, random_state=seed, n_init=3).fit(fit_rows)
        bic = float(model.bic(fit_rows))
        table.append({"components": k, "bic": bic})
        if best is None or bic < best[0]:
            best = (bic, model)
    return best[1], scaler, table


def stability(X, components, scaler, repeats=5, share=.7, covariance="full", seed=0):
    """Mean adjusted Rand index between refits on random subsamples, labels compared on all rows."""
    rng = np.random.default_rng(seed)
    Z = scaler.transform(X)
    labels = []
    for repeat in range(repeats):
        rows = rng.choice(len(Z), int(len(Z) * share), replace=False)
        model = GaussianMixture(components, covariance_type=covariance, random_state=seed + repeat, n_init=3).fit(Z[rows])
        labels.append(model.predict(Z))
    scores = [adjusted_rand_score(labels[i], labels[j]) for i in range(repeats) for j in range(i + 1, repeats)]
    return float(np.mean(scores)), float(np.min(scores))


def mahalanobis_keep(Z, quantile=.98):
    """Keep rows whose squared Mahalanobis distance is below the chi-square ``quantile``.

    Also returns the share kept by the literal ``distance <= 2.58`` rule, which is
    a one-dimensional 99% cutoff and drops far more than 2% in many dimensions.
    """
    from scipy.stats import chi2
    centred = Z - Z.mean(axis=0)
    inverse = np.linalg.pinv(np.cov(centred, rowvar=False))
    squared = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    keep = squared <= chi2.ppf(quantile, df=Z.shape[1])
    return keep, float(np.mean(np.sqrt(squared) <= 2.58))


def principal_components(Z, components=3):
    from sklearn.decomposition import PCA
    model = PCA(components, random_state=0).fit(Z)
    return model, model.transform(Z)


def isometric_bins(scores, bins=3, low=1, high=99):
    """PCIB: equal-width bins on each of the first two principal components (robust range)."""
    labels = np.zeros(len(scores), dtype=int)
    for column in range(2):
        lo, hi = np.percentile(scores[:, column], [low, high])
        edges = np.linspace(lo, hi, bins + 1)[1:-1]
        labels = labels * bins + np.digitize(scores[:, column], edges)
    return labels


def logratio_assign(cluster_sizes, shares, other=False, tolerance=.05):
    """Assign clusters to crops so pairwise log-ratios of assigned area match ``shares``.

    Loss = sum over crop pairs (log z_j/z_k - log zhat_j/zhat_k)^2 (Wang et al. 2019).
    Every crop needs at least one cluster; with ``other`` a cluster may stay
    unassigned (non-target cover). Returns the best assignment, its loss and how
    many assignments reach a loss within ``tolerance`` of it (identifiability).
    """
    from itertools import product
    crops = list(shares)
    target = np.log(np.asarray([shares[c] for c in crops], dtype=float))
    clusters = list(cluster_sizes)
    sizes = np.asarray([cluster_sizes[c] for c in clusters], dtype=float)
    options = list(range(len(crops))) + ([-1] if other else [])
    results = []
    for choice in product(options, repeat=len(clusters)):
        area = np.asarray([sizes[[i for i, k in enumerate(choice) if k == j]].sum() for j in range(len(crops))])
        if (area == 0).any():
            continue
        estimate = np.log(area)
        loss = sum((target[j] - target[k] - estimate[j] + estimate[k]) ** 2
                   for j in range(len(crops)) for k in range(j + 1, len(crops)))
        results.append((float(loss), choice))
    if not results:
        raise ValueError("No assignment gives every crop a cluster")
    results.sort(key=lambda item: item[0])
    best_loss, best = results[0]
    near = sum(loss <= best_loss + tolerance for loss, _ in results)
    assignment = {cluster: (crops[k] if k >= 0 else "other") for cluster, k in zip(clusters, best)}
    return assignment, best_loss, near, len(results)
