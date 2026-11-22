"""Growing degree days and climate similarity between training parcels and a target.

Descriptors use monthly mean air temperature only (no daily data): annual GDD
above ``base`` (days x positive monthly excess), the warmest four-month GDD,
and the coldest and warmest monthly means. Descriptors ignore calendar timing,
so northern and southern hemisphere sites compare on temperature alone.
"""

import calendar

import numpy as np

BASE_C = 5.0
DESCRIPTORS = ("gdd_annual", "gdd_best_4_months", "coldest_month_c", "warmest_month_c")


def monthly_gdd(monthly, year=2021, base=BASE_C):
    days = np.asarray([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    return np.maximum(np.asarray(monthly, dtype=float) - base, 0) * days


def descriptors(monthly, year=2021, base=BASE_C):
    monthly = np.asarray(monthly, dtype=float)
    if monthly.shape != (12,) or not np.isfinite(monthly).all():
        return None
    gdd = monthly_gdd(monthly, year, base)
    best = max(sum(gdd[(start + k) % 12] for k in range(4)) for start in range(12))
    return {"gdd_annual": float(gdd.sum()), "gdd_best_4_months": float(best),
            "coldest_month_c": float(monthly.min()), "warmest_month_c": float(monthly.max())}


def matrix(items):
    return np.asarray([[item[d] for d in DESCRIPTORS] for item in items], dtype=float)


def distances(train_descriptors, target_descriptors):
    """Euclidean distance from each training parcel to the target mean, in training-SD units."""
    train = matrix(train_descriptors)
    target = matrix(target_descriptors).mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    return np.sqrt((((train - target) / scale) ** 2).sum(axis=1))


def nearest_share_weights(rows, parcel_distance, base_weights, keep=.5):
    """Keep the nearest ``keep`` share of parcels within each crop; others get zero weight."""
    groups_by_crop = {}
    for row in rows:
        groups_by_crop.setdefault(row["crop"], set()).add(row["partition_group"])
    kept = set()
    for groups in groups_by_crop.values():
        ranked = sorted(groups, key=lambda g: parcel_distance[g])
        kept.update(ranked[:max(1, int(round(len(ranked) * keep)))])
    weights = np.asarray([w if r["partition_group"] in kept else 0. for r, w in zip(rows, base_weights)])
    return weights * (len(rows) / weights.sum())


def similarity_weights(rows, parcel_distance, base_weights, bandwidth=1.0):
    """Soft weights exp(-d^2 / 2 b^2) times the pipeline weights."""
    factor = np.asarray([np.exp(-parcel_distance[r["partition_group"]] ** 2 / (2 * bandwidth ** 2)) for r in rows])
    weights = np.asarray(base_weights) * factor
    return weights * (len(rows) / weights.sum())
