"""Valid observations, missing intervals and usable coverage in a half-month image cube.

The cube is ``(slots, bands, rows, cols)`` with one slot per half-month (24 per year,
starting in January of ``first_year``) and NaN wherever a composite gave no valid value.
A pixel-slot is *valid* when every band is finite.
"""

import numpy as np
import pandas as pd

SLOTS_PER_YEAR = 24
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def valid_mask(cube):
    """(slots, rows, cols): True where every band is finite."""
    return np.isfinite(cube).all(axis=1)


def slot_index(first_year, count):
    slots = np.arange(count)
    return pd.DataFrame({"year": first_year + slots // SLOTS_PER_YEAR,
                         "month": (slots % SLOTS_PER_YEAR) // 2 + 1,
                         "half": slots % 2 + 1})


def slot_validity(cube, mask, first_year):
    """One row per half-month: share of mask pixels with any data and with valid data.

    A slot with no data anywhere in the mask is a *missing interval* (no composite);
    otherwise invalid pixels are attributed to cloud and shadow masking.
    """
    valid = valid_mask(cube)[:, mask]
    any_data = np.isfinite(cube).any(axis=1)[:, mask]
    table = slot_index(first_year, len(cube))
    table["data_share"] = any_data.mean(axis=1)
    table["valid_share"] = valid.mean(axis=1)
    table["missing_interval"] = table["data_share"] == 0
    return table


def never_observed(cube, mask=None):
    """Pixels without a single valid half-month (outside the imagery footprint)."""
    never = ~valid_mask(cube).any(axis=0)
    return never if mask is None else never & mask


def yearly_quality(cube, mask, first_year, years=None):
    """Valid observations, missing intervals and cloud loss per year for the mask.

    Cloud-related loss is the invalid share of pixel-slots in half-months that do have a
    composite, over pixels that are observed at least once (so footprint gaps are excluded).
    """
    valid = valid_mask(cube)
    observed = mask & ~never_observed(cube)
    slots = slot_validity(cube, mask, first_year)
    rows = []
    for year, part in slots.groupby("year"):
        if years is not None and year not in years:
            continue
        index = part.index.to_numpy()
        present = index[~part["missing_interval"].to_numpy()]
        pixel_valid = valid[index][:, observed]
        cloud_loss = 1 - valid[present][:, observed].mean() if len(present) else np.nan
        rows.append({
            "year": year,
            "half_months": len(index),
            "missing_intervals": int(part["missing_interval"].sum()),
            "valid_observations_per_pixel": float(pixel_valid.sum(axis=0).mean()),
            "valid_share": float(pixel_valid.mean()),
            "cloud_related_loss": float(cloud_loss),
            "median_longest_gap_half_months": float(np.median(longest_invalid_run(pixel_valid))),
        })
    return pd.DataFrame(rows).set_index("year")


def longest_invalid_run(valid):
    """Longest run of consecutive invalid slots for each column of a (slots, pixels) array."""
    longest = np.zeros(valid.shape[1], dtype=int)
    current = np.zeros(valid.shape[1], dtype=int)
    for row in valid:
        current = np.where(row, 0, current + 1)
        longest = np.maximum(longest, current)
    return longest


def usable_coverage(cube, mask, first_year, windows, years, length=8, min_valid=6):
    """Share of mask pixels with at least ``min_valid`` of ``length`` valid half-months.

    ``windows`` maps a start month to a label; each window starts in the first half of
    that month and may run into the next year.
    """
    valid = valid_mask(cube)[:, mask]
    table = {}
    for year in years:
        row = {}
        for month, label in windows.items():
            start = (year - first_year) * SLOTS_PER_YEAR + (month - 1) * 2
            block = valid[start:start + length]
            row[label] = float((block.sum(axis=0) >= min_valid).mean()) if len(block) == length else np.nan
        table[year] = row
    return pd.DataFrame(table).T.rename_axis("year")


def validity_grid(slots, years=None):
    """Year x half-month table of valid share, NaN for missing intervals."""
    part = slots if years is None else slots[slots["year"].isin(years)]
    grid = part.assign(slot=(part["month"] - 1) * 2 + part["half"] - 1,
                       value=part["valid_share"].where(~part["missing_interval"]))
    return grid.pivot(index="year", columns="slot", values="value")
