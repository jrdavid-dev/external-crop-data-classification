"""Exploratory Atok crop maps on the ELUM cropland mask, and their diagnostics (section 7)."""

import numpy as np
import pandas as pd
import rasterio

from .atok import LABEL_ROWS, SLOT_YEAR, SLOTS, window_features
from .config import CLASSES, INTERVAL, LABEL_YEARS, MIN_CLEAR
from .models import THRESHOLDS, aligned_proba, predicted

MAP_WINDOWS = {3: "Mar-Jun (potato season)", 7: "Jul-Oct (cabbage season)", 11: "Nov-Feb (carrot season)"}
CLASS_CODES = "1 carrot, 2 cabbage, 3 potato, 4 unknown"


def mappable_share(grid, cropland):
    """Share of the cropland mask with >= MIN_CLEAR of 8 clear composites, per year and map window."""
    rr, cc = np.nonzero(cropland)
    clear = grid.clear_all
    shares = {}
    for year in LABEL_YEARS:
        starts = [SLOTS.index((year, m, 1)) for m in MAP_WINDOWS]
        shares[year] = [(clear[s:s + 8][:, rr, cc].sum(0) >= MIN_CLEAR).mean() for s in starts]
    table = pd.DataFrame(shares, index=list(MAP_WINDOWS.values())).T.rename_axis("year")
    table["mean"] = table.mean(axis=1)
    return table


def never_clear(grid, last_year=2025):
    return ~grid.clear_all[SLOT_YEAR <= last_year].any(0)


class MapPredictor:
    """Predict a calibrated model over the cropland mask for one 8-half-month window; features are cached."""

    def __init__(self, grid, cropland, distance_to_training, ood_threshold):
        self.grid, self.distance, self.ood_threshold = grid, distance_to_training, ood_threshold
        self.rr, self.cc = np.nonzero(cropland)
        self.cache = {}

    def __call__(self, model, year, month, threshold):
        """(class codes, top probability, kNN distance, low-probability mask, OOD mask) on the grid."""
        if (year, month) not in self.cache:
            features, keep, _ = window_features(self.grid, self.rr, self.cc, SLOTS.index((year, month, 1)))
            self.cache[(year, month)] = (features, keep, self.distance(features))
        features, keep, distance = self.cache[(year, month)]
        proba = aligned_proba(model, features, INTERVAL)
        shape = (self.grid.height, self.grid.width)
        codes, top, dist = (np.full(shape, np.nan, "float32") for _ in range(3))
        low, ood = np.zeros(shape, bool), np.zeros(shape, bool)
        rows, cols = self.rr[keep], self.cc[keep]
        codes[rows, cols], top[rows, cols], dist[rows, cols] = proba.argmax(1), proba.max(1), distance
        low[rows, cols] = proba.max(1) < threshold
        ood[rows, cols] = distance > self.ood_threshold
        return codes, top, dist, low, ood


def write_map(path, grid, codes, top, dist, unknown, model_name, month, threshold, ood_threshold):
    """3-band GeoTIFF: class code (4 = unknown), top probability, kNN distance."""
    band1 = np.where(np.isfinite(codes), np.where(unknown, 4, codes + 1), np.nan).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=3, dtype="float32",
                       crs=grid.crs, transform=grid.transform, nodata=np.nan) as sink:
        sink.write(band1, 1)
        sink.write(top, 2)
        sink.write(dist, 3)
        sink.update_tags(model=model_name, status="exploratory; not validated; provisional labels", classes=CLASS_CODES,
                         window=MAP_WINDOWS[month],
                         unknown_rule=f"calibrated top probability < {threshold} or kNN distance > {ood_threshold:.3f}")


def predict_maps(predictor, map_models, years, output_folder):
    """Predict every model, year and window; write the GeoTIFFs; return the raw layers and a summary table."""
    raw, records = {}, []
    cropland_pixels = len(predictor.rr)
    for year in years:
        for model_name, (slug, model, threshold) in map_models.items():
            for month in MAP_WINDOWS:
                codes, top, dist, low, ood = predictor(model, year, month, threshold)
                unknown = low | ood
                raw[(model_name, year, month)] = (codes, top, dist, low, ood)
                write_map(output_folder / f"atok_{slug}_{year}-{month:02d}.tif", predictor.grid, codes, top, dist, unknown,
                          model_name, month, threshold, predictor.ood_threshold)
                mapped = np.isfinite(codes)
                known = mapped & ~unknown
                shares = pd.Series(np.asarray(CLASSES)[codes[known].astype(int)]).value_counts(normalize=True)
                records.append({"model": model_name, "year": year, "window": MAP_WINDOWS[month],
                                "mapped_share_of_mask": mapped.sum() / cropland_pixels,
                                "threshold": threshold, "unknown_share_of_mapped": float(unknown[mapped].mean()) if mapped.any() else np.nan,
                                "uncertain_share_of_mapped": float(low[mapped].mean()) if mapped.any() else np.nan,
                                "ood_share_of_mapped": float(ood[mapped].mean()) if mapped.any() else np.nan,
                                **{f"share_{c}_of_known": float(shares.get(c, 0)) for c in CLASSES}})
    return raw, pd.DataFrame(records)


def rejection_causes(raw, map_models, year):
    """Per model and window: share of mapped pixels classified, uncertain, out of distribution, or both."""
    rows, masks = [], {}
    for model_name, (_, _, threshold) in map_models.items():
        for month in MAP_WINDOWS:
            codes, _, _, low, ood = raw[(model_name, year, month)]
            mapped = np.isfinite(codes)
            categories = {"classified": mapped & ~low & ~ood, "uncertain only": mapped & low & ~ood,
                          "out of distribution only": mapped & ~low & ood, "both": mapped & low & ood}
            masks[(model_name, month)] = categories
            rows.append({"model": model_name, "window": MAP_WINDOWS[month], "threshold": threshold, "mapped_pixels": int(mapped.sum()),
                         **{k: float(v.sum() / max(mapped.sum(), 1)) for k, v in categories.items()}})
    return pd.DataFrame(rows), masks


def operating_points(raw, map_models, year):
    """Share of mappable cropland classified at each threshold, with and without the OOD check."""
    rows = []
    for model_name in map_models:
        for month in MAP_WINDOWS:
            codes, top, _, _, ood = raw[(model_name, year, month)]
            mapped = np.isfinite(codes)
            for use_ood in (True, False):
                coverage = [float(((top >= t) & (~ood if use_ood else True))[mapped].mean()) for t in THRESHOLDS]
                rows += [{"model": model_name, "window": MAP_WINDOWS[month], "ood_check": use_ood, "threshold": float(t), "coverage": c}
                         for t, c in zip(THRESHOLDS, coverage)]
    return pd.DataFrame(rows)


def model_agreement(raw, map_models, year):
    """Where the two map models agree, disagree, or either abstains; with crop-to-crop transition tables."""
    first, second = list(map_models)
    rows, transitions, masks = [], [], {}
    for month in MAP_WINDOWS:
        (a_codes, *_, a_low, a_ood), (b_codes, *_, b_low, b_ood) = (raw[(n, year, month)] for n in (first, second))
        a_unknown, b_unknown = a_low | a_ood, b_low | b_ood
        both = np.isfinite(a_codes) & np.isfinite(b_codes)
        either_unknown = both & (a_unknown | b_unknown)
        known = both & ~either_unknown
        agree, disagree = known & (a_codes == b_codes), known & (a_codes != b_codes)
        masks[month] = {"agree": agree, "disagree": disagree, "either_unknown": either_unknown}
        rows.append({"window": MAP_WINDOWS[month], "pixels_both_mapped": int(both.sum()), "agree": int(agree.sum()),
                     "disagree": int(disagree.sum()), "either_unknown": int(either_unknown.sum()),
                     "agree_share_of_known": float(agree.sum() / max(known.sum(), 1))})

        def label(codes, unknown):
            return np.where(unknown[both], "unknown", np.asarray(CLASSES + ["?"])[np.nan_to_num(codes[both], nan=3).astype(int)])

        table = pd.crosstab(pd.Series(label(a_codes, a_unknown), name=first), pd.Series(label(b_codes, b_unknown), name=second))
        transitions.append((MAP_WINDOWS[month], table))
    return pd.DataFrame(rows), transitions, masks


def labelled_field_calls(atok, map_models):
    """Each map model's call on the labelled pixel-windows: predicted crop and whether it abstains."""
    proba = {name: aligned_proba(model, atok, INTERVAL) for name, (_, model, _) in map_models.items()}
    unknown = {name: (proba[name].max(1) < map_models[name][2]) | atok["ood"].to_numpy() for name in map_models}
    called = {name: predicted(proba[name]) for name in map_models}
    return called, unknown


def field_errors(atok, called, unknown, year):
    """Per model, window and labelled field: share correct, unknown, and wrong as each crop."""
    rows = []
    for name in called:
        for month in MAP_WINDOWS:
            selected = ((atok["year"] == year) & (atok["start_month"] == month)).to_numpy()
            truth, calls, abstain = atok.loc[selected, "crop"].to_numpy(), called[name][selected], unknown[name][selected]
            for field in sorted(set(atok.loc[selected, "field"])):
                in_field = (atok.loc[selected, "field"] == field).to_numpy()
                rows.append({"model": name, "window": MAP_WINDOWS[month], "field": field, "calendar_crop": truth[in_field][0],
                             "pixels": int(in_field.sum()), "correct": float(((calls == truth) & ~abstain)[in_field].mean()),
                             "unknown": float(abstain[in_field].mean()),
                             **{f"wrong_as_{c}": float(((calls == c) & (c != truth) & ~abstain)[in_field].mean()) for c in CLASSES}})
    return pd.DataFrame(rows)


def field_consistency(atok, called, unknown):
    """Per pixel: share of its labelled windows (2019-2025) called correctly; unknown counts as not correct."""
    shares, rows = {}, []
    for name in called:
        frame = atok[["row", "col", "field"]].assign(correct=(called[name] == atok["crop"].to_numpy()) & ~unknown[name])
        per_pixel = frame.groupby(["row", "col"]).agg(share=("correct", "mean"), windows=("correct", "size")).reset_index()
        shares[name] = per_pixel
        rows.append({"model": name, "pixels": len(per_pixel), "median_windows_per_pixel": per_pixel["windows"].median(),
                     "never_correct": float((per_pixel["share"] == 0).mean()), "always_correct": float((per_pixel["share"] == 1).mean()),
                     "mean_share": per_pixel["share"].mean()})
    return shares, pd.DataFrame(rows)


def labelled_fields():
    return sorted({row[0] for row in LABEL_ROWS})
