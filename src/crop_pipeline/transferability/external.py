"""External tables, CropHarvest catalogue matching and the data profiles of section 1."""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from crop_pipeline.training import external_combined as ext

from .atok import OTHER_CROP_FIELDS
from .config import CLASSES, COLORS, SOURCE_NAMES

MATCH_RADIUS_M = 300
EARTH_RADIUS_M = 6_371_000
ATOK_SOURCE = "Atok (Paoay KML)"

PROVENANCE = {  # source -> (label provenance, label detail)
    "CropHarvest": ("verified: published dataset", "survey or declaration label, one season; no window dates"),
    "Sen4AgriNet": ("verified: published dataset", "Catalonia farmer declaration, 2020 season"),
    "FranceCrops": ("verified: published dataset", "French parcel declaration, 2022 season"),
}
TARGET_PATTERNS = {"carrot": r"carrot|carotte", "cabbage": r"cabbage|^chou$", "potato": r"potato|pomme de terre"}
NOT_TARGET_PATTERN = r"sweet|douce|chinese|fleur"  # sweet potato, Chinese cabbage, cauliflower (chou-fleur)
STARCH_POTATO = "féculière"


def load_data(paths):
    """v1's cached tables with v1's locked test split, plus readable source and region names."""
    tables, _ = ext.cached_tables(None, paths.earth_engine_project, paths.v1 / "inputs", False)
    data = ext.add_derived_features(ext.harmonize(tables))
    data, _ = ext.link_split_units(data)
    data = ext.add_spatial_blocks(data)
    data = ext.lock_test_split(data, paths.v1 / "test_split.csv")  # verifies v1's split; never rewrites it
    data["source"] = data["domain"].map(SOURCE_NAMES)
    data["region_name"] = np.select([data["domain"] == "sen4agrinet", data["domain"] == "francecrops", data["region"] == "France"],
                                    ["Catalonia (ES)", "France", "Île-de-France"], data["region"])
    data["source_region"] = data["source"] + ": " + data["region_name"]
    return tables, data


def split_partitions(data):
    development = data[data["partition"] == "development"].reset_index(drop=True)
    test = data[data["partition"] == "test"].reset_index(drop=True)
    return development, test


def inventory(data, atok, interiors):
    """One row per source and crop: years, regions, parcels, pixels and label provenance."""
    rows = []
    for (source, crop), part in data.groupby(["source", "crop"]):
        per_parcel = part.groupby("partition_group").size()
        rows.append({
            "source": source, "crop": crop, "years": ", ".join(map(str, sorted(part["year"].unique()))),
            "regions": "; ".join(f"{r} {n}" for r, n in part["region_name"].value_counts().items()),
            "parcels": part["partition_group"].nunique(), "pixels": len(part), "samples": len(part),
            "pixels_per_parcel_median": per_parcel.median(), "label_provenance": PROVENANCE[source][0],
            "label_detail": PROVENANCE[source][1]})
    for crop, part in atok.groupby("crop"):
        unique = part.drop_duplicates(["field", "row", "col"])
        per_field = unique.groupby("field").size()
        rows.append({
            "source": ATOK_SOURCE, "crop": crop, "years": "2019-2025 (calendar repeated)",
            "regions": "; ".join(f"{f} {n}" for f, n in per_field.items()),
            "parcels": part["field"].nunique(), "pixels": len(unique), "samples": len(part),
            "pixels_per_parcel_median": per_field.median(), "label_provenance": "advisor-described rotation (provisional)",
            "label_detail": "; ".join(sorted(part["label_basis"].unique()))})
    for field, crop in OTHER_CROP_FIELDS:
        pixels = int(interiors[field].sum())
        rows.append({
            "source": ATOK_SOURCE, "crop": f"other: {crop}", "years": "-", "regions": field, "parcels": 1,
            "pixels": pixels, "samples": 0, "pixels_per_parcel_median": pixels,
            "label_provenance": "advisor-described rotation (provisional)", "label_detail": "not used: no other_crop class"})
    return pd.DataFrame(rows)


def imbalance(inventory_table):
    """Largest ÷ smallest class per source; a source with one crop has ratio 1 and cannot be balanced."""
    used = inventory_table[inventory_table["samples"] > 0]
    pooled = used[used["source"] != ATOK_SOURCE].groupby("crop")[["samples", "parcels"]].sum().reset_index()
    rows = []
    for source, part in list(used.groupby("source")) + [("External pooled", pooled)]:
        rows.append({"source": source,
                     "largest class (samples)": part.loc[part["samples"].idxmax(), "crop"],
                     "smallest class (samples)": part.loc[part["samples"].idxmin(), "crop"],
                     "sample ratio": part["samples"].max() / part["samples"].min(),
                     "parcel ratio": part["parcels"].max() / part["parcels"].min(),
                     "crops": part["crop"].nunique()})
    return pd.DataFrame(rows)


def region_counts(data):
    return data.groupby("source_region").agg(parcels=("partition_group", "nunique"), pixels=("sample_id", "size"),
                                             lon=("lon", "mean"), lat=("lat", "mean"))


def region_crop_parcels(data, atok):
    """Parcels per region and crop (fields for Atok), and the development/test split of each cell."""
    coverage = (data.groupby(["source_region", "crop"])["partition_group"].nunique().unstack(fill_value=0)
                .reindex(columns=CLASSES, fill_value=0))
    coverage.loc["Atok (provisional fields)"] = atok.groupby("crop")["field"].nunique().reindex(CLASSES, fill_value=0)
    dev_test = data.groupby(["source_region", "crop", "partition"])["partition_group"].nunique().unstack(fill_value=0)
    return coverage, dev_test


def external_windows(paths, data):
    """Notebook 04 windows for the parcels in the Sen4AgriNet/FranceCrops table."""
    windows = pd.read_csv(paths.windows_run / "estimated_windows.csv", parse_dates=["window_start"])
    table_groups = data.loc[data["domain"] != "cropharvest", "partition_group"].str.split("|", n=1).str[1]
    return windows[windows["partition_group"].isin(set(table_groups))]


def cropharvest_catalogue(labels_path):
    """CropHarvest labels whose name is a target crop."""
    labels = gpd.read_file(labels_path)
    point = labels.geometry.representative_point()
    labels["lat"], labels["lon"] = labels["lat"].fillna(point.y), labels["lon"].fillna(point.x)
    labels = pd.DataFrame(labels.drop(columns="geometry")).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    label_text = labels["label"].astype(str).str.lower()
    not_target = label_text.str.contains(NOT_TARGET_PATTERN, regex=True)
    labels["target_crop"] = None
    for crop, pattern in TARGET_PATTERNS.items():
        labels.loc[label_text.str.contains(pattern, regex=True) & ~not_target, "target_crop"] = crop
    return labels[labels["target_crop"].notna()].reset_index(drop=True)


def match_cropharvest(data, catalogue):
    """Nearest catalogue label for every CropHarvest table row (for planting dates)."""
    ch_rows = data[data["domain"] == "cropharvest"].reset_index(drop=True)
    distance, index = BallTree(np.radians(catalogue[["lat", "lon"]].to_numpy()), metric="haversine").query(
        np.radians(ch_rows[["lat", "lon"]].to_numpy()), k=1)
    match = catalogue.iloc[index[:, 0]].reset_index(drop=True).assign(
        distance_m=distance[:, 0] * EARTH_RADIUS_M, partition_group=ch_rows["partition_group"], crop=ch_rows["crop"],
        region_name=ch_rows["region_name"])
    return ch_rows, match


def cropharvest_planting(match):
    return (match[match["distance_m"] <= MATCH_RADIUS_M].dropna(subset=["planting_date"])
            .groupby("partition_group").agg(planting=("planting_date", "first"), crop=("crop", "first"),
                                            region=("region_name", "first")))


def half_month_index(date):
    return (date.month - 1) * 2 + (0 if date.day <= 15 else 1)


def coverage_from_starts(starts):
    """Share of windows covering each of the 24 half-months, for 8-half-month windows from `starts`."""
    hits = np.zeros(24)
    for start in starts:
        first = half_month_index(start)
        hits[[(first + k) % 24 for k in range(8)]] += 1
    return hits / max(len(starts), 1)


def calendar_vector(first_month, last_month):
    months = [((first_month - 1 + k) % 12) + 1 for k in range((last_month - first_month) % 12 + 1)]
    vector = np.zeros(24)
    for month in months:
        vector[(month - 1) * 2:(month - 1) * 2 + 2] = 1
    return vector


ATOK_CALENDAR = [  # label, first month, last month, crop colour (None = grey)
    ("Atok B1 carrot", 3, 6, "carrot"),
    ("Atok Area 2 potato", 3, 6, "potato"),
    ("Atok Area 2 cabbage", 7, 11, "cabbage"),
    ("Atok Area 2 carrot", 11, 2, "carrot"),
    ("Atok Area 3 cabbage (year-round)", 1, 12, "cabbage"),
    ("Atok B1 cabbage (not labelled: 2 months)", 1, 2, None),
]


def window_coverage_rows(windows, ch_rows, ch_planting):
    """(label, 24 half-month coverage, colour, note) for Atok's calendar and each external source."""
    rows = [(label, calendar_vector(first, last), COLORS[crop] if crop else "#AAAAAA", "assumed: advisor calendar, undated")
            for label, first, last, crop in ATOK_CALENDAR]
    for (source, crop), part in windows.groupby(["source_dataset", "crop"]):
        rows.append((f"{SOURCE_NAMES[source]} {crop}", coverage_from_starts(part["window_start"]), COLORS[crop],
                     f"observed: detected onset, {len(part)} windows"))
    for (region, crop), part in ch_planting.groupby(["region", "crop"]):
        rows.append((f"CropHarvest {region} {crop}", coverage_from_starts(pd.to_datetime(part["planting"])), COLORS[crop],
                     f"approximate: planting date + 4 months, {len(part)} parcels" + (" (southern hemisphere)" if region == "Madagascar" else "")))
    undated = ch_rows[~ch_rows["partition_group"].isin(ch_planting.index)].groupby("region_name")["partition_group"].nunique()
    for region, count in undated.items():
        rows.append((f"CropHarvest {region}", np.zeros(24), "#FFFFFF", f"{count} parcels: no window or planting dates"))
    return rows


def preprocessing_audits(paths):
    """Notebook 04 audits behind the Sen4AgriNet/FranceCrops table."""
    run = paths.windows_run
    return {"timing": pd.read_csv(run / "automatic_timing_audit.csv"),
            "selection": pd.read_csv(run / "cycle_selection_audit.csv"),
            "usable": pd.read_csv(run / "usable_pixel_selection.csv"),
            "windows": pd.read_csv(run / "estimated_windows.csv")}


def window_means(frame, variables):
    """Mean of the 8 half-months for each variable."""
    return pd.DataFrame({v: frame[[f"bin_{b:02d}_{v}" for b in range(1, 9)]].mean(axis=1).to_numpy() for v in variables})


def short_region(name):
    return name.split(": ")[-1].replace(" (FR)", "").replace(" (ES)", "")
