"""Count candidate crop references from local public releases."""

import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio

from crop_pipeline.sources import cropharvest, cropharvest_review, francecrops

CROPS = ("carrot", "cabbage", "potato")
CDL_CODES = dict(zip(CROPS, (206, 243, 43)))
ALIASES = {
    "carrot": "carrot", "carrots": "carrot", "carotte": "carrot",
    "cabbage": "cabbage", "cabbages": "cabbage", "chou": "cabbage",
    "potato": "potato", "potatoes": "potato", "table potato": "potato",
    "starch potato": "potato", "pomme de terre de consommation": "potato",
    "pomme de terre féculière": "potato",
}


def audit_cropharvest(path, output_dir):
    """Return release size, stage counts, and eligible counts by source."""
    cropharvest.validate_label_checksum(path, cropharvest.LABELS_MD5)
    with path.open(encoding="utf-8") as stream:
        features = json.load(stream)["features"]
    raw = pd.Series([item["properties"].get("label") for item in features], dtype="string")
    raw_counts = raw.str.strip().str.casefold().map(ALIASES).value_counts()
    total = len(features)
    del features
    reviewed = cropharvest_review.review_labels(
        path, output_dir / "unused_country_boundaries.zip", output_dir
    )
    summary = reviewed["summary"].reindex(CROPS).assign(raw_alias_matches=raw_counts)
    eligible = []
    screened = reviewed["reviewed_labels"]
    for row in screened.loc[screened.reliable].drop_duplicates(
        ["crop", "is_test", "dataset", "geom_wkb"]
    ).itertuples():
        feature = cropharvest.parcel_feature(
            row.uid, row.geometry, row.crop, row.observation_year,
            row.country, row.dataset, cropharvest.LABELS_RECORD,
        )
        if francecrops.parcel_pixel_candidates(feature, maximum_pixels=1):
            eligible.append((row.crop, row.dataset, row.is_test))
    by_source = pd.DataFrame(eligible, columns=["crop", "source", "upstream_test"])
    by_source = by_source.groupby(["crop", "source", "upstream_test"]).size().rename(
        "interior_pixel_parcels"
    ).reset_index()
    summary["interior_pixel_parcels"] = (
        by_source.groupby("crop")["interior_pixel_parcels"].sum().reindex(CROPS, fill_value=0)
    )
    return total, summary, by_source


def audit_eurocrops(paths, mapping_path):
    """Count distinct declared parcel IDs in annual GSA files by region and year."""
    mapping = pd.read_csv(mapping_path, dtype="string", keep_default_na=False)
    mapping["crop"] = mapping["translated_name"].str.strip().str.casefold().map(ALIASES)
    counts = []
    seen = set()
    for path in paths:
        match = re.fullmatch(r"([A-Z0-9]+)_(\d{4})", path.stem)
        if not match:
            continue
        region, year = match.groups()
        if (region, year) in seen:
            raise ValueError(f"Duplicate annual EuroCrops input for {region} {year}")
        seen.add((region, year))
        source_codes = mapping.loc[mapping["nuts"].str.upper().eq(region)]
        if source_codes.empty:
            raise ValueError(f"No crop-code mapping for {region} in {mapping_path}")
        targets = source_codes.dropna(subset=["crop"])[["original_code", "crop"]]
        parcels = {crop: set() for crop in CROPS}
        for batch in pq.ParquetFile(path).iter_batches(columns=["cropfield", "original_code"]):
            rows = batch.to_pandas().dropna()
            rows["original_code"] = rows["original_code"].astype("string")
            matched = rows.merge(targets, on="original_code")
            for crop, group in matched.groupby("crop"):
                parcels[crop].update(group["cropfield"])
        counts.extend((region, int(year), crop, len(ids)) for crop, ids in parcels.items())
    return pd.DataFrame(counts, columns=["region", "year", "crop", "declared_parcels"])


def audit_usda_cdl(path):
    """Count unmasked pixels in one official USDA CDL raster."""
    counts = dict.fromkeys(CROPS, 0)
    with rasterio.open(path) as raster:
        for _, window in raster.block_windows(1):
            pixels = raster.read(1, window=window, masked=True).compressed()
            for crop, code in CDL_CODES.items():
                counts[crop] += int(np.count_nonzero(pixels == code))
    return pd.Series(counts, name="classified_pixels").rename_axis("crop")
