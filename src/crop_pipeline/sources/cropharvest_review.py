"""CropHarvest label audit extracted without changing the notebook's eligibility rules.

Reference dates resolve observation years, not planting dates. Source grades and
placeholder flags remain screening diagnostics, not independent label validation.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from crop_pipeline.sources import cropharvest as cropharvest_source


def review_labels(
    labels_path,
    country_boundaries_path,
    run_folder,
    minimum_reference_year=2017,
    maximum_reference_year=2100,
    observation_year_overrides=None,
):
    """Screen labels and save the same year-decision and target-label CSV audits.

    The returned target table includes both passed and excluded labels. Upstream
    test labels are retained in this audit; callers must exclude them before
    preparing development parcels, as the original notebook did.
    """
    labels_path, country_boundaries_path = Path(labels_path), Path(
        country_boundaries_path
    )
    run_folder = Path(run_folder)
    observation_year_overrides = observation_year_overrides or {}
    crop_order = ["cabbage", "carrot", "potato"]
    screened_labels_output = run_folder / "labels/target_crop_labels_screened.csv"
    with labels_path.open(encoding="utf-8") as label_stream:
        source_geojson = json.load(label_stream)
    raw_reference_features = source_geojson["features"]
    label_table = pd.DataFrame(
        [
            reference_feature["properties"]
            for reference_feature in raw_reference_features
        ]
    )
    label_table["geometry"] = [
        (
            shape(reference_feature["geometry"])
            if reference_feature.get("geometry")
            else None
        )
        for reference_feature in raw_reference_features
    ]
    label_table["geom_type"] = [
        reference_geometry.geom_type if reference_geometry is not None else "missing"
        for reference_geometry in label_table["geometry"]
    ]
    label_table["uid"] = (
        label_table["dataset"].astype(str) + ":" + label_table["index"].astype(str)
    )
    for column in [
        "collection_date",
        "planting_date",
        "harvest_date",
        "export_end_date",
    ]:
        label_table[column] = pd.to_datetime(label_table[column], errors="coerce")
    label_table["label_norm"] = (
        label_table["label"].astype("string").str.strip().str.casefold()
    )
    original_label_table = label_table.copy()
    label_table = original_label_table.copy()
    for column in ["collection_date", "export_end_date"]:
        label_table[column] = pd.to_datetime(label_table[column], errors="coerce")
    label_table["is_crop"] = pd.to_numeric(label_table["is_crop"], errors="coerce")
    label_table["label_norm"] = (
        label_table["label"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )
    year_decisions = [
        cropharvest_source.resolve_observation_year(
            row.collection_date,
            row.export_end_date,
            observation_year_overrides.get(row.uid),
        )
        for row in label_table.itertuples()
    ]
    label_table["observation_year"] = pd.array(
        [decision["year"] for decision in year_decisions], dtype="Int64"
    )
    label_table["year_resolution_status"] = [
        decision["status"] for decision in year_decisions
    ]
    label_table["year_resolution_reason"] = [
        decision["reason"] for decision in year_decisions
    ]
    year_audit_path = run_folder / "labels/observation_year_audit.csv"
    year_audit_path.parent.mkdir(parents=True, exist_ok=True)
    label_table[
        [
            "uid",
            "collection_date",
            "export_end_date",
            "observation_year",
            "year_resolution_status",
            "year_resolution_reason",
        ]
    ].to_csv(year_audit_path, index=False)
    valid_year = (
        label_table["observation_year"]
        .between(minimum_reference_year, maximum_reference_year)
        .fillna(False)
    )
    label_table = label_table.loc[valid_year].copy()
    target_crop_aliases = {
        "potato": {
            "potato": "",
            "potatoes": "",
            "pomme de terre de consommation": "table potato (FR)",
            "pomme de terre féculière": "starch potato (FR)",
        },
        "carrot": {"carrot": "", "carrots": "", "carotte": "carrot (FR)"},
        "cabbage": {"cabbage": "", "chou": "cabbage (FR); FranceCrops CHU"},
    }
    review_label_aliases = {}
    alias_to_crop = {
        alias: crop
        for crop, variants in target_crop_aliases.items()
        for alias in variants
    }
    alias_notes = {
        alias: note
        for variants in target_crop_aliases.values()
        for alias, note in variants.items()
    }
    label_table["crop"] = label_table["label_norm"].map(alias_to_crop).astype("string")
    label_table["crop_note"] = label_table["label_norm"].map(alias_notes).fillna("")
    mixed_label_pattern = "\\bmix(?:ed|ture|ing)?\\b|\\binter[\\s-]?crop\\w*\\b|\\bm[ée]lange\\w*\\b|\\bassociation\\b|\\bpolyculture\\b|\\bassoc\\w*\\b|\\bwith other crops\\b"
    multi_crop_labels = {
        "café / cacao",
        "cauliflower and brocoli",
        "canola/rapeseed (with other crops)",
    }
    label_table["mixed_label"] = (
        label_table["label_norm"].str.contains(
            mixed_label_pattern, regex=True, na=False
        )
        | label_table["label_norm"].isin(multi_crop_labels)
    ).fillna(False)
    generic_crop_labels = {
        "annual crop",
        "vegetables",
        "vegetables (undiff)",
        "leafy or stem vegetable",
        "root",
        "root crops",
        "root, bulb or tuberous vegetable",
        "root/tuber crop with high starch or inulin content",
        "other",
        "other agriculture (undiff)",
        "market gardening",
        "autre légume ou fruit annuel",
        "légume sous abri",
    } | set(review_label_aliases)
    label_table["model_class"] = pd.Series(
        "unknown_crop", index=label_table.index, dtype="string"
    )
    label_table.loc[label_table["is_crop"].eq(0), "model_class"] = "non_crop"
    target_mask = label_table["is_crop"].eq(1) & label_table["crop"].notna()
    label_table.loc[target_mask, "model_class"] = label_table.loc[target_mask, "crop"]
    other_mask = (
        label_table["is_crop"].eq(1)
        & label_table["label_norm"].notna()
        & label_table["crop"].isna()
        & ~label_table["label_norm"].isin(generic_crop_labels)
    )
    label_table.loc[other_mask, "model_class"] = "other_crop"
    label_table["geom_wkb"] = [
        (
            reference_geometry.wkb_hex
            if reference_geometry is not None and (not reference_geometry.is_empty)
            else None
        )
        for reference_geometry in label_table.geometry
    ]
    identity = label_table["label_norm"].copy()
    identity.loc[label_table["is_crop"].eq(0)] = "non_crop"
    check = pd.DataFrame(
        {
            "geom": label_table["geom_wkb"],
            "season": label_table["export_end_date"],
            "identity": identity,
        },
        index=label_table.index,
    )
    valid = (
        check["geom"].notna()
        & check["season"].notna()
        & check["identity"].notna()
        & ~label_table["mixed_label"]
    )
    names_per_field = check.loc[valid].groupby(["geom", "season"])["identity"].nunique()
    conflict_keys = set(names_per_field[names_per_field > 1].index)
    label_table["season_conflict"] = [
        (geo, season) in conflict_keys
        for geo, season in zip(label_table["geom_wkb"], label_table["export_end_date"])
    ]
    keep = (
        ~label_table["mixed_label"]
        & ~label_table["season_conflict"]
        & label_table["model_class"].ne("unknown_crop")
        & label_table["export_end_date"].notna()
        & label_table["geom_wkb"].notna()
    )
    lab_clean = label_table.loc[keep].copy()
    target_labels = lab_clean.loc[lab_clean["crop"].isin(crop_order)].copy()
    target_labels["geom_type"] = target_labels["geometry"].apply(
        lambda reviewed_labels: getattr(reviewed_labels, "geom_type", None)
    )
    is_polygon = target_labels["geom_type"].isin(["Polygon", "MultiPolygon"])
    target_labels = target_labels.loc[is_polygon].copy()
    if country_boundaries_path.exists():
        with zipfile.ZipFile(country_boundaries_path) as archive:
            shape_bytes = io.BytesIO(archive.read("ne_50m_admin_0_countries.shp"))
            attribute_bytes = io.BytesIO(archive.read("ne_50m_admin_0_countries.dbf"))
            reader = shapefile.Reader(shp=shape_bytes, dbf=attribute_bytes)
            country_shapes = [
                (shape(record.shape.__geo_interface__), record.record["NAME"])
                for record in reader.iterShapeRecords()
            ]
        polygons = [geometry for geometry, _ in country_shapes]
        tree = STRtree(polygons)
        geometry_indices = {
            id(geometry): index for index, geometry in enumerate(polygons)
        }

    def polygon_index(candidate):
        """Support either spatial-index return format without guessing the country."""
        if isinstance(candidate, (int, np.integer)):
            return int(candidate)
        matched_index = geometry_indices.get(id(candidate))
        if matched_index is not None:
            return matched_index
        return next(
            (
                index
                for index, geometry in enumerate(polygons)
                if geometry.equals(candidate)
            )
        )

    def country_of(longitude, latitude):
        """Use the saved boundaries, retaining the original nearest-country fallback."""
        point = Point(float(longitude), float(latitude))
        for candidate in tree.query(point):
            country_index = polygon_index(candidate)
            if polygons[country_index].contains(point):
                return country_shapes[country_index][1]
        return country_shapes[polygon_index(tree.nearest(point))][1]

    if country_boundaries_path.exists():
        TERRITORY = {
            "reunion-france": "Réunion (FR)",
            "martinique-france": "Martinique (FR)",
        }
        target_labels["country"] = [
            TERRITORY.get(dataset) or country_of(lon, lat)
            for dataset, lon, lat in zip(
                target_labels.dataset, target_labels.lon, target_labels.lat
            )
        ]
    else:
        target_labels["country"] = target_labels["dataset"].astype("string")
    target_labels["cell_1deg"] = (
        np.floor(target_labels.lat).astype(int).astype(str)
        + ","
        + np.floor(target_labels.lon).astype(int).astype(str)
    )
    source_methods = {
        "jecam": ("in-situ field survey (JECAM sites)", "high"),
        "canada": ("AAFC ground-truth points (Annual Crop Inventory)", "high"),
        "ile-de-france": ("farmer parcel declarations (French RPG)", "high"),
        "reunion-france": ("farmer parcel declarations (French RPG)", "high"),
        "martinique-france": ("farmer parcel declarations (French RPG)", "high"),
        "kenya": ("field survey", "high"),
        "croplands": (
            "GFSAD / croplands reference samples (field + image interpretation)",
            "medium",
        ),
    }
    reviewed_labels = target_labels.copy()
    meters_per_degree = 111320.0
    reviewed_labels["area_ha"] = [
        (
            reference_geometry.area
            * meters_per_degree**2
            * np.cos(np.radians(lat))
            / 10000.0
            if kind == "Polygon"
            else np.nan
        )
        for reference_geometry, lat, kind in zip(
            reviewed_labels.geometry, reviewed_labels.lat, reviewed_labels.geom_type
        )
    ]
    reviewed_labels["invalid_geom"] = [
        kind == "Polygon" and (not reference_geometry.is_valid)
        for reference_geometry, kind in zip(
            reviewed_labels.geometry, reviewed_labels.geom_type
        )
    ]
    reviewed_labels["dup_in_source"] = reviewed_labels.duplicated(
        ["dataset", "geom_wkb"], keep=False
    )
    dup_source_count = label_table.groupby("geom_wkb").dataset.nunique()
    reviewed_labels["dup_across_sources"] = reviewed_labels.geom_wkb.map(
        dup_source_count
    ).gt(1)
    expected_group = {
        "potato": "root_tuber",
        "carrot": "vegetables_melons",
        "cabbage": "vegetables_melons",
    }
    reviewed_labels["group_mismatch"] = reviewed_labels.classification_label.ne(
        reviewed_labels.crop.map(expected_group)
    )
    other_season_count = (
        label_table.dropna(subset=["label_norm"])
        .groupby("geom_wkb")
        .label_norm.nunique()
    )
    reviewed_labels["relabelled_other_season"] = reviewed_labels.geom_wkb.map(
        other_season_count
    ).gt(1)
    reviewed_labels["placeholder_date"] = reviewed_labels.collection_date.dt.month.eq(
        1
    ) & reviewed_labels.collection_date.dt.day.eq(1)
    start = reviewed_labels.export_end_date - pd.DateOffset(months=12)
    inside = reviewed_labels.collection_date.between(
        start, reviewed_labels.export_end_date
    )
    same_year = reviewed_labels.collection_date.dt.year.eq(start.dt.year)
    reviewed_labels["season_class"] = np.select(
        [
            reviewed_labels.planting_date.notna(),
            inside & ~reviewed_labels.placeholder_date,
            reviewed_labels.placeholder_date & (inside | same_year),
        ],
        ["planting / harvest dated", "in-season observation", "season-year only"],
        default="inconsistent",
    )
    reviewed_labels["season_year"] = start.dt.year
    planting_days_by_source = reviewed_labels.groupby(
        "dataset"
    ).planting_date.transform(
        lambda series: series.dropna().dt.strftime("%m-%d").nunique()
    )
    reviewed_labels["planting_date_placeholder_suspected"] = (
        reviewed_labels.planting_date.notna() & planting_days_by_source.le(2)
    )
    reviewed_labels["source_method"] = reviewed_labels.dataset.map(
        lambda x: source_methods.get(x, ("unknown", "unknown"))[0]
    )
    reviewed_labels["source_grade"] = reviewed_labels.dataset.map(
        lambda x: source_methods.get(x, ("unknown", "unknown"))[1]
    )
    reviewed_labels["reliable"] = (
        reviewed_labels.source_grade.eq("high")
        & ~reviewed_labels.invalid_geom
        & ~reviewed_labels.dup_across_sources
        & reviewed_labels.season_class.ne("inconsistent")
    )
    checks = [
        ("Source not high grade", ~reviewed_labels.source_grade.eq("high")),
        ("Invalid polygon", reviewed_labels.invalid_geom),
        ("Duplicate across sources", reviewed_labels.dup_across_sources),
        ("Season inconsistent", reviewed_labels.season_class.eq("inconsistent")),
    ]
    reviewed_labels["fail_reason"] = "passed"
    for reason, mask in reversed(checks):
        reviewed_labels.loc[mask.fillna(False), "fail_reason"] = reason
    assert reviewed_labels.reliable.equals(
        reviewed_labels.fail_reason.eq("passed")
    ), "Check screening and exclusion reasons"
    original_columns = [
        column
        for column in reviewed_labels.columns
        if column
        not in {
            "geometry",
            "geom_wkb",
            "source_method",
            "source_grade",
            "planting_date_placeholder_suspected",
            "fail_reason",
        }
    ]
    to_export = reviewed_labels[original_columns]
    screened_labels_output.parent.mkdir(parents=True, exist_ok=True)
    to_export.to_csv(screened_labels_output, index=False)
    summary = (
        reviewed_labels.groupby("crop")
        .agg(
            candidates=("uid", "size"),
            passed=("reliable", "sum"),
        )
        .reindex(crop_order, fill_value=0)
    )
    summary["excluded"] = summary["candidates"] - summary["passed"]
    return {
        "reviewed_labels": reviewed_labels,
        "label_table": label_table,
        "crop_order": crop_order,
        "observation_year_audit": year_audit_path,
        "screened_labels_output": screened_labels_output,
        "summary": summary,
    }
