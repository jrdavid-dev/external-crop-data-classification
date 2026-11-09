"""Actual-date summaries on the exact saved pixel selection and imagery grid."""

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from crop_pipeline.imagery import sampling_mask
from crop_pipeline.imagery.quality import save_json, write_table

SUMMARY_RECIPE = "selected_pixel_grid_v3"
INDEX_BANDS = (("NDVI", "B4"), ("NDRE", "B5"), ("NDMI", "B11"))
TIMELINE_COLUMNS = [
    "region_id",
    "partition_group",
    "crop",
    "date",
    "source_kind",
    "valid_pixels",
    "valid_fraction",
    "eligible",
    "NDVI",
    "NDRE",
    "NDMI",
]


def summary_contract(plan, selection, minimum_fraction):
    """Timing recipe and mask changes invalidate daily summaries, not imagery requests."""
    return {
        "recipe": SUMMARY_RECIPE,
        "settings": plan["settings"],
        "mask_sha256": selection["mask_sha256"],
        "selected_pixels": selection["selected_pixels"],
        "minimum_fraction": minimum_fraction,
        "acquisitions": plan["acquisitions"],
    }


def checkpoint_path(destination, region_id, contract):
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()[:20]
    return (
        Path(destination)
        / "observed_checkpoints"
        / f"{region_id}_{SUMMARY_RECIPE}_{fingerprint}.json"
    )


def retrieve_observed_timelines(
    plans, mask_acquisition, ee, destination, minimum_fraction=0.8
):
    """Checkpoint corrected summaries; a failed region cannot masquerade as completion."""
    if not 0 < minimum_fraction <= 1:
        raise ValueError("Daily-summary minimum valid fraction must be in (0, 1].")
    destination = Path(destination)
    all_rows, failures = [], []
    for plan in plans:
        try:
            all_rows.extend(
                retrieve_region(
                    plan, mask_acquisition, ee, destination, minimum_fraction
                )
            )
        except Exception as error:
            failures.append({"region_id": plan["fragment_id"], "error": str(error)})
            save_aggregates(destination, all_rows, failures)
            raise RuntimeError(
                f"Daily summaries failed for {plan['fragment_id']}: {error}. Correct the error and resume collection."
            ) from error
    save_aggregates(destination, all_rows, failures)
    return all_rows


def retrieve_region(
    plan, mask_acquisition, earth_engine, destination, minimum_fraction
):
    selection = sampling_mask.load_selection(plan)
    contract = summary_contract(plan, selection, minimum_fraction)
    path = checkpoint_path(destination, plan["fragment_id"], contract)
    if path.is_file():
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("contract") != contract:
            raise ValueError(
                "Daily-summary checkpoint contract differs from its request."
            )
        validate_cached_rows(checkpoint["rows"], plan, minimum_fraction)
        print(plan["fragment_id"], "masked daily summaries reused")
        return checkpoint["rows"]
    features = daily_features(plan, selection, mask_acquisition, earth_engine)
    payload = earth_engine.FeatureCollection(features).getInfo()
    rows = summary_rows(payload["features"], plan, minimum_fraction)
    save_json(
        path,
        {
            "settings": plan["settings"],
            "contract": contract,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        },
    )
    print(plan["fragment_id"], len(rows), "masked daily summaries")
    return rows


def matched_dates(plan):
    return sorted(
        {
            record["date"]
            for record in plan["acquisitions"]
            if record.get("cloud_probability_id")
        }
    )


def daily_features(plan, selection, mask_acquisition, earth_engine):
    grid = selection["grid"]
    selected = sampling_mask.selection_image(
        earth_engine, grid, selection["selected_ids"]
    )
    footprint = sampling_mask.grid_footprint(earth_engine, grid)
    # Count and median use pixel centres, with no fractional parcel-edge weighting.
    reducer = (
        earth_engine.Reducer.median()
        .unweighted()
        .combine(earth_engine.Reducer.count(), sharedInputs=True)
    )
    return [
        daily_feature(
            plan,
            observation_date,
            mask_acquisition,
            earth_engine,
            selected,
            footprint,
            reducer,
        )
        for observation_date in matched_dates(plan)
    ]


def daily_feature(
    plan, observation_date, mask_acquisition, earth_engine, selected, footprint, reducer
):
    granules = sorted(
        [
            record
            for record in plan["acquisitions"]
            if record.get("cloud_probability_id") and record["date"] == observation_date
        ],
        key=lambda record: record["image_index"],
    )
    images = [
        mask_acquisition(
            earth_engine.Image(record["image_id"]),
            earth_engine.Image(record["cloud_probability_id"]),
        )
        for record in granules
    ]
    daily = earth_engine.ImageCollection.fromImages(images).mosaic()
    image = daily_index_image(daily, earth_engine).updateMask(selected)
    grid = plan["settings"]["grid"]
    statistics = image.reduceRegion(
        reducer=reducer,
        geometry=footprint,
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=1000000,
    )
    return earth_engine.Feature(None, statistics.set("date", observation_date))


def daily_index_image(daily, earth_engine):
    return earth_engine.Image.cat(
        [index_image(daily, name, other) for name, other in INDEX_BANDS]
    )


def index_image(daily, name, other_band):
    left, right = daily.select("B8"), daily.select(other_band)
    denominator = left.add(right)
    return (
        left.subtract(right)
        .divide(denominator)
        .updateMask(denominator.neq(0))
        .rename(name)
    )


def validated_count(value, selected_pixels):
    number = float(value or 0)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(
            "Daily-summary clear-pixel count must be a finite nonnegative integer."
        )
    if number > selected_pixels:
        raise ValueError("Region reduction included more pixels than the sampling mask")
    return int(number)


def row_from_statistics(values, properties, region_id, minimum_fraction):
    selected_pixels = int(properties["selected_pixels"])
    counts = [
        validated_count(values.get(name + "_count"), selected_pixels)
        for name, _ in INDEX_BANDS
    ]
    count = min(counts)
    fraction = count / selected_pixels
    return {
        "region_id": region_id,
        "partition_group": properties["partition_group"],
        "crop": properties["crop"],
        "date": values["date"],
        "source_kind": "observed_daily",
        "valid_pixels": count,
        "valid_fraction": fraction,
        "eligible": fraction >= minimum_fraction,
        **{name: values.get(name + "_median") for name, _ in INDEX_BANDS},
    }


def summary_rows(features, plan, minimum_fraction):
    values = [feature["properties"] for feature in features]
    values.sort(key=lambda record: record["date"])
    if [record["date"] for record in values] != matched_dates(plan):
        raise ValueError(
            "Daily-summary dates differ from the cloud-matched acquisition dates."
        )
    return [
        row_from_statistics(
            record,
            plan["fragment"]["properties"],
            plan["fragment_id"],
            minimum_fraction,
        )
        for record in values
    ]


def validate_cached_rows(rows, plan, minimum_fraction):
    if [row["date"] for row in rows] != matched_dates(plan):
        raise ValueError("Cached daily-summary dates differ from the request.")
    properties = plan["fragment"]["properties"]
    for row in rows:
        count = validated_count(row["valid_pixels"], properties["selected_pixels"])
        expected_fraction = count / properties["selected_pixels"]
        if (
            row["region_id"] != plan["fragment_id"]
            or row["partition_group"] != properties["partition_group"]
            or row["crop"] != properties["crop"]
        ):
            raise ValueError("Cached daily-summary identity differs from the request.")
        if row["valid_fraction"] != expected_fraction or row["eligible"] != (
            expected_fraction >= minimum_fraction
        ):
            raise ValueError(
                "Cached daily-summary coverage differs from the configured quality rule."
            )


def preserve_aggregate(destination, path):
    """Keep content-addressed snapshots before replacing convenience aggregate files."""
    if not path.is_file():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    backup = destination / "summary_history" / digest / path.name
    if not backup.is_file():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def save_aggregates(destination, rows, failures):
    destination = Path(destination)
    for filename in (
        "observed_timelines.json",
        "observed_timelines.csv",
        "observed_errors.json",
    ):
        preserve_aggregate(destination, destination / filename)
    save_json(destination / "observed_timelines.json", rows)
    save_json(destination / "observed_errors.json", failures)
    write_table(destination / "observed_timelines.csv", rows, TIMELINE_COLUMNS)
