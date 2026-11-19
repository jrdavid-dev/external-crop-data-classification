"""Prepare separate test cohorts from California DWR and Denmark Marker."""
import json
from pathlib import Path

import requests
from shapely.geometry import mapping, shape

from crop_pipeline.common import save_json
from crop_pipeline.sources import adaptive, cropharvest, test_cohort
from crop_pipeline.windows.seasons import file_sha256

CALIFORNIA = "california_dwr"
DWR_QUERY = (
    "https://utility.arcgis.com/usrsvcs/servers/d94e891e00364e49a2ed9e9e2e27837d"
    "/rest/services/Planning/i15_Crop_Mapping_2023/MapServer/0/query"
)
DWR_WHERE = "MAIN_CROP='T6' AND MULTIUSE='S' AND MAIN_CROP_DATE BETWEEN 121 AND 243"
SPECS = {
    CALIFORNIA: ("carrot", "california_dwr_carrot_2023_v1"),
    "denmark_marker": ("cabbage", "denmark_marker_cabbage_2023_v1"),
}


def policy_for(source):
    _, cohort = SPECS[source]
    return {**adaptive.DEFAULT_POLICY, "test_source": source, "cohort": cohort}


def load_california(path, *, download_if_missing=True):
    """Fetch the bounded DWR query and reuse its saved reference snapshot."""
    path = Path(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        response = requests.get(DWR_QUERY, params={
            "where": DWR_WHERE, "outFields": "UniqueID,MAIN_CROP,MAIN_CROP_DATE,MULTIUSE,COUNTY",
            "outSR": 4326, "orderByFields": "OBJECTID", "f": "geojson",
        }, timeout=60)
        response.raise_for_status()
        payload = response.json()
        _california_features(payload)
        save_json(path, payload)
    return _california_features(json.loads(path.read_text()))


def _california_features(payload):
    if payload.get("type") != "FeatureCollection" or payload.get("exceededTransferLimit"):
        raise ValueError("California reference query failed or returned a partial result")
    features = {}
    for item in payload["features"]:
        p = item["properties"]
        day = p.get("MAIN_CROP_DATE")
        if (p.get("MAIN_CROP") != "T6" or p.get("MULTIUSE") != "S"
                or not isinstance(day, (int, float)) or not 121 <= day <= 243):
            continue
        if not p.get("UniqueID") or not item.get("geometry"):
            continue
        geom = shape(item["geometry"])
        if geom.is_empty or not geom.is_valid or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        identifier = str(p["UniqueID"])
        if identifier in features:
            raise ValueError("Duplicate California field identifier")
        group = f"{CALIFORNIA}|US|2023|{identifier}"
        longitude, latitude = geom.centroid.coords[0]
        features[identifier] = {"type": "Feature", "geometry": mapping(geom), "properties": {
            "fragment_id": group, "partition_group": group, "parcel_id": identifier,
            "crop": "carrot", "country": "US", "year": 2023, "source_split": "train",
            "source_crs": cropharvest.utm_crs(longitude, latitude), "label_status": "reported",
            "label_evidence": f"DWR 2023 single crop, T6, main crop date day {day}",
        }}
    if not features:
        raise ValueError("No valid California carrot polygons")
    return list(features.values())


def _reserved_test_footprints(report):
    """Read registered footprints and prepared plans, including pending JECAM."""
    policies = (test_cohort.TEST_POLICY, test_cohort.CABBAGE_SUPPLEMENT_POLICY,
                test_cohort.CARROT_POTATO_SUPPLEMENT_POLICY)
    for policy in policies:
        campaign = adaptive.root(report, policy)
        for batch in sorted(campaign.glob("rounds/*/*")):
            receipt, context = batch / "completed.json", batch / "context.json"
            if receipt.is_file():
                from crop_pipeline.sources import pilot_sources

                paths = []
                for record in json.loads(receipt.read_text())["handoffs"]:
                    path = Path(record["path"])
                    if file_sha256(path) != record["sha256"]:
                        raise ValueError(f"Test handoff changed: {path}")
                    paths.append(path)
                yield from test_cohort.source_footprints(pilot_sources.load_sources(paths))
            if context.is_file():
                plan_path = json.loads(context.read_text()).get("plan_path")
                if plan_path:
                    plan = json.loads(Path(plan_path).read_text())
                    for inputs in plan["years"].values():
                        footprint = Path(inputs["field_folder"]) / "source_footprints.geojson"
                        yield from json.loads(footprint.read_text())["features"]


def screen(report, features):
    development = test_cohort.completed_sources(report)
    occupied = [*test_cohort.source_footprints(development), *_reserved_test_footprints(report)]
    return test_cohort.exclude_overlap(features, occupied)


def prepare(source, report, archive, features, references, *, limit=50, extra_batches=0):
    crop, cohort = SPECS[source]
    candidates, excluded = screen(report, features)
    if not candidates:
        raise ValueError("No supplemental parcels passed the overlap screen")
    print(f"{cohort}: {len(candidates)} eligible references, {len(excluded)} overlaps excluded")
    return adaptive.prepare_round_batches(
        source, candidates, archive, report,
        {"cohort": cohort, "references_sha256": file_sha256(references),
         "excluded_overlap_groups": sorted(excluded)},
        policy=policy_for(source), parcels_per_crop=limit,
        extra_batches=extra_batches, candidate_crops=(crop,),
    )
