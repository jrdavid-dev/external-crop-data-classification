"""Audit the saved JECAM adaptation split and reproduce its parcel predictions."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations, product
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import transform, union_all
from shapely.geometry import shape
from sklearn.base import clone

from crop_pipeline.paoay.predictions import load_classifier
from crop_pipeline.sources import adaptive, sampling, test_cohort
from crop_pipeline.training import exploratory, model_comparisons, source_holdout

read_json = model_comparisons.read_json
SITE_CRS = {"Madagascar": "EPSG:32738", "Senegal": "EPSG:32628"}
COHORT = "aligned_matched"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_separation(split, footprints, adapted, held):
    """Audit original boundaries and year-independent identities for the scored split."""
    partitions = ("adaptation", "held_out")
    groups = {name: {group for group, row in split.items() if row["partition"] == name}
              for name in partitions}
    for name, rows in zip(partitions, (adapted, held), strict=True):
        labels = {row["partition_group"]: row["crop"] for row in rows}
        if labels != {group: split[group]["crop"] for group in groups[name]}:
            raise ValueError(f"{name} feature membership differs from the saved split")
    if {sampling.pixel_location(row) for row in adapted} & {sampling.pixel_location(row) for row in held}:
        raise ValueError("Adaptation and holdout share a pixel location")

    shapes, physical_keys = defaultdict(list), defaultdict(set)
    for item in footprints:
        props = item["properties"]
        group = props["partition_group"]
        if group not in split:
            continue
        geometry = shape(item["geometry"])
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Missing a valid original field boundary for {group}")
        if props["crop"] != split[group]["crop"] or not props.get("parcel_id"):
            raise ValueError(f"Missing or inconsistent reference identity for {group}")
        shapes[group].append(geometry)
        physical_keys[split[group]["partition"]].add(adaptive.physical_key(test_cohort.TEST, props))
    if shapes.keys() != split.keys():
        raise ValueError("Original footprints do not cover every field in the split")
    if physical_keys["adaptation"] & physical_keys["held_out"]:
        raise ValueError("Adaptation and holdout share a source parcel ID across observations")
    geometry = {group: union_all(parts) for group, parts in shapes.items()}
    if any(geometry[a].intersects(geometry[b]) for a, b in product(*groups.values())):
        raise ValueError("An original field boundary intersects across the split")

    block_partitions = defaultdict(set)
    for row in split.values():
        block_partitions[row["site"], row["block"]].add(row["partition"])
    if any(not block or len(assignments) != 1 for (_, block), assignments in block_partitions.items()):
        raise ValueError("A geographic block is missing or split across partitions")
    if {row["site"] for row in split.values()} != SITE_CRS.keys():
        raise ValueError("Unexpected JECAM adaptation sites")
    for site, crs in SITE_CRS.items():
        site_groups = [group for group, row in split.items() if row["site"] == site]
        project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        projected = {group: transform(geometry[group], project, interleaved=False) for group in site_groups}
        neighbors = {group: set() for group in site_groups}
        for a, b in combinations(site_groups, 2):
            if projected[a].centroid.distance(projected[b].centroid) <= 500:
                neighbors[a].add(b)
                neighbors[b].add(a)
        remaining = set(site_groups)
        while remaining:
            group = next(iter(remaining))
            component = {group}
            pending = [group]
            while pending:
                unseen = neighbors[pending.pop()] - component
                pending.extend(unseen)
                component.update(unseen)
            if component != {g for g in site_groups if split[g]["block"] == split[group]["block"]}:
                raise ValueError(f"{site} archived blocks differ from the 500 m centroid clusters")
            remaining -= component
        distance = min(projected[a].distance(projected[b])
                       for a, b in product(*(g & projected.keys() for g in groups.values())))
        print(f"{site}: {len(set(g['block'] for g in split.values() if g['site'] == site))} "
              f"blocks; closest original boundaries {distance:.1f} m")


def verify_weights(rows, weights, development_pixels, multiplier):
    """Check each pixel against its intended field and crop share after scaling."""
    fields = defaultdict(set)
    confidences = Counter()
    for row in rows:
        group, crop = row["partition_group"], row["crop"]
        fields[crop].add(group)
        confidences[group] += model_comparisons.WEIGHTS[row["label_status"]]
    expected = np.array([
        (1 if i < development_pixels else multiplier) * model_comparisons.WEIGHTS[row["label_status"]]
        / confidences[row["partition_group"]] / len(fields[row["crop"]])
        for i, row in enumerate(rows)
    ])
    np.testing.assert_allclose(weights, expected / expected.mean(), rtol=1e-12, atol=0)


def verify(run_dir, quality_handoff):
    metadata = read_json(run_dir / "jecam_adaptation.json")
    manifest_path = Path(metadata["frozen_model"])
    manifest = read_json(manifest_path)
    forest, recipe = load_classifier(manifest_path)
    columns = recipe["feature_columns"]
    development = read_json(Path(metadata["development_features"]))
    window = development["window_sha256"]
    dev = source_holdout.checked_rows(development["rows"][COHORT], False)
    del development
    assignments = {row["partition_group"]: row["partition"] for row in dev}
    if assignments != manifest["development_split"]:
        raise ValueError("Development features differ from the frozen partition")
    train = [row for row in dev if row["partition"] == "train"]
    del dev
    test = read_json(Path(metadata["test_features"]))
    if test["window_sha256"] != window:
        raise ValueError("Development and JECAM features come from different windows snapshots")
    jecam = [row for row in source_holdout.checked_rows(test["rows"][COHORT], True)
             if row.get("test_cohort") == "jecam"]
    del test
    records = read_csv(run_dir / "jecam_site_split.csv")
    split = {row["partition_group"]: row for row in records}
    if (len(split) != len(records) or split.keys() != {row["partition_group"] for row in jecam}
            or split.keys() & assignments.keys()):
        raise ValueError("The saved split does not uniquely partition the matched JECAM fields")
    counts = Counter(row["partition"] for row in records)
    if counts != {"adaptation": 23, "held_out": 88} or sum(p == "train" for p in assignments.values()) != 472:
        raise ValueError("Unexpected development, adaptation, or holdout field counts")
    adapted = [row for row in jecam if split[row["partition_group"]]["partition"] == "adaptation"]
    held = [row for row in jecam if split[row["partition_group"]]["partition"] == "held_out"]
    sources = [source for source in read_json(quality_handoff)["sources"]
               if test_cohort.source_role(source["source_name"]) == test_cohort.TEST]
    verify_separation(split, test_cohort.source_footprints(sources), adapted, held)

    rows = train + adapted
    matrix = exploratory.feature_matrix(rows, columns)
    labels = [row["crop"] for row in rows]
    base_weights = exploratory.normalized_training_weights(rows, model_comparisons.WEIGHTS)
    for multiplier in range(5):
        model = forest
        name = "frozen"
        if multiplier:
            name = f"{multiplier}x"
            weights = base_weights.copy()
            weights[len(train):] *= multiplier
            weights *= len(weights) / weights.sum()
            verify_weights(rows, weights, len(train), multiplier)
            model = clone(forest).fit(matrix, labels, sample_weight=weights)

        score, scored = source_holdout.parcel_metrics(model, held, columns)
        saved = read_csv(run_dir / f"jecam_predictions_{name}.csv")
        model_comparisons.require_shared_parcels(saved, scored)
        observed = {row["partition_group"]: row["prediction"] for row in scored}
        if (any(observed[row["partition_group"]] != row["prediction"] for row in saved)
                or not np.isclose(score["macro_f1"], metadata["metrics"][name]["macro_f1"], rtol=0, atol=1e-12)):
            raise ValueError(f"{name} did not reproduce the saved holdout predictions")
        print(f"{name}: {len(scored)} predictions reproduced; macro-F1 {score['macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptation_run", type=Path)
    parser.add_argument("quality_handoff", type=Path)
    args = parser.parse_args()
    verify(args.adaptation_run, args.quality_handoff)


if __name__ == "__main__":
    main()
