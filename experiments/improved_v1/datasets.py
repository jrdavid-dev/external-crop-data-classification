"""Development-data variants: the groupmate's parcels, yours, or both.

Each development parcel group gets one origin:

- ``shared``: in the groupmate's package and in your own collection,
- ``groupmate_only``: only in the groupmate's package,
- ``yours_only``: only in your own collection (the extra parcels you sent him).

A variant keeps the development (train/validation) rows whose origin it lists. Test rows
(JECAM and the supplements) are never filtered, so every variant is scored on the same
external parcels. Locked partitions, labels and quality rules are unchanged; a variant
only drops whole development parcel groups.

Origins are saved once as a sealed ``origins`` stage, so later runs do not depend on the
groupmate's package or your pre-merge backup staying where they are now.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from crop_pipeline.common import save_json
from crop_pipeline.windows import seasons

ORIGINS = ("shared", "groupmate_only", "yours_only")
VARIANTS = {
    "groupmate": ("shared", "groupmate_only"),
    "yours": ("shared", "yours_only"),
    "combined": ORIGINS,
}
DEVELOPMENT = ("train", "validation")


def classify_origins(groupmate_groups, your_groups):
    """{partition_group: origin} for every group in either collection."""
    groupmate_groups, your_groups = set(groupmate_groups), set(your_groups)
    return {group: "shared" if group in groupmate_groups and group in your_groups
            else "groupmate_only" if group in groupmate_groups else "yours_only"
            for group in sorted(groupmate_groups | your_groups)}


def groupmate_groups(package):
    """Parcel groups in every bundle listed by a share package's ``package.json``."""
    package = Path(package)
    listing = json.loads((package / "package.json").read_text())
    groups = set()
    for bundle in listing["bundles"]:
        record = json.loads((package / bundle["path"] / "bundle.json").read_text())
        groups.update(record["selection"]["groups"])
    return groups


def your_groups(campaign_root):
    """Parcel groups in every batch selection under a campaign folder (e.g. your pre-merge backup)."""
    groups = set()
    for path in sorted(Path(campaign_root).glob("*/rounds/*/*/selection.json")):
        groups.update(json.loads(path.read_text()).get("groups", []))
    return groups


def variant_mask(rows, origins, variant):
    """Boolean mask over rows: every test row, plus development rows whose origin the variant keeps."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown dataset variant {variant!r}; choose one of {sorted(VARIANTS)}")
    keep = set(VARIANTS[variant])
    development = {r["partition_group"] for r in rows if r["partition"] in DEVELOPMENT}
    unknown = sorted(development - set(origins))
    if unknown:
        raise ValueError(f"{len(unknown)} development parcel groups have no recorded origin "
                         f"(first: {unknown[0]}); rebuild the origins stage")
    mask = np.asarray([r["partition"] not in DEVELOPMENT or origins[r["partition_group"]] in keep for r in rows])
    crops = {r["crop"] for r, m in zip(rows, mask) if m and r["partition"] == "train"}
    missing = sorted({r["crop"] for r in rows if r["partition"] in DEVELOPMENT} - crops)
    if missing:
        raise ValueError(f"Variant {variant!r} has no training parcels for {missing}")
    return mask


def subset(rows, arrays, mask):
    return [r for r, m in zip(rows, mask) if m], {key: value[mask] for key, value in arrays.items()}


def summary(rows, origins):
    """Development parcel counts per (variant, partition, crop) for the rows given."""
    parcels = {r["partition_group"]: (r["partition"], r["crop"]) for r in rows if r["partition"] in DEVELOPMENT}
    counts = Counter()
    for group, (partition, crop) in parcels.items():
        for variant, kept in VARIANTS.items():
            if origins.get(group) in kept:
                counts[variant, partition, crop] += 1
    return [{"variant": v, "partition": p, "crop": c, "parcels": n} for (v, p, c), n in sorted(counts.items())]


def build_origins(repository, groupmate_package, your_campaign, progress=print):
    """Seal and publish the origins stage (``current_origins.json``)."""
    from experiments.improved_v1 import workflow

    workflow._start(repository, "origins")
    package, campaign = Path(groupmate_package).resolve(), Path(your_campaign).resolve()
    origins = classify_origins(groupmate_groups(package), your_groups(campaign))
    run = workflow._new_run(repository, "origins")
    save_json(run / "origins.json", origins)
    counts = dict(Counter(origins.values()))
    receipt = workflow.seal(run, "origins", groupmate_package=str(package),
                            groupmate_package_sha256=seasons.file_sha256(package / "package.json"),
                            your_campaign=str(campaign), counts=counts)
    if progress:
        progress(f"Origins: {counts}")
    return workflow._publish(repository, "origins", receipt)


def load_origins(repository, origins_handoff=None):
    from experiments.improved_v1 import workflow

    receipt, _ = workflow._input(repository, origins_handoff, "origins")
    return receipt, json.loads((receipt.parent / "origins.json").read_text())
