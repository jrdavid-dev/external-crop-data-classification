"""Build the external test parcel reference audit that Testing 04 reads.

Testing 04 joins each scored test parcel to its raw reference record, mainly to recover the
observation date. The original audit (``test_audit_2026-11-19/parcel_audit.json``) was made on
another machine by a script that is not in the repository. This tool rebuilds an equivalent audit
for a local testing run:

- CropHarvest parcels (JECAM and the non-JECAM supplements): the matching ``labels.geojson``
  record (``<dataset>:<index>``), whose ``collection_date`` is the observation date. The labels
  file must be the revision named in each parcel's label evidence.
- California DWR: ``MAIN_CROP_DATE`` from the label evidence ("main crop date day N").

Label evidence is read from the quality audit behind the test features, not from the feature rows.
- Denmark Marker: the declaration only; it has no observation date.

One row per model cohort and test parcel, with the parcel's pixel count and reference crop.
It is written to ``pipeline_outputs/testing/<run>_reference_audit/parcel_audit.json``.

    .venv/bin/python tools/build_test_reference_audit.py pipeline_outputs/testing/run_<id> \\
        data/external/cropharvest/labels.geojson
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_pipeline.common import save_json  # noqa: E402
from crop_pipeline.training import notebook_workflow  # noqa: E402
from crop_pipeline.training.external_report import MODELS  # noqa: E402

DWR_DAY = re.compile(r"main crop date day (\d+)")
REVISION = re.compile(r"labels\.geojson sha256:([0-9a-f]{64})")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_matches(row, evidence, labels, labels_sha):
    source = row["partition_group"].split("|", 1)[0]
    if source == "cropharvest":
        revision = REVISION.search(evidence)
        if not revision or revision.group(1) != labels_sha:
            raise ValueError(f"{row['partition_group']}: labels.geojson revision differs from the label evidence")
        dataset, index = row["partition_group"].rsplit("|", 1)[1].rsplit(":", 1)
        record = labels.get((dataset, int(index)))
        if record is None:
            raise ValueError(f"{row['partition_group']}: no matching labels.geojson record")
        return [record]
    if source == "california_dwr":
        day = DWR_DAY.search(evidence)
        return [{"MAIN_CROP_DATE": int(day.group(1)), "label_evidence": evidence}] if day else []
    return [{"label_evidence": evidence}]


def build(testing_run, labels_path):
    testing_run = Path(testing_run)
    score_stage, _ = notebook_workflow._verify(testing_run / "scores.json", "scores")
    test = json.loads((score_stage.parent / "features_test.json").read_text())
    labels_sha = sha256(labels_path)
    labels = {(f["properties"]["dataset"], f["properties"]["index"]): f["properties"]
              for f in json.loads(Path(labels_path).read_text())["features"]}
    # Aligned-window rows carry a generic evidence note; the quality audit keeps each parcel's own.
    quality = json.loads(Path(test["quality"]).read_text())
    evidence = {}
    for region in quality["regions"]:
        p = region["properties"]
        if p.get("partition") == "test":
            evidence.setdefault(p["partition_group"], p.get("label_evidence", ""))
    rows = []
    for cohort in MODELS:
        pixels = Counter(r["partition_group"] for r in test["rows"][cohort])
        first = {}
        for r in test["rows"][cohort]:
            first.setdefault(r["partition_group"], r)
        for group, count in sorted(pixels.items()):
            row = first[group]
            rows.append({"cohort": cohort, "group": group, "crop": row["crop"], "pixels": count,
                         "test_cohort": row["test_cohort"], "source_year": int(row["source_year"]),
                         "raw_matches": raw_matches(row, evidence[group], labels, labels_sha)})
    output = testing_run.parent / f"{testing_run.name}_reference_audit"
    output.mkdir(exist_ok=False)
    save_json(output / "parcel_audit.json", rows)
    save_json(output / "inputs.json", {"testing_run": str(testing_run.resolve()), "labels_sha256": labels_sha,
                                       "features_test": str((score_stage.parent / "features_test.json").resolve()),
                                       "rows": len(rows), "built_by": "tools/build_test_reference_audit.py"})
    return output / "parcel_audit.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("testing_run", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    print("Audit:", build(args.testing_run, args.labels))


if __name__ == "__main__":
    main()
