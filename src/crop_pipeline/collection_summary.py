"""Local collection inventory; never select fields, alter jobs, or score imagery."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from crop_pipeline.output_layout import SOURCES, refresh

SOURCE_ROLES = {
    "sen4agrinet": "development: mostly potato, some carrot",
    "francecrops": "development: potato and cabbage",
    "eurocrops": "development: additional carrot",
    "cropharvest": "external test: eligible JECAM carrot, cabbage and potato",
}


def inspect_collection(repository):
    """Read saved metadata, including incomplete batches; no network or mutations."""
    root = Path(repository).resolve()
    inputs, issues = {}, []

    def read(path, required=False):
        path = Path(path)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            if required:
                issues.append({"path": str(path), "reason": "missing_metadata"})
            return {}
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        try:
            return json.loads(raw)
        except ValueError:
            issues.append({"path": str(path), "reason": "invalid_json"})
            return {}

    # Settings locate data, but their contents (which may contain credentials) are
    # never copied into the report or its input list.
    settings_path = root / "settings.local.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    report_root = (root / Path(settings.get("report_root", "sen4agrinet_work/inventory")).expanduser()).resolve()
    campaigns = report_root.parent / "adaptive_training"
    batches, members, policies = [], {}, []
    for campaign in sorted(campaigns.glob("*")):
        if not campaign.is_dir():
            continue
        policy = read(campaign / "policy.json")
        if policy:
            policies.append({"campaign": campaign.name, "policy": policy,
                             "feedback": read(campaign / "feedback.json")})
        for folder in sorted((campaign / "rounds").glob("*/*")):
            if not folder.is_dir():
                continue
            selection = read(folder / "selection.json")
            context = read(folder / "context.json")
            completed = read(folder / "completed.json")
            state = read(folder / "state.json")
            source = selection.get("source") or context.get("source") or completed.get("source")
            if source not in SOURCES:
                continue
            status = state.get("status", "not_recorded")
            fields = []
            plan = read(context["plan_path"], required=True) if context.get("plan_path") else {}
            fields.extend((year, info["field_folder"]) for year, info in plan.get("years", {}).items()
                          if info.get("field_folder"))
            registered_tiles = 0
            for handoff in completed.get("handoffs", []):
                manifest = read(handoff["path"], required=True)
                path = str((root / handoff["path"]).resolve())
                if manifest and inputs.get(path) != handoff.get("sha256"):
                    issues.append({"path": path, "reason": "handoff_checksum_mismatch"})
                    continue
                if manifest:
                    fields.append((str(manifest["year"]), manifest["field_folder"]))
                    registered_tiles += len(manifest.get("annual_tif_index", {}))
            wanted = set(selection.get("groups", []))
            found = set()
            for year, field_folder in sorted(set(fields)):
                footprints = read(Path(field_folder) / "source_footprints.geojson", required=True)
                for feature in footprints.get("features", []):
                    props = feature["properties"]
                    group = props["partition_group"]
                    # Imported completed batches may have no local selection file.
                    if selection and group not in wanted:
                        continue
                    found.add(group)
                    partition = selection.get("assignments", {}).get(group, "unavailable")
                    if source == "cropharvest":
                        partition = "test"
                    role = ("external_test" if source == "cropharvest" else
                            "development" if partition in {"train", "validation"} else
                            "reserved_development" if partition == "test" else "unresolved")
                    key = (campaign.name, source, group, year)
                    row = {"campaign": campaign.name, "source": source, "group": group,
                           "year": year, "crop": props.get("crop", "unavailable"),
                           "country_or_source_context": props.get("country", "unavailable"),
                           "parcel_id": str(props.get("parcel_id", group)),
                           "partition": partition, "role": role,
                           "label_status": props.get("label_status", "unavailable"),
                           "label_evidence": props.get("label_evidence", "unavailable"),
                           "batch": str(folder.relative_to(campaigns)), "collection_status": status}
                    if key in members and members[key] != row:
                        issues.append({"path": str(folder), "reason": "repeated_reference_record", "group": group})
                    else:
                        members[key] = row
            missing = wanted - found
            if missing:
                issues.append({"path": str(folder), "reason": "selected_groups_without_reference_metadata",
                               "groups": sorted(missing)})
            batches.append({"campaign": campaign.name, "source": source,
                            "batch": str(folder.relative_to(campaigns)), "status": status,
                            "selected_groups": len(wanted) if selection else None,
                            "profiled_groups": len(found), "missing_groups": len(missing),
                            "registered_annual_tiles": registered_tiles,
                            "plan_path": context.get("plan_path"),
                            "selection_path": str(folder / "selection.json") if selection else None,
                            "completed_path": str(folder / "completed.json") if completed else None})
    counts = defaultdict(set)
    for row in members.values():
        key = (row["source"], row["crop"], row["partition"], row["role"])
        counts[key].add((row["campaign"], row["group"]))
    profile = [{"source": source, "crop": crop, "partition": partition, "role": role,
                "selected_parcel_groups": len(groups)}
               for (source, crop, partition, role), groups in sorted(counts.items())]
    return {"kind": "collection_metadata_snapshot", "version": 1, "source_roles": SOURCE_ROLES,
            "report_root": str(report_root), "inputs": inputs, "batches": batches,
            "references": list(members.values()), "profile": profile,
            "policies_and_feedback": policies, "issues": issues,
            "limitations": ["Counts describe saved selections, not all available source references.",
                            "Parcel groups are not proven independent physical fields across sources/years.",
                            "Country/source context is retained as recorded; fallback names are not verified countries.",
                            "Registered TIFF counts and saved job states are metadata, not fresh imagery or cloud checks.",
                            "Quality targets and development/test overlap require preprocessing/05.",
                            "This report does not change eligibility, labels, partitions or collection budgets."]}


def write_summary(repository):
    """Publish a content-keyed report and refresh navigation, preserving all inputs."""
    root = Path(repository).resolve()
    snapshot = inspect_collection(root)
    snapshot["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(snapshot, sort_keys=True, indent=2) + "\n"
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    destination = root / "pipeline_outputs/collection/reports" / f"summary_{key}"
    refresh(root)
    for path, expected in snapshot["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Collection metadata changed during inspection; rerun summary: {path}")
    if destination.exists():
        receipt = json.loads((destination / "files.json").read_text())
        if (destination / "collection_manifest.json").read_text() != payload or any(
            hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest
            for name, digest in receipt.items()
        ):
            raise ValueError(f"Existing summary changed; refusing to overwrite {destination}")
        return destination / "collection_manifest.json"
    with TemporaryDirectory(prefix=".summary-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        (staging / "collection_manifest.json").write_text(payload)
        for name, rows in (("batches", snapshot["batches"]), ("references", snapshot["references"]),
                           ("dataset_profile", snapshot["profile"])):
            with (staging / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["status"])
                writer.writeheader()
                writer.writerows(rows)
        states = dict(Counter(row["status"] for row in snapshot["batches"]))
        text = ["# Collection summary", "", f"Saved batches: {len(snapshot['batches'])}",
                f"Saved status counts: {states}", f"Metadata issues: {len(snapshot['issues'])}", "",
                "| Source | Crop | Partition | Role | Selected parcel groups |",
                "| --- | --- | --- | --- | ---: |"]
        text += [f"| {r['source']} | {r['crop']} | {r['partition']} | {r['role']} | {r['selected_parcel_groups']} |"
                 for r in snapshot["profile"]]
        text += ["", *["- " + item for item in snapshot["limitations"]], "",
                 "Review issues in collection_manifest.json. Resume pending source batches before preprocessing/05.",
                 "The next notebook discovers the existing completed handoffs; this report does not replace them."]
        (staging / "report.md").write_text("\n".join(text) + "\n")
        receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
        (staging / "files.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staging.rename(destination)
    return destination / "collection_manifest.json"
