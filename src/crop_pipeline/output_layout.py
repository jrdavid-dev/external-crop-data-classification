"""Non-destructive, refreshable output navigation for existing pipeline archives.

The original files remain authoritative. Relative symbolic links preserve saved
absolute paths, receipt hashes, collection identities, and resumable jobs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

SOURCES = ("sen4agrinet", "francecrops", "eurocrops", "cropharvest")


def layout_plan(repository):
    repository = Path(repository).resolve()
    settings_path = repository / "settings.local.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    def local(value):
        return (repository / Path(value).expanduser()).resolve()

    work = local(settings.get("report_root", "sen4agrinet_work/inventory")).parent
    archive = local(settings.get("annual_archive_root", str(work / "annual_archive")))
    root = repository / "pipeline_outputs"
    links = {}
    directories = {root / name for name in ("collection/reports", "preprocessing", "training", "testing")}

    def link(name, target):
        target = Path(target)
        if target.exists():
            destination = root / name
            if destination in links and links[destination] != target:
                raise ValueError(f"Ambiguous output destination: {destination}")
            links[destination] = target

    link("README.md", repository / "docs/output_layout.md")
    for source in SOURCES:
        for name in ("references", "imagery", "state"):
            directories.add(root / "collection" / source / name)
        link(f"collection/{source}/references/downloads", repository / "data/external" / (
            "eurocrops_v2" if source == "eurocrops" else source))
        link(f"collection/{source}/imagery/annual", archive.parent / "annual_tifs" / source)
    link("collection/sen4agrinet/references/release", local(settings.get(
        "sen4agrinet_reference_root", str(archive / "sen4agrinet_references"))))
    link("collection/francecrops/references/preparation", archive / "francecrops")
    link("collection/francecrops/references/source_metadata", work / "francecrops/source_metadata")
    link("collection/cropharvest/references/reviews", archive / "cropharvest_label_review")
    link("collection/reports/reference_audit", repository / "data/external/dataset_audit")
    # Policies and feedback belong to the whole campaign, not to one source.
    link("collection/reports/campaigns", work / "adaptive_training")
    link("collection/reports/logs", work / "progress")
    for context_path in sorted((work / "adaptive_training").glob("*/rounds/*/*/context.json")):
        context = json.loads(context_path.read_text())
        source = context.get("source")
        if source not in SOURCES:
            continue
        batch = context_path.parent
        key = f"{batch.parents[2].name}_{batch.parent.name}_{batch.name}"
        link(f"collection/{source}/state/{key}", batch)
        if context.get("plan_path"):
            link(f"collection/{source}/state/{key}_archive", local(context["plan_path"]).parent)

    external = work / "analysis/external"
    for run in sorted(external.glob("external_*")):
        if not run.is_dir():
            continue
        prefix = f"preprocessing/{run.name}"
        for name in ("quality", "windows", "features"):
            directories.add(root / prefix / name)
        link(f"{prefix}/quality/audit", run / "quality")
        link(f"{prefix}/quality/timelines", run / "timelines")
        link(f"{prefix}/windows/seasons", run / "seasons")
        link(f"{prefix}/windows/greenup", run / "greenup")
        for item in sorted(run.iterdir()):
            if not item.is_file():
                continue
            section = ("features" if item.name.startswith("features") or item.name in {"usable_samples.csv", "preprocessing_report.md"} else
                       "windows" if item.name.startswith(("windows", "window_")) or item.name == "fixed_losses.csv"
                       else "quality")
            link(f"{prefix}/{section}/{item.name}", item)
        for models in sorted(run.glob("models_*")):
            key = f"{run.name}_{models.name}"
            link(f"training/{key}/models", models)
            directories.add(root / "training" / key / "validation")
            for manifest in models.glob("*/model_manifest.json"):
                link(f"training/{key}/validation/{manifest.parent.name}.json", manifest)
            for result in models.glob("*_test_parcels.csv"):
                link(f"testing/{key}/predictions/{result.name}", result)
            link(f"testing/{key}/reports/test_scores.json", models / "test_scores.json")
    # Historical fitted packages stay identifiable and separate from new experiments.
    for models in sorted((work / "analysis").glob("*/seasons/workflow/pilot/models")):
        run_name = models.parents[3].name
        link(f"training/legacy_{run_name}/models", models)
    return root, links, directories


def inventory(links):
    """Fingerprint every referenced regular file once, including shared targets."""
    files = set()
    for target in links.values():
        candidates = target.rglob("*") if target.is_dir() else [target]
        files.update(path.resolve() for path in candidates if path.is_file())
    records = {}
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records[str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return records


def refresh(repository, verify=False):
    root, links, directories = layout_plan(repository)
    manifest = root / "layout_manifest.json"
    # Preflight the entire plan before creating anything. Never adopt/replace user files.
    if manifest.exists():
        if json.loads(manifest.read_text()).get("kind") != "linked_pipeline_output_layout":
            raise FileExistsError(f"Unrecognized layout manifest: {manifest}")
    for destination, target in links.items():
        if destination.is_symlink() and destination.resolve() == target.resolve():
            continue
        if os.path.lexists(destination):
            raise FileExistsError(f"Refusing to replace existing output: {destination}")
    for destination in [*directories, *(p.parent for p in links), root]:
        for parent in [destination, *destination.parents]:
            if parent == root.parent:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise FileExistsError(f"Output parent is not an ordinary directory: {parent}")
    before = inventory(links) if verify else None
    for directory in sorted(directories):
        directory.mkdir(parents=True, exist_ok=True)
    for destination, target in links.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_symlink():
            try:
                destination.symlink_to(os.path.relpath(target, destination.parent), target_is_directory=target.is_dir())
            except FileExistsError:
                if not destination.is_symlink() or destination.resolve() != target.resolve():
                    raise
    if verify and before != inventory(links):
        raise RuntimeError("Source files changed during layout creation; inspect active writers before retrying")
    report = {"kind": "linked_pipeline_output_layout", "version": 1,
              "links": {str(p.relative_to(root)): os.path.relpath(t, root) for p, t in links.items()},
              "verified_file_count": len(before) if verify else None,
              "verified_bytes": sum(v["bytes"] for v in before.values()) if verify else None}
    with NamedTemporaryFile(mode="w", prefix=".layout-", suffix=".tmp", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(report, indent=2) + "\n")
    temporary.replace(manifest)
    if verify:
        # Preserve dated evidence across later refreshes without verification.
        from datetime import datetime, timezone
        receipt = root / ("integrity_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        with receipt.open("x") as stream:
            json.dump({"files": before, "verified_file_count": len(before)}, stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--verify", action="store_true", help="Hash source files before and after creating links")
    args = parser.parse_args()
    report = refresh(args.repository, verify=args.verify)
    print(f"Output layout refreshed: {len(report['links'])} links; source files remain in place.")
    if args.verify:
        print(f"Integrity verified: {report['verified_file_count']} files, {report['verified_bytes']} bytes.")


if __name__ == "__main__":
    main()
