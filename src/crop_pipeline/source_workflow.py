"""Shared notebook presentation without changing collection or campaign identity."""

from collections import Counter
import json
from pathlib import Path

ROUND_PARCELS_PER_CROP = 100
STAGES = [
    "Obtain references",
    "Validate references",
    "Prepare collection round",
    "Collect annual imagery",
    "Summary and next action",
]


def reference_options(mode):
    """Map the same visible reference modes onto source-specific retrieval flags."""
    if mode not in {"reuse", "download_missing", "offline"}:
        raise ValueError("reference_mode must be reuse, download_missing or offline")
    download = mode == "download_missing"
    return {"download_if_missing": download, "cached_only": not download}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _crop_counts(batches):
    """Count physical selected parcels, never raster tiles or repeated years."""
    crops = {}
    unavailable = False
    for plan, context in batches:
        if plan is None:
            continue
        selection = _read_json(Path(context["folder"]) / "selection.json")
        wanted = set(selection["groups"]) if selection is not None else None
        found = set()
        for inputs in plan.get("years", {}).values():
            if not inputs.get("field_folder"):
                continue
            features = _read_json(Path(inputs["field_folder"]) / "source_footprints.geojson")
            if features is None:
                continue
            for feature in features.get("features", []):
                properties = feature["properties"]
                group = properties["partition_group"]
                if wanted is not None and group not in wanted:
                    continue
                found.add(group)
                key = (context.get("source"), str(properties.get("country", "")),
                       str(properties.get("parcel_id") or group))
                crop = properties.get("crop")
                if crop:
                    crops[key] = crop
        if wanted is not None:
            unavailable |= bool(wanted - found)
        else:
            unavailable |= len(found) != context.get("candidate_parcels", len(found))
    return Counter(crops.values()), unavailable


def print_summary(source, batches, result, reference_mode, buffer_metres, reference_note=None):
    """Print one concise summary; inspect only local metadata, never alter saved work."""
    reference_options(reference_mode)
    batches = list(batches)
    print(f"{source} — {result.get('status', 'unavailable')}")
    print(f"References: {reference_mode} | Inward buffer: {buffer_metres} m")
    if reference_note:
        print(f"Reference coverage: {reference_note}")
    counts, incomplete_counts = _crop_counts(batches)
    if counts:
        print("Selected parcels: " + ", ".join(f"{crop} {count}" for crop, count in sorted(counts.items()))
              + (" (partial; some saved reference counts unavailable)" if incomplete_counts else ""))
    else:
        selected = sum(context.get("candidate_parcels", 0) for _, context in batches)
        print(f"Selected parcels: {selected}; crop breakdown unavailable" if selected
              else "Selected parcels: none")

    results = result.get("batches", [])
    if results:
        print("Batches: " + "; ".join(
            f"{entry['batch']}: {entry['result'].get('status', 'unavailable')}"
            for entry in results))
    saved_states = []
    if result.get("status") == "disabled":
        for _, context in batches:
            saved = _read_json(Path(context["folder"]) / "state.json") or {}
            if saved.get("status") not in {None, "disabled", "skipped"}:
                saved_states.append(f"{Path(context['folder']).name}: {saved['status']}")
        if saved_states:
            print("Saved operational state (unchanged): " + "; ".join(saved_states))
    snapshots = list(result.get("progress_by_year", {}).values())
    if snapshots and all("tiles" in snapshot for snapshot in snapshots):
        complete = sum(snapshot["tiles"]["complete"] for snapshot in snapshots)
        total = sum(snapshot["tiles"]["total"] for snapshot in snapshots)
        print(f"Verified local tiles: {complete}/{total} (reported batch ranges)")
    else:
        print("Verified local tiles: unavailable")
    if snapshots and all("jobs" in snapshot for snapshot in snapshots):
        jobs = Counter()
        for snapshot in snapshots:
            jobs.update(snapshot["jobs"])
        print("Cloud jobs: " + ", ".join(
            f"{jobs[key]} {label}" for key, label in (
                ("queued", "queued"), ("running", "running"),
                ("ready_to_download", "ready to download"), ("failed", "failed"),
                ("cancelled", "cancelled"), ("unknown", "unknown"))))
    else:
        jobs = Counter()
        print("Cloud jobs: unavailable")

    for label, key in (("Campaign", "campaign"), ("Annual TIFF root", "annual_tif_root")):
        paths = sorted({str(context[key]) for _, context in batches if context.get(key)})
        if paths:
            print(f"{label}: " + ", ".join(paths))
    registries = sorted({entry["result"]["registry"] for entry in results
                         if entry["result"].get("registry")})
    if registries:
        print("Saved registries: " + ", ".join(registries))
    from crop_pipeline import progress
    monitor = progress.active_monitor()
    if monitor is not None:
        print(f"Detail log: {monitor.output.path}")

    states = {entry["result"].get("status") for entry in results}
    if result.get("status") == "failed" or "failed" in states or jobs["failed"] or jobs["cancelled"]:
        action = "Inspect the reported errors and detail log, then rerun to resume saved work."
    elif result.get("status") in {"pending_retry", "pending"} or states - {"completed", "skipped", "disabled"}:
        action = "Rerun this notebook to collect or retry existing jobs; finish launched batches before notebook 04."
    elif result.get("status") == "disabled":
        action = "Preparation only. Enable satellite collection when ready, then rerun."
        if reference_mode == "offline":
            action += " Leave offline mode to collect; existing cloud jobs are unchanged."
    else:
        action = "Run notebook 04 to assess completed samples and measured deficits; finish other launched batches first."
    print(f"Next: {action}")
    if any(context.get("selection_status") == "no_candidates_for_deficits" for _, context in batches):
        print("No unselected reference candidates remain for these deficits. Review notebook 04 before expanding references.")
        if source == "sen4agrinet":
            print("To expand Sen4AgriNet references deliberately, use download_missing and increase reference_target_parcels_per_crop and, if needed, reference_scan_max_patches; preserve existing selections and jobs.")
