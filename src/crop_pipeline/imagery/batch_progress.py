"""Artifact-based collection progress; queued task counts are never completion."""
from datetime import datetime, timezone

from crop_pipeline.imagery import batch_export as batch

QUEUED = {"SUBMITTED", "PENDING", "READY"}
CANCELLED = {"CANCELLED", "CANCELED", "CANCELLING", "CANCEL_REQUESTED"}


def snapshot(plans, missing_images, missing_daily, ledger, table_mode=True):
    periods = {(plan["fragment_id"], interval["name"]) for plan in plans
               for interval in plan["download_intervals"]}
    missing = {(plan["fragment_id"], interval["name"]) for plan, intervals in missing_images
               for interval in intervals}
    if not missing <= periods:
        raise ValueError("Progress contains periods outside the selected workload")
    identifiers = {plan["fragment_id"] for plan in plans}
    daily_missing = {plan["fragment_id"] for plan in missing_daily}
    if not daily_missing <= identifiers:
        raise ValueError("Daily progress contains unselected tiles")
    data_missing = {region for region, _ in missing}
    data_done = identifiers - data_missing
    daily_done = identifiers - daily_missing
    complete = data_done & daily_done
    kind = "pixel_table" if table_mode else "composite"
    # Compute the fingerprint once per tile, not once per period/mask read.
    pending_items = set()
    for plan in plans:
        region = plan["fragment_id"]
        if region not in daily_missing and region not in data_missing:
            continue
        fingerprint = batch.request_fingerprint(plan)
        if region in daily_missing:
            pending_items.add(f"daily|{region}|{fingerprint}")
        for interval in plan["download_intervals"]:
            if (region, interval["name"]) in missing:
                pending_items.add(f"{kind}|{region}|{interval['name']}|{fingerprint}")
    # The newest attempt owns an item; old failed attempts must not inflate failure counts.
    latest = {}
    for index, entry in enumerate(ledger):
        for item in entry.get("items", []):
            if item in pending_items:
                latest[item] = index
    current = set(latest.values())
    jobs = dict.fromkeys(["queued", "running", "ready_to_download", "failed", "cancelled", "unknown"], 0)
    unscheduled = pending_items - latest.keys()
    for index in current:
        entry = ledger[index]
        if entry.get("collected"):
            # The old asset was collected, but its local result is missing or invalid now.
            unscheduled.update(item for item, owner in latest.items() if owner == index)
            continue
        state = entry.get("state", "UNKNOWN")
        category = ("queued" if state in QUEUED else "running" if state == "RUNNING"
                    else "ready_to_download" if state in batch.FINISHED_STATES
                    else "cancelled" if state in CANCELLED
                    else "failed" if state == "FAILED" else "unknown")
        jobs[category] += 1
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "tiles": {"total": len(identifiers), "complete": len(complete),
                  "percent_complete": round(100 * len(complete) / len(identifiers), 1) if identifiers else 100.0,
                  "pixel_data_complete": len(data_done), "daily_summaries_complete": len(daily_done),
                  "pixel_data_only": len(data_done - daily_done),
                  "daily_summaries_only": len(daily_done - data_done),
                  "neither_complete": len(identifiers - (data_done | daily_done))},
        "half_month_periods": {"total": len(periods), "verified_local": len(periods - missing),
                               "pending": len(missing)},
        "jobs": {**jobs, "work_items_not_submitted": len(unscheduled)},
        "historical_attempts": {"total": len(ledger),
            "failed_or_cancelled": sum(entry.get("state") in {"FAILED", *CANCELLED} for entry in ledger)},
        "percent_basis": "Tiles with both verified local pixel data and current daily checkpoints; not elapsed runtime.",
    }


def print_snapshot(progress):
    tiles, periods, jobs = (progress[key] for key in ("tiles", "half_month_periods", "jobs"))
    print(f"Complete tiles: {tiles['complete']}/{tiles['total']} ({tiles['percent_complete']:.1f}%)", flush=True)
    print(f"  Pixel data: {tiles['pixel_data_complete']}/{tiles['total']} | "
          f"Daily summaries: {tiles['daily_summaries_complete']}/{tiles['total']}", flush=True)
    print(f"  Half-month periods verified locally: {periods['verified_local']}/{periods['total']} | "
          f"Pending: {periods['pending']}", flush=True)
    print(f"  Jobs: {jobs['running']} running, {jobs['queued']} queued, "
          f"{jobs['ready_to_download']} ready to download, {jobs['failed']} failed, "
          f"{jobs['cancelled']} cancelled, {jobs['unknown']} unknown", flush=True)
    if jobs["work_items_not_submitted"]:
        print(f"  Work items awaiting submission: {jobs['work_items_not_submitted']}", flush=True)
    print("  Completion measures saved results, not a percentage of processing time. ETA: unavailable.", flush=True)


def save_snapshot(path, progress):
    """Watch and collection processes may publish the same report concurrently."""
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                            suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(progress, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
