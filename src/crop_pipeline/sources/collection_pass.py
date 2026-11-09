"""Durable collection-pass reports and retry queues, separate from image quality."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from crop_pipeline.common import save_json


def start_attempt(plan, report_folder):
    identifier = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    folder = Path(plan["run_folder"]) / "collection_attempts" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "collecting",
        "source": plan["source"],
        "revision": plan["revision"],
        "attempt_folder": str(folder.resolve()),
    }
    save_json(state_path(report_folder, plan["source"]), state)
    return folder


def state_path(report_folder, source):
    return (
        Path(report_folder).parent
        / "source_handoffs/full_year_all_parcels"
        / f"{source}.state.json"
    )


def write_event(attempt_folder, year, result):
    if attempt_folder is None:
        return
    path = Path(attempt_folder) / "tile_results.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"year": str(year), **result}, allow_nan=False) + "\n")


def finish_attempt(plan, report_folder, attempt_folder, year_reports):
    pending = [entry for report in year_reports for entry in report["retry_queue"]]
    result = {
        "status": "pending_retry" if pending else "completed",
        "source": plan["source"],
        "revision": plan["revision"],
        "planned_tiles": sum(report["planned_tiles"] for report in year_reports),
        "completed_tiles": sum(report["completed_tiles"] for report in year_reports),
        "downloaded_images": sum(
            report["downloaded_images"] for report in year_reports
        ),
        "reused_images": sum(report["reused_images"] for report in year_reports),
        "pending_images": sum(report["pending_images"] for report in year_reports),
        "retry_tiles": len(pending),
        "retry_queue": str((Path(attempt_folder) / "retry_queue.json").resolve()),
    }
    if any("progress" in report for report in year_reports):
        result["progress_by_year"] = {report["year"]: report["progress"]
                                      for report in year_reports if "progress" in report}
    if any("selected_batch" in report for report in year_reports):
        result["selected_batches"] = [report["selected_batch"] for report in year_reports
                                      if "selected_batch" in report]
    save_json(Path(attempt_folder) / "retry_queue.json", pending)
    save_json(Path(attempt_folder) / "pass_summary.json", result)
    save_json(Path(plan["run_folder"]) / "latest_collection_pass.json", result)
    save_json(
        state_path(report_folder, plan["source"]),
        {**result, "status": "pending_retry" if pending else "awaiting_registration"},
    )
    return result


def image_failures(plan, results, report_folder):
    completed = {
        record["name"]
        for record in results
        if record["status"] in {"downloaded", "reused"}
    }
    request_id = hashlib.sha256(
        json.dumps(plan["settings"], sort_keys=True).encode()
    ).hexdigest()[:16]
    manifest_path = (
        Path(report_folder)
        / "composite_progress"
        / f"{plan['fragment_id']}_{request_id}.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return [
        {
            "interval": interval["name"],
            "error": manifest.get("files", {})
            .get(interval["name"], {})
            .get("error", "Image missing or did not pass completion checks"),
        }
        for interval in plan["download_intervals"]
        if interval["name"] not in completed
    ]


def estimated_required_space(plan):
    """Use missing raw-grid volume for storage planning; reuse still verifies checksums."""
    completed_raw_bytes = 0
    for inputs in plan["years"].values():
        completed_raw_bytes += saved_raw_volume(
            Path(inputs["report_folder"]) / "composite_progress"
        )
    remaining = max(0, plan["estimate"]["raw_raster_bytes"] - completed_raw_bytes)
    return remaining * 2 + 64 * 1024**2


def saved_raw_volume(progress_folder):
    volume = 0
    for path in progress_folder.glob("*.json"):
        manifest = json.loads(path.read_text())
        grid = manifest["settings"]["grid"]
        completed = sum(
            record.get("status") == "complete"
            and Path(record.get("path", "")).is_file()
            for record in manifest.get("files", {}).values()
        )
        volume += completed * grid["width"] * grid["height"] * 7 * 4
    return volume
