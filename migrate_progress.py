"""Update local paths after moving PCs; do not change request IDs or checksums."""
import json
from pathlib import Path

from crop_workflow import save_json


def main():
    settings = json.loads(Path("settings.local.json").read_text(encoding="utf-8"))
    imagery = Path(settings["imagery_root"]).expanduser().resolve()
    reports = Path(settings.get("report_root", "./sen4agrinet_work/inventory")).resolve().parent
    progress = reports / "l2a" / "2020" / "composite_progress"
    updated, missing = 0, 0
    for path in progress.glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        region_id, request_id = path.stem.rsplit("_", 1)
        for name, record in manifest["files"].items():
            destination = imagery / region_id / request_id / (name + ".tif")
            record["path"] = str(destination)
            if record.get("status") == "complete" and not destination.is_file():
                missing += 1
        save_json(path, manifest)
        updated += 1
    print("Progress manifests updated:", updated)
    print("Completed records without local files:", missing)


if __name__ == "__main__":
    main()
