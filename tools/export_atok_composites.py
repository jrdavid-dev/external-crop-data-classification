"""Plan, submit, track and arrange the all-Atok Sentinel-2 half-month composite exports.

    python tools/export_atok_composites.py plan                 # no Earth Engine calls
    python tools/export_atok_composites.py submit               # starts Drive exports
    python tools/export_atok_composites.py status               # refreshes task states
    python tools/export_atok_composites.py arrange SOURCE DEST  # flat Drive folder -> DEST/YYYY/
    python tools/export_atok_composites.py check ROOT           # verify files against the grid

Paths come from settings.local.json (atok_boundary_path, earth_engine_project). The ledger
in analysis/atok/imagery_export/ records every task, so reruns never resubmit a half-month.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from crop_pipeline.imagery import atok_composites as ac

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "analysis/atok/imagery_export/ledger.json"


def settings():
    path = ROOT / "settings.local.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def make_plan(args, local):
    boundary = Path(args.boundary or local.get("atok_boundary_path", "./data/atok/paoay_boundary_2023.geojson"))
    boundary = boundary if boundary.is_absolute() else ROOT / boundary
    folder = args.drive_folder or f"{boundary.stem}_sentinel2_csplus"
    return ac.plan_exports(boundary, date.fromisoformat(args.first), date.fromisoformat(args.last),
                           folder, buffer_m=args.buffer)


def initialize(local):
    import ee

    ee.Initialize(project=local["earth_engine_project"])
    return ee


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "submit", "status", "arrange", "check"])
    parser.add_argument("paths", nargs="*", help="arrange: SOURCE DEST; check: ROOT")
    parser.add_argument("--boundary", help="boundary file (default: atok_boundary_path)")
    parser.add_argument("--first", default="2019-01-01")
    parser.add_argument("--last", default=date.today().isoformat())
    parser.add_argument("--drive-folder", help="Drive folder (default: <boundary file name>_sentinel2_csplus)")
    parser.add_argument("--buffer", type=float, default=ac.BUFFER_M, help="metres around the boundary")
    parser.add_argument("--existing", help="local YYYY/ folder; half-months already there on this grid are skipped")
    args = parser.parse_args()
    local = settings()

    if args.command == "arrange":
        source, destination = args.paths
        copied = ac.arrange_by_year(source, destination)
        print(f"Copied {len(copied)} files into {destination}/YYYY/")
        return
    if args.command == "status":
        print(ac.refresh(initialize(local), LEDGER))
        return

    plan = make_plan(args, local)
    grid = plan["grid"]
    if args.command == "check":
        files = sorted(Path(args.paths[0]).glob("*/Sentinel2_CSPlus_*.tif"))
        bad = {path.name: problems for path in files if (problems := ac.check_file(path, grid))}
        missing = sorted({item["stem"] for item in plan["items"]} - {path.stem for path in files})
        print(f"{len(files)} files, {len(bad)} with problems, {len(missing)} planned half-months missing")
        for name, problems in bad.items():
            print(f"  {name}: {'; '.join(problems)}")
        return

    print(f"Grid {grid['width']} x {grid['height']} px at {ac.SCALE} m, {ac.CRS}, origin {grid['transform'][2]}, {grid['transform'][5]}")
    print(f"Half-months {len(plan['items'])}: {plan['items'][0]['stem']} .. {plan['items'][-1]['stem']}")
    print(f"Up to {grid['width'] * grid['height'] * 5 * 4 * len(plan['items']) / 1e9:.1f} GB uncompressed "
          f"to Google Drive folder '{plan['drive_folder']}' of the Earth Engine account")
    already = ac.covered(ac.load_ledger(LEDGER), plan["settings_key"])
    print(f"Already exported or running with these settings: {len(already)}")
    if args.command == "submit":
        submitted = ac.submit(initialize(local), plan, LEDGER, args.existing)
        print(f"Submitted {len(submitted)} exports; ledger: {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
