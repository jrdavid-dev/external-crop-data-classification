#!/usr/bin/env python3
"""Inspect S4A masks and acquisition metadata without reading spectral arrays.

Install: python -m pip install numpy netCDF4
Run: python inspect_crops.py "D:\\S4A\\data\\2020"
Default codes are original S4A codes. Use --encoding linear only if verified.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

CODES = {"original": {"cabbage": 213, "carrot": 231, "potato": 510},
         "linear": {"cabbage": 17, "carrot": 32, "potato": 101}}
BANDS = ("B03", "B04", "B05", "B08", "B11")


def variables(group):
    for name, var in group.variables.items():
        yield var.name, var
    for child in group.groups.values():
        yield from variables(child)


def find_var(ds, name):
    matches = [v for n, v in variables(ds) if n.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} variable, found {len(matches)}")
    return matches[0]


def mask(var):
    data = np.ma.asarray(var[:]).squeeze()
    if data.ndim != 2:
        raise ValueError(f"{var.name}: expected a 2D mask, got {data.shape}")
    valid = ~np.ma.getmaskarray(data)
    raw = np.asarray(data.data)
    valid &= np.isfinite(raw)
    if np.any(raw[valid] != np.floor(raw[valid])):
        raise ValueError(f"{var.name}: noninteger mask values")
    return raw, valid


def dates(var):
    values = np.ma.asarray(var[:]).compressed()
    units = getattr(var, "units", "")
    if "since" not in units:
        raise ValueError(f"{var.name}: cannot decode time without CF 'since' units")
    return num2date(values, units, calendar=getattr(var, "calendar", "standard"))


def inspect(path, codes, background):
    row = {"file": str(path), "status": "ok", "warnings": []}
    with Dataset(path) as ds:
        row["metadata"] = {key: str(ds.getncattr(key)) for key in ds.ncattrs()}
        for short, attr in (("year", "patch_year"), ("country", "patch_country_code"),
                            ("tile", "patch_tile"), ("patch", "patch_full_name")):
            row[short] = str(getattr(ds, attr, "unknown"))
        labels, valid = mask(find_var(ds, "labels"))
        row["shape"] = list(labels.shape)
        row["valid_label_pixels"] = int(valid.sum())
        row["masked_label_pixels"] = int(valid.size - valid.sum())
        ids, counts = np.unique(labels[valid], return_counts=True)
        row["label_codes"] = {str(int(i)): int(c) for i, c in zip(ids, counts)}
        parcels = parcel_valid = None
        try:
            parcels, parcel_valid = mask(find_var(ds, "parcels"))
            if parcels.shape != labels.shape:
                raise ValueError(f"Parcel shape {parcels.shape} differs from labels {labels.shape}")
        except Exception as exc:
            parcels = parcel_valid = None
            row["warnings"].append(str(exc))
        row["crops"] = {}
        for crop, code in codes.items():
            selected = valid & (labels == code)
            info = {"code": code, "pixels": int(selected.sum()),
                    "parcel_occurrences": None, "pixels_without_valid_parcel": None,
                    "parcel_pixel_min": None, "parcel_pixel_median": None,
                    "parcel_pixel_max": None, "mixed_label_parcel_occurrences": None}
            if parcels is not None:
                usable = parcel_valid & ~np.isin(parcels, background)
                pids, sizes = np.unique(parcels[selected & usable], return_counts=True)
                info.update(parcel_occurrences=len(pids),
                            pixels_without_valid_parcel=int((selected & ~usable).sum()),
                            parcel_pixel_min=int(sizes.min()) if len(sizes) else None,
                            parcel_pixel_median=float(np.median(sizes)) if len(sizes) else None,
                            parcel_pixel_max=int(sizes.max()) if len(sizes) else None,
                            mixed_label_parcel_occurrences=sum(
                                bool(np.any(valid & (parcels == pid) & (labels != code)))
                                for pid in pids))
            row["crops"][crop] = info
        all_vars = list(variables(ds))
        row["required_bands"] = {}
        for band in BANDS:
            found = [v for n, v in all_vars if n.upper() == band]
            row["required_bands"][band] = [{"shape": list(v.shape), "dimensions": list(v.dimensions)} for v in found]
        row["time_coordinates"] = []
        for name, var in all_vars:
            if name.lower() not in ("time", "timestamp"):
                continue
            entry = {"variable": var.group().path + "/" + name, "shape": list(var.shape)}
            try:
                acquired = dates(var)
                bins = Counter(f"{d.year:04d}-{d.month:02d}-{'01-15' if d.day <= 15 else '16-end'}" for d in acquired)
                entry.update(observations=len(acquired),
                             first=min(str(d) for d in acquired) if len(acquired) else None,
                             last=max(str(d) for d in acquired) if len(acquired) else None,
                             semi_monthly_observation_counts=dict(sorted(bins.items())))
            except Exception as exc:
                entry["error"] = str(exc)
            row["time_coordinates"].append(entry)
        if not row["time_coordinates"]:
            row["warnings"].append("No time/timestamp coordinates found")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".", help="Folder scanned recursively, or one .nc file")
    parser.add_argument("--encoding", choices=CODES, default="original")
    parser.add_argument("--output", default="crop_inspection", help="Report directory (default: crop_inspection in current folder)")
    parser.add_argument("--parcel-background", type=int, nargs="+", default=[0, -1], help="Excluded parcel IDs; default: 0 -1")
    args = parser.parse_args()
    root = Path(args.folder).resolve()
    if not root.exists():
        parser.error(f"Path does not exist: {root}")
    files = [root] if root.is_file() and root.suffix.lower() == ".nc" else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".nc")
    if not files:
        parser.error(f"No .nc files found in {root}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    codes = CODES[args.encoding]
    totals = {c: {"pixels": 0, "files_present": 0, "parcel_occurrences": 0, "files_with_parcel_counts": 0, "pixels_without_valid_parcel": 0} for c in codes}
    label_counts = Counter()
    groups = {}
    failed = 0
    csv_rows = []
    with (output / "file_details.jsonl").open("w", encoding="utf-8") as details:
        for index, path in enumerate(files, 1):
            try:
                row = inspect(path, codes, args.parcel_background)
            except Exception as exc:
                row = {"file": str(path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                failed += 1
            details.write(json.dumps(row) + "\n")
            if row["status"] == "ok":
                label_counts.update(row["label_codes"])
                key = " / ".join(row[k] for k in ("year", "country", "tile"))
                group = groups.setdefault(key, {c: 0 for c in codes})
                for crop, info in row["crops"].items():
                    total = totals[crop]
                    total["pixels"] += info["pixels"]
                    total["files_present"] += int(info["pixels"] > 0)
                    if info["parcel_occurrences"] is not None:
                        total["parcel_occurrences"] += info["parcel_occurrences"]
                        total["files_with_parcel_counts"] += 1
                        total["pixels_without_valid_parcel"] += info["pixels_without_valid_parcel"]
                    group[crop] += info["pixels"]
                    csv_rows.append({"file": str(path), "year": row["year"], "country": row["country"], "tile": row["tile"], "crop": crop, **info})
            print(f"[{index}/{len(files)}] {path.name}: " + (", ".join(f"{c}={v['pixels']}" for c, v in row["crops"].items()) if row["status"] == "ok" else row["error"]), flush=True)
    with (output / "crop_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["file", "year", "country", "tile", "crop", "code", "pixels", "parcel_occurrences", "pixels_without_valid_parcel", "parcel_pixel_min", "parcel_pixel_median", "parcel_pixel_max", "mixed_label_parcel_occurrences"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {"input": str(root), "encoding": args.encoding, "codes": codes,
               "files_found": len(files), "files_successful": len(files)-failed,
               "files_failed": failed, "parcel_background": args.parcel_background,
               "crops": totals, "observed_label_codes": dict(sorted(label_counts.items(), key=lambda x: int(x[0]))),
               "pixels_by_year_country_tile": groups}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["S4A crop inspection", f"Input: {root}", f"Encoding: {args.encoding}; codes: {codes}",
             f"Files: {len(files)} found, {len(files)-failed} successful, {failed} failed", ""]
    for crop, total in totals.items():
        lines.append(f"{crop}: {total['pixels']:,} pixels; {total['files_present']} files containing crop; {total['parcel_occurrences']:,} parcel occurrences ({total['files_with_parcel_counts']} files with usable parcel masks)")
    lines += ["", f"Observed label codes and pixel counts: {dict(sorted(label_counts.items(), key=lambda x: int(x[0])))}", "",
              "Interpretation and next steps:",
              "- Verify the encoding before interpreting absence. Codes are never auto-detected.",
              "- Parcel occurrences are per-file counts, NOT independent fields. ID scope and cross-patch duplication remain unverified.",
              "- Parcel pixel sizes refer to the crop portion visible in a patch, not complete field size.",
              "- Pixel counts are not unique land area; no area conversion is assumed.",
              "- Review crop_counts.csv for candidate patches, mixed-label parcels, and pixels lacking parcel IDs.",
              "- Review file_details.jsonl for errors, band shapes, spatial metadata, and semi-monthly acquisition counts.",
              "- Acquisition counts do not measure clear pixels or cloud-free coverage; missing bins are absent from each coordinate's dictionary.",
              "- Confirm L1C/L2A compatibility, georeferencing, and crop label relevance to your four-month window.",
              "- Confirm eight semi-monthly bins have usable observations after cloud masking for all five bands.",
              "- Keep all pixels/windows of an independent field in one partition; evaluate on independent Benguet fields.",
              "- Zero counts apply only to successfully scanned files under the selected encoding."]
    report = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report + f"Reports saved to: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
