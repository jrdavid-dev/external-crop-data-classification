"""Compare how each mapped model behaves on Paoay and the Atok boundary, without local labels.

Reads saved maps only: the experiment rerun's annual-family maps (``rerun_experiments``) and, for
Paoay, the window models' fixed January-April map (Paoay 04) and aligned exploratory map
(Paoay 07). No dated local crop labels exist, so nothing here is accuracy. The diagnostics are:

- confident share of cropland: cropland pixels with a confident prediction (top probability
  >= 0.60 and margin >= 0.10);
- outside training range: share of predicted pixels with more than 20% of predictors outside the
  training 5th-95th percentile range (annual-family maps only);
- crop mix distance: total variation distance between the confident crop mix in cropland and PSA's
  2024 Benguet harvested-area shares (0 = same mix, 1 = disjoint). Province statistics are a
  weak reference for one barangay or municipality;
- calendar consistency: share of confident pixels in the KMZ reference areas whose crop the area's
  calendar names for months the prediction covers (whole year for annual maps, so it only checks
  the crop name there);
- farmland concentration: confident crop share inside NAMRIA ELUM 2023 farmland minus outside it.

    .venv/bin/python -m experiments.annual.application_review \\
        pipeline_outputs/experiments/<rerun>/experiments.json pipeline_outputs/experiments/<rerun>/application
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from crop_pipeline.paoay import aligned, area_review

ROOT = Path(__file__).resolve().parents[2]
CROPS = area_review.CROPS


def settings():
    return json.loads((ROOT / "settings.local.json").read_text())


def local(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def province_shares():
    with (ROOT / "data/paoay/benguet_vegetable_area_2024.csv").open() as stream:
        table = {r["crop"]: float(r["area_ha"]) for r in csv.DictReader(stream)}
    total = sum(table[c] for c in CROPS)
    return {c: table[c] / total for c in CROPS}


def diagnostics(item, cropland, study, farm, masks, entries, outside_share=None, reference=None):
    labels = item["labels"]
    mapped = study & (labels > 0)
    confident = mapped & (labels <= 3)
    crop_cells = mapped & cropland
    counts = {c: int(((labels == k) & crop_cells).sum()) for k, c in enumerate(CROPS, 1)}
    total = sum(counts.values())
    mix = {c: counts[c] / total for c in CROPS} if total else {c: 0.0 for c in CROPS}
    in_farm = confident[farm & mapped].mean() if (farm & mapped).any() else np.nan
    out_farm = confident[~farm & mapped].mean() if (~farm & mapped).any() else np.nan
    row = {"map": item["name"], "period": item["period"], "mapped_pixels": int(mapped.sum()),
           "confident_share_of_cropland": float(total / crop_cells.sum()) if crop_cells.any() else None,
           **{f"{c}_share_of_confident_cropland": round(mix[c], 4) for c in CROPS},
           "crop_mix_distance_to_benguet_2024": (round(0.5 * sum(abs(mix[c] - reference[c]) for c in CROPS), 4)
                                                 if total and reference else None),
           "outside_training_range_share": outside_share,
           "farmland_confident_share_minus_outside": float(in_farm - out_farm)}
    if masks:
        comparison = area_review.calendar_comparison(item, masks, entries)
        confident_total = sum(r["confident_pixels"] for r in comparison)
        consistent = sum(r["confident_pixels"] * (r["calendar_consistent_percent_of_confident"] or 0) / 100
                         for r in comparison if r["confident_pixels"])
        row["reference_area_confident_pixels"] = confident_total
        row["calendar_consistent_share"] = round(consistent / confident_total, 4) if confident_total else None
        row["dominant_crop_by_area"] = "; ".join(f"{r['area']}={r['dominant_prediction'] or '-'}" for r in comparison)
    return row


def study_masks(area, grid, s):
    kmz_root = local(s.get("paoay_kmz_root", "./data/paoay"))
    kmz = [kmz_root / n for n in ("00 - Region of Interest.kmz", "00 - Atok Boundary.kmz", "Crop Area.kmz")]
    roi_name = "Atok" if area == "atok" else (s.get("paoay_roi_name") or "Region of Interest")
    context = [n for n in s.get("paoay_context_names", []) if n != roi_name]
    roi, areas, _ = aligned.study_area(kmz, roi_name, context, grid["crs"])
    # With the Atok boundary as the study area, the Paoay study boundary is not a reference area.
    paoay_roi = s.get("paoay_roi_name") or "Region of Interest"
    areas = [a for a in areas if a["name"] != paoay_roi]
    return area_review.area_masks([{"name": "study", "geometry": roi}], grid)["study"], areas, roi


def review(summary_path, output):
    summary_path, output = Path(summary_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    s = settings()
    summary = json.loads(summary_path.read_text())
    reference = province_shares()
    paoay_root = local(s.get("report_root", "./data/pipeline_work/inventory")).parent / "paoay"
    calendar = local(s.get("paoay_calendar_file", paoay_root / "review/calendars/kmz_notes_calendar.csv"))
    landuse_path = local(s["atok_landuse_path"])
    rows = {"paoay": [], "atok": []}
    for area in rows:
        items = []
        for label, record in summary["experiments"].items():
            run = Path(record["maps"][area]["run"])
            item = area_review.product("annual", run, name=label)
            with rasterio.open(run / "crop_map.tif") as source:
                cropland = source.read(2) == 2
            items.append((item, cropland, record["maps"][area]["outside_training_range_share"]))
        if area == "paoay":
            baseline = resolve_stage_pointer(paoay_root / "crop_baseline/crop_baseline_stage.json")
            land = resolve_stage_pointer(paoay_root / "land_activity/land_activity_stage.json").parent / "land_activity.tif"
            with rasterio.open(land) as source:
                active = source.read(1) == 1
            window = [area_review.product("fixed", baseline, name="window model: fixed Jan-Apr (Paoay 04)")]
            folder = area_review.aligned_bundle_for(paoay_root / "aligned_exploratory", aligned.default_model(ROOT))
            if folder is not None:
                window.append(area_review.product("aligned", folder, name="window model: aligned (Paoay 07)"))
            items += [(item, active, None) for item in window]
        grid = items[0][0]["grid"]
        study, areas, roi = study_masks(area, grid, s)
        masks = area_review.area_masks(areas, grid)
        entries = area_review.read_calendar(calendar, set(masks))
        a, _, x0, _, e, y0 = grid["transform"]
        height, width = grid["shape"]
        farm = area_review.polygon_mask(area_review.farmland(area_review.read_landuse(
            landuse_path, (x0, y0 + e * height, x0 + a * width, y0), grid["crs"])), grid)
        for item, cropland, outside in items:
            if item["grid"] != grid:
                raise ValueError(f"{item['name']} uses a different grid")
            rows[area].append(diagnostics(item, cropland, study, farm, masks, entries, outside, reference))
        area_review.write_csv(output / f"application_{area}.csv", rows[area])
    (output / "province_reference.json").write_text(json.dumps(reference, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for area, table in review(args.summary, args.output).items():
        print(f"\n== {area}")
        for row in table:
            print({k: row[k] for k in ("map", "confident_share_of_cropland", "crop_mix_distance_to_benguet_2024",
                                       "outside_training_range_share", "calendar_consistent_share")})


if __name__ == "__main__":
    main()
