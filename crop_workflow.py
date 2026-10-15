"""Portable pilot quality checks, timelines and exploratory green-up detection.

No imagery downloads occur in this module. Input GeoTIFFs remain unchanged.
"""
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


STUDY_BANDS = ["B3", "B4", "B5", "B8", "B11", "B2"]


def write_table(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def resolve_mask(region, field_folder):
    # Prefer portable project paths over old absolute paths from another PC.
    filename = region["properties"]["fragment_id"] + ".tif"
    portable = Path(field_folder) / "balanced_pilot" / "sampling_masks" / filename
    if portable.is_file():
        return portable
    old_path = Path(region["properties"].get("sampling_mask", ""))
    if old_path.is_file():
        return old_path
    raise FileNotFoundError(f"Sampling mask missing: {filename}")


def resolve_image(record, manifest_path, imagery_folder):
    # Request IDs in existing manifests survive migration; no redownload needed.
    stem = manifest_path.stem
    fragment_id, request_id = stem.rsplit("_", 1)
    portable = Path(imagery_folder) / fragment_id / request_id / (record["name"] + ".tif")
    if portable.is_file():
        return portable
    original = Path(record["path"])
    if original.is_file():
        return original
    raise FileNotFoundError(f"Composite missing: {portable}")


def interval_bins(year=2020):
    result = []
    for month in range(1, 13):
        first, middle = date(year, month, 1), date(year, month, 16)
        end = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
        result.extend([
            {"name": f"{year}_{month:02d}_01_15", "start": first.isoformat(), "end": middle.isoformat()},
            {"name": f"{year}_{month:02d}_16_end", "start": middle.isoformat(), "end": end.isoformat()},
        ])
    return result



def compatible_request(settings, year, cloud_threshold, required_version):
    if settings.get("version") not in {3, 4}:
        return False
    if required_version is not None and settings.get("version") != required_version:
        return False
    if settings.get("cloud_probability_threshold") != cloud_threshold:
        return False
    if settings.get("start") != f"{year}-01-01" or settings.get("end") != f"{year + 1}-01-01":
        return False
    if settings["version"] == 4:
        from satellite_workflow import RESAMPLING_POLICY
        return settings.get("resampling") == RESAMPLING_POLICY
    return True

def quality_check(regions, progress_folder, imagery_folder, field_folder, output_folder,
                  cloud_threshold=40, minimum_fraction=0.8, year=2020, required_request_version=None):
    """Audit all 24 bins, preserving missing/failed records instead of filling."""
    progress_folder, output_folder = Path(progress_folder), Path(output_folder)
    region_map = {item["properties"]["fragment_id"]: item for item in regions}
    # Never silently select one of multiple competing versions.
    compatible = defaultdict(list)
    for path in sorted(progress_folder.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        settings = manifest.get("settings", {})
        if manifest["fragment_id"] in region_map and compatible_request(
                settings, year, cloud_threshold, required_request_version):
            compatible[manifest["fragment_id"]].append((path, manifest))
    quality_rows, pixel_rows, errors = [], [], []
    for region_id, region in region_map.items():
        properties = region["properties"]
        versions = compatible.get(region_id, [])
        if len(versions) > 1:
            raise ValueError(f"Multiple compatible requests for {region_id}; choose one explicitly.")
        mask_path = resolve_mask(region, field_folder)
        with rasterio.open(mask_path) as mask_raster:
            selected = mask_raster.read(1) == 1
            mask_transform, mask_crs = mask_raster.transform, mask_raster.crs
        total_selected = int(selected.sum())
        if total_selected != properties["selected_pixels"]:
            raise ValueError(f"Sampling-mask count differs for {region_id}")
        for interval in interval_bins(year):
            row = {"region_id": region_id, "crop": properties["crop"],
                   "partition_group": properties["partition_group"], **interval,
                   "selected_pixels": total_selected, "valid_pixels": 0,
                   "valid_fraction": 0.0, "status": "missing", "image_path": "", "error": ""}
            try:
                if not versions:
                    raise FileNotFoundError("No compatible combined-cloud-mask download manifest")
                manifest_path, manifest = versions[0]
                record = manifest["files"].get(interval["name"])
                if not record or record.get("status") != "complete":
                    raise FileNotFoundError("Composite has not completed downloading")
                image_path = resolve_image(record, manifest_path, imagery_folder)
                with rasterio.open(image_path) as image:
                    if image.count != 7 or list(image.descriptions) != STUDY_BANDS + ["clear_count"]:
                        raise ValueError("Unexpected band count or band order")
                    if image.crs != mask_crs or image.transform != mask_transform or image.shape != selected.shape:
                        raise ValueError("Sampling mask and composite grids differ")
                    expected_grid = properties["sampling_grid"]
                    if not image.transform.almost_equals(Affine(*expected_grid["transform"])):
                        raise ValueError("Composite transform differs from stored sampling grid")
                    values = image.read(masked=True).astype(float).filled(np.nan)
                spectral, clear_count = values[:6], values[6]
                valid = selected & (clear_count > 0) & np.isfinite(spectral).all(axis=0)
                # Nodata is handled before scaling; counts are never scaled.
                valid &= (spectral != -9999).all(axis=0)
                if np.any(valid & (clear_count != np.floor(clear_count))):
                    raise ValueError("Noninteger clear-date count")
                spectral *= 0.0001
                row.update(image_path=str(image_path), valid_pixels=int(valid.sum()))
                row["valid_fraction"] = row["valid_pixels"] / total_selected
                row["status"] = "quality_pass" if row["valid_fraction"] >= minimum_fraction else "low_coverage"
                ys, xs = np.where(selected)
                for raster_row, column in zip(ys, xs):
                    x, y = mask_transform * (int(column) + 0.5, int(raster_row) + 0.5)
                    entry = {"region_id": region_id, "crop": properties["crop"],
                             "partition_group": properties["partition_group"],
                             "name": interval["name"], "x": x, "y": y,
                             "source_crs": mask_crs.to_string(), "valid": int(valid[raster_row, column]),
                             "clear_count": int(clear_count[raster_row, column]) if np.isfinite(clear_count[raster_row, column]) else 0}
                    for band_index, band in enumerate(STUDY_BANDS):
                        entry[band] = float(spectral[band_index, raster_row, column]) if entry["valid"] else ""
                    pixel_rows.append(entry)
            except Exception as error:
                row["error"] = str(error)
                row["status"] = "missing" if isinstance(error, FileNotFoundError) else "invalid"
                errors.append({"region_id": region_id, "interval": interval["name"], "error": str(error)})
            quality_rows.append(row)
        print(region_id, "quality audit completed")
    write_table(output_folder / "composite_quality.csv", quality_rows,
                ["region_id", "crop", "partition_group", "name", "start", "end", "selected_pixels", "valid_pixels", "valid_fraction", "status", "image_path", "error"])
    write_table(output_folder / "selected_pixel_bands.csv", pixel_rows,
                ["region_id", "crop", "partition_group", "name", "x", "y", "source_crs", "valid", "clear_count"] + STUDY_BANDS)
    save_json(output_folder / "quality_errors.json", errors)
    save_json(output_folder / "quality_settings.json", {"cloud_threshold": cloud_threshold,
              "minimum_valid_fraction": minimum_fraction, "reflectance_scale": 0.0001,
              "status": "Exploratory coverage setting; not calibrated", "bins_audited": len(quality_rows)})
    return quality_rows, pixel_rows


def ratio(left, right):
    denominator = left + right
    return None if not np.isfinite(denominator) or abs(denominator) < 1e-10 else float((left - right) / denominator)


def composite_timelines(quality_rows, pixel_rows, output_folder):
    """Use index-of-composite-bands, then a median across selected pixels."""
    by_bin = defaultdict(list)
    index_rows = []
    for row in pixel_rows:
        if not int(row["valid"]):
            continue
        indices = {"NDVI": ratio(float(row["B8"]), float(row["B4"])),
                   "NDRE": ratio(float(row["B8"]), float(row["B5"])),
                   "NDMI": ratio(float(row["B8"]), float(row["B11"]))}
        by_bin[(row["region_id"], row["name"])].append(indices)
        index_rows.append({**row, **indices})
    timeline_rows = []
    for row in quality_rows:
        start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
        result = {**row, "date": (start + (end - start) // 2).isoformat(),
                  "source_kind": "semi_monthly", "eligible": row["status"] == "quality_pass"}
        for index in ["NDVI", "NDRE", "NDMI"]:
            values = [entry[index] for entry in by_bin[(row["region_id"], row["name"])] if entry[index] is not None]
            result[index] = float(np.median(values)) if values else None
        timeline_rows.append(result)
    write_table(Path(output_folder) / "composite_timelines.csv", timeline_rows,
                ["region_id", "crop", "partition_group", "name", "start", "end", "date", "source_kind", "eligible", "valid_fraction", "valid_pixels", "NDVI", "NDRE", "NDMI", "status"])
    write_table(Path(output_folder) / "selected_pixel_indices.csv", index_rows,
                ["region_id", "crop", "partition_group", "name", "x", "y", "NDVI", "NDRE", "NDMI"])
    return timeline_rows


def plot_timelines(rows, destination, title="Selected-pixel vegetation timeline"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for row in rows:
        groups[row["region_id"]].append(row)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for region_id, entries in groups.items():
        entries.sort(key=lambda row: row["date"])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        dates = [date.fromisoformat(row["date"]) for row in entries]
        for index in ["NDVI", "NDRE", "NDMI"]:
            values = [float(row[index]) if row.get(index) is not None else np.nan for row in entries]
            axes[0].plot(dates, values, ".-", label=index)
        axes[1].plot(dates, [row.get("valid_fraction", 0) for row in entries], ".-", color="black")
        axes[0].legend(); axes[0].set_ylabel("Vegetation index")
        axes[1].set_ylabel("Valid sample fraction"); axes[1].set_ylim(0, 1.05)
        fig.suptitle(title + " | " + region_id)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(destination / (region_id + ".png"), dpi=140)
        plt.close(fig)


DEFAULT_DETECTOR = {"relative_threshold": 0.20, "minimum_amplitude": 0.15,
                    "maximum_onset_gap_days": 30, "baseline_lookback_days": 120,
                    "confirmation_days": 30, "minimum_confirmations": 2,
                    "minimum_points": 6, "minimum_valid_fraction": 0.8,
                    "smoothing_neighbour_days": 15}


def detect_greenup(rows, settings=None):
    """Dispatch by recorded algorithm version so archived results remain reproducible."""
    config = {**DEFAULT_DETECTOR, **(settings or {})}
    method = config.get("cycle_detector", "legacy_local_peaks")
    if method == "legacy_local_peaks":
        return detect_greenup_legacy(rows, settings)
    from full_cycle_detector import METHOD, detect_region
    if method != METHOD:
        raise ValueError("Unknown cycle detector version.")
    if not 0 < config["relative_threshold"] < 1:
        raise ValueError("Relative threshold must lie between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[row["region_id"]].append(row)
    return [result for region_id, source_rows in groups.items()
            for result in detect_region(region_id, source_rows, config)]


def detect_greenup_legacy(rows, settings=None):
    """Candidate cycles only; never turn bin midpoints into observed dates."""
    config = {**DEFAULT_DETECTOR, **(settings or {})}
    if not 0 < config["relative_threshold"] < 1:
        raise ValueError("Relative threshold must lie between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[row["region_id"]].append(row)
    results = []
    for region_id, source_rows in groups.items():
        eligible = [row for row in source_rows if row.get("NDVI") is not None
                    and np.isfinite(float(row["NDVI"]))
                    and row.get("valid_fraction", 0) >= config["minimum_valid_fraction"]]
        eligible.sort(key=lambda row: row["date"])
        if len(eligible) < config["minimum_points"]:
            results.append({"region_id": region_id, "status": "rejected", "reason": "insufficient_supported_observations"})
            continue
        dates = [date.fromisoformat(row["date"]) for row in eligible]
        values = np.array([float(row["NDVI"]) for row in eligible])
        # Restrict smoothing to close neighbours; never bridge long gaps.
        smooth = np.array([np.median(values[max(0, i-1):min(len(values), i+2)][
            [abs((dates[j]-dates[i]).days) <= config["smoothing_neighbour_days"]
             for j in range(max(0, i-1), min(len(values), i+2))]]) for i in range(len(values))])
        peaks = [i for i in range(1, len(values)-1) if smooth[i] > smooth[i-1] and smooth[i] >= smooth[i+1]]
        if not peaks:
            peaks = [int(np.argmax(smooth))]
        region_results = []
        seen_crossings = set()
        previous_peak = 0
        for peak in peaks:
            baseline_indices = [i for i in range(previous_peak, peak) if 0 < (dates[peak]-dates[i]).days <= config["baseline_lookback_days"]]
            previous_peak = peak
            if len(baseline_indices) < 2:
                continue
            baseline = float(np.quantile(smooth[baseline_indices], 0.10))
            amplitude = float(smooth[peak] - baseline)
            if amplitude < config["minimum_amplitude"]:
                continue
            threshold = baseline + config["relative_threshold"] * amplitude
            lows = [i for i in baseline_indices if smooth[i] <= threshold and values[i] <= threshold]
            if not lows:
                continue
            low = lows[-1]
            crossings = [i for i in range(low+1, peak+1) if smooth[i] > threshold and values[i] > threshold]
            if not crossings:
                continue
            crossing = crossings[0]
            if (low, crossing) in seen_crossings:
                continue
            seen_crossings.add((low, crossing))
            confirmations = [i for i in range(crossing+1, len(values)) if 0 < (dates[i]-dates[crossing]).days <= config["confirmation_days"] and values[i] > threshold]
            gap = (dates[crossing]-dates[low]).days
            reason = "supported_rise"
            status = "candidate"
            if gap > config["maximum_onset_gap_days"]:
                status, reason = "rejected", "gap_hides_onset"
            elif len(confirmations) < config["minimum_confirmations"]:
                status, reason = "rejected", "insufficient_sustained_growth"
            observed = all(row.get("source_kind") == "observed_daily" for row in eligible)
            interval_start = eligible[low]["date"] if observed else eligible[low]["start"]
            interval_end = eligible[crossing]["date"] if observed else eligible[crossing]["end"]
            if not observed and status == "candidate":
                status, reason = "provisional", "composite_only_requires_observed_confirmation"
            region_results.append({"region_id": region_id, "partition_group": eligible[0].get("partition_group"),
                "status": status, "reason": reason, "source_kind": "observed_daily" if observed else "semi_monthly",
                "onset_interval_start": interval_start, "onset_interval_end": interval_end,
                "peak_date": dates[peak].isoformat(), "baseline": baseline, "amplitude": amplitude,
                "ndvi_threshold": threshold, "gap_days": gap, "confirmations": len(confirmations),
                "month_boundary_uncertain": interval_start[:7] != interval_end[:7],
                "label_relevance": "not_verified", "validated": False})
        candidates = [result for result in region_results if result["status"] in ["candidate", "provisional"]]
        if len(candidates) > 1:
            for result in candidates:
                result.update(status="ambiguous", reason="multiple_candidate_cycles_require_review")
        results.extend(region_results or [{"region_id": region_id, "status": "rejected", "reason": "no_supported_low_to_growth_transition"}])
    return results


def plot_greenup(rows, results, destination):
    """Draw uncertainty intervals without presenting them as planting dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    outcomes = defaultdict(list)
    for row in rows:
        groups[row["region_id"]].append(row)
    for result in results:
        outcomes[result["region_id"]].append(result)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for region_id, entries in groups.items():
        entries.sort(key=lambda row: row["date"])
        fig, axis = plt.subplots(figsize=(10, 4))
        dates = [date.fromisoformat(row["date"]) for row in entries]
        values = [row.get("NDVI") if row.get("NDVI") is not None else np.nan for row in entries]
        axis.plot(dates, values, ".-", label="NDVI summary")
        for result in outcomes[region_id]:
            if not result.get("onset_interval_start"):
                continue
            left = date.fromisoformat(result["onset_interval_start"])
            right = date.fromisoformat(result["onset_interval_end"])
            axis.axvspan(left, right, alpha=0.15, label=result["status"] + " onset interval")
            axis.axhline(result["ndvi_threshold"], linestyle="--", alpha=0.5)
        axis.set_title(region_id + " | exploratory green-up, not verified planting")
        axis.set_ylabel("NDVI"); axis.legend(fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(destination / (region_id + ".png"), dpi=140)
        plt.close(fig)
