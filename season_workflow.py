"""Review field-seasons, build complete features, and compare training sources.

Human evidence and field identities are inputs: satellite curves cannot supply
them. All dates use start-inclusive, end-exclusive calendar intervals.
"""

import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from crop_workflow import interval_bins, ratio, save_json, write_table


FEATURE_BANDS = ("B3", "B4", "B5", "B8", "B11")
FEATURE_INDICES = ("NDVI", "NDRE", "NDMI")
REVIEW_COLUMNS = [
    "candidate_id", "region_id", "onset_interval_start", "onset_interval_end",
    "status", "reason", "source_kind", "peak_date", "gap_days", "confirmations",
    "decision", "season_id", "crop", "label_status", "label_evidence",
    "timing_evidence", "cycle_evidence", "reviewer", "review_notes",
    "resolved_onset_month", "month_resolution_evidence", "reference_greenup_start",
    "reference_greenup_end", "detector_fingerprint",
]
FIELD_COLUMNS = [
    "partition_group", "field_id", "source", "partition", "identity_evidence",
]


def read_table(path):
    """Read CSV records without silently converting identifiers into numbers."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def record_fingerprint(record):
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def identify_candidates(candidates):
    """Bind review decisions to exact detector outputs, including parameters' effects."""
    identified = [{**candidate, "candidate_id": record_fingerprint(candidate)}
                  for candidate in candidates]
    identifiers = [candidate["candidate_id"] for candidate in identified]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate detector outcomes; remove duplicate inputs first.")
    return identified


def prepare_review_files(candidates, regions, detector_settings, output_folder):
    """Write fresh templates separately so reruns cannot overwrite human decisions."""
    output_folder = Path(output_folder)
    detector_fingerprint = record_fingerprint(detector_settings)
    properties_by_region = {region["properties"]["fragment_id"]: region["properties"]
                            for region in regions}
    review_rows = []
    for candidate in identify_candidates(candidates):
        properties = properties_by_region[candidate["region_id"]]
        review_rows.append({**candidate, "crop": properties["crop"],
                            "decision": "pending", "label_status": "unknown",
                            "detector_fingerprint": detector_fingerprint})
    write_table(output_folder / "season_reviews.template.csv", review_rows, REVIEW_COLUMNS)
    groups = sorted({properties["partition_group"]
                     for properties in properties_by_region.values()})
    write_table(output_folder / "field_registry.template.csv",
                [{"partition_group": group} for group in groups], FIELD_COLUMNS)
    configuration = {
        "frozen": False, "detector_parameters": detector_settings,
        "development_evidence": "", "dataset_revisions": {}, "source_terms": {},
        "map_reference_period": "", "study_classes": ["carrot", "potato"],
        "window_rule": "onset_month_plus_three", "missing_data_policy": "complete_only",
        "minimum_valid_fraction": 0.8, "random_seed": 42,
        "forest_parameters": {"n_estimators": 300, "min_samples_leaf": 2,
                              "class_weight": None, "n_jobs": -1},
        "confidence_weights": {"confirmed": 1.0, "reported": 0.5},
        "limitations": ["Cabbage has no configured verified source.",
                        "Green-up is not a verified planting date."],
    }
    save_json(output_folder / "frozen_configuration.template.json", configuration)
    return review_rows


def validate_configuration(configuration):
    """Require an explicit development freeze before accepting training seasons."""
    if configuration.get("frozen") is not True or not configuration.get("development_evidence"):
        raise ValueError("Freeze settings with development_evidence before accepting seasons.")
    if configuration.get("window_rule") != "onset_month_plus_three":
        raise ValueError("Only the four-calendar-month onset rule is implemented.")
    if configuration.get("missing_data_policy") != "complete_only":
        raise ValueError("Only complete-data features are accepted; no imputation is applied.")
    minimum_fraction = configuration["minimum_valid_fraction"]
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1].")
    classes = configuration["study_classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Provide at least two distinct study_classes.")
    weights = configuration["confidence_weights"]
    if set(weights) != {"confirmed", "reported"}:
        raise ValueError("Provide confidence weights for confirmed and reported labels only.")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("Confidence weights must be finite and positive.")


def add_calendar_months(first_day, month_count):
    month_number = first_day.year * 12 + first_day.month - 1 + month_count
    return date(month_number // 12, month_number % 12 + 1, 1)


def review_problem(review, candidate, properties, configuration):
    """Return one actionable exclusion reason instead of guessing missing evidence."""
    if review.get("decision") != "accept":
        return "review_" + (review.get("decision") or "pending")
    if candidate.get("status") == "rejected":
        return "detector_rejected_requires_new_detection"
    required = ("season_id", "reviewer", "review_notes", "label_evidence",
                "timing_evidence", "cycle_evidence")
    if any(not review.get(column, "").strip() for column in required):
        return "missing_review_evidence"
    if review.get("detector_fingerprint") != record_fingerprint(configuration["detector_parameters"]):
        return "review_uses_different_detector_settings"
    if review.get("crop") != properties["crop"] or review["crop"] not in configuration["study_classes"]:
        return "crop_conflict_or_outside_study"
    if review.get("label_status") not in configuration["confidence_weights"]:
        return "unsupported_label_status"
    return ""


def choose_onset_month(review, candidate):
    onset_start = date.fromisoformat(candidate["onset_interval_start"])
    onset_end = date.fromisoformat(candidate["onset_interval_end"])
    if onset_end < onset_start:
        raise ValueError("Onset interval ends before it starts.")
    first_month = onset_start.replace(day=1)
    last_month = onset_end.replace(day=1)
    resolved_month = review.get("resolved_onset_month", "")
    if first_month == last_month and not resolved_month:
        return first_month
    if not resolved_month or not review.get("month_resolution_evidence", "").strip():
        raise ValueError("Month uncertainty requires resolved_onset_month and supporting evidence.")
    selected_month = date.fromisoformat(resolved_month + "-01")
    if not first_month <= selected_month <= last_month:
        raise ValueError("Resolved month lies outside the candidate onset interval.")
    return selected_month


def timing_validation(review, candidate):
    """Compare green-up intervals only when independent green-up dates are supplied."""
    reference_start = review.get("reference_greenup_start", "")
    reference_end = review.get("reference_greenup_end", "")
    if not reference_start and not reference_end:
        return {"timing_validation": "review_only", "interval_distance_days": ""}
    reference_start = date.fromisoformat(reference_start)
    reference_end = date.fromisoformat(reference_end)
    if reference_end < reference_start:
        raise ValueError("Independent green-up interval ends before it starts.")
    candidate_start = date.fromisoformat(candidate["onset_interval_start"])
    candidate_end = date.fromisoformat(candidate["onset_interval_end"])
    distance = max(0, (candidate_start - reference_end).days,
                   (reference_start - candidate_end).days)
    return {"timing_validation": "independent_greenup_interval",
            "interval_distance_days": distance}


def accept_seasons(candidates, reviews, regions, configuration):
    """Apply Steps 9–10 and retain an audit row for every detector outcome."""
    validate_configuration(configuration)
    identified = identify_candidates(candidates)
    candidate_identifiers = {candidate["candidate_id"] for candidate in identified}
    review_map = unique_records(reviews, "candidate_id")
    if set(review_map) - candidate_identifiers:
        raise ValueError("Stale reviews do not match current detector outcomes.")
    properties_by_region = {region["properties"]["fragment_id"]: region["properties"]
                            for region in regions}
    accepted, audit = [], []
    for candidate in identified:
        review = review_map.get(candidate["candidate_id"], {})
        properties = properties_by_region[candidate["region_id"]]
        problem = review_problem(review, candidate, properties, configuration)
        if problem:
            audit.append({**review, **candidate, "crop": properties["crop"],
                          "partition_group": properties["partition_group"],
                          "decision": "excluded", "exclusion_reason": problem})
            continue
        season, outcome = construct_season(candidate, review, properties)
        audit.append(outcome)
        if season is not None:
            accepted.append(season)
    unique_records(accepted, "season_key")
    return accepted, audit


def construct_season(candidate, review, properties):
    try:
        window_start = choose_onset_month(review, candidate)
        validation = timing_validation(review, candidate)
    except (ValueError, KeyError) as error:
        return None, {**review, **candidate, "decision": "excluded", "exclusion_reason": str(error)}
    window_end = add_calendar_months(window_start, 4)
    season = {**review, **candidate, **validation, "crop": properties["crop"],
              "partition_group": properties["partition_group"],
              "season_id": review["season_id"],
              "season_key": candidate["region_id"] + "|" + review["season_id"],
              "window_start": window_start.isoformat(), "window_end": window_end.isoformat(),
              "selected_pixels": properties["selected_pixels"],
              "month_boundary_uncertain": candidate["onset_interval_start"][:7]
              != candidate["onset_interval_end"][:7]}
    return season, {**season, "decision": "accepted", "exclusion_reason": ""}


def unique_records(records, key):
    result = {}
    for record in records:
        identifier = record[key]
        if not identifier or identifier in result:
            raise ValueError(f"Empty or duplicate {key}: {identifier!r}")
        result[identifier] = record
    return result


def window_intervals(season):
    first_day = date.fromisoformat(season["window_start"])
    end_day = date.fromisoformat(season["window_end"])
    intervals = []
    for year in range(first_day.year, end_day.year + 1):
        intervals.extend(interval for interval in interval_bins(year)
                         if season["window_start"] <= interval["start"] < season["window_end"])
    if len(intervals) != 8:
        raise ValueError("A season must contain eight consecutive calendar half-month bins.")
    return intervals


def feature_columns(include_indices=True):
    variables = FEATURE_BANDS + FEATURE_INDICES if include_indices else FEATURE_BANDS
    return [f"bin_{bin_number:02d}_{variable}"
            for bin_number in range(1, 9) for variable in variables]


def shifted_window_intervals(season, bin_shift):
    """Keep eight real calendar bins when shifting the development window."""
    primary_intervals = window_intervals(season)
    if bin_shift not in {-1, 0, 1}:
        raise ValueError("Window sensitivity supports shifts of -1, 0 or 1 bin.")
    first_day = date.fromisoformat(primary_intervals[0]["start"])
    first_bin_number = first_day.year * 24 + (first_day.month - 1) * 2 + bin_shift
    intervals = []
    for bin_offset in range(8):
        bin_number = first_bin_number + bin_offset
        intervals.append(interval_bins(bin_number // 24)[bin_number % 24])
    return intervals


def finite_number(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) and numeric_value != -9999 else None


def pixel_features(pixel):
    if pixel is None or str(pixel.get("valid")) not in {"1", "True"}:
        return None
    clear_count = finite_number(pixel.get("clear_count"))
    if clear_count is None or clear_count < 1 or not clear_count.is_integer():
        return None
    bands = {band: finite_number(pixel.get(band)) for band in FEATURE_BANDS}
    if any(value is None for value in bands.values()):
        return None
    indices = {"NDVI": ratio(bands["B8"], bands["B4"]),
               "NDRE": ratio(bands["B8"], bands["B5"]),
               "NDMI": ratio(bands["B8"], bands["B11"])}
    if any(value is None or not math.isfinite(value) for value in indices.values()):
        return None
    return {**bands, **indices, "clear_count": int(clear_count)}


def coordinate_key(pixel):
    # Numeric normalization makes CSV round trips preserve spatial identities.
    return (pixel["source_crs"], float(pixel["x"]), float(pixel["y"]))


def index_pixels(pixel_rows):
    indexed = defaultdict(dict)
    for pixel in pixel_rows:
        identifier = (pixel["region_id"], pixel["name"])
        location = coordinate_key(pixel)
        if location in indexed[identifier]:
            raise ValueError(f"Duplicate pixel in region/bin: {identifier}, {location}")
        indexed[identifier][location] = pixel
    return indexed


def quality_problem(quality, minimum_fraction):
    if quality is None:
        return "missing_composite"
    fraction = finite_number(quality.get("valid_fraction"))
    if quality.get("status") != "quality_pass" or fraction is None or fraction < minimum_fraction:
        return "failed_composite_quality"
    return ""


def build_pixel_window(season, location, intervals, indexed_pixels, quality_map, minimum_fraction):
    features, missing_bins = {}, []
    for bin_number, interval in enumerate(intervals, start=1):
        identifier = (season["region_id"], interval["name"])
        values = pixel_features(indexed_pixels.get(identifier, {}).get(location))
        problem = quality_problem(quality_map.get(identifier), minimum_fraction)
        if problem or values is None:
            missing_bins.append(interval["name"])
            continue
        features.update({f"bin_{bin_number:02d}_{variable}": value
                         for variable, value in values.items()})
    record = {**season, "source_crs": location[0], "x": location[1], "y": location[2],
              "reconstructed": False, "missing_bins": ";".join(missing_bins), **features}
    return record, missing_bins


def build_features(seasons, pixel_rows, quality_rows, minimum_fraction=0.8, bin_shift=0):
    """Steps 11–12: select existing medians; retain only complete 64-feature rows."""
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0, 1].")
    indexed_pixels = index_pixels(pixel_rows)
    quality_map = unique_records([{**row, "identifier": (row["region_id"], row["name"])}
                                  for row in quality_rows], "identifier")
    accepted, exclusions, losses, bin_quality = [], [], [], []
    for season in seasons:
        results = build_season_features(season, indexed_pixels, quality_map, minimum_fraction, bin_shift)
        accepted.extend(results[0])
        exclusions.extend(results[1])
        losses.append(results[2])
        bin_quality.extend(results[3])
    return accepted, exclusions, losses, bin_quality


def build_season_features(season, indexed_pixels, quality_map, minimum_fraction, bin_shift=0):
    intervals = shifted_window_intervals(season, bin_shift)
    season = {**season, "primary_window_start": season["window_start"],
              "primary_window_end": season["window_end"],
              "window_variant": "primary" if bin_shift == 0 else f"shift_{bin_shift:+d}_bin",
              "window_start": intervals[0]["start"], "window_end": intervals[-1]["end"]}
    locations = set()
    for identifier, pixels in indexed_pixels.items():
        if identifier[0] == season["region_id"]:
            locations.update(pixels)
    selected_count = int(season["selected_pixels"])
    if len(locations) > selected_count:
        raise ValueError("Pixel locations exceed the sampling mask count; check grid consistency.")
    accepted, exclusions = [], []
    missing_counts = Counter()
    for location in sorted(locations):
        record, missing_bins = build_pixel_window(
            season, location, intervals, indexed_pixels, quality_map, minimum_fraction)
        missing_counts.update(missing_bins)
        if missing_bins:
            exclusions.append({**record, "exclusion_reason": "incomplete_features"})
            continue
        accepted.append(record)
    unlocated_count = selected_count - len(locations)
    metadata = {column: season[column] for column in
                ("season_key", "region_id", "partition_group", "crop", "selected_pixels",
                 "window_variant", "window_start", "window_end")}
    losses = {**metadata, "accepted_pixels": len(accepted),
              "excluded_pixels": selected_count - len(accepted),
              "pixels_without_records": unlocated_count}
    bin_quality = [{**metadata, **interval,
                    "usable_pixels": selected_count - missing_counts[interval["name"]] - unlocated_count,
                    "exclusion_reason": quality_problem(
                        quality_map.get((season["region_id"], interval["name"])), minimum_fraction)}
                   for interval in intervals]
    return accepted, exclusions, losses, bin_quality


def summarize_coverage_thresholds(timeline_rows, thresholds=(0.7, 0.8, 0.9)):
    """Compare clear-pixel coverage on saved summaries without downloading imagery.

    This changes the coverage acceptance rule only, not acquisition cloud masks.
    Original row flags are preserved separately from the experimental decision.
    """
    decisions, grouped_decisions = [], defaultdict(list)
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("Coverage thresholds must be in (0, 1].")
        for row in timeline_rows:
            fraction = finite_number(row.get("valid_fraction"))
            accepted = (fraction is not None and fraction >= threshold
                        and finite_number(row.get("NDVI")) is not None)
            decision = {**row, "coverage_threshold": threshold,
                        "coverage_experiment_accepted": accepted}
            decisions.append(decision)
            grouped_decisions[(threshold, row.get("crop", "unknown"), row["region_id"])].append(decision)
    summary = [{"coverage_threshold": key[0], "crop": key[1], "region_id": key[2],
                "observations": len(entries),
                "accepted": sum(row["coverage_experiment_accepted"] for row in entries),
                "excluded": sum(not row["coverage_experiment_accepted"] for row in entries)}
               for key, entries in sorted(grouped_decisions.items())]
    return decisions, summary


def validate_field_registry(registry):
    """Explicit identities allow fragments and seasons to share one partition."""
    groups = unique_records(registry, "partition_group")
    assignments = {}
    for record in registry:
        if any(not record.get(column, "").strip() for column in FIELD_COLUMNS):
            raise ValueError("Every field registry column requires a value and identity evidence.")
        if record["source"] not in {"external", "local"}:
            raise ValueError("Field source must be external or local.")
        if record["partition"] not in {"train", "validation", "test"}:
            raise ValueError("Field partition must be train, validation or test.")
        assignment = (record["source"], record["partition"])
        previous = assignments.setdefault(record["field_id"], assignment)
        if previous != assignment:
            raise ValueError("An independent field crosses sources or partitions.")
        if record["partition"] == "test" and record["source"] != "local":
            raise ValueError("Final test fields must be independent local Benguet fields.")
    return groups


def development_region_ids(regions, registry):
    """Keep optional parameter checks away from reserved final test fields."""
    groups = validate_field_registry(registry)
    return {region["properties"]["fragment_id"] for region in regions
            if groups.get(region["properties"]["partition_group"], {}).get("partition")
            in {"train", "validation"}}


def assign_partitions(feature_rows, registry):
    """Step 13: require reviewed identities; never randomly split correlated pixels."""
    groups = validate_field_registry(registry)
    partitioned, seen_locations = [], {}
    for row in feature_rows:
        if row["partition_group"] not in groups:
            raise ValueError(f"Field registry missing {row['partition_group']}")
        identity = groups[row["partition_group"]]
        location = coordinate_key(row)
        previous_field = seen_locations.setdefault(location, identity["field_id"])
        if previous_field != identity["field_id"]:
            raise ValueError("A duplicate location has conflicting independent-field identities.")
        if identity["partition"] == "test" and row["label_status"] != "confirmed":
            raise ValueError("Final test rows require confirmed field-season crop labels.")
        partitioned.append({**row, **identity})
    identifiers = [(row["field_id"], row["season_id"], coordinate_key(row)) for row in partitioned]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate field-season pixels; deduplicate footprints before training.")
    return partitioned


def field_sample_weights(rows, confidence_weights):
    """Share one confidence budget across all pixels and seasons of each field."""
    field_counts = Counter(row["field_id"] for row in rows)
    return [confidence_weights[row["label_status"]] / field_counts[row["field_id"]]
            for row in rows]


def sample_counts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["partition"], row["crop"], row["label_status"])].append(row)
    return [{"source": key[0], "partition": key[1], "crop": key[2], "label_status": key[3],
             "pixels": len(entries), "independent_fields": len({row["field_id"] for row in entries})}
            for key, entries in sorted(groups.items())]


def require_class_coverage(rows, classes, description):
    present = {row["crop"] for row in rows}
    if present != set(classes):
        raise ValueError(f"{description} needs exactly the configured crops: {classes}; found {sorted(present)}")


def train_comparisons(rows, configuration, output_folder, include_indices=True):
    """Step 14: fit three fresh forests against the same untouched local test rows."""
    validate_configuration(configuration)
    if not rows:
        raise ValueError("No complete, reviewed feature rows are available for training.")
    # Revalidate imported CSV tables, not only tables built in this process.
    registry = {row["partition_group"]: {column: row[column] for column in FIELD_COLUMNS}
                for row in rows}
    validated_rows = assign_partitions(rows, list(registry.values()))
    if validated_rows != rows:
        raise ValueError("Inconsistent field metadata in training rows.")
    classes = configuration["study_classes"]
    test_rows = [row for row in rows if row["partition"] == "test"]
    external_rows = [row for row in rows if row["partition"] == "train" and row["source"] == "external"]
    local_rows = [row for row in rows if row["partition"] == "train" and row["source"] == "local"]
    training_sets = {"external_only": external_rows, "local_only": local_rows,
                     "combined": external_rows + local_rows}
    require_class_coverage(test_rows, classes, "Independent local test set")
    for name, training_rows in training_sets.items():
        require_class_coverage(training_rows, classes, name)
    columns = feature_columns(include_indices)
    for row in rows:
        if row.get("label_status") not in configuration["confidence_weights"]:
            raise ValueError("Training rows require confirmed or reported labels.")
        if any(finite_number(row.get(column)) is None for column in columns):
            raise ValueError("Training tables must contain complete finite features.")
    return fit_comparisons(training_sets, test_rows, configuration, Path(output_folder), columns)


def fit_comparisons(training_sets, test_rows, configuration, output_folder, columns):
    # Keep scikit-learn optional for review/feature steps on data-preparation PCs.
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

    classes = configuration["study_classes"]
    test_features = [[float(row[column]) for column in columns] for row in test_rows]
    test_labels = [row["crop"] for row in test_rows]
    output_folder.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, training_rows in training_sets.items():
        classifier = RandomForestClassifier(
            random_state=configuration["random_seed"], **configuration["forest_parameters"])
        classifier.fit([[float(row[column]) for column in columns] for row in training_rows],
                       [row["crop"] for row in training_rows],
                       sample_weight=field_sample_weights(training_rows, configuration["confidence_weights"]))
        predictions = classifier.predict(test_features)
        precision, recall, scores, support = precision_recall_fscore_support(
            test_labels, predictions, labels=classes, zero_division=0)
        metrics[name] = {
            "classes": classes, "confusion_matrix": confusion_matrix(test_labels, predictions, labels=classes).tolist(),
            "macro_f1": float(scores.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
            "per_crop": {crop: {"precision": float(precision[index]), "recall": float(recall[index]),
                                "f1": float(scores[index]), "support": int(support[index])}
                         for index, crop in enumerate(classes)},
            "training_pixels": len(training_rows),
            "training_fields": len({row["field_id"] for row in training_rows}),
            "test_pixels": len(test_rows), "test_fields": len({row["field_id"] for row in test_rows}),
        }
        joblib.dump({"classifier": classifier, "feature_columns": columns,
                     "configuration": configuration}, output_folder / f"{name}.joblib")
        write_records(output_folder / f"{name}_predictions.csv",
                      [{**row, "prediction": str(prediction)}
                       for row, prediction in zip(test_rows, predictions)])
    save_json(output_folder / "metrics.json", metrics)
    save_json(output_folder / "model_settings.json", {"configuration": configuration,
              "feature_columns": columns, "evaluation_unit": "pixel; correlated within fields",
              "test_used_for_tuning": False})
    return metrics


def write_records(path, records):
    columns = list(dict.fromkeys(column for record in records for column in record))
    write_table(path, records, columns or ["status"])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_run(output_folder, configuration, input_paths, artifact_paths, coverage):
    """Step 15: inventory actual files; never imply an unrun export completed."""
    output_folder = Path(output_folder)
    inputs = artifact_inventory(input_paths)
    artifacts = artifact_inventory(artifact_paths)
    packages = {}
    for package in ("numpy", "rasterio", "scikit-learn", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(), "packages": packages,
                "configuration": configuration, "configuration_fingerprint": record_fingerprint(configuration),
                "inputs": inputs, "artifacts": artifacts, "coverage": coverage,
                "missing_files": [record["path"] for record in inputs + artifacts if not record["exists"]],
                "mapping_status": "No wall-to-wall mapping or imagery export is performed by Steps 9–15."}
    save_json(output_folder / "run_manifest.json", manifest)
    summary = ("# Run record\n\n"
               f"- Created: {manifest['created_at_utc']}\n"
               f"- Settings frozen: {configuration.get('frozen', False)}\n"
               f"- Missing input/output files: {len(manifest['missing_files'])}\n\n"
               "## Coverage\n\n```json\n" + json.dumps(coverage, indent=2) + "\n```\n\n"
               "## Interpretation\n\nReview `run_manifest.json` for settings, file hashes, source terms, "
               "revisions and limitations. Pixel metrics describe only accepted complete windows. "
               "They do not establish full-map accuracy or independently verify planting dates.\n")
    (output_folder / "RUN_README.md").write_text(summary, encoding="utf-8")
    return manifest


def artifact_inventory(paths):
    return [{"path": str(Path(path).resolve()), "exists": Path(path).is_file(),
             "sha256": file_sha256(path) if Path(path).is_file() else None,
             "bytes": Path(path).stat().st_size if Path(path).is_file() else None}
            for path in paths]
