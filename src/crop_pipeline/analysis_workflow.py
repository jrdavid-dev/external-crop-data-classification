"""Immutable window handoffs for the explicit pilot and strict notebook workflows.

Pilot mode waives collection budgets only. Complete observations, qualified timing,
source identities and locked parcel partitions remain required in either mode.
"""

import json
from pathlib import Path

from crop_pipeline.common import save_json
from crop_pipeline.features import experiments
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY
from crop_pipeline.sources import adaptive
from crop_pipeline.training import exploratory
from crop_pipeline.windows import seasons

MODES = {"pilot", "strict"}
WINDOW_FILES = {"window_features.csv", "configuration.json", "coverage.json", "provenance.json"}


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError("analysis_mode must be pilot or strict")
    return mode


def require_collection_readiness(report_folder, mode):
    """Report measured deficits; only strict mode requires collection targets."""
    validate_mode(mode)
    campaign = adaptive.root(report_folder)
    feedback_path = campaign / "feedback.json"
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else {}
    if mode == "strict":
        adaptive.require_targets(report_folder)
    return {
        "analysis_mode": mode,
        "targets_met": feedback.get("targets_met") is True,
        "counts": feedback.get("counts", {}),
        "collection_budget_required": mode == "strict",
        "split_path": str(campaign / "locked_split.json"),
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": seasons.file_sha256(path)}


def validate_coverage(coverage, mode):
    if coverage.get("imagery_request_version") != REQUEST_VERSION or coverage.get("resampling") != RESAMPLING_POLICY:
        raise ValueError("Window handoff requires the current bilinear imagery contract; do not relabel old data")
    minimum = coverage.get("minimum_valid_fraction")
    value = seasons.finite_number(minimum)
    if isinstance(minimum, bool) or value is None or not 0.8 <= value <= 1:
        raise ValueError("Window coverage must record a minimum valid fraction between 0.8 and 1")
    recipe = coverage.get("window_recipe", {})
    if not isinstance(recipe, dict):
        raise ValueError("Window recipe must be a recorded settings object")
    if "minimum_valid_fraction" in recipe:
        recipe_minimum = recipe["minimum_valid_fraction"]
        if isinstance(recipe_minimum, bool) or seasons.finite_number(recipe_minimum) != value:
            raise ValueError("Window recipe and coverage minimum valid fractions differ")
    if mode == "strict" and coverage.get("targets_met") is not True:
        raise ValueError("Strict analysis requires measured collection targets to be met")


def _complete_windows(rows):
    """Check the saved complete-window contract without filling or changing values."""
    result = []
    for row in rows:
        row = dict(row)
        required = ("crop", "partition_group", "source_crs", "x", "y", "window_start", "window_end", "label_status")
        if any(row.get(name) in (None, "") for name in required):
            raise ValueError("Window rows need complete crop, parcel, coordinate, date and label identities")
        for name in ("x", "y"):
            value = seasons.finite_number(row[name])
            if value is None:
                raise ValueError("Window coordinates must be finite")
            row[name] = value
        if row["label_status"] not in {"confirmed", "reported"}:
            raise ValueError("Window label confidence must retain its supported source status")
        if row.get("window_status") not in {"reference_supported", "satellite_estimated"}:
            raise ValueError("Every window needs its original reference-supported or satellite-estimated status")
        if not seasons.timing_is_qualified(row):
            raise ValueError("Unqualified timing cannot enter the pilot or strict window handoff")
        if str(row.get("reconstructed", "")).strip().lower() not in {"", "false", "0", "none"}:
            raise ValueError("Reconstructed windows cannot enter a complete-only model handoff")
        if row.get("missing_bins"):
            raise ValueError("Incomplete windows cannot enter a model handoff")
        intervals = seasons.window_intervals(row)
        if (intervals[0]["start"] != row["window_start"]
                or intervals[-1]["end"] != row["window_end"]
                or row["window_start"][8:] != "01" or row["window_end"][8:] != "01"):
            raise ValueError("Windows must span four calendar months with eight complete half-month bins")
        for column in seasons.feature_columns():
            value = seasons.finite_number(row.get(column))
            if value is None:
                raise ValueError(f"Incomplete or nonfinite base feature: {column}")
            row[column] = value
        row["season_key"] = row.get("season_key") or row.get("season_id")
        if not row["season_key"]:
            raise ValueError("Every window needs a stable season identity")
        result.append(row)
    return result


def _prepare_rows(rows, split):
    if split.get("kind") != "exploratory_existing_source_parcel_evaluation":
        raise ValueError("Use the saved locked source-parcel split")
    previous = split.get("group_assignments", {})
    groups = {row["partition_group"] for row in rows}
    if groups - previous.keys():
        raise ValueError("Window parcels are absent from the locked split; do not assign new holdouts here")
    for row in rows:
        if row.get("partition") and row["partition"] != previous[row["partition_group"]]:
            raise ValueError("Window partition differs from the locked parcel assignment")
    unpartitioned = [{key: value for key, value in row.items() if key != "partition"} for row in rows]
    assignments = exploratory.parcel_partitions(unpartitioned, previous)
    prepared = [exploratory.prepare_identity(row, assignments[row["partition_group"]]) for row in unpartitioned]
    exploratory.assert_partition_separation({name: [row for row in prepared if row["partition"] == name]
                                            for name in ("train", "validation", "test")})
    for row in prepared:
        row["sample_id"] = experiments.sample_identifier(row)
    if len({row["sample_id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate pixel-window identities in window inputs")
    return sorted(prepared, key=lambda row: row["sample_id"])


def export_window_handoff(reference_paths, output_root, split_path, coverage_path,
                          analysis_mode="pilot", collection_name="adaptive_training", source_inputs_path=None):
    """Publish selected windows under an input-keyed directory, never replace old runs."""
    validate_mode(analysis_mode)
    reference_paths = list(map(Path, reference_paths))
    if not reference_paths:
        raise ValueError("No selected complete-window tables were supplied")
    coverage = _read_json(coverage_path)
    validate_coverage(coverage, analysis_mode)
    raw = [row for path in reference_paths for row in seasons.read_table(path)]
    rows = _prepare_rows(_complete_windows(raw), _read_json(split_path))
    provenance = {
        "window_inputs": [_file_record(path) for path in reference_paths],
        "locked_split": _file_record(split_path),
        "coverage": _file_record(coverage_path),
        "source_inputs": _file_record(source_inputs_path) if source_inputs_path is not None else None,
        "source_input_snapshot": _read_json(source_inputs_path) if source_inputs_path is not None else None,
        "implementation_sha256": seasons.file_sha256(Path(__file__)),
    }
    configuration = {
        "analysis_mode": analysis_mode,
        "collection_name": collection_name,
        "purpose": f"external_parcel_{analysis_mode}",
        "backend": "scikit_learn",
        "frozen": False,
        "study_classes": list(exploratory.CLASSES),
        "window_policy": "reference_supported_and_qualified_estimates",
        "window_recipe": coverage.get("window_recipe", {"window_rule": "four_calendar_months", "half_month_bins": 8,
                                                          "missing_data_policy": "complete_only",
                                                          "minimum_valid_fraction": coverage["minimum_valid_fraction"]}),
        "imagery_policy": "verified_current_bilinear_annual_sources",
        "split_policy": {**exploratory.SPLIT_POLICY, "assignments": "preserved_locked_source_parcels"},
        "confidence_weights": {"confirmed": 1.0, "reported": 0.5},
        "forest_parameters": {"n_estimators": 1000, "random_state": 42, "max_features": "sqrt",
                              "min_samples_leaf": 2, "bootstrap": True, "max_samples": 0.5, "n_jobs": -1},
        "weighting": "equal_crop_then_equal_source_parcel_confidence_within_parcel_v1",
        "weight_normalization": "mean_one_preserves_relative_weights_and_half_sample_bootstrap",
        "limitations": ["Held-out source parcels are not independent Paoay ground truth.",
                        "Satellite-estimated timing does not independently verify crop-cycle relevance.",
                        "Source parcel groups are identity proxies, not independently verified fields.",
                        "Pilot mode permits unmet collection budgets; quality and split requirements remain unchanged."],
    }
    key = seasons.record_fingerprint({"provenance": provenance, "configuration": configuration})[:20]
    destination = Path(output_root) / key
    manifest_path = destination / "window_handoff.json"
    if manifest_path.exists():
        load_window_handoff(manifest_path)
        return manifest_path
    seasons.write_records(destination / "window_features.csv", rows)
    for name, value in (("configuration.json", configuration), ("coverage.json", coverage), ("provenance.json", provenance)):
        save_json(destination / name, value)
    manifest = {
        "format_version": 1, "kind": "selected_four_month_windows", "status": "prepared",
        "analysis_mode": analysis_mode, "row_count": len(rows),
        "sample_fingerprint": seasons.record_fingerprint([row["sample_id"] for row in rows]),
        "files": {name: {"sha256": seasons.file_sha256(destination / name)} for name in sorted(WINDOW_FILES)},
    }
    save_json(manifest_path, manifest)
    return manifest_path


def load_window_handoff(path):
    """Verify the portable snapshot before reading its windows and partitions."""
    path = Path(path)
    manifest = _read_json(path)
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "selected_four_month_windows"
            or manifest.get("status") != "prepared" or set(manifest.get("files", {})) != WINDOW_FILES):
        raise ValueError("Unsupported window handoff or incomplete file inventory")
    for name, record in manifest["files"].items():
        if seasons.file_sha256(path.parent / name) != record["sha256"]:
            raise ValueError(f"Window handoff file changed: {name}")
    configuration, coverage = (_read_json(path.parent / name) for name in ("configuration.json", "coverage.json"))
    mode = validate_mode(configuration.get("analysis_mode"))
    if mode != manifest.get("analysis_mode"):
        raise ValueError("Window analysis mode differs between configuration and manifest")
    validate_coverage(coverage, mode)
    rows = _complete_windows(seasons.read_table(path.parent / "window_features.csv"))
    if len(rows) != manifest["row_count"] or seasons.record_fingerprint([row["sample_id"] for row in rows]) != manifest["sample_fingerprint"]:
        raise ValueError("Window sample identities differ from the exported manifest")
    exploratory.assert_partition_separation({name: [row for row in rows if row["partition"] == name]
                                            for name in ("train", "validation", "test")})
    for row in rows:
        if row["sample_id"] != experiments.sample_identifier(row):
            raise ValueError("Window sample identity differs from its coordinates or season")
    return rows, configuration, coverage, manifest


def write_stage_pointer(pointer_path, manifest_path=None, status="prepared", reason=None):
    """Record the current stage explicitly; a pending stage cannot fall back to old outputs."""
    if status not in {"prepared", "pending"}:
        raise ValueError("Stage status must be prepared or pending")
    if status == "prepared" and manifest_path is None:
        raise ValueError("Prepared stage requires a manifest")
    record = {"format_version": 1, "status": status}
    if status == "prepared":
        record.update(_file_record(manifest_path))
    else:
        if not reason:
            raise ValueError("Pending stage requires a reason")
        record["reason"] = reason
    save_json(pointer_path, record)
    return Path(pointer_path)


def resolve_stage_pointer(pointer_path):
    path = Path(pointer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage pointer missing: {path}. Finish the previous notebook first")
    record = _read_json(path)
    if record.get("format_version") != 1 or record.get("status") not in {"prepared", "pending"}:
        raise ValueError("Unsupported stage pointer")
    if record["status"] == "pending":
        raise RuntimeError("Previous notebook stage is pending: " + record.get("reason", "inspect its summary"))
    manifest = Path(record["path"])
    if seasons.file_sha256(manifest) != record["sha256"]:
        raise ValueError("Stage manifest changed; rerun the previous notebook")
    return manifest
