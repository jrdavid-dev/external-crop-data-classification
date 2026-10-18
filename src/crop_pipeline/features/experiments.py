"""Derive comparable SAVI and temporal-shape designs from selected pixel windows."""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from crop_pipeline.windows import seasons as season_workflow
from crop_pipeline.imagery.quality import save_json
from crop_pipeline.imagery.retrieval import REQUEST_VERSION, RESAMPLING_POLICY

INDEX_NAMES = ("NDVI", "NDRE", "NDMI", "SAVI")
FEATURE_POLICY = {
    "version": 1,
    "savi_soil_adjustment": 0.5,
    "time_axis": "actual_calendar_bin_midpoints_in_days",
    "differences": "consecutive_index_value_differences",
    "phase_slopes": "least_squares_before_and_after_first_maximum_per_day",
    "boundary_peak": "missing_phase_slope_zero_with_observed_phase_metadata",
    "integral": "trapezoidal_between_first_and_last_bin_midpoints",
    "peak_interval": "one_based_first_maximum",
}
FEATURE_DESIGNS = ("interval_only", "interval_and_shape", "shape_only")


def interval_columns():
    variables = season_workflow.FEATURE_BANDS + INDEX_NAMES
    return [
        f"bin_{bin_number:02d}_{variable}" for bin_number in range(1, 9) for variable in variables
    ]


def shape_columns():
    return [column for index_name in INDEX_NAMES for column in index_shape_columns(index_name)]


def index_shape_columns(index_name):
    prefix = f"shape_{index_name}"
    differences = [
        f"{prefix}_difference_{bin_number:02d}_{bin_number + 1:02d}" for bin_number in range(1, 8)
    ]
    return differences + [
        f"{prefix}_{name}"
        for name in ("greenup_slope", "senescence_slope", "peak_interval", "amplitude", "integral")
    ]


def feature_sets():
    return {
        "interval_only": interval_columns(),
        "interval_and_shape": interval_columns() + shape_columns(),
        "shape_only": shape_columns(),
    }


def validate_policy(policy):
    if set(policy) != set(FEATURE_POLICY):
        raise ValueError("Feature policy fields differ from the supported schema.")
    fixed_rules = {
        name: value for name, value in FEATURE_POLICY.items() if name != "savi_soil_adjustment"
    }
    if any(policy[name] != value for name, value in fixed_rules.items()):
        raise ValueError("Unsupported temporal feature definition; do not mix feature recipes.")
    adjustment = float(policy["savi_soil_adjustment"])
    if not math.isfinite(adjustment) or not 0 <= adjustment <= 1:
        raise ValueError("SAVI soil adjustment must be finite and between zero and one.")


def calendar_positions(row):
    intervals = season_workflow.window_intervals(row)
    # Keep half-day positions even when a calendar interval has an odd day count.
    first_day = date.fromisoformat(intervals[0]["start"])
    return np.asarray(
        [
            (date.fromisoformat(interval["start"]) - first_day).days
            + (date.fromisoformat(interval["end"]) - date.fromisoformat(interval["start"])).days / 2
            for interval in intervals
        ],
        dtype=float,
    )


def finite_value(row, column):
    value = season_workflow.finite_number(row.get(column))
    if value is None:
        raise ValueError(
            f"Nonfinite or missing feature {column}; export complete windows before feature engineering."
        )
    return value


def savi_value(near_infrared, red, adjustment):
    denominator = near_infrared + red + adjustment
    if abs(denominator) < 1e-10:
        raise ValueError(
            "SAVI denominator is zero; inspect reflectance rather than fill a synthetic value."
        )
    return float((1 + adjustment) * (near_infrared - red) / denominator)


def fitted_slope(positions, values):
    if len(values) < 2:
        return 0.0
    centered_positions = positions - positions.mean()
    return float(
        np.dot(centered_positions, values - values.mean())
        / np.dot(centered_positions, centered_positions)
    )


def temporal_shape(values, positions, index_name):
    values = np.asarray(values, dtype=float)
    peak_number = int(np.argmax(values))
    prefix = f"shape_{index_name}"
    features = {
        f"{prefix}_difference_{bin_number:02d}_{bin_number + 1:02d}": float(difference)
        for bin_number, difference in enumerate(np.diff(values), start=1)
    }
    features.update(
        {
            f"{prefix}_greenup_slope": fitted_slope(
                positions[: peak_number + 1], values[: peak_number + 1]
            ),
            f"{prefix}_senescence_slope": fitted_slope(
                positions[peak_number:], values[peak_number:]
            ),
            f"{prefix}_peak_interval": peak_number + 1,
            f"{prefix}_amplitude": float(values.max() - values.min()),
            f"{prefix}_integral": float(np.trapezoid(values, positions)),
            f"{prefix}_greenup_observed": peak_number > 0,
            f"{prefix}_senescence_observed": peak_number < len(values) - 1,
        }
    )
    return features


def enhance_window(row, policy):
    positions = calendar_positions(row)
    enhanced = dict(row)
    for column in season_workflow.feature_columns():
        enhanced[column] = finite_value(row, column)
    for bin_number in range(1, 9):
        prefix = f"bin_{bin_number:02d}"
        enhanced[f"{prefix}_SAVI"] = savi_value(
            enhanced[f"{prefix}_B8"],
            enhanced[f"{prefix}_B4"],
            float(policy["savi_soil_adjustment"]),
        )
    for index_name in INDEX_NAMES:
        values = [enhanced[f"bin_{bin_number:02d}_{index_name}"] for bin_number in range(1, 9)]
        enhanced.update(temporal_shape(values, positions, index_name))
    enhanced["sample_id"] = sample_identifier(row)
    for column in feature_sets()["interval_and_shape"]:
        finite_value(enhanced, column)
    return enhanced


def sample_identifier(row):
    identity = [
        row["field_id"],
        row.get("season_key", row.get("season_id")),
        row["source_crs"],
        float(row["x"]),
        float(row["y"]),
    ]
    return season_workflow.record_fingerprint(identity)


def enhance_rows(rows, policy):
    validate_policy(policy)
    enhanced = [enhance_window(row, policy) for row in rows]
    if len({row["sample_id"] for row in enhanced}) != len(enhanced):
        raise ValueError("Duplicate pixel-window identities in feature inputs.")
    return enhanced


def export_feature_handoff(
    source_handoff, output_folder, policy=None, frozen=False, development_evidence=""
):
    """Snapshot one common row set and three explicit predictor schemas for fair experiments."""
    if not isinstance(frozen, bool):
        raise ValueError("Feature freeze must be a boolean, not a text value.")
    policy = dict(FEATURE_POLICY if policy is None else policy)
    rows, configuration, coverage, base_manifest = season_workflow.load_training_handoff(
        source_handoff
    )
    validate_policy(policy)
    if (
        coverage.get("imagery_request_version") != REQUEST_VERSION
        or coverage.get("resampling") != RESAMPLING_POLICY
    ):
        raise ValueError(
            "Feature experiments require bilinear B5/B11 imagery. Rerun updated source retrieval and window export; do not relabel nearest-neighbor data."
        )
    enhanced = enhance_rows(rows, policy)
    configuration = {
        **configuration,
        "training_backend": "earth_engine",
        "earth_engine_forest": {
            **season_workflow.EARTH_ENGINE_FOREST,
            **configuration.get("earth_engine_forest", {}),
        },
        "feature_policy": policy,
        "feature_policy_frozen": bool(frozen),
        "feature_development_evidence": development_evidence,
    }
    output_folder = Path(output_folder)
    season_workflow.write_records(output_folder / "experiment_features.csv", enhanced)
    save_json(output_folder / "configuration.json", configuration)
    save_json(output_folder / "coverage.json", coverage)
    source_path = Path(source_handoff)
    provenance = {
        "manifest_sha256": season_workflow.file_sha256(source_path),
        "files": base_manifest["files"],
        "row_count": base_manifest["row_count"],
        "window_policy": base_manifest["window_policy"],
    }
    save_json(output_folder / "window_input_provenance.json", provenance)
    filenames = [
        "experiment_features.csv",
        "configuration.json",
        "coverage.json",
        "window_input_provenance.json",
    ]
    ready = bool(
        enhanced and configuration.get("frozen") is True and frozen and development_evidence.strip()
    )
    manifest = {
        "format_version": 2,
        "kind": "savi_temporal_feature_experiments",
        "status": "prepared" if ready else "pending_inputs",
        "row_count": len(enhanced),
        "feature_sets": feature_sets(),
        "implementation_sha256": season_workflow.file_sha256(Path(__file__)),
        "feature_policy": policy,
        "sample_fingerprint": season_workflow.record_fingerprint(
            [row["sample_id"] for row in enhanced]
        ),
        "window_policy": configuration.get("window_policy", "reference_supported_only"),
        "files": {
            filename: {"sha256": season_workflow.file_sha256(output_folder / filename)}
            for filename in filenames
        },
    }
    manifest_path = output_folder / "feature_handoff.json"
    save_json(manifest_path, manifest)
    return manifest_path


def load_feature_handoff(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Feature handoff missing: {manifest_path}. Run notebooks/04_feature_engineering.ipynb first."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 2 or manifest.get("feature_sets") != feature_sets():
        raise ValueError("Unsupported feature handoff or predictor schema.")
    expected = {
        "experiment_features.csv",
        "configuration.json",
        "coverage.json",
        "window_input_provenance.json",
    }
    if set(manifest.get("files", {})) != expected:
        raise ValueError("Feature handoff is missing required files.")
    for filename, record in manifest["files"].items():
        if season_workflow.file_sha256(manifest_path.parent / filename) != record["sha256"]:
            raise ValueError(f"Feature input changed: {filename}. Re-export feature engineering.")
    configuration = json.loads((manifest_path.parent / "configuration.json").read_text())
    coverage = json.loads((manifest_path.parent / "coverage.json").read_text())
    rows = season_workflow.read_table(manifest_path.parent / "experiment_features.csv")
    validate_policy(configuration["feature_policy"])
    if configuration["feature_policy"] != manifest["feature_policy"]:
        raise ValueError("Feature recipe differs between configuration and manifest.")
    if (
        len(rows) != manifest["row_count"]
        or season_workflow.record_fingerprint([row["sample_id"] for row in rows])
        != manifest["sample_fingerprint"]
    ):
        raise ValueError("Feature rows no longer match the exported sample identities.")
    return rows, configuration, coverage, manifest
