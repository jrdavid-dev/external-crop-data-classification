"""Package or verify the saved external classification study without fitting models."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from crop_pipeline.training import notebook_workflow
from crop_pipeline.training.external_report import (
    MODELS, REVISED_POPULATION_NAMES, read_predictions, retained_reference,
)
from crop_pipeline.windows.seasons import file_sha256

TESTING = Path("pipeline_outputs/testing")
TRAINING = Path("pipeline_outputs/training/run_fa40f5d6bb31")
TEST = TESTING / "run_81a0d0a2812e"
PREPROCESSING = Path("pipeline_outputs/preprocessing/external_xxu0iz4o")
RESULT_DIRS = {
    "original_reporting": TESTING / "run_81a0d0a2812e_batch1",
    "revised_reporting": TESTING / "run_81a0d0a2812e_batch1_revised",
    "comparators": TESTING / "run_81a0d0a2812e_batch2",
    "error_diagnostics": TESTING / "run_81a0d0a2812e_batch3",
    "transfer_followups": TESTING / "run_81a0d0a2812e_transfer_followups",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def preprocessing_source_drift(root, receipts):
    """Compare current source files with historical receipt hashes, using portable names."""
    sources = {}
    for receipt in receipts:
        data = read_json(root / receipt)
        for name, recorded in data["files"].items():
            parts = Path(name).parts
            start = next((i for i in range(len(parts) - 2, -1, -1)
                          if parts[i:i + 2] == ("src", "crop_pipeline")), None)
            if start is not None:
                relative = Path(*parts[start:]).as_posix()
                sources.setdefault(relative, {})[data["stage"]] = recorded
    return {
        relative: {
            "recorded_sha256_by_stage": recorded,
            "current_sha256": (current := file_sha256(root / relative)),
            "matching_stages": [stage for stage, digest in recorded.items() if digest == current],
        }
        for relative, recorded in sorted(sources.items())
    }


def build(root, output):
    root = root.resolve()
    output = output.resolve()
    if any(output.is_relative_to((root / directory).resolve())
           for directory in (TRAINING, TEST, PREPROCESSING, *RESULT_DIRS.values())):
        raise ValueError("Write the study manifest outside the saved input and result directories")
    if output.exists():
        raise FileExistsError(output)
    model_receipt, model_stage = notebook_workflow._verify(root / TRAINING / "models/models_stage.json",
                                                          "models_stage")
    score_receipt, score_stage = notebook_workflow._verify(root / TEST / "scores.json", "scores")
    development = TRAINING / "features_development.json"
    preprocessing_receipts = [
        PREPROCESSING / stage / f"{stage}_stage_receipt.json"
        for stage in ("quality", "windows", "features")
    ]
    frozen_development = read_json(root / TRAINING / "models/development.json")
    if (file_sha256(model_receipt) != score_stage["models_handoff_sha256"]
            or file_sha256(root / development) != frozen_development["feature_sha256"]):
        raise ValueError("Frozen development or model receipt differs from the study run")
    saved_export = read_json(root / preprocessing_receipts[-1])["files"]
    export_hashes = {Path(name).name: digest for name, digest in saved_export.items()}
    for name, snapshot in (("features_development.json", development),
                           ("features_test.json", TEST / "features_test.json")):
        if file_sha256(root / snapshot) != export_hashes[name]:
            raise ValueError(f"Feature snapshot differs from the recorded preprocessing export: {name}")

    groups = {}
    artifacts = {}

    def record(role, paths):
        names = []
        for path in sorted(set(paths)):
            name = path.as_posix()
            if name not in artifacts:
                artifacts[name] = file_sha256(root / path)
            names.append(name)
        groups[role] = names

    record("preprocessing", [
        *preprocessing_receipts,
        *(PREPROCESSING / name for name in (
            "quality/quality.json", "quality/collection_snapshot.json", "quality/source_inputs.json",
            "quality/development_overlap_exclusions.json", "quality/test_overlap_exclusions.json",
            "windows/windows.json", "windows/seasons/cycle_selection_audit.csv",
            "windows/seasons/cycle_candidates/complete_features.csv", "features/usable_samples.csv",
        )),
        Path("pipeline_outputs/test_audit_2026-11-19/parcel_audit.json"),
        Path("pipeline_outputs/development_error_review_2026-11-19/parcel_diagnostics.json"),
    ])
    record("frozen_development", [
        development, TRAINING / "training.json", TRAINING / "validation/validation_candidates.json",
        TRAINING / "matched_geographic_feature_comparison.json",
        TRAINING / "matched_geographic_strongest_cycle.json",
        TRAINING / "shape_external_comparison/external_comparison.json",
        TRAINING / "shape_external_comparison/shape_models.json",
        *(TRAINING / "shape_external_comparison/frozen" / model / name
          for model in ("fixed_matched", "aligned_matched")
          for name in ("model_manifest.json", "recipe.json", "model.joblib")),
        *(TRAINING / "shape_external_comparison/combined/shape" / f"{model}_test_parcels.csv"
          for model in ("fixed_matched", "aligned_matched")),
    ])
    record("frozen_models", [TRAINING / "models/models_stage.json",
                            *(TRAINING / "models" / name for name in model_stage["files"])])
    record("frozen_test", [TEST / "scores.json", *(TEST / name for name in score_stage["files"])])
    for role, directory in RESULT_DIRS.items():
        paths = [p.relative_to(root) for p in (root / directory).rglob("*") if p.is_file()]
        if not paths or not (root / directory / "report.md").is_file():
            raise FileNotFoundError(root / directory / "report.md")
        record(role, paths)
    record("documentation", [Path("docs/classification_dataset_register.md"),
                             Path("docs/classification_external_study.md")])
    drift = preprocessing_source_drift(root, preprocessing_receipts)
    record("reproduction_entry_points", [
        *(Path(name) for name in drift),
        *(Path("src/crop_pipeline/training") / f"{name}.py" for name in (
            "study_manifest", "external_report", "model_comparisons", "model_followups", "transfer_followups",
            "error_analysis", "forest_diagnostics", "source_holdout", "notebook_workflow",
        )),
        Path("notebooks/testing/02_results_and_diagnostics.ipynb"),
        Path("notebooks/testing/03_model_comparisons.ipynb"),
        Path("notebooks/testing/04_error_analysis_and_diagnostics.ipynb"),
        Path("notebooks/training/03_matched_model_tuning.ipynb"),
        Path("requirements-notebooks.txt"),
    ])

    cohorts = {}
    for model in MODELS:
        manifest_path = TRAINING / "models" / model / "model_manifest.json"
        manifest = read_json(root / manifest_path)
        rows = read_predictions(root / TEST / "predictions/combined" / f"{model}_test_parcels.csv")
        split = manifest["development_split"]
        if set(split) != set(manifest["development_groups"]) or set(split) & rows.keys():
            raise ValueError(f"Frozen development and test split integrity failed for {model}")
        cohorts[model] = {
            "model_manifest": manifest_path.as_posix(),
            "development_train_parcels": sorted(key for key, role in split.items() if role == "train"),
            "development_validation_parcels": sorted(key for key, role in split.items() if role == "validation"),
            "original_external_parcels": sorted(rows),
            "reported_external_parcels": sorted(key for key, row in rows.items() if retained_reference(*row[:2])),
        }
    for key in ("development_train_parcels", "development_validation_parcels",
                "original_external_parcels", "reported_external_parcels"):
        if cohorts["fixed_matched"][key] != cohorts["aligned_matched"][key]:
            raise ValueError(f"Matched parcel identities differ for {key}")

    revised_methods = read_json(root / RESULT_DIRS["revised_reporting"] / "methods.json")
    comparison_runtime = read_json(root / RESULT_DIRS["comparators"] / "comparators.json")["runtime"]
    manifest = {
        "schema_version": 1, "path_base": "repository_root", "digest": "sha256",
        "scope": "external-to-external classification of carrot, cabbage, and potato",
        "primary_population": "jecam", "reported_populations": list(REVISED_POPULATION_NAMES),
        "reporting_revision": revised_methods["reporting_scope_revision"],
        "paoay": {"status": "deferred", "designated_application_model": None},
        "runtime_used_to_package": {"python": platform.python_version(),
                                    **{name: version(name) for name in (
                                        "numpy", "scikit-learn", "joblib", "scipy", "shapely", "pyproj")}},
        "recorded_comparator_runtime": comparison_runtime,
        "recorded_revised_bootstrap_numpy": revised_methods["numpy_version"],
        "preprocessing_source_comparison": drift,
        "original_cohorts": cohorts, "artifact_groups": groups, "artifact_sha256": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return len(artifacts)


def verify(root, manifest_path):
    artifacts = read_json(manifest_path)["artifact_sha256"]
    if not artifacts:
        raise ValueError("Manifest contains no artifacts")
    failures = []
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a repository-relative artifact path: {name}")
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            failures.append(name)
    if failures:
        raise ValueError("Missing or changed study artifacts:\n" + "\n".join(failures))
    return len(artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    operation = build if options.action == "build" else verify
    count = operation(options.root, options.manifest)
    print(f"{options.action}: {count} study artifacts")


if __name__ == "__main__":
    main()
