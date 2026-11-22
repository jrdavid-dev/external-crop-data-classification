"""Rebuild a separate 25-day feature run with the current matched-first split."""

from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from crop_pipeline import lean_preprocessing as lean
from crop_pipeline import preprocessing_workflow as workflow
from crop_pipeline.common import save_json
from crop_pipeline.windows import external, seasons


def resplit(indexed):
    exclusions = {name: external.exclude_shared_development_locations(rows)
                  for name, rows in indexed.items()}
    fixed, aligned = (indexed[name] for name in lean.RECIPES)
    matched = list(external.capped_keys(key for key in fixed if key in aligned))
    split = external.split_development(
        {name: [details for _, details in rows.values()] for name, rows in indexed.items()},
        [fixed[key][1] for key in matched],
    )
    assignments = split["assignments"]
    selections = {"fixed_full": external.capped_keys(fixed), "fixed_matched": matched,
                  "aligned_full": external.capped_keys(aligned), "aligned_matched": matched}
    selected = {}
    for name, keys in selections.items():
        rows = indexed[name.split("_")[0]]
        cohort = {}
        for key in keys:
            pointer, details = rows[key]
            if details["partition"] == "test":
                if key[0] in assignments:
                    raise ValueError(f"Reserved test field is assigned to development: {key[0]}")
            else:
                details = {**details, "partition": assignments[key[0]]}
            cohort[key] = (pointer, details)
        selected[name] = cohort
    return split, selected, exclusions


def run(repository, baseline_features=None):
    root, report, output = workflow.paths(repository)
    baseline_handoff, published = lean.baseline_source(root, baseline_features)
    current = workflow.collection_signature(report)
    if (published and lean.load_json(published).get("split_mode") == "fresh_matched_first"
            and lean.load_json(published.parent / "collection_snapshot.json")["campaign_metadata"] == current):
        return workflow.verify(published, "features")

    workflow.verify(baseline_handoff, "features")
    baseline = baseline_handoff.parent
    saved = lean.load_json(baseline / "quality.json")
    snapshot = lean.load_json(baseline / "collection_snapshot.json")["campaign_metadata"]
    lean.confirm_completed_inputs(snapshot, current, report)
    batches, sources = lean.timing.new_sources(report, snapshot)
    detector_policy = lean.load_json(baseline / "seasons/automatic_timing_policy.json")
    detector, policy = detector_policy["detector_parameters"], detector_policy["quality_policy"]
    if policy["maximum_onset_gap_days"] != 20 or policy["maximum_support_gap_days"] != 20:
        raise ValueError("Expected the saved strict 20-day timing baseline")
    review = baseline / "seasons/review"
    frozen = review / "frozen_configuration.json"
    configuration = lean.load_json(frozen if frozen.exists() else review / "frozen_configuration.template.json")
    configuration.update(cycle_selection_policy="earliest_complete", automatic_timing_policy=policy)

    with TemporaryDirectory(prefix="lean_resplit_", dir=output) as temporary:
        indexed, props, _ = lean.finalize_indices(
            baseline / "windows.json", saved["regions"], saved, report, Path(temporary),
            detector, policy, configuration, sources
        )
        split, selected, exclusions = resplit(indexed)
        if workflow.collection_signature(report) != current:
            raise RuntimeError("Collection changed during preparation; rerun with the new snapshot")
        destination = Path(mkdtemp(prefix="external_25d_resplit_", dir=output))
        counts = {name: sum(split["assignments"][group] == name for group in split["matched_parcel_groups"])
                  for name in ("train", "validation")}
        provenance = {"baseline_features": str(baseline_handoff), "timing_policy": lean.POLICY,
                      "split_mode": "fresh_matched_first", "matched_development_fields": counts,
                      "new_completed_development_batches": batches}
        windows = destination / "windows.json"
        save_json(windows, {**provenance, "selected": {
            name: [list(key) for key in rows] for name, rows in selected.items()}})
        metadata = {**provenance, "window_sha256": seasons.file_sha256(windows),
                    "quality": str(baseline / "quality.json"), "fixed_start_month": external.FIXED_START_MONTH}
        cohorts = lean.load_json(baseline / "features_test.json")["test_cohorts"]
        development, test = lean.write_packages(destination, selected, {name: {} for name in lean.COHORTS},
                                                metadata, cohorts, props)
        save_json(destination / external.SPLIT_FILENAME, {**split, "window_sha256": metadata["window_sha256"]})
        save_json(destination / "collection_snapshot.json", {"campaign_metadata": current, "report_root": str(report)})
        save_json(destination / "development_overlap_exclusions.json", exclusions)
        save_json(destination / "window_coverage.json", lean.window_coverage(development, test))
        handoff = destination / "features.json"
        save_json(handoff, metadata)
        workflow.write_feature_reports(handoff, development, test)
        if workflow.collection_signature(report) != current:
            raise RuntimeError("Collection changed while writing features; previous pointers were preserved")
        implementation = [Path(__file__), Path(lean.__file__), Path(lean.timing.__file__)]
        workflow.publish(root, "windows", windows, [windows, destination / "collection_snapshot.json",
                                                     *implementation])
        return workflow.publish(root, "features", handoff, [*destination.iterdir(), *implementation])
