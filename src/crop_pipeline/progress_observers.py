"""Observe completed calls without modifying processing functions or their files.

Keeping observers outside hashed collectors avoids new download revisions for a
UI-only change. Arguments, return values, exceptions and model settings pass
through unchanged; counters advance only after successful work returns.
"""

import functools
import inspect
from pathlib import Path
from contextvars import ContextVar

_counters = ContextVar("pipeline_progress_counters", default={})
_installed = False


def watch_parent(
    module, parent_name, child_module, child_name, total, description, unit
):
    parent = getattr(module, parent_name)
    child_key = (child_module.__name__, child_name)
    watch_child(child_module, child_name, child_key)

    @functools.wraps(parent)
    def observed_parent(*arguments, **keywords):
        from crop_pipeline import progress

        if progress.active_monitor() is None:
            return parent(*arguments, **keywords)
        bound = inspect.signature(parent).bind(*arguments, **keywords)
        bound.apply_defaults()
        if parent_name == "collect_source" and not bound.arguments.get(
            "enabled", False
        ):
            return parent(*arguments, **keywords)
        if parent_name == "collect_source" and bound.arguments.get("collection_backend") == "batch":
            # Remote jobs do not call the serial tile child. Their artifact snapshot
            # reports actual completion; a synchronous attempted-tile bar would stay at zero.
            return parent(*arguments, **keywords)
        with progress.WorkCounter(total(bound.arguments), description, unit) as counter:
            token = _counters.set({**_counters.get(), child_key: counter})
            try:
                return call_parent(parent, parent_name, arguments, keywords)
            finally:
                _counters.reset(token)

    setattr(module, parent_name, observed_parent)


def watch_child(module, name, key):
    child = getattr(module, name)
    if getattr(child, "_pipeline_progress_child", False):
        return

    @functools.wraps(child)
    def observed_child(*arguments, **keywords):
        result = call_child(child, name, arguments, keywords)
        counter = _counters.get().get(key)
        if counter is not None:
            counter.update()
        return result

    observed_child._pipeline_progress_child = True
    setattr(module, name, observed_child)


def enable():
    global _installed
    if _installed:
        return
    from crop_pipeline.sources import annual, sen4agrinet, francecrops, bounded_references, annual_batch
    from crop_pipeline.windows import annual_archive, seasons, detector
    from crop_pipeline.features import experiments
    from crop_pipeline.training import exploratory, earth_engine
    from crop_pipeline.imagery.retrieval import SatelliteRetriever
    from crop_pipeline.imagery import quality
    from sklearn.ensemble import RandomForestClassifier

    # Observe current bounded workflows outside their fingerprinted helpers.
    watch_parent(
        bounded_references, "prepare", sen4agrinet, "index_checkpoint",
        lambda arguments: None, "Gathering reference patches", "patch",
    )
    from crop_pipeline.imagery.accelerated import AcceleratedRetriever
    watch_parent(
        annual_batch, "collect_batch_year", AcceleratedRetriever, "inventory_region",
        lambda arguments: None, "Gathering annual imagery", "tile",
    )
    watch_parent(
        sen4agrinet,
        "discover_manifest",
        sen4agrinet,
        "repository_file_record",
        lambda arguments: None,
        "Repository entries discovered",
        "entry",
    )
    watch_parent(
        sen4agrinet,
        "publish_inventory",
        sen4agrinet,
        "index_checkpoint",
        lambda arguments: len(arguments["manifest"]["files"]),
        "Consolidating references",
        "patch",
    )
    watch_parent(
        quality,
        "detect_greenup",
        detector,
        "detect_region",
        lambda arguments: len({row["region_id"] for row in arguments["rows"]}),
        "Candidate timing detection",
        "tile",
    )
    watch_parent(
        seasons,
        "select_season_windows",
        seasons,
        "select_candidate_window",
        lambda arguments: len(arguments["candidates"]),
        "Candidate window assessment",
        "cycle",
    )
    watch_parent(
        sen4agrinet,
        "build_reference_inventory",
        sen4agrinet,
        "process_patch",
        lambda arguments: len(arguments["manifest"]["files"]),
        "Reference patches",
        "patch",
    )
    watch_parent(
        annual,
        "prepare_cohort",
        annual,
        "prepare_parcel_tiles",
        lambda arguments: len(arguments["records"]),
        "Eligible parcel masks",
        "parcel",
    )
    watch_parent(
        annual,
        "collect_source",
        annual,
        "collect_tile_safely",
        lambda arguments: sum(
            inputs["regions"] for inputs in arguments["plan"]["years"].values()
        ),
        "Annual imagery tiles attempted",
        "tile",
    )
    watch_parent(
        SatelliteRetriever,
        "download_region",
        SatelliteRetriever,
        "download_interval",
        lambda arguments: len(arguments["plan"].get("download_intervals", [])) or 24,
        "Half-month composites",
        "file",
    )
    watch_parent(
        francecrops,
        "download_reference_tables",
        francecrops,
        "download_missing_table",
        lambda arguments: sum(
            not (Path(arguments["destination"]) / filename).is_file()
            for filename in francecrops.TABLE_FILES.values()
        ),
        "Missing reference tables",
        "table",
    )
    watch_parent(
        annual_archive,
        "audit_source",
        annual_archive.quality,
        "quality_check",
        lambda arguments: len(annual.read_regions(arguments["source"])),
        "Saved tiles audited",
        "tile",
    )
    watch_parent(
        annual_archive,
        "partitioned_timelines",
        annual_archive.quality,
        "composite_timelines",
        lambda arguments: len({row["region_id"] for row in arguments["quality_rows"]}),
        "Vegetation timelines",
        "tile",
    )
    watch_parent(
        annual_archive,
        "build_partitioned_features",
        seasons,
        "build_features",
        lambda arguments: len({row["region_id"] for row in arguments["seasons"]}),
        "Complete pixel windows",
        "tile",
    )
    watch_parent(
        experiments,
        "enhance_rows",
        experiments,
        "enhance_window",
        lambda arguments: len(arguments["rows"]),
        "Temporal features",
        "pixel",
    )
    watch_parent(
        exploratory,
        "run_experiments",
        RandomForestClassifier,
        "fit",
        lambda arguments: len(experiments.feature_sets()),
        "Random Forest designs fitted",
        "model",
    )
    watch_parent(
        earth_engine,
        "train_feature_experiments",
        earth_engine,
        "fit_forest",
        lambda arguments: None,
        "Earth Engine forests fitted",
        "model",
    )
    _installed = True


def call_parent(parent, name, arguments, keywords):
    from crop_pipeline import progress

    if name in {"build_reference_inventory", "audit_source", "download_region",
                "prepare", "collect_batch_year"}:
        with progress.quiet_output():
            return parent(*arguments, **keywords)
    return parent(*arguments, **keywords)


def call_child(child, name, arguments, keywords):
    from crop_pipeline import progress

    if name == "collect_tile_safely" and progress.active_monitor() is not None:
        with progress.quiet_output():
            return child(*arguments, **keywords)
    return child(*arguments, **keywords)
