"""Keep imagery archives resumable when orchestration or daily-summary code changes.

The planner, reference content and imagery recipe determine compatibility. Full
code hashes remain in original configurations and per-pass runtime receipts.
"""

import ast
import hashlib
import json
from pathlib import Path

from crop_pipeline.common import save_json

PLANNING_FUNCTIONS = {
    "normalize_sen4agrinet_feature",
    "grouped_footprints",
    "merge_footprints",
    "projected_cohorts",
    "uncontested_interior",
    "tile_bounds",
    "save_tile",
    "valid_request_geometry",
    "prepare_cohort",
    "prepare_parcel_tiles",
    "save_reference_footprints",
    "save_year_inputs",
    "count_pixels_by_crop",
}
# Verified migration from the previously shipped planner. Other unknown old code
# is deliberately not considered equivalent merely because its settings match.
LEGACY_PLANNERS = {
    "1777f8c92dc245ed17b2df769dbdc62738d3ee5849d44332bf59d6c12e90a1b9": "eae345246c06adace06ca2d1afe3838b24c683721d71b1ba3018b1fc50d72390",
}


PLANNER_BINDINGS = {
    "np",
    "rasterio",
    "Affine",
    "Transformer",
    "rasterize",
    "box",
    "mapping",
    "shape",
    "transform_geometry",
    "unary_union",
    "STRtree",
    "sampling",
}


def planning_imports(tree):
    return [
        filtered
        for node in tree.body
        if (filtered := planning_import(node)) is not None
    ]


def planning_import(node):
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return None
    selected = [
        alias
        for alias in node.names
        if (alias.asname or alias.name) in PLANNER_BINDINGS
    ]
    if not selected:
        return None
    return (
        ast.Import(names=selected)
        if isinstance(node, ast.Import)
        else ast.ImportFrom(module=node.module, names=selected, level=node.level)
    )


def planning_fingerprint(path):
    tree = ast.parse(Path(path).read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in PLANNING_FUNCTIONS
    ]
    if {node.name for node in functions} != PLANNING_FUNCTIONS:
        raise ValueError("Archive identity is missing a required planning function.")
    normalized = ast.dump(
        ast.Module(body=planning_imports(tree) + functions, type_ignores=[]),
        include_attributes=False,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def imagery_identity(configuration):
    identity = {
        name: value
        for name, value in configuration.items()
        if name not in {"helper_sha256", "timelines_sha256", "planning_sha256"}
    }
    identity["planning_sha256"] = configuration.get(
        "planning_sha256"
    ) or LEGACY_PLANNERS.get(configuration.get("helper_sha256"))
    return identity


def initialize_archive(source_root, collection_name, configuration):
    """Resume a compatible archive without relabelling its original configuration."""
    identity = imagery_identity(configuration)
    base = Path(source_root) / "runs" / collection_name
    configurations = sorted(
        base.glob("*/run_configuration.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in configurations:
        old = json.loads(path.read_text())
        if imagery_identity(old) == identity:
            return path.parent, path.parent.name
    revision = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]
    folder = base / revision
    configuration_path = folder / "run_configuration.json"
    if (
        configuration_path.exists()
        and imagery_identity(json.loads(configuration_path.read_text())) != identity
    ):
        raise ValueError("Archive identity collision; existing outputs were preserved.")
    if not configuration_path.exists():
        save_json(configuration_path, configuration)
    return folder, revision
