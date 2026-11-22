"""Load the current outputs the walkthrough draws from, following each pipeline's stage pointers.

Every loader resolves the published ``current_*.json`` pointer (or the receipt it names) so the
notebook always describes the latest verified run, and returns the run folder it used for provenance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_pipeline.analysis_workflow import resolve_stage_pointer
from experiments.improved_v1 import workflow
from experiments.pilots import data

CROPS = ("carrot", "cabbage", "potato")


def repository_root(start=None):
    root = Path(start or Path.cwd()).resolve()
    while not (root / "pyproject.toml").is_file():
        if root == root.parent:
            raise RuntimeError("Run this inside the project repository")
        root = root.parent
    return root


def _pointer(root, relative):
    return resolve_stage_pointer(Path(root) / relative)


# Improved pipeline

def improved_inputs(root):
    """Per-pixel rows, Sentinel-2 [n, 24, 5], clear flags [n, 24] and Sentinel-1 dB [n, 24, 2]."""
    receipt, record = workflow._input(root, None, "inputs")
    rows, arrays = data.load(receipt.parent)
    return {"run": receipt.parent, "quality_handoff": record.get("quality_handoff"), "rows": rows,
            "s2": arrays["s2_values"], "valid": arrays["s2_valid"], "s1": arrays["s1_values"],
            "frame": pd.DataFrame(rows)}


def improved_model(root):
    receipt, _ = workflow._input(root, None, "models")
    model, name, info, record = workflow.load_model(receipt)
    return {"run": receipt.parent, "model": model, "candidate": name, "info": info, "record": record}


def improved_selection(root):
    receipt, _ = workflow._input(root, None, "selection")
    return {"run": receipt.parent, "scores": pd.read_csv(receipt.parent / "candidate_scores.csv"),
            "selection": json.loads((receipt.parent / "selection.json").read_text())}


def improved_test(root):
    receipt, _ = workflow._input(root, None, "test")
    return {"run": receipt.parent, "scores": json.loads((receipt.parent / "test_scores.json").read_text()),
            "predictions": pd.read_csv(receipt.parent / "test_parcels.csv")}


def improved_paoay(root):
    """Current improved Paoay map: class codes, cropland flag, probabilities and the summary."""
    import rasterio

    receipt, _ = workflow._input(root, None, "paoay")
    run = receipt.parent
    with rasterio.open(run / "crop_map.tif") as source:
        codes, cropland = source.read(1), source.read(2)
        transform, crs = source.transform, source.crs
    with rasterio.open(run / "probabilities.tif") as source:
        probabilities = source.read()
        classes = list(source.descriptions)
    return {"run": run, "codes": codes, "cropland": cropland == 2, "inside": codes > 0,
            "probabilities": probabilities, "classes": classes, "transform": transform, "crs": crs,
            "summary": json.loads((run / "paoay_summary.json").read_text()),
            "areas": pd.read_csv(run / "area_summary.csv")}


class _CachedEarthEngine:
    """Stand-in that fails loudly if a Paoay input would need a new Earth Engine request."""

    def __getattr__(self, name):
        raise RuntimeError("Paoay Sentinel-1 is not cached; run improved notebook 04 first")


def paoay_study(root):
    """Paoay study pixels on the improved map grid: S2 [n, 24, 5], S1 [n, 24, 2], boundary and reference areas.

    Uses only local composites and the cached Sentinel-1 grid; no Earth Engine request is made.
    """
    return workflow.paoay_inputs(root, ee=_CachedEarthEngine())


# Main pipeline

def main_quality_folder(root):
    return _pointer(root, "pipeline_outputs/preprocessing/current_quality.json").parent


def composite_quality(root):
    """One row per parcel fragment and half-month: selected and valid pixels and the quality status."""
    return pd.read_csv(main_quality_folder(root) / "quality" / "composite_quality.csv",
                       usecols=["region_id", "crop", "partition_group", "name", "start", "selected_pixels",
                                "valid_pixels", "valid_fraction", "status", "image_path"])


def usable_samples(root):
    return pd.read_csv(main_quality_folder(root) / "usable_samples.csv")


def main_test(root):
    """Main-pipeline JECAM scores per cohort and parcel predictions (current testing run)."""
    scores_path = _pointer(root, "pipeline_outputs/testing/current_scores.json")
    run = scores_path.parent
    scores = json.loads((run / "reports" / "test_scores.json").read_text())
    combined = json.loads((run / "reports" / "combined" / "test_scores.json").read_text())
    predictions = {}
    for path in sorted((run / "predictions").glob("*_test_parcels.csv")):
        predictions[path.name.replace("_test_parcels.csv", "")] = pd.read_csv(path)
    combined_predictions = {path.name.replace("_test_parcels.csv", ""): pd.read_csv(path)
                            for path in sorted((run / "predictions" / "combined").glob("*_test_parcels.csv"))}
    return {"run": run, "jecam": scores, "combined": combined, "predictions": predictions,
            "combined_predictions": combined_predictions}


def main_paoay_aligned(root, model_run=None):
    """The main pipeline's exploratory aligned Paoay map built from the current frozen models."""
    import rasterio

    model_run = model_run or _pointer(root, "pipeline_outputs/training/current_models.json").parent.parent.name
    for manifest in sorted((Path(root) / "sen4agrinet_work/paoay/aligned_exploratory").glob("*/manifest.json")):
        record = json.loads(manifest.read_text())
        if f"/{model_run}/" in record.get("model_manifest", ""):
            with rasterio.open(manifest.parent / "aligned_map.tif") as source:
                codes = source.read(1)
            return {"run": manifest.parent, "codes": codes, "manifest": record,
                    "areas": pd.read_csv(manifest.parent / "area_summary.csv")}
    raise FileNotFoundError(f"No aligned Paoay map was built from the current main models ({model_run})")


def parcel_archive(image_path):
    """One parcel's annual archive: per half-month bands and the before/after mask counts."""
    import rasterio

    with rasterio.open(image_path) as source:
        stack = source.read().astype(float)
        names = list(source.descriptions)
    layers = {}
    for index, name in enumerate(names):
        interval, layer = name.split("__")
        layers.setdefault(interval, {})[layer] = stack[index]
    return layers


def world_countries(root):
    """Natural Earth 1:50m countries shipped with the CropHarvest download (no internet needed)."""
    import geopandas as gpd

    path = Path(root) / "data/external/cropharvest/ne_50m_admin_0_countries.zip"
    return gpd.read_file(f"zip://{path}")


def main_population_predictions(main, cohort):
    """All external-test parcel predictions of one main-pipeline cohort: JECAM plus every supplement."""
    run = main["run"] / "predictions"
    frames = [main["predictions"][cohort]]
    frames += [pd.read_csv(path) for path in sorted(run.glob(f"*/{cohort}_test_parcels.csv")) if path.parent.name != "combined"]
    return pd.concat(frames, ignore_index=True)


def example_pixel(frame, valid, crop="cabbage", partition="train", minimum_clear=16):
    """Index of the first stable-ordered development pixel of ``crop`` with at least ``minimum_clear`` clear half-months."""
    chosen = ((frame.crop == crop) & (frame.partition == partition)).to_numpy() & (valid.sum(axis=1) >= minimum_clear)
    return int(np.nonzero(chosen)[0][0])
