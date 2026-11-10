"""Fixed analysis settings and input/output paths for notebook 07."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from crop_pipeline.features import experiments as feature_experiments
from crop_pipeline.training import external_combined as ext

SEED = 42
CLASSES = list(ext.CLASSES)
YEARS = list(range(2019, 2027))
LABEL_YEARS = range(2019, 2026)
BANDS = ["B3", "B4", "B5", "B8", "B11"]
VARS = BANDS + ["NDVI", "NDRE", "NDMI", "SAVI"]
INTERVAL = feature_experiments.feature_sets()["interval_only"]
SHAPE = feature_experiments.feature_sets()["shape_only"]

MIN_CLEAR = 6  # of 8 half-month composites in an Atok window
ATOK_FLOOR = 5  # samples are built down to 5 clear composites for the clear-data sensitivity test
MIN_PARCELS_PER_CROP = 5  # a region set is eligible for the climate experiment only with >= 5 parcels of every crop
TARGET_SELECTIVE_ACCURACY = 0.70  # abstention: lowest top-class probability whose kept predictions reach this accuracy
OOD_QUANTILE = 0.95  # out of distribution: farther from the training data than 95% of the external test samples
N_TREES_SEARCH = 500  # trees for the many-fit experiments (tuning grid, region subsets); final models use 1,000
ATOK_ELEVATION_M = 2255
LAPSE_C_PER_M = 0.0065

PROTOTYPE = "RF, interval (prespecified prototype)"
LR_SHAPE = "LR, shape (v1 CV-selected)"

COLORS = {"carrot": "#F58518", "cabbage": "#54A24B", "potato": "#B279A2"}
SOURCE_COLORS = {"Atok": "#E45756", "CropHarvest": "#4C78A8", "Sen4AgriNet": "#72B7B2", "FranceCrops": "#B79A20"}
SOURCE_NAMES = {"cropharvest": "CropHarvest", "sen4agrinet": "Sen4AgriNet", "francecrops": "FranceCrops"}


@dataclass(frozen=True)
class Paths:
    """Inputs read and outputs written by notebook 07, resolved from settings.local.json."""

    earth_engine_project: str
    v1: Path  # notebook 07 v1: cached tables, locked test split, CV selection
    windows_run: Path  # notebook 04 run behind the francecrops_s4a table
    output: Path
    figures: Path
    imagery_root: Path
    kml: Path
    landuse: Path  # NAMRIA ELUM CY-2023
    cropharvest_labels: Path
    countries_zip: Path
    land_activity: Path  # notebook 09, for comparison only
    cube_caches: tuple
    pagasa: Path | None  # CSV: year, month, tmax, tmin (°C)
    pagasa_elevation_m: float  # Baguio synoptic station


def chdir_to_repository():
    """Run from the repository root, whether the kernel starts there or in notebooks/."""
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)


def load_paths(settings_path="settings.local.json"):
    settings_path = Path(settings_path)
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    report_folder = Path(settings.get("report_root", "./sen4agrinet_work/inventory")).expanduser().resolve()
    analysis_folder = report_folder.parent / "analysis"  # notebooks 04-06 (pipeline runs)
    analysis_root = Path(settings.get("analysis_root", "./analysis")).expanduser().resolve()  # notebooks 07-10
    output = analysis_root / "transferability"
    pagasa = settings.get("pagasa_baguio_path")
    return Paths(
        earth_engine_project=settings.get("earth_engine_project", "thesis-18"),
        v1=analysis_root / "external",
        windows_run=analysis_folder / settings.get("external_windows_run", "bilinear_expanded_carrot_gate_waived") / "seasons",
        output=output,
        figures=output / "figures",
        imagery_root=Path(settings.get("atok_imagery_root", "")).expanduser(),
        kml=Path(settings.get("atok_kml_path", "./data/atok/atok_benguet.kml")).expanduser(),
        landuse=Path(settings.get("atok_landuse_path", "./data/atok/elum2023_benguet.gpkg")).expanduser(),
        cropharvest_labels=Path(settings.get("cropharvest_labels_path", "./data/external/cropharvest/labels.geojson")).expanduser(),
        countries_zip=Path("./data/external/cropharvest/ne_50m_admin_0_countries.zip"),
        land_activity=analysis_root / "atok" / "regimes" / "land_activity.tif",
        cube_caches=(analysis_root / "atok" / "weak_labels" / "roi_cube.npz", analysis_root / "atok" / "regimes" / "roi_cube.npz"),
        pagasa=Path(pagasa).expanduser() if pagasa else None,
        pagasa_elevation_m=float(settings.get("pagasa_station_elevation_m", 1500)),
    )


def check_inputs(paths):
    """Stop early, naming the settings key, when a required input is missing; create the output folders."""
    checks = (
        ("Imagery folder", "atok_imagery_root", paths.imagery_root, paths.imagery_root.is_dir() and str(paths.imagery_root) != "."),
        ("KML", "atok_kml_path", paths.kml, paths.kml.is_file()),
        ("Land-use map", "atok_landuse_path", paths.landuse, paths.landuse.is_file()),
        ("CropHarvest labels", "cropharvest_labels_path", paths.cropharvest_labels, paths.cropharvest_labels.is_file()),
        ("Notebook 07 v1 outputs", "-", paths.v1 / "test_split.csv", (paths.v1 / "test_split.csv").is_file()),
    )
    missing = [f"{name} ({key}): {path}" for name, key, path, found in checks if not found]
    if missing:
        raise FileNotFoundError("Missing input(s); see settings.example.json:\n  " + "\n  ".join(missing))
    paths.figures.mkdir(parents=True, exist_ok=True)
