# External crop data classification

Prepare a repeatable classification workflow for **carrot, cabbage and potato**. The default collection, `full_year_all_parcels`, uses every eligible parcel in the supplied Sen4AgriNet, FranceCrops and CropHarvest references. Earth Engine processes full-year observations; local storage holds 24 half-month composites and dated vegetation summaries. Window selection happens afterward.

Collection is **configured for the next Run All**, but no work starts until cells execute. Old pilot datasets and results remain separate. See [full-year collection](docs/full_year_collection.md) for reference coverage and storage planning.

## Setup

**A fresh clone needs no existing datasets or settings JSON.** Open the repository in your notebook editor, select a Python 3.12+ kernel (3.13 recommended), and Run All in source notebooks 01–03. They install missing dependencies in the active environment, make the cloned package available to fresh kernels, create portable local settings, and obtain missing reference inputs automatically.

Enter your registered Earth Engine project ID once if prompted, then complete sign-in. A compatible Python kernel, pip, internet access and an Earth Engine-enabled account/project are still required. Use a virtual environment to keep dependencies isolated. Nothing starts merely by opening the notebooks.

See the [fresh-clone guide](docs/fresh_clone.md) for the complete sequence and storage paths. Manual installation remains available:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebooks.txt
.venv/bin/python -m pip install -e .
```

On Windows, use `py -3.13` and `.venv\Scripts\python.exe`. Local storage is the default; choose another local drive with `annual_archive_root` in the automatically created settings file.

## Notebook order

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [Sen4AgriNet preparation](notebooks/01_sen4agrinet_preparation.ipynb) | Discover all upstream years/tiles, extract target references and annual imagery |
| 2 | [FranceCrops preparation](notebooks/02_francecrops_preparation.ipynb) | Prepare cabbage/table-potato references and annual imagery |
| 3 | [CropHarvest preparation](notebooks/03_cropharvest_preparation.ipynb) | Screen references and prepare annual imagery |
| 4 | [Four-month windows](notebooks/04_four_month_windows.ipynb) | Combine selected sources, choose windows and usable pixels |
| 5 | [Feature engineering](notebooks/05_feature_engineering.ipynb) | Prepare the feature experiments |
| 6 | [Model training](notebooks/06_model_training.ipynb) | Fit models and evaluate parcel-separated predictions |

Finish all three source collections before window selection. Missing handoffs stop downstream processing. Sources have no pilot pixel budgets; notebook 04 retains every usable pixel while keeping the existing timing and quality checks. FranceCrops supplies explicit cabbage and table-potato classes, with no carrot class.

## Project folders

Notebooks are in `notebooks/`; shared Python code is in `src/crop_pipeline/`, including the `visualization` helpers. The source-notebook bootstrap installs this clone as an editable package for fresh downstream kernels. Settings, datasets and generated results stay in their existing root-level locations; they do not move into the notebook folder.

## Collection and evaluation

Set `annual_archive_root` in local settings to choose local or external-SSD storage. Run sources with collection disabled to inspect reference scope and exact grid-based raster estimates. Sen4AgriNet builds a complete pinned-release reference inventory using selective mask reads, avoiding full NetCDF downloads and preserving the old subset. CropHarvest fetches its pinned labels only when missing; FranceCrops fetches missing pinned reference tables automatically.

Notebooks 04–06 share the selected collection. Local exploratory training keeps parcels separate and balances each crop's total training weight. Formal production exports retain their scientific evidence and freeze requirements. External-source accuracy does not establish Benguet performance.

## Essential guides

Start with the [documentation index](docs/README.md) for a short glossary.

- [Progress and estimates](docs/progress.md): measured stage ETAs and saved notebook timings.
- [Fresh clone](docs/fresh_clone.md): Run All without existing datasets or settings files.
- [Sen4AgriNet references](docs/sen4agrinet.md): dataset-wide discovery, Run All, checkpoints and raw-file preservation.
- [Full-year collection](docs/full_year_collection.md): scope, storage estimates, switches and completed handoffs.
- [CropHarvest](docs/cropharvest.md): run modes, observation years, safe batch collection and supplemental handoffs.
- [Workflow](docs/workflow.md): run modes, notebook steps, saved files and common problems.
- [Methodology](docs/methodology.md): imagery rules, four-month timing, usable pixels and limitations.
- [Feature experiments](docs/feature_experiments.md): SAVI, temporal shape, bilinear processing and model comparisons.
- [Training and evaluation](docs/training.md): field partitions, required evidence, model inputs and metrics.

## Development

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.\.venv\Scripts\python.exe` instead. Tests verify software behavior; they do not establish crop-label or phenology accuracy.

Code, workflow notebooks, tests and documentation belong in Git. Raw data, imagery, generated results, credentials, `.venv` and local settings are excluded. Attribute the upstream datasets when using their data; their licenses do not assign a license to this project's code.
