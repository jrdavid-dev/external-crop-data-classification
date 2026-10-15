# External crop data classification

Test a repeatable workflow from labelled parcels to saved model evaluation for **carrot, cabbage and potato**. The active bilinear collection targets **3,400 carrot, 7,000 cabbage and 10,000 potato candidates**, then retain **up to 500 usable unique pixels per crop** after quality and four-month window checks.

| Crop | Reference dataset | Imagery year |
|---|---|---|
| Carrot and potato | Sen4AgriNet | 2020 |
| Cabbage | FranceCrops | 2022 |

## Setup

Use Python 3.13 and a notebook editor. Run commands from the repository folder.

**macOS / Linux**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebooks.txt
cp settings.example.json settings.local.json
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
Copy-Item settings.example.json settings.local.json
```

Edit `settings.local.json` with your data folders and registered Earth Engine project. Select `.venv` as the notebook kernel. Earth Engine access is required for satellite retrieval; window and feature preparation use saved files. The exploratory training mode runs locally; production fitting and evaluation use Earth Engine.

## Notebook order

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [Sen4AgriNet preparation](Sen4AgriNet_Satellite_Preparation.ipynb) | Prepare carrot/potato candidates and retrieve imagery |
| 2 | [FranceCrops preparation](FranceCrops_Satellite_Preparation.ipynb) | Prepare cabbage candidates and retrieve imagery |
| 3 | [Four-month windows](Crop_Four_Month_Windows.ipynb) | Check quality, select windows and usable pixels, export features |
| 4 | [Feature engineering](Crop_Feature_Engineering.ipynb) | Derive SAVI and temporal-shape features; export experiment inputs |
| 5 | [Model training](Crop_Model_Training.ipynb) | Evaluate saved features locally, or run production Earth Engine experiments |

The first two notebooks prepare independent sources. Complete both before preparing the full three-crop pilot. Each notebook can start in a fresh kernel; run its cells in order.

**Sen4AgriNet defaults to reusing saved fields.** NetCDF downloading and field rebuilding stay off. Daily-summary retrieval, satellite downloading, and Earth Engine sign-in are enabled for the next collection run. Capacity is checked before network access. A new user without the saved inventory must obtain the source files and build the inventory first. See [workflow](docs/workflow.md).

## Active prototype run

All five notebooks use `bilinear_expanded`. New outputs stay separate from historical pilot runs. Finish both source handoffs before running the window notebook; it stops if either is missing.

Training defaults to local exploratory Random Forest evaluation, with parcel-separated train/validation/test groups. It reads this run's complete pixel tables, computes the feature designs and saves models and metrics. Formal feature exports may remain pending scientific review; those production requirements are separate from local exploratory testing.

The prototype checks the repeatable path from labelled parcels to saved model evaluation. Software tests and external-source accuracy do not establish performance in Benguet.

## Essential guides

Start with the [documentation index](docs/README.md) for a short glossary.

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

Code, five workflow notebooks, tests and documentation belong in Git. Raw data, imagery, generated results, credentials, `.venv` and local settings are excluded. Attribute the upstream datasets when using their data; their licenses do not assign a license to this project's code.
