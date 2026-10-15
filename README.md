# External crop data classification

Prepare Sentinel-2 data for a three-crop prototype: **carrot, cabbage and potato**. Select **3,000 candidate pixels per crop**, then retain **up to 1,000 usable unique pixels per crop** after quality and four-month window checks.

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

Edit `settings.local.json` with your data folders and registered Earth Engine project. Select `.venv` as the notebook kernel. Earth Engine access is required for satellite retrieval; window preparation and training use saved files.

## Notebook order

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [Sen4AgriNet preparation](Sen4AgriNet_Satellite_Preparation.ipynb) | Prepare carrot/potato candidates and retrieve imagery |
| 2 | [FranceCrops preparation](FranceCrops_Satellite_Preparation.ipynb) | Prepare cabbage candidates and retrieve imagery |
| 3 | [Four-month windows](Crop_Four_Month_Windows.ipynb) | Check quality, select windows and usable pixels, export features |
| 4 | [Model training](Crop_Model_Training.ipynb) | Load prepared inputs, train and evaluate |

The first two notebooks prepare independent sources. Complete both before preparing the full three-crop pilot. Each notebook can start in a fresh kernel; run its cells in order.

**Sen4AgriNet defaults to reusing saved fields.** Run All leaves NetCDF downloads and field rebuilding disabled. A new user without the saved inventory must obtain the source files and build the inventory first. See [workflow](docs/workflow.md).

## Essential guides

- [Workflow](docs/workflow.md): run modes, notebook steps, saved files and common problems.
- [Methodology](docs/methodology.md): imagery rules, four-month timing, usable pixels and limitations.
- [Training and evaluation](docs/training.md): field partitions, required evidence, model inputs and metrics.

## Development

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.\.venv\Scripts\python.exe` instead. Tests verify software behavior; they do not establish crop-label or phenology accuracy.

Code, four notebooks, tests and documentation belong in Git. Raw data, imagery, generated results, credentials, `.venv` and local settings are excluded. Attribute the upstream datasets when using their data; their licenses do not assign a license to this project's code.
