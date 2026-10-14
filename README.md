# External crop data classification

Prepare external crop-reference data for exploratory crop classification in Benguet. The current Sen4AgriNet notebook implements Steps 1–8: inventory, inspection, field-fragment extraction, deduplication, balanced pixel sampling, cloud-screened semi-monthly L2A composites, quality checks, timelines and candidate green-up detection. Steps 9–15 remain methodology outlines.

The pilot selects 1,000 carrot and 1,000 potato pixels across 20 conservative parcel groups per crop. It downloads 24 annual composites per region: 960 planned GeoTIFFs. SCL and cloud probability below 40 are applied before daily mosaicking and median reduction. Sampling masks distinguish selected training pixels from surrounding image context. Cabbage is not present in the inspected S4A 2020 subset; FranceCrops has a companion outline.

## Setup

Use Python 3.13, Git and a notebook editor such as VS Code with the Python and Jupyter extensions. From the repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
Copy-Item settings.example.json settings.local.json
```

Edit `settings.local.json` with your dataset root, imagery folder and registered Earth Engine project. The dataset root is the parent of `data`, so raw 2020 files resolve to `dataset_root/data/2020/<tile>/`. Select `.venv` as the notebook kernel. Local settings are ignored by Git.

## Running

Open `Sen4AgriNet_Workflow_Outline.ipynb` from this folder. Run settings first. Download cells are optional when the relevant files already exist. Run Steps 3–4 to reconstruct inventories and the balanced pilot, or restore their generated outputs from a previous machine. Step 5 requires Earth Engine authentication. Its masks, grid and temporal settings are fingerprinted; a changed cloud threshold creates a different request folder.

Steps 6–8 use `crop_workflow.py` and `observed_timelines.py`. Step 6 reports missing/failed intervals, applies reflectance scaling and checks sampling alignment. Step 7 builds composite curves and optionally retrieves small actual-date vegetation summaries without more image downloads. Step 8 prefers observed summaries; composite-only green-up remains provisional. Output folders are excluded from Git.

The detector parameters are development settings, not validated vegetable phenology rules. Outputs are candidate intervals, not planting dates or accepted training seasons. Step 9 must validate onset estimates, resolve crop-cycle label relevance and freeze parameters before final testing. Same parcel groups must stay together in later splits; their scope and independent-field identities still need verification.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover scaling/nodata, moved-PC paths, missing composites, weak coverage, flat signals, long onset gaps, multi-cycle ambiguity and composite-only uncertainty. They do not independently validate crop labels, cloud screening or phenology accuracy.

## What belongs in this repository

Include notebooks with outputs cleared, source modules, tests, dependency versions, setup examples and methodology notes. Do not upload raw NetCDF files, GeoTIFFs, generated spatial inventories, progress files, local settings, credentials or `.venv`. The public repository contains workflow code, not downloaded datasets. See `MOVING_PC.md` for the transfer checklist.

## Sources

- [Sen4AgriNet](https://huggingface.co/datasets/orion-ai-lab/S4A)
- [Sentinel-2 L2A](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Cloud probability and matching](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY)

Dataset access and redistribution remain subject to their own terms. No project license is assigned automatically; choose one before encouraging third-party redistribution of the code.

The active notebook uses `crop_workflow.py` and `observed_timelines.py`; `migrate_progress.py` supports moving PCs. Inspection and boundary extraction are implemented in the notebook itself. Older standalone inspection/visualization scripts and exploratory outputs are archived locally under `archive/legacy_2026-10-14/` and excluded from Git. They remain recoverable locally and, where previously tracked, in repository history.
