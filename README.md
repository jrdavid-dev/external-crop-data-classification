# External crop data classification

Prepare Sen4AgriNet crop-reference data and Sentinel-2 imagery for a Benguet crop-classification study. The notebook implements inventory, geometry checks, balanced sampling, cloud-screened composites, quality checks, timelines and candidate green-up detection.

The current pilot contains **1,000 carrot pixels and 1,000 potato pixels**, with 20 conservative parcel groups per crop. It plans 960 annual half-month composite GeoTIFFs. Cabbage remains a target crop, but no verified cabbage source is configured.

## Quick setup

Use Python 3.13, Git and a notebook editor. From the repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
Copy-Item settings.example.json settings.local.json
```

Edit local settings with your data paths and registered Earth Engine project. Select `.venv` as the kernel, then open [the notebook](Sen4AgriNet_Workflow_Outline.ipynb). Run settings before later sections. Existing raw files do not need to be downloaded again.

## Guides

- [Documentation index](docs/README.md)
- [Study plan](docs/study_plan.md)
- [Execution order and outputs](docs/workflow.md)
- [Crop-label evidence](docs/label_quality.md)
- [Moving to another PC](docs/moving_pc.md)

Steps 1–8 are implemented. Steps 9–15 remain plans. Detector outputs are candidate intervals, not verified planting dates or accepted training seasons.

## Supporting files

| File | Purpose |
|---|---|
| `crop_workflow.py` | Quality checks, scaling, indices, plots and candidate detection |
| `observed_timelines.py` | Small actual-date summaries without more imagery downloads |
| `migrate_progress.py` | Update saved paths after moving PCs |
| `settings.example.json` | Portable configuration template |
| `requirements-notebooks.txt` | Environment package versions |
| `tests/` | Processing and detector checks |

Run tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Repository contents

Git contains code, the notebook, tests and documents. Raw NetCDF files, GeoTIFFs, generated outputs, `.venv`, credentials and local settings are excluded. Transfer data and `sen4agrinet_work` separately when moving PCs.

Software tests do not validate crop identity, cloud screening or phenology accuracy. Review independent evidence before accepting field-seasons. Dataset access and redistribution remain subject to source terms; no project code license is assigned automatically.

[Sen4AgriNet dataset](https://huggingface.co/datasets/orion-ai-lab/S4A)
