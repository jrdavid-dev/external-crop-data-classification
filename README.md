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

Steps 1–15 have executable code. Steps 9–15 cover evidence review, four-month features, complete-data filtering, independent-field partitions, Random Forest comparisons and run packaging. Follow the [review and training guide](docs/remaining_steps.md) to supply evidence and independent local data. Detector outputs remain candidates until reviewed; no real training result is implied by implementing these steps.

## Supporting files

| File | Purpose |
|---|---|
| `crop_workflow.py` | Quality checks, scaling, indices, plots and candidate detection |
| `season_workflow.py` | Season review, 40/64 features, field partitions, models and provenance |
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
