# Move the project to another PC

The repository contains code and documents. Data, generated reports, local settings and `.venv` are not uploaded.

## 1. Get the code

Clone [the repository](https://github.com/jrdavid-dev/external-crop-data-classification), or copy its small source files. Keep the notebook, helper modules, tests, requirements and example settings together.

## 2. Transfer data separately

Stop downloads before copying progress. Move the existing drive or copy these folders, preserving their structure:

| Folder | Purpose |
|---|---|
| `ground_truth/data/2020/` | Reinspection and rebuilding source geometry |
| `ground_truth/sentinel2_pilot_composites/` | Current pilot imagery |
| Project `sen4agrinet_work/` | Masks, selections, inventories, progress and results |

Raw NetCDFs are not needed to continue Steps 5–8 if required inventories and masks are restored. Keep masks with selection records and progress/checksums with imagery.

Abandoned full-acquisition imagery is optional; the pilot does not use it. Current work does not require 2019. Do not copy obsolete downloads merely because they share a parent folder.

## 3. Rebuild the environment

Install Python 3.13, Git and a notebook editor such as VS Code with Python/Jupyter support. In the new repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
Copy-Item settings.example.json settings.local.json
```

Do not copy `.venv` between PCs. Edit local settings for the new data roots, report folder and registered Earth Engine project. The dataset root is the parent of `data`. Select the new kernel and authenticate Earth Engine when needed. Never commit tokens.

## 4. Update saved paths

After restoring outputs and imagery, run:

```powershell
.\.venv\Scripts\python.exe migrate_progress.py
```

This updates local paths while preserving request IDs, settings and checksums. Check its missing-file count before downloading again.

## 5. Resume and verify

Run Step 2 settings, then all Step 5 settings/connection/helpers/inventory cells. Reuse compatible inventories and observed-summary checkpoints. Step 6 tries current roots before old absolute paths and reports absent or invalid files.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use the [workflow guide](workflow.md) for execution order. Scientific validation is still required even when software tests pass.
