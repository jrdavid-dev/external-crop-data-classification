# Move the workflow to another PC

## Clone or copy the small code repository

Bring the notebooks, Python source modules, tests, `requirements-notebooks.txt`, `settings.example.json` and methodology notes. These belong in Git. Notebook cell outputs are deliberately cleared in the repository to keep images and long download logs out of commits.

## Transfer data separately

Stop running downloads cleanly before copying progress. Use the existing data drive or a separate external drive. Preserve directory structures:

- `ground_truth/data/2020/`: raw Sen4AgriNet files. Useful for reinspection and rebuilding geometry; not required to continue Steps 5–8 if all generated inventories/masks are restored.
- `ground_truth/sentinel2_pilot_composites/`: current pilot GeoTIFFs, retaining fragment/request folders and filenames.
- `sen4agrinet_work/`: generated geometry, pixel-selection records, sampling masks, acquisition inventories, cloud-match records, download-progress/checksum files, timelines and green-up results. This folder is not in Git; copy it into the new project folder to resume.
- Your old `settings.local.json`: use as a reference only; update drive letters and paths on the new PC.

Previously abandoned full-acquisition imagery and full-dataset composite downloads are optional archives; the pilot does not use them. Do not delete them automatically. 2019 data is deferred and not required for the current 2020 pilot.

Keep sampling masks and selected-pixel records together. They define exactly which pixels belong to the balanced pilot. Keep progress files with their associated imagery; otherwise the notebook cannot verify reuse reliably.

## Recreate the environment

Install Python 3.13 and Git; install your preferred notebook editor and its Python/Jupyter support. Create a new `.venv` and install `requirements-notebooks.txt` using the commands in README. Do not copy `.venv` between PCs: its interpreter paths and platform files are machine-specific.

Copy `settings.example.json` to `settings.local.json`, edit the storage roots/project, and select the new `.venv` kernel. Authenticate Earth Engine again when required; do not copy authentication tokens into the repository.

## Resume and verify

Run the Step 2 settings cell, then Step 5 settings/connection/helpers/inventory. Acquisition metadata can be reused after the corresponding output folders are restored. Before running downloads, rebuild progress paths with `python migrate_progress.py` if drive letters have changed. That utility changes only recorded local paths, preserving request IDs, source settings and checksums.

Steps 6–8 resolve masks and GeoTIFFs from current project/data roots first, so old absolute paths do not prevent local analysis. Rerun Step 6 to identify absent or invalid composites. Observed-summary checkpoints avoid repeating successful metadata retrieval. No code should treat missing bins as cloud-free observations.

The old local archive has been deleted and is not needed for the notebook or the PC move. Previously tracked standalone scripts can be recovered from Git history if needed.
