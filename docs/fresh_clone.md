# Run from a fresh clone

No existing NetCDFs, GeoJSON labels, Parquet tables, local settings or generated results are required.

## First run

1. Clone the repository and open its folder in your notebook editor.
2. Choose a writable Python 3.12+ kernel; Python 3.13 in a virtual environment is recommended. Internet access and pip are required for missing packages and datasets.
3. Open source notebooks **01, 02 and 03**, then **Restart Kernel → Run All**. Each has its download switches enabled. Their first cell installs missing dependencies in the active kernel, installs this clone as an editable package for later kernels, and prepares local paths.
4. If no project is configured, enter your registered Earth Engine Google Cloud project ID when prompted. It is saved in ignored `settings.local.json`. Complete the Earth Engine sign-in prompt. Project registration and account access must already be available; the notebooks do not create a Google account or Cloud project.
5. Wait for the launched **small candidate batches** to report `completed`. A `pending_retry` pass can include cloud jobs still running. Run All again to collect existing work. Each source selects at most 25 new eligible parcels per crop.
6. Run **04 Four-month windows** to assess usable pixel and parcel counts. If targets are short, repeat source notebooks → 04; only deficit crop/partition combinations receive new candidates. Run **05–06** after the targets are met. See [adaptive training](adaptive_training.md).

You do not need to copy settings or datasets from another computer. Existing settings are preserved. Installed dependencies are reused; only missing packages are installed automatically. If package installation fails, fix the displayed environment error before continuing. Let the first source finish environment setup before starting others, so two kernels do not install packages simultaneously. Use separate kernels for concurrent collections, and authenticate one at a time.

## Automatic source inputs

| Notebook | What it obtains |
|---|---|
| 01 Sen4AgriNet | Complete pinned year/tile manifest and selective label/parcel-mask reads; no full imagery NetCDF downloads |
| 02 FranceCrops | Pinned class-map, metadata and parcel-footprint tables |
| 03 CropHarvest | Pinned `labels.geojson` if missing |

FranceCrops does not fetch its interpolated imagery arrays. CropHarvest does not fetch its prebuilt imagery or feature archives. All three use Earth Engine to process the new Sentinel-2 archive, then save the results locally.

The optional CropHarvest country-boundary ZIP is not a prerequisite. Without it, the existing dataset-scoped identity fallback is used and recorded through the reference-processing run.

## Default storage and preservation

- `settings.local.json`: created on first run; kept out of Git.
- `sen4agrinet_work/annual_archive/`: versioned references, sampling masks, transport caches and summaries.
- `sen4agrinet_work/annual_tifs/`: organized annual TIFFs by source, year, campaign and round.
- `sen4agrinet_work/adaptive_training/`: batch selections, locked partitions and measured deficits.
- `sen4agrinet_work/francecrops/source_metadata/`: reusable FranceCrops tables.
- `data/external/cropharvest/labels.geojson`: reusable CropHarvest labels.

Change `annual_archive_root` before large transfers if you prefer another local drive. Old inputs and results remain preserved. Missing files are retrieved; complete verified files/checkpoints are reused. Sen4AgriNet caches small reference masks; reference-only mode does not stage complete NetCDF files.

An empty clone removes the need to prepare data manually; it does not remove download time, local disk requirements or Earth Engine access requirements. Formal production exports also retain their scientific evidence and freeze requirements.
