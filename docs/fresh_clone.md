# Run from a fresh clone

Choose one input route from the [workflow guide](workflow.md): collect source imagery, or [import shared imagery](sharing_imagery.md) and start preprocessing 01. Supplemental collection notebooks are optional.

Use Python 3.12+. New imagery collection also requires an Earth Engine-enabled Google Cloud project. Local preprocessing, training and testing on imported imagery do not. See the repository [setup commands](../README.md#setup).

1. Use `download_missing` where needed in Collection 02, 03, and 05. Collection 04 expects the local `data/external/eurocrops_v2/NL_2021.parquet` file.
2. Review the saved crop filters, limits and local input paths. On a fresh clone select `collection_cohort = "jecam"` in 05 for the primary test. Review its crop filter against the planned cohort.
3. Run the collection notebooks. Create local settings and authenticate if prompted. Rerun pending collections to resume. Run Collection 08 to review the saved progress.
4. After collection completes, follow the [workflow](workflow.md) through preprocessing, training, and testing.

## Inputs and storage

Sen4AgriNet reads selective label/parcel masks; FranceCrops obtains reference tables; CropHarvest obtains `labels.geojson`. Full upstream imagery or feature archives are not required. CropHarvest country borders are optional.

| Default path | Contents |
| --- | --- |
| `settings.local.json` | Local configuration; kept out of Git |
| `data/pipeline_work/annual_archive/` | Versioned references, sampling masks, transport caches and summaries |
| `data/pipeline_work/annual_tifs/` | Completed annual TIFFs by source/year/campaign/round |
| `data/pipeline_work/adaptive_training/` | Selections, locked partitions, states and feedback |
| `data/pipeline_work/francecrops/source_metadata/` | Reusable reference tables |
| `data/external/cropharvest/labels.geojson` | Reusable labels |

Set `annual_archive_root` before large transfers to choose another drive; `annual_tifs` is a sibling folder. Preserve existing archives and campaign files. `reuse` avoids new reference downloads; `offline` also prevents package installation and satellite calls and requires all inputs/dependencies locally.

See [workflow](workflow.md) for resume behavior and [adaptive training](collection.md) for targets and reference expansion.

## Use shared imagery

To run locally without recollecting imagery, use the [TIFF export/import guide](sharing_imagery.md). Import the complete package, then start preprocessing 01.

See [local data storage](storage_layout.md) for the consolidated data root and saved-path migration records.
