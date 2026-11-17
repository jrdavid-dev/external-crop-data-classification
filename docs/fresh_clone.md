# Run from a fresh clone

Batch 2 update: active preprocessing is now [01–03](../notebooks/preprocessing/README.md). The old 05–07 filenames alias these new entry points. Finish pending collection before a live audit.

Current source entry points are collection 02 (Sen4AgriNet), 03 (FranceCrops), 04 (local EuroCrops), and 05 (JECAM test). Review each saved setup instead of treating historical limits below as defaults. Run collection 06, then preprocessing 05–07. See the [current collection guide](../notebooks/data_collection/README.md).

Use a writable Python 3.12+ notebook kernel, internet access and an Earth Engine-enabled Google Cloud project. Python 3.13 in a virtual environment is recommended.

1. Open a source notebook **02–04** and set `reference_mode = "download_missing"` in its setup cell. The saved `reuse` mode expects local references.
2. Review local paths and source limits. Sen4AgriNet has a separate 100-per-crop reference target and a **1,500-patch cap**. Its new imagery batches wait for 50 eligible unused parcels of each crop.
3. **Restart Kernel → Run All.** Setup installs missing dependencies and creates ignored `settings.local.json` if absent. Existing settings are preserved. Enter your Earth Engine-enabled project ID and complete sign-in when prompted.
4. Let one notebook finish environment setup before starting another. Use separate kernels for concurrent source collections and authenticate one at a time.
5. Rerun pending collections to retrieve saved cloud results and create annual TIFFs. Check the final summary for errors and the next action.
6. Once launched batches finish, run **04**. Repeat relevant sources → 04 for measured deficits; continue to **05–06** when pilot inputs are ready. Strict mode requires all collection targets.

All three notebooks use the same six sections. FranceCrops allows up to **100 cabbage parcels per batch**. The other two use a limit of **50 parcels per crop**. Sen4AgriNet supplies carrot/potato, FranceCrops cabbage for new batches, and CropHarvest whichever target crops pass its checks. Candidate limits do not guarantee usable yield or runtime.

## Inputs and storage

Sen4AgriNet reads selective label/parcel masks; FranceCrops obtains reference tables; CropHarvest obtains `labels.geojson`. Full upstream imagery or feature archives are not required. CropHarvest country borders are optional.

| Default path | Contents |
| --- | --- |
| `settings.local.json` | Local configuration; kept out of Git |
| `sen4agrinet_work/annual_archive/` | Versioned references, sampling masks, transport caches and summaries |
| `sen4agrinet_work/annual_tifs/` | Completed annual TIFFs by source/year/campaign/round |
| `sen4agrinet_work/adaptive_training/` | Selections, locked partitions, states and feedback |
| `sen4agrinet_work/francecrops/source_metadata/` | Reusable reference tables |
| `data/external/cropharvest/labels.geojson` | Reusable labels |

Set `annual_archive_root` before large transfers to choose another drive; `annual_tifs` is a sibling folder. Preserve existing archives and campaign files. `reuse` avoids new reference downloads; `offline` also prevents package installation and satellite calls and requires all inputs/dependencies locally.

See [workflow](workflow.md) for resume behavior and [adaptive training](adaptive_training.md) for targets and reference expansion.
