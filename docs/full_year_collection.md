Current default: [small adaptive training batches with organized annual TIFFs](adaptive_training.md). The full-catalogue and benchmark modes below remain optional.

# Full-year collection

The new collection is **`full_year_all_parcels`**. The three source notebooks archive satellite observations first. Notebook 04 chooses usable four-month windows afterward. Timing changes can reuse saved imagery without repeating collection.

## What is collected

- Every eligible declared carrot, cabbage or potato parcel in the supplied reference inventories. No pilot candidate budgets, 100-pixel parcel quotas or 500-pixel usable caps apply.
- All 24 calendar half-month median composites in each parcel's reference year, with six spectral bands and a clear-date count.
- Actual-date NDVI, NDRE and NDMI summaries for timing detection.
- Every eligible interior pixel. Large parcels use bounded tiles, which retain their original parcel group for training splits.

This is a processed annual archive, **not every original Sentinel-2 scene**. It supports changing window selection and later pixel selection. Changing cloud masks, bands or compositing requires a new archive revision.

Start with the [fresh-clone guide](fresh_clone.md) when no local files or settings are present.

## Before collection

1. Run the source notebooks to obtain missing reference labels and boundaries automatically. Sen4AgriNet discovers every patch across all years and tiles in a pinned upstream release, then reads label/parcel masks through bounded HTTP byte ranges. Its new complete reference inventory replaces the old subset as input, while preserving old files. See [complete Sen4AgriNet collection](sen4agrinet.md). FranceCrops automatically fetches missing pinned reference tables, including cabbage and table potato; it has no carrot class. CropHarvest automatically retrieves its pinned label file when missing; existing labels are reused. Eligibility still requires polygons and resolved observation years.
2. Choose `annual_archive_root` in `settings.local.json`. An external SSD is suitable. Use an absolute mounted-drive path, and keep it connected throughout collection and window processing. The default is `sen4agrinet_work/annual_archive`.
3. For preparation only, set `run_satellite_collection = False`. Sen4AgriNet still retrieves references while its reference-mask retrieval switch is enabled. This writes local masks, audits and an exact grid-based raster estimate without querying imagery.
4. Read `estimate`: parcels, tiles, composite files, uncompressed raster GiB and available space. Seven float32 bands × 24 bins require **672 bytes per archived grid cell per year**, before compression and file headers. Allow additional space for summaries, features and models; collection checks a two-times-raster free-space allowance, which is a working allowance, not an exact total-size prediction.
5. When ready, explicitly enable `run_satellite_collection` in the three source notebooks. Set `authenticate_now = True` only if sign-in is needed. The three notebooks are currently configured with collection and authentication enabled for the next Run All; no collection happens until cells execute.

No imagery is collected just by opening a notebook. Sen4AgriNet reference-only retrieval is enabled for Run All; full source NetCDF files are not fetched. CropHarvest label retrieval is also enabled and fetches only missing labels. FranceCrops now retrieves missing pinned reference tables automatically.

## Processing and local storage

Earth Engine applies the existing SCL, cloud-probability and scene-edge masks, aligns B5/B11 with bilinear resampling, mosaics each date and computes half-month medians. Small bounded results are retrieved directly to local GeoTIFFs; this workflow does not require permanent Earth Engine assets or Drive copies.

The retriever caches acquisition metadata, fingerprints requests, validates saved rasters and checkpoints each interval. A pass continues through tile failures; restarting collection reuses verified files and retries missing or damaged intervals. Keep the collection files and masks together; local manifests contain absolute paths.

Outputs are versioned under:

```text
<annual_archive_root>/<source>/runs/full_year_all_parcels/<revision>/
  fields/<year>/                 parcel tile masks and region inventory
  imagery/<year>/                half-month GeoTIFFs
  reports/<year>/                acquisition metadata and completion records
  timelines/<year>/tiles/        dated index summaries
  handoffs/<year>.json           completed acquisition contract
  collection_plan.json           scope, provenance and storage estimate
```

Source revisions include reference content, archive settings and helper-code hashes. Historical `bilinear_expanded` imagery, datasets, models and thesis-team guides are not moved or deleted.

## Windows, features and training

After all three sources finish, notebook 04 requires their hash-checked handoffs. A content-specific analysis folder keeps different archive selections separate. All downstream notebooks default to this collection. Set `analysis_collection` to `bilinear_expanded` only when deliberately reopening the old pilot.

Notebook 04 audits all saved intervals using the existing 80% coverage rule. Pixel records are partitioned on disk in Parquet, and quality/feature processing reads one tile at a time. It uses the existing detector and timing checks, evaluates cycles separately, and chooses the earliest date range with complete usable pixels per parcel. All eight bins and finite features are required; no imputation is added. All usable pixels are retained. Annual runs plot one example per crop; expensive optional sensitivity exports require `run_development_experiments = True`. Those optional exports do not replace or disable the primary timing checks.

FranceCrops' original validation/test roles are preserved. Their windows are saved separately in `reserved_upstream_features.csv`, outside model-development tables. CropHarvest upstream test labels retain the existing exclusion.

Run notebooks 05 and 06 afterward for feature designs and parcel-separated evaluation. Final feature tables and model fitting still use memory proportional to the usable training set; disk-backed raster auditing does not make model fitting out-of-core.

## Remaining timing limitation

A full January–December archive does not supply the adjacent year. A four-month window crossing its boundary remains incomplete, while valid in-year alternatives can still pass. Additional dates require an explicit later collection extension. Annual crop labels and satellite rises still do not establish exact planting or harvesting dates.

## Daily summaries use the saved pixel selection

The direct collector now applies the saved binary sampling mask on the exact imagery CRS, transform and dimensions before reducing daily NDVI, NDRE and NDMI. Geometry outlines no longer determine which pixels enter these summaries. Cloud/edge masks and zero-denominator exclusions remain active; coverage still requires 80% of selected pixels. Any index reporting more than the selected count remains an error.

Corrected daily checkpoints use the `selected_pixel_grid_v2` recipe and include the mask content, acquisition list and coverage threshold in their identity. Legacy checkpoints remain untouched and are not reused as corrected summaries. Previous aggregate JSON/CSV files are preserved under `summary_history/` before the convenience files are updated. Imagery request settings and saved GeoTIFFs are unchanged.

To resume an already prepared run in the current kernel after this fix, reload the timeline helper, then rerun only **Collect annual imagery** when ready:

```python
import importlib
from crop_pipeline.imagery import timelines
importlib.reload(timelines)
```

This keeps the existing `annual_plan` and paths. No reference preparation is needed for that resume. Restart Kernel → Run All can now resume a compatible imagery archive even after retry or daily-summary fixes. Changed inputs, imagery rules or planning logic still create a separate archive; prior runs remain preserved.

## Continue through failures, then retry on Run All

All three source notebooks share the same collection-pass behavior:

1. Attempt every tile across every planned year. A failed image, daily summary or tile inventory is recorded without stopping later tiles.
2. Attempt imagery even if that tile's daily summary failed. Both must still complete before its handoff is accepted.
3. Return `pending_retry` with completed-tile counts, downloaded/reused-image counts and a retry-queue path if any work remains. This is not a completed source collection.
4. On the next **Restart Kernel → Run All**, reuse verified images and corrected daily checkpoints. Only failed, missing or damaged imagery is fetched again.
5. Register the source only when the unchanged all-interval completion gate and daily-summary requirements pass for every tile. Notebook 04 blocks while a source's active state is pending, even if an older source registry remains on disk.

Each pass creates its own `collection_attempts/<run-id>/` folder with tile-event JSONL, an immutable retry queue, a pass summary and the actual collector/timeline code hashes. `latest_collection_pass.json` points to the most recent outcome. Earlier reports, images and results remain preserved. An interrupted pass retains the individual image manifests and daily checkpoints, so the next Run All can verify and resume them even before a final retry queue was written.

Imagery identity now follows reference content, the verified planning implementation and the imagery recipe. A retry-orchestration or daily-summary-only change can reuse a compatible archive. A known previous planner is recognized explicitly; unknown legacy code, changed references, changed imagery settings or changed planning logic create a separate run. Original run configurations are not rewritten. Corrected daily summaries remain independently versioned by their mask-specific contract.

Reference validation, authentication and insufficient-storage errors remain setup blockers. The continue-through-failures behavior applies to tile collection after setup succeeds. No incomplete source is promoted to training input.

## Faster source collection

All three source notebooks use the same optimized collector. The imagery recipe and archive paths remain compatible with existing downloads.

| Setting | Default | Purpose |
| --- | --- | --- |
| `download_workers` | `2` | Run at most two grouped image requests at once, within one notebook. |
| `intervals_per_request` | `6` | Request up to six half-month composites together, then save the original seven-band files. |

Groups automatically shrink to a **24 MiB uncompressed payload budget**. A new 24-image tile normally needs four grouped requests instead of 24 individual requests. This reduces request overhead; it does not guarantee a particular speedup. Daily summaries and Earth Engine processing still take time.

- Existing images must pass the same grid and checksum checks before reuse. Only missing, failed or damaged intervals enter a new request.
- Nearby tiles share cached yearly scene metadata. Each tile still filters both reflectance and cloud-probability footprints separately and requires an exact image-index match.
- Unchanged parcel preparation is reused after checking the saved plan, outlines, region inventory and mask checksums. Older runs without this receipt are prepared once before they gain this cache.
- A transient HTTP failure gets one delayed retry. Persistent failures remain in the retry queue while other groups continue. The next Run All requests only unfinished images.
- Workers do not write the manifest concurrently. Each completed group's records are checkpointed by the main collector thread. An interrupted group may need to be requested again.

**To use the changes:** stop the old kernel, reopen the notebook if needed, then Restart Kernel → Run All when ready. Do not run two copies of the same source archive simultaneously. If running all three sources together, `download_workers = 1` in each notebook is a conservative starting point. Worker limits apply per notebook, not across kernels.

The main progress bar still counts attempted tiles. `downloaded_images`, `reused_images` and `pending_images` count individual output images, even when requests are grouped. Transport settings and code hashes are saved in each collection attempt. They do not change the scientific archive identity or CropHarvest label-review identity.

No live speed benchmark is included: tests use synthetic rasters and mocked services. Measure new tiles, rather than cached tiles, before extrapolating a full-run duration.

## Background batch collection (default in notebooks 01–03)

The source notebooks set `collection_backend = "batch"`. Existing direct collection remains available with `"direct"`.

1. Restart the kernel and Run All in a source notebook. After authentication, the collector inventories scenes, collects any finished exports, and submits missing daily summaries and composites. Submission itself requires the notebook to stay open.
2. Once the collection cell returns, cloud tasks continue without your laptop. A `pending_retry` result can mean jobs are queued/running, rather than failed. Do not start downstream notebooks yet.
3. Run All again later in the same clone. The collector checks the durable ledger, saves completed results locally, verifies them, deletes their temporary assets, and submits retries for failed tasks. Running jobs are not duplicated.
4. When all tiles pass the existing completion checks, the source handoff is registered. Finish all three sources before notebook 04.

Temporary table assets live under `projects/<earth_engine_project>/assets/crop_pipeline_batch/<source>_<revision>_<year>`. Composites are exported as pixel tables and reconstructed into the existing GeoTIFF layout; daily summaries use the same saved pixel mask and reduction helpers as direct collection. No Google Drive connection or Cloud Storage bucket is required. Your Earth Engine identity needs permission to create/read/delete assets in that project and sufficient task/storage quota.

Keep `<source run folder>/batch/<year>.json` and your local archive: these ledgers track submitted task IDs and prevent duplicate submissions. Do not run concurrent copies of the same source collection. Temporary results remain in Earth Engine storage until a collection pass successfully saves and checks them locally. Local disk is still needed for the completed archive. Empty temporary year folders are removed after successful completion.

Previously, each tile's daily summaries and imagery were requested synchronously while your notebook stayed running. Now the notebook submits asynchronous exports and returns; a later pass retrieves results. This improves resilience and lets the computer sleep between passes. Queue time and project quotas still determine throughput; batch mode is not a guaranteed speedup. The progress bar measures local orchestration, not remote task completion.

A [small live validation](live_batch_validation.md) completed successfully: both composite rasters and corrected daily summaries matched the direct method exactly. Full-dataset throughput and additional edge cases remain unbenchmarked. See Google's [table asset export documentation](https://developers.google.com/earth-engine/apidocs/export-table-toasset).

### Select 5,000 tiles at a time

Notebook 02 (FranceCrops) now defaults to `tile_start = 0` and `tile_count = 5000`. The controls select a slice of the saved tile list **before scene inventory**, without changing the source archive identity or parcel definitions. They apply separately to each observation year; notebooks 01 and 03 expose the same controls but default to all tiles.

| Manual range | `tile_start` | `tile_count` | FranceCrops tiles |
| --- | --- | --- | --- |
| 1 | 0 | 5000 | 1–5,000 |
| 2 | 5000 | 5000 | 5,001–10,000 |
| 3 | 10000 | 5000 | 10,001–15,000 |
| 4 | 15000 | 5000 | 15,001–18,012 |

Restart Kernel → Run All with the chosen settings. Rerun the **same** range until its `selected_batches` entry reports `completed`, then advance `tile_start`. A selected range is a workload boundary, not one Earth Engine task: the existing outstanding-task limit still applies, so it may need several submit/collect passes. No automatic polling or queue replenishment was added.

Ranges keep independent ledgers and asset folders (`<year>_tiles_<start>_<end>`). Starting an overlapping range while another ledger has outstanding tasks is rejected before scene inventory, to avoid duplicate exports. Resume the original range first. Keep the batch directory with the archive.

Completed range receipts accumulate locally. Finishing the last range triggers full-year verification using the existing file/checksum and daily-summary checks. Only successful full-source verification publishes the handoff for notebook 04. Until then, the overall status remains `pending_retry`, even when the selected range is complete. Saved reports distinguish pending selected tiles from tiles outside the chosen range. These range controls were verified with offline regression tests; they do not launch a live full collection merely by editing the notebook.


## Selected-pixel Parquet archives

Source notebooks now default to `archive_format = "pixel_tables"`. These retain every selected pixel across all 24 periods, with one annual file per tile, and skip surrounding rectangle pixels. This supersedes the GeoTIFF reconstruction described above when table mode is selected. The earlier mode remains available as `archive_format = "rasters"`. See [annual pixel tables and the 100-tile benchmark](pixel_tables.md) for schema, compatibility, run instructions and performance limits.
