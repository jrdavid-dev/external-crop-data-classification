Current default: [small adaptive training batches with organized annual TIFFs](adaptive_training.md). The full-catalogue and benchmark modes below remain optional.

# Annual selected-pixel tables

Source notebooks 01–03 use `archive_format = "pixel_tables"` with the batch backend. The full year is retained: 24 calendar half-month composites, six spectral bands and a clear-observation count, for every pixel in each saved binary parcel mask. Window selection happens afterward and can change without downloading the same measurements again.

The export omits surrounding rectangle pixels, not cloudy observations. A selected cell has a record in every period, including explicit nodata values and zero clear counts when appropriate. Cloud masks, daily mosaics, bilinear resampling, half-month medians, grids and parcel-separated identities retain the existing imagery recipe. Raw reflectance is stored as float32 and scaled only when read by the window pipeline.

## Files and identity

Each tile has one annual Zstandard-compressed Parquet file under its imagery folder's `pixel_tables/<tile ID>/` directory, plus a JSON receipt. The columns are `interval`, `row`, `column`, `B3`, `B4`, `B5`, `B8`, `B11`, `B2` and `clear_count`. The tile ID plus integer row/column identifies a geographic pixel; the receipt preserves the CRS, transform, crop, parcel identity, mask checksum, acquisition fingerprint and interval definitions. Thus location and identity do not need to be repeated as strings in every row.

Before a partition is published, the collector requires exactly the saved selected cells in every requested period, rejects duplicates and extra cells, validates clear counts, reads the saved Parquet back, and records a checksum. Changed measurements or masks use separate content-specific paths. Older archives remain intact.

Changing a four-month window or a formula using saved bands reuses the archive. Changing the cloud mask, composite recipe, spatial selection or required bands needs new source processing. The tables do not contain every daily source observation or its cloud probability.

## Run a bounded range

In notebook 02, the first manual range remains:

```python
collection_backend = "batch"
archive_format = "pixel_tables"
tile_start = 0
tile_count = 5000
```

Run the same range again to collect completed jobs and submit pending work. Advance `tile_start` to 5000, 10000 and 15000 only when `selected_batches` reports the current range completed. Cloud task limits still apply; one manual range can require several submit/collect passes. There is no automatic queue replenishment while the notebook is closed.

Pixel-table tasks and coverage receipts are isolated in `batch/pixel_tables/`. The final range triggers deep full-year table verification and daily-checkpoint verification before publishing a source handoff. Notebook 04 consumes either this format or the earlier GeoTIFF format. The preprocessing preview reads selected pixels from Parquet and leaves surrounding pixels blank. It does not recover imagery omitted from the export.

Set `archive_format = "rasters"` to use the previous transport. Complete verified raster years can be converted locally to selected-pixel tables without downloading them again; partial raster years remain preserved.

## Live 100-tile benchmark

The reproducible runner is:

```bash
.venv/bin/python tools/benchmark_pixel_tables.py
```

It uses the first 100 prepared FranceCrops 2022 tiles and a separate directory, `sen4agrinet_work/pixel_table_benchmark_100`. It submits/collects temporary Earth Engine asset exports and returns when jobs remain queued. Repeat the command to resume. It never starts the 5,000-tile range or changes production source registrations.

On completion it audits all 100 tiles through the downstream reader, compares all 24 periods and seven bands for the first tile with the original direct GeoTIFF transport, compares annual daily summaries, and compares the window adapter with the raster quality auditor. The machine-readable `benchmark.json` records row counts, Parquet/receipt sizes, task download/write durations, EECU consumption, pass durations and comparison results. Temporary assets are deleted only after verified local collection.

Table download duration includes Earth Engine's feature-serving requests and client parsing; it is not a pure network speed measurement. Parquet compression happens locally, so final Parquet size is not the number of bytes transferred from Earth Engine. Scene inventory, batch queues, computation and retrieval must all be considered when estimating a full collection. A first-100 sample need not represent every geographic area or tile size.

## Check actual progress

With the venv active, check once without submitting, downloading or deleting cloud results:

```bash
python tools/benchmark_pixel_tables.py --status
```

Or refresh the read-only status every 30 seconds:

```bash
python tools/benchmark_pixel_tables.py --watch
```

Ctrl+C stops the watcher; it does not cancel Earth Engine jobs. A watcher does not collect completed assets, so its local completion count advances only when a collection process saves results. To collect results and continue the benchmark, run the original command without flags. Run one collector at a time; a read-only watcher can run in another terminal.

The report separates:

- **Complete tiles:** both annual pixel data and the current daily checkpoint are local.
- **Pixel-data tiles / daily-summary tiles:** each prerequisite counted independently.
- **Half-month periods:** verified local periods plus pending periods always equals the selected workload (2,400 for 100 annual tiles).
- **Jobs:** current running, queued, ready-to-download, failed, cancelled or unknown attempts. Superseded failures do not count as current failures.
- **Benchmark verification:** independent numerical checks are reported separately from collection completion.

Thus 85 pixel-data tiles and 50 daily-summary tiles can mean only 50 complete tiles. If 2,040 periods are local, exactly 360 of the 2,400 remain pending. A successful remote export is still pending locally until downloaded and verified. Period percentages do not estimate a percentage of running time, and no unsupported ETA is displayed.

A status snapshot is saved atomically to `sen4agrinet_work/pixel_table_benchmark_100/progress.json`; normal collection also writes one. Snapshot timestamps indicate when local state was inspected. `--status`/`--watch` refresh remote task states through a read-only project task-list request. If an active task is absent from that response, it is shown as unknown rather than assumed queued. Normal collection reports the latest states observed during that pass, including newly submitted tasks.

Source notebooks use the same accounting and save per-year snapshots under the ledger directory's `progress/` subdirectory. The old synchronous attempted-tile bar is disabled for the batch backend, and notebook output uses the new labelled counts. Legacy `reused_images` and `pending_images` fields remain in saved reports for compatibility; in table mode they are aliases for verified-local and pending half-month periods, not counts of GeoTIFF files or newly downloaded files.

Table retrieval now requests 5,000 rows per page, prints each page's saved row count, and checkpoints pages under the ledger directory's `download_pages/<task_id>/`. Requests use a 120-second network timeout and one visible retry for transient failures; this is not a strict total wall-clock limit. Rerunning collection verifies cached page checksums and continues from the saved page token. Temporary assets and checkpoints are retained if retrieval or local verification fails. Checkpoints are removed only after verified local output, remote asset deletion and ledger persistence. Restart an existing notebook kernel to load this change; partial transfers made by the older downloader have no page checkpoints.
