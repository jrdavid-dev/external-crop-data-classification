# Sen4AgriNet: notebook 01

Batch 2 update: active preprocessing is now [01–03](../notebooks/preprocessing/README.md). The old 05–07 filenames alias these new entry points. Finish pending collection before a live audit.

Active entry point: [collection 02](../notebooks/data_collection/02_sen4agrinet_collection.ipynb). Saved new-batch settings currently target potato; existing carrot batches are retained. Follow collection 06, then preprocessing 05–07.

[Notebook 01](../notebooks/data_collection/02_sen4agrinet_collection.ipynb) prepares carrot and potato candidate parcels, then collects full-calendar-year Sentinel-2 histories through Earth Engine. Cabbage is supplied by FranceCrops. The default workflow uses bounded reference subsets and records their coverage.

## Review these settings

```python
candidate_parcels_per_crop = 50
reference_target_crops = ("carrot", "potato")
reference_scan_max_patches = 1500
reference_mode = "download_missing"
reference_target_parcels_per_crop = 100
```

`reuse` consolidates valid local checkpoints or uses a complete inventory without new reference transfers. On a fresh clone use `reference_mode = "download_missing"`; `offline` also prevents package installation and satellite calls. Missing local inputs produce an error.

The reference target and patch cap are separate from the 50-per-crop imagery batch limit. To expand an exhausted pool, increase `reference_target_parcels_per_crop` and, if needed, `reference_scan_max_patches` deliberately. Extraction stops at the reference target or patch cap. A new imagery batch waits until it can plan 50 unused carrot parcels and 50 unused potato parcels. The cap limits patches, not time.

`run_satellite_collection = False` prepares references/plans without satellite work. Existing job states are preserved. Sen4AgriNet retains declared mask cells with **0 m additional buffer**; this documented rule is unchanged.

## Run All steps

1. **Setup:** install missing dependencies, preserve or create local settings, and resolve the Earth Engine project.
2. **Discover:** pin the upstream release and list its patches. Full NetCDF sizes are context, not planned transfer sizes.
3. **Prepare references:** reuse checkpoints or selectively read label/parcel masks. Validate crop codes, grids, coordinates, identities and geometry; exclude known mixed/conflicting groups and upstream held-out references.
4. **Select candidates:** spread 50 physical parcels per crop across geographic/year strata when both crops have enough eligible parcels. Existing selections, pending batches and locked partitions are preserved.
5. **Collect:** submit or collect Earth Engine jobs using original saved plans.
6. **Summarize:** inspect verified tiles, job states and the next action. Rerun pending work, then assess usable windows in notebook 04.

Unseen patches have not been audited for label conflicts or neighboring geometry. The inventory records scanned files, counts, deficits and the stop reason; subset coverage is not full-release coverage. No usable cached checkpoints produces an error rather than an empty imagery batch.

## Reference reads and storage

When new reads are enabled, small HTTP ranges access HDF5 metadata, labels and coordinates; parcel masks are read when target labels occur. Spectral arrays are not accessed. A server that ignores Range is rejected before its full response body is read. Verified complete local files can also be reused.

Partial reads check source offsets and file size but cannot verify the entire upstream NetCDF checksum. Local mask caches have their own SHA-256 checks. Printed received bytes measure reference transfers only.

References default to `<annual_archive_root>/sen4agrinet_references/`, with a pinned manifest, mask caches and versioned extraction checkpoints. Change `sen4agrinet_reference_root` to use another location. Existing archives remain preserved.

Completed annual TIFFs are organized by source, year, campaign and round; sampling masks and Parquet caches remain separate. See [workflow](workflow.md) for completion states and [adaptive training](adaptive_training.md) for the collection/assessment loop.
