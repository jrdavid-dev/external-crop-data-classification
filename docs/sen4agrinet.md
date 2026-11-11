# Sen4AgriNet: notebook 01

[Notebook 01](../notebooks/preprocessing/02_sen4agrinet_preparation.ipynb) prepares carrot and potato candidate parcels, then collects full-calendar-year Sentinel-2 histories through Earth Engine. Cabbage is supplied by FranceCrops. The default workflow uses bounded reference subsets and records their coverage.

## Review these settings

```python
candidate_parcels_per_crop = 100
reference_target_crops = ("carrot", "potato")
reference_scan_max_patches = 350
reference_mode = "reuse"
reference_target_parcels_per_crop = 100
```

`reuse` consolidates valid local checkpoints or uses a complete inventory without new reference transfers. On a fresh clone use `reference_mode = "download_missing"`; `offline` also prevents package installation and satellite calls. Missing local inputs produce an error.

The reference target and patch cap are separate from the shared 100-per-crop imagery round limit. To expand an exhausted pool, increase `reference_target_parcels_per_crop` and, if needed, `reference_scan_max_patches` deliberately. Extraction stops at the target or cap and proceeds with available candidates. The cap limits patches, not time.

`run_satellite_collection = False` prepares references/plans without satellite work. Existing job states are preserved. Sen4AgriNet retains declared mask cells with **0 m additional buffer**; this documented rule is unchanged.

## Run All steps

1. **Setup:** install missing dependencies, preserve or create local settings, and resolve the Earth Engine project.
2. **Discover:** pin the upstream release and list its patches. Full NetCDF sizes are context, not planned transfer sizes.
3. **Prepare references:** reuse checkpoints or selectively read label/parcel masks. Validate crop codes, grids, coordinates, identities and geometry; exclude known mixed/conflicting groups and upstream held-out references.
4. **Select candidates:** spread up to 100 physical parcels per crop across geographic/year strata. Existing selections, pending batches and locked partitions are preserved.
5. **Collect:** submit or collect Earth Engine jobs using original saved plans.
6. **Summarize:** inspect verified tiles, job states and the next action. Rerun pending work, then assess usable windows in notebook 04.

Unseen patches have not been audited for label conflicts or neighboring geometry. The inventory records scanned files, counts, deficits and the stop reason; subset coverage is not full-release coverage. No usable cached checkpoints produces an error rather than an empty imagery batch.

## Reference reads and storage

When new reads are enabled, small HTTP ranges access HDF5 metadata, labels and coordinates; parcel masks are read when target labels occur. Spectral arrays are not accessed. A server that ignores Range is rejected before its full response body is read. Verified complete local files can also be reused.

Partial reads check source offsets and file size but cannot verify the entire upstream NetCDF checksum. Local mask caches have their own SHA-256 checks. Printed received bytes measure reference transfers only.

References default to `<annual_archive_root>/sen4agrinet_references/`, with a pinned manifest, mask caches and versioned extraction checkpoints. Change `sen4agrinet_reference_root` to use another location. Existing archives remain preserved.

Completed annual TIFFs are organized by source, year, campaign and round; sampling masks and Parquet caches remain separate. See [workflow](workflow.md) for completion states and [adaptive training](adaptive_training.md) for the collection/assessment loop.
