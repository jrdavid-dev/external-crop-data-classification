# Live batch validation — 5 November 2026

A real Earth Engine test completed successfully in the configured project, using one FranceCrops cabbage parcel (`annual_b0c04cfcfd9706ae69b412e3`) in June 2022. No full source collection was started.

The test compared the original synchronous `SatelliteRetriever.download_region` implementation with temporary table exports, collection and GeoTIFF reconstruction. Both paths used the same scene inventory, cloud rules and 10 m sampling grid. Daily summaries were independently evaluated through the direct and batch transports.

| Check | Result |
| --- | --- |
| Grid | EPSG:32630, 14 columns × 24 rows |
| Saved parcel mask | 133 selected pixels |
| Daily observations | 12 dates; 6 with valid observations, 6 cloud-masked |
| Daily indices, coverage, eligibility and identities | Exactly equal after the shared mask correction below |
| June 1–15 composite | All 7 bands exactly equal; maximum absolute difference 0 |
| June 16–30 composite | All 7 bands exactly equal; maximum absolute difference 0 |
| Raster metadata and nodata masks | Equal |
| Temporary test assets | Deleted after verified local collection |
| Automated regression suite | 239 tests passed |

## Shared daily-mask correction

Live evaluation exposed an existing error in the daily sampling mask: `ee.Image.pixelCoordinates` returned centres such as 0.5 and 1.5, while the saved row/column IDs were integers. Without flooring each coordinate before constructing pixel IDs, the mask selected zero pixels. Both the old direct and new batch daily methods therefore agreed on empty observations before this fix.

`sampling_mask.selection_image` now floors each coordinate before constructing pixel IDs. A live reduction confirmed that it selects exactly the saved 133 pixels. The daily summary recipe is now `selected_pixel_grid_v3`, so older checkpoints are preserved and corrected daily summaries are recomputed. The numeric Earth Engine test substitute now uses half-pixel centres to cover this regression. Composite imagery and its cache identity are unchanged by this daily-mask correction.

A second real daily batch export matched all corrected direct rows exactly, including valid-pixel counts, coverage, eligibility, NDVI, NDRE and NDMI. Thus imagery matches the previous method exactly, while daily summaries match the corrected direct method rather than preserving the old empty-result bug.

## Reproduce

From the repository root, with existing Earth Engine credentials and prepared FranceCrops 2022 references:

```bash
.venv/bin/python tools/validate_live_batch.py
```

The script uses one parcel and two half-month intervals, isolates all local output under `sen4agrinet_work/live_batch_validation`, tracks remote tasks in its own ledger, and returns if tasks are still queued/running. Run it again later to collect and compare. It never starts a full collection. Keep its plan and ledger together; remove neither while tasks are active. `--diagnose` only inspects coordinates and mask reductions for the test area.

The machine-readable result is `sen4agrinet_work/live_batch_validation/comparison.json`; an earlier result before the mask fix is preserved as `comparison_before_mask_fix.json`. Raw test data and task ledgers remain local and excluded from Git.

This small test establishes live transport and numerical agreement for the selected case, not full-dataset throughput or guaranteed equivalence for every geometry, year or edge case. Batch queue delays mean this is not evidence of a speedup.
