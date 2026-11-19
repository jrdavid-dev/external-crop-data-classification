# Annual imagery transport reference

For normal collection use [the collection guide](../collection.md). This page
explains saved formats and maintenance tools, not an alternative notebook order.

## Measurements and schemas

An annual selected-pixel archive has 24 half-month bins on the saved grid. The
base measurements are B3, B4, B5, B8, B11, B2 and clear-observation count. B2 and
counts support inspection; the classifier uses five bands and four indices in
eight selected bins (72 predictors). Surrounding rectangle cells are not sampled.

| Format | Stored measurements |
| --- | --- |
| Legacy annual TIFF | 24 × 7 = 168 labelled float32 bands |
| TIFF with observation diagnostics | 24 × 10 = 240 bands |
| Selected-pixel Parquet | `interval`, integer `row`/`column`, six bands, `clear_count`, and diagnostic columns when available |

Diagnostic columns are `pre_mask_count`, `cloud_shadow_count` and
`other_rejection_count`. Counts are unscaled; reflectance is scaled by 0.0001
when read for features. Use actual TIFF band descriptions and receipts, not an
assumption that every archive has the newer diagnostics. Missing legacy
cloud-loss counts remain unavailable, not zero.

Each selected cell has a record in every interval, including explicit nodata and
zero clear counts. Parquet uses Zstandard compression. Its receipt stores the
contract, CRS/grid, crop/parcel identity, binary mask checksum, acquisitions and
intervals. Tile ID plus integer row/column locates a pixel without repeating
coordinates in every row.

Collection validates selected cells, duplicates, period coverage, finite transport
values and counts, reads the saved table back, and records checksums. Cloud masks,
daily mosaics, resampling and half-month medians follow [methodology](../methodology.md).
The tables do not retain all daily source scenes or cloud probabilities; daily
index timelines are separate. They cannot reconstruct observations never exported.

Window changes can reuse annual measurements. Changed cloud rules, masks, bands
or compositing require new processing. Exact-contract cache lookup and checksums
allow relocated caches to retain their original filenames. Invalid or conflicting
caches are not silently accepted. Complete verified legacy raster years can be
converted locally through the retained import helpers; partial years remain
preserved.

## Maintenance tools and evidence

`tools/benchmark_pixel_tables.py` and `tools/validate_live_batch.py` are deliberate
Earth Engine operations with separate plans/ledgers. They are not routine offline
workflow checks. Read their CLI/help and retained plans before running them;
keep task ledgers until local results verify. `--status` on the benchmark
is a read-only inspection route; its normal invocation can submit/collect jobs.

The [small live comparison](../history/live_batch_validation.md) records transport
agreement and the mask correction for its selected case. The
[historical pixel-table notes](../history/pixel_tables.md) retain benchmark context,
and [full-catalogue notes](../history/full_year_collection.md) retain old controls.
Their retired notebook numbering and 5,000-tile procedures are not current source
notebook instructions. No full-catalogue notebook run is prescribed here.

Downloaded Parquet size is not network transfer volume or Earth Engine cost.
Client parsing, computation, queue delays and scene inventories affect elapsed
time. Small cached or first-100-tile comparisons do not establish whole-dataset
throughput, universal equivalence or crop accuracy.
