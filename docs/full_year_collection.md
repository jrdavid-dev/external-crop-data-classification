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

The retriever caches acquisition metadata, fingerprints requests, validates saved rasters and checkpoints each interval. Restarting collection reuses verified files and retries missing or damaged intervals. Keep the collection files and masks together; local manifests contain absolute paths.

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
