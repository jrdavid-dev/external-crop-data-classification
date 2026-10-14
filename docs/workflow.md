# Notebook workflow

Open [the notebook](../Sen4AgriNet_Workflow_Outline.ipynb) from the repository folder. Read each explanation before running its code. **Steps 1–8 are implemented; Steps 9–15 are plans.**

## Before running

Follow the root [README](../README.md), configure `settings.local.json`, and select `.venv` as the kernel. The dataset root is the parent of `data`: `E:/ground_truth` resolves to `E:/ground_truth/data/2020`. Collaborators set their own paths.

Run settings and helper cells before later cells. Skip Step 2's preview/download cells when raw files already exist. When restoring existing data, use the [moving-PC guide](moving_pc.md).

## Implemented steps

| Step | Work | Output |
|---|---|---|
| 1 | Define scope and open decisions | Study specification |
| 2 | Download one raw-data batch | File plan and batch CSV |
| 3 | Inspect all local 2020 files sequentially | Counts, checkpoints and candidate patches |
| 4 | Review imagery, extract geometry, deduplicate and sample | Polygons, review records, pilot regions and masks |
| 5 | Retrieve cloud-screened half-month medians | GeoTIFFs and download progress |
| 6 | Check quality, scale bands and verify grids | Quality and selected-pixel tables |
| 7 | Calculate curves and retrieve daily summaries | Timelines, checkpoints and plots |
| 8 | Detect candidate green-up | Onset records, sensitivity results and plots |

### Batches and field groups

Download batches control workload; they are not analysis or training partitions. Step 3 scans every downloaded 2020 file. Step 4 selects the pilot separately.

Remove redundant footprints only when country, year, crop, parcel ID and geometry agree. Keep different footprints sharing an ID for review. Preserve removed-to-retained links.

The pilot uses equal pixel and group budgets for carrot and potato. A partition group combines country, year and parcel ID to keep possible field fragments together. It does not prove independent-field identity. Later splits must be field-based, never random pixels.

### Cloud screening before medians

Apply these checks to each acquisition:

1. Exclude SCL classes 0, 1, 3, 8, 9, 10 and 11: invalid pixels, saturation, shadows, clouds, cirrus and snow.
2. Retain cloud-probability scores below 40. This is an exploratory per-pixel cutoff, not whole-scene cloud coverage.
3. Require valid spectral bands, probability pixels and B8A/B9 edge masks.
4. Exclude and report acquisitions without an exact cloud-probability match.

Retain bare soil for low-to-growth detection. Shadows are screened using SCL; a separate projected-shadow detector is not implemented.

Mask granules first, then mosaic each date. Sorted image-index order gives the last valid granule priority in overlaps. Median-composite distinct dates with equal weight. This deterministic ordering is not cloud-score ranking.

Each GeoTIFF contains B3, B4, B5, B8, B11, B2 and `clear_count`. Spectral medians are unscaled float32 values; missing values are −9999. Count zero remains valid quality information. Empty bins are retained, not filled or dropped.

Use a fixed 10 m grid. Resampling originally 20 m bands does not add detail. Use only pixels marked 1 in sampling masks for pilot training; surrounding pixels are context.

### Quality and curves

Step 6 initially requires valid values at 80% of selected pixels for an interval to pass. Calibrate this development rule. Preserve missing, invalid and low-coverage records. Multiply spectral bands by 0.0001 after nodata handling; never scale counts.

Calculate indices from median bands, then summarize selected pixels. An index of median bands differs from a median of acquisition-level indices; the feature design uses the former.

Composite midpoints are display dates. Step 7 also retrieves small summaries at actual observation dates, with the same masks and footprints. These are sample-region summaries, not whole-field averages.

### Candidate detector

The detector prefers observed daily summaries and falls back to composites where necessary. Development defaults are:

| Setting | Value |
|---|---:|
| Relative-amplitude crossing | 20% |
| Minimum NDVI amplitude | 0.15 |
| Maximum onset gap | 30 days |
| Later confirmations | 2 within 30 days |
| Baseline lookback | 120 days |
| Minimum valid fraction | 80% |

These are experimental settings. Composite-only outcomes remain provisional. Long gaps, weak signals and insufficient confirmations can cause rejection. Multiple plausible cycles are flagged. Preserve intervals and month-boundary uncertainty; do not report exact planting dates.

## Remaining steps

| Step | Work still required |
|---|---|
| 9 | Validate timing, crop-label relevance and parameters |
| 10 | Select four consecutive months using the frozen rule |
| 11 | Select the matching eight existing composites and derive indices |
| 12 | Apply missing-data rules and report sample losses |
| 13 | Build tables and independent-field partitions |
| 14 | Train/evaluate the three training-source comparisons |
| 15 | Record versions, parameters, splits, exports and limitations |

Step 11 selects existing medians; do not composite them again. Begin with complete-data samples. Any gap treatment must preserve reconstruction flags and avoid inventing onset evidence.

## Folders and helper modules

| Location | Contents |
|---|---|
| Data drive: `data/2020/` | Original NetCDFs |
| Data drive: `sentinel2_pilot_composites/` | Large GeoTIFFs |
| Project: `sen4agrinet_work/inventory/2020/` | File and crop reports |
| Project: `sen4agrinet_work/fields/2020/` | Geometry, selection and masks |
| Project: `sen4agrinet_work/l2a/2020/` | Acquisition inventory and progress |
| Project: `sen4agrinet_work/analysis/2020/` | Quality, timelines and candidate results |

`crop_workflow.py` handles quality, indices, plots and detection. `observed_timelines.py` retrieves dated summaries. `migrate_progress.py` updates local paths. Generated data and reports are excluded from Git; preserve them separately.

## References

- [L2A collection](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Cloud probability](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY)
- [Download limits](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl)
- [Study plan](study_plan.md)
