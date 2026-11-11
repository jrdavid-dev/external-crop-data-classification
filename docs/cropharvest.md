# CropHarvest: notebook 03

[Preprocessing notebook 04](../notebooks/preprocessing/04_cropharvest_test_preparation.ipynb) audits carrot, cabbage and potato labels, selects up to **100 eligible parcels per available crop per round**, and collects annual Sentinel-2 histories through Earth Engine.

The active notebook uses a separate JECAM campaign for testing. It excludes test footprints that overlap existing Sen4AgriNet or FranceCrops development footprints and assigns selected parcels to the test role. Collection is disabled in the notebook until explicitly enabled. The preprocessing audit rechecks overlap when development batches have changed.

## Run All steps

1. **Setup:** install missing dependencies, create or preserve local settings, and configure Earth Engine access.
2. **Obtain labels:** obtain missing `data/external/cropharvest/labels.geojson` in `download_missing` mode from the pinned release. Existing labels are validated and reused. Imagery and feature archives are not downloaded.
3. **Audit labels:** apply explicit crop aliases and mixed-label exclusions; review source reliability, polygon geometry, duplicates and dates. Save the label audit and observation-year decisions.
4. **Prepare candidates:** exclude upstream test labels, require a resolved observation year, and select a bounded, geographically/year-stratified batch. Apply the 10 m inward buffer and contested-location exclusions before imagery requests.
5. **Collect:** submit or retrieve imagery using original saved plans.
6. **Summarize:** inspect verified tiles, job states, saved paths and the next action. Finish pending work before notebook 04.

A source date can support the observation year; it is not automatically a planting date. January 1 placeholders need a supported season year. Source reliability grades are screening judgments, not independent proof of crop accuracy. Optional country borders can be omitted; dataset names provide the recorded fallback.

## Switches and outputs

The shared `reference_mode` is `reuse` by default. Choose `download_missing` to obtain missing labels, or `offline` to use local inputs and installed dependencies without sign-in, installation or satellite calls. `run_satellite_collection = False` allows reference preparation without satellite work; it preserves existing operational job states.

The notebook uses the same six sections as 01–02. Label screening lives in a reusable helper; one crop-level summary and saved audit paths replace the previous lengthy inline audit. Its eligibility rules are unchanged. The existing **10 m inward buffer** on polygon parcels is preserved.

Label downloads verify the pinned release MD5 and GeoJSON structure before atomic publication. A `.download.json` receipt records the source and local SHA-256. Label-content hashes identify versioned review runs; observation-year decisions are saved in `labels/observation_year_audit.csv`.

Completed annual TIFFs are organized by source, year, campaign and round, with separate masks and Parquet transport caches. `completed` confirms saved collection files; notebook 04 still applies timing, completeness and quality checks. Earlier selections, locked partitions and archives are preserved. See [workflow](workflow.md) for resuming jobs and [adaptive training](adaptive_training.md) for sample targets.
