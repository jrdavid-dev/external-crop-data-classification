# Benguet crop classification: S4A to four-month inputs

> Methodology notes include earlier standalone examples. Those scripts are now archived locally; use the Sen4AgriNet notebook for the current executable workflow.

Prepared October 13, 2026.

## Purpose

Use external crop-reference labels for cabbage, carrot, and potato to support a Random Forest classifier for Benguet. Recover crop labels and field locations from S4A NetCDF files, retrieve matching Sentinel-2 L2A imagery, clean the observations, estimate vegetation green-up, and create eight semi-monthly composites over four consecutive calendar months.

Each pixel will have five bands (B3, B4, B5, B8, B11) and three indices (NDVI, NDRE, NDMI) in each interval: **8 × 8 = 64 features**.

This document describes the planned workflow. The existing inspection scripts implement exploratory inspection only; they do not yet download L2A, mask clouds, or implement a validated growth detector.

## Important distinctions

- S4A contains satellite imagery, crop-label masks, and parcel-ID masks. Its source imagery is L1C; this project's final inputs require L2A surface reflectance.
- Crop labels are annual farmer declarations, not date-specific planting or harvest records.
- Green-up is the appearance/increase of vegetation detected from imagery. It is not necessarily the planting date.
- A declining canopy is not proof of harvest. Stress, senescence, clouds, and other disturbances can cause declines.
- A four-month input need not contain the entire crop cycle.
- Detecting a vegetation cycle does not establish which crop occupied that cycle. Annual declarations must be checked for temporal relevance.

## Step 1 — Download S4A NetCDF files

Install the downloader:

```powershell
python -m pip install -U huggingface_hub
```

Open PowerShell in the destination folder on your hard disk. Download 2020:

```powershell
hf download orion-ai-lab/S4A --repo-type dataset --include "data/2020/*" --local-dir "."
```

To include both years:

```powershell
hf download orion-ai-lab/S4A --repo-type dataset --include "data/2019/*" "data/2020/*" --local-dir "."
```

Expected structure:

```text
E:\ground_truth\data\2020\31TBF\*.nc
E:\ground_truth\data\2020\31TCF\*.nc
```

The absolute drive and destination are examples. Rerun inspection after the download finishes: a scan started during downloading may miss new files or encounter incomplete files.

**Output:** completed `.nc` files and a record of the downloaded years/tiles.

## Step 2 — Count labels and identify candidate patches

Install inspection dependencies:

```powershell
python -m pip install numpy netCDF4 matplotlib
```

From the THS folder, run:

```powershell
python inspect_crops.py "E:\ground_truth\data\2020"
```

The reports are saved to `crop_inspection` in the current working folder unless `--output` is supplied.

Review:

- `summary.txt`: overall counts, observed codes, failures, and limitations.
- `crop_counts.csv`: candidate files and per-file parcel statistics.
- `file_details.jsonl`: acquisition dates, band dimensions, and metadata.

Default original codes: cabbage **213**, carrot **231**, potato **510**. Do not switch to linear codes without confirming the file encoding.

Count independent fields separately from pixels. Parcel occurrences summed across files are not independent field counts. Complete the scan before interpreting class availability. If cabbage is absent, inspect other tiles/years or obtain a compatible additional label source. Seek more independent carrot fields rather than simply extracting more pixels from one field.

**Output:** candidate patch list, target-crop counts, and unresolved data-quality issues.

## Step 3 — Inspect candidate patches visually

Place a candidate `.nc` file in the THS folder. If it is the only `.nc` file and contains carrot:

```powershell
python visualize_crop.py
```

For a potato file:

```powershell
python visualize_crop.py "path\to\potato_patch.nc" --crop potato
```

Select an acquisition near a date or another parcel:

```powershell
python visualize_crop.py "path\to\patch.nc" --crop carrot --date 2020-06-15 --parcel 1667452
```

Inspect `visual_inspection`:

- `05_satellite_labels.png`: satellite RGB, colored labels, parcel boundaries, and a parcel zoom.
- `03_timeline.png`: parcel band and index trajectories.
- `04_acquisitions_*.png`: dated satellite previews.
- `parcel_inventory.csv`: parcel IDs available for the selected crop.
- `parcel_timeseries.csv`: exploratory measurements.
- `inspection_metadata.json`: metadata and four-month acquisition-bin counts.

Check whether masks follow visible field boundaries, whether the parcel contains conflicting labels, and whether it is large enough for meaningful interior samples.

These plots use the supplied L1C imagery without cloud masking. They are screening tools, not evidence of exact planting/harvest dates. Category names not present in the script's limited lookup are explicitly shown as unmapped codes.

**Output:** manually reviewed candidate parcels and rejection reasons.

## Step 4 — Recover spatial information and deduplicate fields

Read the NetCDF `x`/`y` coordinates, CRS, and spatial transform from the imagery and label variables. The inspected carrot file uses UTM coordinates in meters, EPSG:32631, rather than explicit latitude/longitude arrays. Check other files individually.

1. Recover the patch bounds using the transform and raster dimensions; distinguish pixel centers from outer edges.
2. Recover each target parcel's raster footprint, retaining its original crop code and parcel ID.
3. Export a georeferenced label/parcel raster or polygonize the parcel mask.
4. Transform boundaries to EPSG:4326 when querying Earth Engine.
5. Retain the original projected grid for precise pixel alignment.
6. Check parcel-ID scope and duplicated geometry across patches/tiles. Merge fragments only when evidence establishes they represent the same field.
7. Assign a stable `field_id`; preserve source IDs and provenance.

Do not label all pixels in a patch as carrot just because it contains one carrot field. Exclude background and other crop categories from target-class samples.

**Output:** georeferenced crop/parcel masks and a deduplicated field inventory.

## Step 5 — Set up Earth Engine and retrieve L2A

Create or select a Google Cloud project, enable Earth Engine, and register it for the applicable access category. Follow the current noncommercial verification requirements if eligible.

Install the Python client:

```powershell
python -m pip install earthengine-api
```

Authenticate and initialize in Python:

```python
import ee

ee.Authenticate()
ee.Initialize(project="YOUR_REGISTERED_PROJECT_ID")
```

Use:

```python
ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
```

For each candidate field:

1. Filter by geographic footprint and year.
2. Retrieve a full-year timeline, extending into adjacent years for boundary-crossing seasons.
3. Keep B3, B4, B5, B8, B11 and the bands/products required for quality masking. Retain B2 for true-color inspection.
4. Record acquisition dates and source image IDs.
5. Retrieve all suitable available acquisitions, not just the dates retained in S4A.
6. Export manageable patches, clipped imagery, or tables instead of downloading whole tiles unnecessarily.

The L2A archive may supply observations missing from the filtered S4A archive, but clouds can still leave gaps. Spatial bounds locate the data; downloading is not achieved by changing a NetCDF filename.

**Output:** dated L2A observations, quality information, and provenance for each field.

## Step 6 — Preprocess consistently

Apply the same pipeline to external and Benguet imagery.

### 6.1 Mask invalid observations

Use pixel-level cloud/shadow information, such as SCL together with a documented cloud-probability or Cloud Score+ method. Exclude invalid pixels, cloud shadows, clouds, cirrus, and snow as appropriate. Do not rely only on whole-scene cloud percentage.

Retain valid bare soil: removing every nonvegetated pixel would prevent observing the low baseline before green-up. Document thresholds and any spatial buffer around cloud masks, then inspect examples.

### 6.2 Scale values and align grids

- Apply the collection's documented reflectance scale (0.0001 for the spectral bands in the harmonized L2A collection).
- Fix CRS, grid origin, pixel size, extent, and resampling policy.
- B3/B4/B8 are 10 m; B5/B11 are 20 m. A 10 m output grid does not create new 10 m detail in those bands.
- Use nearest-neighbor for categorical crop and parcel masks. Never interpolate crop codes.
- Specify the reflectance resampling method and verify label/image overlays after alignment.
- Consider a fixed field-interior buffer to reduce mixed pixels. Check how it affects small fields; do not silently delete them.

### 6.3 Preserve quality records

For every date/field, record clear-pixel counts, clear fractions, and excluded observations. A finite numerical value is not necessarily a clear observation.

**Output:** comparable, georeferenced, cloud-masked reflectance time series.

## Step 7 — Build parcel-level growth curves

For each clear acquisition, calculate pixel indices:

```text
NDVI = (B8 - B4) / (B8 + B4)
NDRE = (B8 - B5) / (B8 + B5)
NDMI = (B8 - B11) / (B8 + B11)
```

Treat invalid/zero denominators as missing. Summarize each index using the median of valid field-interior pixels. Record how many pixels support each value. Reject dates with inadequate clear-field coverage using a documented criterion established during the pilot.

Plot chronological NDVI, NDRE, NDMI and dated RGB images. Use NDVI as the primary green-up detector; other indices support interpretation. NDMI does not independently prove crop identity or planting.

**Output:** a quality-controlled field-level timeline spanning the candidate growing season.

## Step 8 — Detect green-up using a relative threshold

This is a proposed detector to develop and validate, not an already validated carrot/cabbage/potato algorithm.

1. Identify candidate cycles separately, each with a preceding low-vegetation period and a sustained rise toward a peak.
2. Smooth minor noise using a documented method that respects irregular acquisition dates. A robust smoother or double-logistic fit for a well-observed single cycle is a candidate; check residuals and do not force that curve onto multiple unresolved cycles.
3. Estimate a robust pre-growth baseline and seasonal peak for each cycle. Define these estimators before batch processing; do not use arbitrary yearly minima and maxima across several cycles.
4. Define seasonal amplitude as `peak - baseline`.
5. Test a small predefined set of relative thresholds, for example 15%, 20%, and 25% of amplitude above baseline.
6. Identify the rising crossing, supported by actual clear observations.
7. As an initial confirmation rule to test, require at least two subsequent clear observations supporting sustained growth, within a predefined time horizon. Establish that horizon and minimum signal strength during validation; widely separated points cannot automatically confirm a rise.
8. Check NDRE and RGB for agreement or contamination.
9. Return an estimated date, observed crossing interval, quality flags, and an acceptance decision.

Example only:

```text
Baseline NDVI = 0.15
Peak NDVI = 0.65
Amplitude = 0.50
Proposed 20% threshold = 0.15 + 0.20 × 0.50 = 0.25
```

If a clear observation is below the threshold on March 8 and above it on March 18, report **green-up between March 8 and March 18**. A fitted point estimate does not eliminate that uncertainty.

Copernicus uses seasonal-amplitude thresholds for start of season, but its 25% definition uses PPI. This supports the method family, not a universal 25% NDVI threshold for these crops.

### Reject or flag uncertain cases

- No observed baseline before the rise.
- Weak seasonal signal or poor curve fit.
- Long gaps around the threshold crossing.
- Unresolved multiple cycles or contradictory indices/images.
- Estimates change substantially across reasonable detector settings.
- Annual crop labels cannot be linked to the detected cycle.

Do not force a date onto these fields. Save why detection failed.

This detector uses the eventual seasonal peak, so it is retrospective. A real-time deployment would need a separately validated method that uses only information available at prediction time.

**Output:** accepted candidate green-up estimates and explicitly uncertain/rejected cycles.

## Step 9 — Validate and freeze the detector

Research supports an approach; reliability for these fields must be measured.

1. Create a development set spanning crops, regions, field sizes, and observation quality.
2. Obtain independent emergence/green-up evidence where possible: dated field observations, farmer records with emergence information, or independently reviewed high-resolution imagery.
3. Use planting records as supporting context, recognizing that planting precedes detectable green-up.
4. Compare detector estimates with reference intervals. Where point reference dates are reliable, report mean absolute error and bias.
5. Report detection/acceptance rates and timing errors by crop and region, not just overall.
6. Test threshold sensitivity, masking choices, acceptable gaps, and confirmation settings on development fields.
7. Freeze the chosen rule before evaluating held-out fields. Do not tune it to final Benguet test accuracy.
8. Where only manual satellite review is available, report agreement with that review; do not call it validated planting-date accuracy.

No independently dated reference evidence means the dates remain estimates. Literature crop durations are plausibility checks, not a way to manufacture field-specific ground truth.

**Output:** detector settings, validation results, limitations, and a frozen selection rule.

## Step 10 — Select four consecutive calendar months

Proposed consistent rule:

> For an accepted vegetation cycle whose crop identity is supported, start on the first day of the month containing estimated green-up. Extract that month and the next three calendar months using fixed semi-monthly bins.

Example: accepted green-up on March 18 → March 1 through June 30.

If the green-up uncertainty interval crosses a month boundary, flag the window. Test both possible starts in a predefined sensitivity analysis, or exclude it from the initial clean experiment.

Use the same rule for all crops and regions. Do not choose windows using the known crop category or whichever window yields the best classifier result.

Check that the selected window corresponds to the supported crop cycle. It may include baseline or post-harvest observations, but must not be assigned the annual label when a different crop dominates the selected period. Reject unresolved rotations or mixtures. Do not concatenate separated seasons or compress longer seasons into four months.

For multiple verified cycles in one field, separate samples are possible; all remain in the same field partition.

**Output:** accepted field-season windows with evidence and uncertainty metadata.

## Step 11 — Create eight semi-monthly composites

For March–June:

| Interval | Dates |
|---|---|
| 1 | March 1–15 |
| 2 | March 16–31 |
| 3 | April 1–15 |
| 4 | April 16–30 |
| 5 | May 1–15 |
| 6 | May 16–31 |
| 7 | June 1–15 |
| 8 | June 16–30 |

For each bin:

1. Select dated observations using unambiguous start-inclusive/end-exclusive boundaries.
2. Apply quality masks consistently across required inputs.
3. Calculate per-pixel median B3, B4, B5, B8, B11 from valid observations.
4. Calculate NDVI, NDRE, NDMI from those median bands.
5. Save per-pixel observation counts and missing-data flags.

This guide chooses **median bands followed by indices**. Median per-image indices is a different calculation; do not mix conventions across datasets.

**Output:** a spatial feature stack with eight intervals, eight variables per interval, and quality layers.

## Step 12 — Handle missing data

Start with a conservative complete-data baseline: accept pixels only when all 64 required features are valid across the eight bins. An acquisition somewhere in the patch does not guarantee clear coverage of the selected field.

If this removes too much data:

- Report losses by crop, region, and field size.
- Evaluate a separately specified short-gap policy; fix the maximum allowable gap and forbidden extrapolation before applying it.
- Do not bridge long gaps around green-up or harvest and describe the reconstructed transition as observed.
- Fit any learned imputer using training data only.
- Compare imputed results with the complete-data subset.
- Document any added missingness features as changes beyond the initial 64-feature design.

**Output:** accepted training pixels, excluded pixels, and an auditable missing-data policy.

## Step 13 — Build the Random Forest table

Use one row per accepted pixel-window:

```text
field_id, season_id, country, year, crop_code,
greenup_interval, window_start, window_end,
B3_t1, B4_t1, B5_t1, B8_t1, B11_t1, NDVI_t1, NDRE_t1, NDMI_t1,
... same variables through t8 ...,
quality_metadata
```

The 64 satellite variables are classifier inputs. Crop code is the target. IDs and provenance are metadata, not accidental classifier features.

Split by independent field before pixel sampling. Keep all field fragments, duplicated locations, overlapping windows, and seasons of the same field in one partition. Limit pixels from large fields so they do not dominate simply through size. Compare balanced sampling or class weighting on development data; neither replaces missing classes or additional independent fields.

**Output:** field-separated training/validation tables and untouched Benguet test fields.

## Step 14 — Evaluate the thesis experiment

Train new Random Forests for:

1. External-only training.
2. Limited verified Benguet-only training.
3. External plus the same limited Benguet training set.

Use the same independent Benguet test fields for all comparisons. Report class distributions by pixels and fields, per-class precision/recall/F1, macro-F1, balanced accuracy, and the confusion matrix. Where feasible, quantify uncertainty by resampling fields rather than correlated pixels.

Apply the frozen growth detector and preprocessing at inference too. Field-level growth alignment requires field boundaries or a separately justified spatial-unit method in Benguet. Evaluate detector failures; do not silently omit difficult test fields and claim performance for all fields.

The detector may use a longer observation timeline than the classifier's four months. State that distinction in the methodology. This workflow produces retrospective field-season classification; a single-date regional map requires a clear rule for assigning those field seasons to the map's reference period.

**Output:** a defensible comparison of whether external data improve Benguet classification.

## Immediate pilot for the current carrot file

Candidate: `2020_31TBF_patch_27_00.nc`, carrot parcel `1667452`, 495 labeled pixels.

The exploratory unmasked timeline has strong vegetation in May–early July and a marked decline between July 4 and July 19. Planting and green-up remain unresolved, particularly because of the March–May observation gap. Counting backward from literature crop duration does not verify a planting date.

Next actions:

1. Recover this parcel's footprint and verify its overlay.
2. Retrieve the full 2020 L2A timeline, extending dates if needed.
3. Mask clouds/shadows and recreate the parcel curves.
4. Check whether an observed low-to-growth transition is recovered.
5. If onset remains uncertain, retain the field as an inspection example and exclude it from the initial green-up-aligned training set.
6. Repeat with several potato and additional carrot fields; obtain cabbage candidates.
7. Validate the detector before bulk generation of training windows.

## Sources and what they support

- [S4A dataset](https://huggingface.co/datasets/orion-ai-lab/S4A): download source.
- [S4A documentation](https://github.com/Orion-AI-Lab/S4A): grouped imagery, crop masks, parcel masks, and inspection examples.
- [S4A crop taxonomy](https://github.com/Orion-AI-Lab/S4A/blob/main/utils/encodings_en.py): original crop codes.
- [S4A paper](https://arxiv.org/abs/2204.00951): L1C source imagery and annual-reference dataset structure.
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli): filtered downloads and local destinations.
- [Earth Engine access](https://developers.google.com/earth-engine/guides/access): project registration and access requirements.
- [Earth Engine L2A collection](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED): scaling, resolutions, quality information, and cloud-mask products.
- [Copernicus start-of-season definition](https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/start-season-date-2017-present-raster-10-m-europe-yearly): seasonal-amplitude threshold approach, specifically 25% with PPI; not a validated NDVI threshold for these crops.
- [Forecasting carrot yield with optimal timing of Sentinel-2 image acquisition](https://link.springer.com/article/10.1007/s11119-023-10083-z): carrot index trajectories and variation in development timing across regions; not a universal onset detector.
- [Optimal Timing of Carrot Crop Monitoring and Yield Assessment](https://www.mdpi.com/2076-3417/14/9/3636): carrot observations through development and harvest at 125–140 days in the studied setting; not field-specific dates for S4A.
- [Monitoring Irrigation Events and Crop Dynamics](https://www.mdpi.com/2072-4292/14/5/1205): carrot cycle interpretation with farming-record support.
- [Operational Framework for Field-Scale Crop Sowing and Emergence Date Estimation](https://spj.science.org/doi/10.34133/remotesensing.0878): independent validation of satellite-derived timing estimates.

Numerical detector settings in this guide are proposed experiments unless explicitly attributed to a source. They must be calibrated and validated before being presented as reliable for Benguet or the external crop fields.
