# External Crop Data for the Benguet Thesis

> Methodology notes include earlier standalone examples. Those scripts have been removed from the working tree and remain in Git history; use the Sen4AgriNet notebook for the current executable workflow.

## Study requirements

- Target crops: **cabbage, carrot, and potato**.
- Classifier: **Random Forest**.
- Satellite inputs: Sentinel-2 **L2A surface reflectance**.
- Temporal input: a **four-month window containing eight semi-monthly median composites**.
- Semi-monthly bins: days 1–15 and days 16–month-end.
- Selected bands: B3, B4, B5, B8, and B11; derived indices include NDVI, NDRE, and NDMI.
- With five bands and three indices per interval, each pixel has 64 interval-wise features before any additional temporal features.

## Proposed research direction

Evaluate whether external crop-reference data can improve crop classification in Benguet when verified local training data are scarce.

This is cross-region adaptation under limited labels. It differs from the original proposal's spatial expansion of a Paoay-trained model to Atok and Benguet.

### Practical experiment

| Model | Training source | Purpose |
|---|---|---|
| External baseline | External samples only | Measure transfer to Benguet |
| Local baseline | Limited verified Benguet samples only | Establish local-only performance |
| Combined model | External + the same limited Benguet samples | Test the benefit of external data |

Evaluate all models on the same independent Benguet test fields. Increase the local training budget where data permit and measure improvement. Count independent fields separately from extracted pixels. Keep every pixel and overlapping temporal window from the same field in the same partition.

Possible extension: compare randomly selected local training fields against uncertain and geographically diverse fields selected through active learning. Verify their labels, then retrain the forest.

Standard Random Forests do not support neural-network-style fine-tuning of their existing trees. The combined-data route trains a **new forest**. Borrowing a finished forest and borrowing its training data are different approaches.

## Sen4AgriNet: what was verified

Sen4AgriNet supplies dated Sentinel-2 observations and crop-reference masks from LPIS farmer declarations for Catalonia (2019–2020) and France (2019).

### Data fields

| Field | Meaning |
|---|---|
| Band arrays | Satellite observations with dimensions time × height × width |
| `timestamp` | Image acquisition dates |
| `labels` | Crop-category mask |
| `parcels` | Parcel-identifier mask |
| Patch metadata | Year, country, tile, and patch identifiers |

Farmer-declared LPIS labels are ground-reference data rather than predictions from another crop classifier. They are not necessarily independently field-verified and may contain reporting errors or mixed pixels.

### Confirmed crop definitions

The repository's `utils/encodings_en.py` and example COCO annotation file define all three target categories:

| Crop | Dataset name | Original code | Linear category ID |
|---|---|---:|---:|
| Cabbage | Cabbages | 213 | 17 |
| Carrot | Carrots | 231 | 32 |
| Potato | Potatoes | 510 | 101 |

**Defined categories do not establish available samples.** The example COCO file contains one patch with 205 annotations and none for these three crops. This is only an example, not a dataset-wide result. The published 11-class benchmark includes potato; cabbage and carrot are excluded from that experiment.

Local inspection has found carrot and potato in the downloaded 2020 subset. Cabbage was absent from the earlier 446-file scan; that does not establish absence across S4A. Counts can change as downloads finish. Dataset-wide availability and deduplicated independent-field counts remain unverified. Use FranceCrops as a candidate additional cabbage source.

## Compatibility and limitations

1. **Temporal format:** dated observations can be grouped into your eight composites. The README example uses six monthly composites and must be modified. Monthly medians cannot be split back into two true semi-monthly medians.
2. **Processing level:** the documented source imagery is **L1C**, whereas your workflow uses **L2A**. Harmonize levels before combining data. One route is recovering georeferenced reference labels and retrieving corresponding L2A imagery, subject to verifying the spatial metadata needed.
3. **Spatial resolution:** B3/B4/B8 are 10 m; B5/B11 are 20 m. Align them consistently. Resampling does not add spatial detail.
4. **Crop timing:** annual declarations and image timestamps do not provide parcel-specific planting and harvest dates. Establish that each label applies to the chosen four-month window; do not label arbitrary windows from an annual declaration.
5. **Growth-stage alignment:** eight intervals from different regions may represent different stages. Document how external and Benguet windows are aligned.
6. **Cloud gaps:** use a consistent masking and missing-observation policy. The README's optional interpolation is not automatically suitable for your workflow.
7. **Validation scope:** Paoay-only testing supports local claims. Wider Benguet predictions remain exploratory without wider independent validation.

**Current verdict:** temporally adaptable and a documented potato candidate, but not yet confirmed as a complete three-crop training source.

## Downloading one tile folder on Windows

Open PowerShell in the folder where downloads should be saved.

Install the downloader:

```powershell
python -m pip install -U huggingface_hub
```

Download only `data/2019/31TBF`:

```powershell
hf download orion-ai-lab/S4A --repo-type dataset --include "data/2019/31TBF/*" --local-dir S4A
```

Files are saved under `S4A\data\2019\31TBF`. Re-running the download command reuses completed files.

Alternatively, browse the dataset files and download individual `.nc` files.

## Inspecting downloaded files

Use the supplied `inspect_crops.py` script. It reads label and parcel masks without loading spectral imagery.

Install dependencies:

```powershell
python -m pip install numpy netCDF4
```

Run from the folder containing the script:

```powershell
python inspect_crops.py "S4A\data\2019\31TBF"
```

Files can be on another disk:

```powershell
python inspect_crops.py "D:\S4A\data\2019\31TBF"
```

Replace the example path with the actual folder. The script scans subfolders recursively. If `python` is unavailable but the Python launcher is installed, use `py` instead.

The script defaults to original codes. Use `--encoding linear` only after confirming that the masks use remapped category IDs.

### Output interpretation

- Per-file crop pixel counts and distinct parcel IDs.
- A final summary of crop pixels, parcel occurrences, files containing each crop, and successful/failed file counts.
- **Parcel occurrences are not unique field totals:** a field may cross patches.
- Pixel totals across repeated locations/years are not unique land area.
- Zero counts establish absence only in the scanned files under the selected encoding.
- A dataset-wide independent-field count requires confirming parcel-ID scope and deduplicating by appropriate region/year identifiers.

Inspection dependencies are available in the current workspace. The mask-counting script was tested on synthetic NetCDF files; a real carrot patch was subsequently read and visualized. Full-dataset correctness, deduplication, temporal label relevance, and final preprocessing remain separate checks.

## FranceCrops: additional cabbage and potato source

FranceCrops supplies Sentinel-2 L2A time series for French parcels in 2022, with labels from IGN's Registre Parcellaire Graphique (RPG). Its published benchmark class map includes **cabbage (`CHU`, y=4)** and **table potato (`PTC`, y=12)**; carrot is absent from the 20-class benchmark. Confirm that CHU matches the study's intended cabbage definition.

### What the files contain

| File/config | Contents | Planned use |
|---|---|---|
| Benchmark Parquet, such as train.parquet | x, y, parcel_id; all benchmark crop classes | Inspect class counts and exploratory timelines |
| class_map | Label IDs and crop names | Confirm cabbage/potato mapping |
| metadata | Labels and split membership without large x arrays | Select candidates and preserve provenance |
| geolocation | Parcel geometries linked by parcel_id | Recover field footprints for original L2A retrieval |

One benchmark row represents a parcel: **100 sampled pixels × 60 aligned dates × 12 bands**. Dates run every five days from February 1 to November 23, 2022. The five study bands are included. These sampled pixels are not a spatial image grid and cannot reconstruct an RGB patch.

The published training split reports 5,000 parcels per class. This is a publisher-reported count, not an independently audited count in this project. Downloaded counts and independent-field totals must be checked.

Cloudy/missing observations were filled by linear interpolation. Therefore, 60 steps are not 60 guaranteed clear acquisitions, and composites made from these arrays include reconstructed values.

### FranceCrops workflow

1. Download or inspect class_map, metadata and geolocation first; inventory benchmark files and preserve split membership.
2. Confirm y=4 cabbage and y=12 potato against the actual release. Count rows and inspect duplicates.
3. Join labels to geolocation by verified parcel_id, not table row position.
4. Decode WKB geometry where needed. Confirm the source CRS from documentation before converting coordinates or querying Earth Engine.
5. Check polygon validity and deduplicate fields. Keep the label source, year and original IDs.
6. Retrieve **original** Sentinel-2 L2A acquisitions for those footprints, including a full-year growth-inspection timeline.
7. Apply the same masking, scaling, spatial alignment, growth detector and four-month construction as S4A and Benguet.
8. Use the supplied interpolated series for exploratory inspection; evaluate it separately if used for final classification.
9. Check that the annual declared crop applies to the selected cycle. The label is supplied; planting/harvest dates are not.

Download from the desired working folder:

```powershell
hf download saget-antoine/francecrops --repo-type dataset --local-dir ".\FranceCrops"
```

This downloads the whole repository, not only cabbage. Inspect benchmark files using `inspect_francecrops.py`; geolocation decoding/mapping and original-image retrieval are separate steps.

**Source combination:** candidate S4A carrot/potato plus FranceCrops cabbage/potato. Harmonize imagery and class definitions; do not assume that combining sources guarantees improved Benguet performance. Count independent fields by crop and source.

## Growth-based four-month selection

Develop and validate a crop-independent green-up detector using cloud-masked parcel NDVI. Identify a preceding low period and a sustained rise for each cycle; use robust cycle-specific baseline and peak estimators. Test relative-amplitude thresholds as development settings, confirm crossings with observed clear acquisitions, and record date intervals rather than unjustified exact dates. NDRE and RGB provide supporting evidence.

Proposed frozen selection rule: **start on the first day of the month containing accepted green-up and take that month plus the next three calendar months**. Flag uncertainty intervals crossing month boundaries. Reject ambiguous cycles, long gaps hiding onset, weak signals and unresolved crop rotations.

Research supports amplitude-based phenology methods, but no threshold is automatically validated for these three crops. Green-up is not planting; decline is not necessarily harvest. Peak-based onset detection is retrospective and may require imagery outside the classifier's four-month window. Use the same detector at inference, with field boundaries or a separately justified spatial-unit method in Benguet.

## Research-informed checks and experiments

### 1. Field-level cloud coverage

Record clear fractions and supporting pixels for every field/acquisition. Calibrate an acceptance threshold on development fields, inspect residual contamination, and reject unreliable observations. The current retrieval script's 80% setting is exploratory; a cited cereal-yield method used 90% for one pipeline, but neither is a universal rule. Track sample losses by crop and region.

### 2. Bands versus bands plus indices

Compare the same eight intervals using:

- **40 features:** five bands × eight intervals.
- **64 features:** five bands plus NDVI, NDRE, NDMI × eight intervals.

Hold fields, sample budgets, preprocessing and evaluation fixed. Use development data for feature selection, not the final test set. Report which intervals and features contribute, while recognizing correlated predictors.

### 3. Four-month window sensitivity

Use the frozen green-up rule for the primary result. As a proposed sensitivity analysis, shift the selected window one semi-monthly bin earlier/later while preserving eight consecutive calendar half-month bins. Identify these variants separately from the primary four-whole-month definition. Keep field partitions fixed and report coverage changes. Do not choose the variant that performs best on the final test set.

### 4. Three training-source comparisons

Keep external-only, limited Benguet-only, and external plus the same Benguet training fields as the main experiment. Train new Random Forests and use the same independent Benguet test fields. Do not mix pixels or windows from the same field across partitions.

### 5. Per-crop performance

Report cabbage, carrot and potato precision, recall and F1, alongside macro-F1, balanced accuracy, a confusion matrix and overall accuracy. Report pixel and independent-field counts, detector acceptance rates, missing-data exclusions and eligible mapping coverage. High overall accuracy alone can hide a failing minority crop.

### Optional: gap-filling comparison

Start with complete-data samples. If necessary, compare against a predefined short-gap treatment; preserve observed/reconstructed flags and fit learned imputation on training data only. Avoid inventing onset or harvest evidence across long gaps.

These additions are proposed project experiments. The cited articles do not establish the exact four-month/eight-bin design or a validated vegetable green-up detector.

## Next steps

1. Finish S4A inventory and inspect independent carrot/potato candidates.
2. Verify FranceCrops cabbage labels, field counts, geolocation schema and CRS.
3. Recover and deduplicate labeled field footprints for both sources.
4. Retrieve full-year original L2A timelines; mask clouds and inspect alignment.
5. Develop, validate and freeze a green-up detector and window rule.
6. Accept only cycles with supported crop identity and adequate observations.
7. Create eight semi-monthly composites and 40/64-feature comparison tables.
8. Split by independent field and run the three training-source experiments.
9. Evaluate per crop, window sensitivity, coverage and uncertainty on independent Benguet fields.

## Other sources considered

- **USDA Cropland Data Layer:** separately defines potato (43), carrot (206), and cabbage (243). Public crop maps can supply potentially noisy external reference labels, but are predictions rather than direct field ground truth. A downloadable trained forest was not verified.
- **EuroCrops:** supplies farmer-declared crop labels and field boundaries. Target-class availability and counts in selected country files still need checking.
- **ZueriCrop:** includes potato but groups vegetables; released imagery contains only blue, green, red, and NIR. It does not match the complete target class/band requirements as released.

## Research sources for the added experiments

- [Reconstructed Sentinel-2 crop mapping](https://www.sciencedirect.com/science/article/pii/S0168169923006154): temporal inputs and missing-data treatment; accessible preview, not a validated method for these vegetables.
- [Sentinel crop-mapping review](https://doi.org/10.1080/10106049.2026.2690363): transfer, preprocessing and spatial validation context.
- [L2A bands/NDVI and classifier comparison](https://link.springer.com/article/10.1007/s12518-025-00659-x): basic feature-stack pipeline; abstract reviewed.
- [Multitemporal feature optimization](https://www.sciencedirect.com/science/article/pii/S1569843224005284): feature comparison inspiration; abstract reviewed.
- [Pixel-based yield study](https://ira.agroscope.ch/de-CH/Page/Einzelpublikation/Download?einzelpublikationId=56073): cloud quality and cross-year evaluation; cereal yield, not vegetable classification.
- The supplied article identifier S2772375526006301 remains unverified and is not used as methodological evidence.

## Sources

- [FranceCrops dataset and processing](https://huggingface.co/datasets/saget-antoine/francecrops)
- [Copernicus start-of-season definition](https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/start-season-date-2017-present-raster-10-m-europe-yearly)
- [Carrot Sentinel-2 development study](https://link.springer.com/article/10.1007/s11119-023-10083-z)
- [Sen4AgriNet dataset and download folders](https://huggingface.co/datasets/orion-ai-lab/S4A)
- [Sen4AgriNet README](https://github.com/Orion-AI-Lab/S4A/blob/main/README.md)
- [Original crop-code definitions](https://github.com/Orion-AI-Lab/S4A/blob/main/utils/encodings_en.py)
- [Encoding configuration](https://github.com/Orion-AI-Lab/S4A/blob/main/utils/config.py)
- [Example COCO annotations](https://github.com/Orion-AI-Lab/S4A/blob/main/coco_files/coco_example.json)
- [Sen4AgriNet paper](https://arxiv.org/abs/2204.00951)
- [Dataset exploration notebook](https://github.com/Orion-AI-Lab/S4A/blob/main/patch_aggregation_visualization.ipynb)
- [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [USDA CDL in Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL)
- [EuroCrops](https://github.com/maja601/EuroCrops)
- [Random Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Prepared from the proposal, supplied dataset documents, and repository inspection in this conversation on October 13, 2026. No dataset-wide crop count has been computed yet.

Updated in the THS workspace on October 13, 2026 with FranceCrops, growth-based window selection and research-informed checks. Earlier statements of dataset-wide availability remain provisional.
