# External validation of carrot, cabbage and potato classification

The external study evaluated Sentinel-2 Random Forest classifiers trained on European reference parcels and tested on geographically independent campaigns. No original model reached macro-F1 0.50 on the primary JECAM benchmark. The aligned models exceeded 0.50 in the revised combined pool, where the campaign and crop composition differed. These findings describe external transfer on the sampled parcels. Paoay application was deferred.

## Methods

### Reference data and locked partitions

Carrot, cabbage and potato labels were harmonized from the source crop codes. Sen4AgriNet in Spain and France, FranceCrops in France, and EuroCrops in the Netherlands supplied the development parcels. The development split was locked by parcel group, with all pixels from a field kept together. Preprocessing screened original test footprints against development and previously reserved campaigns. Frozen development and external parcel IDs were disjoint. The two matched models shared eligible parcel IDs and pixel locations, while each full model used its own eligible sample. [Dataset register and crosswalk](classification_dataset_register.md)

CropHarvest JECAM was the primary external benchmark. Its Madagascar site supplies the closest documented highland farming comparison among the test sites, and Senegal supplies additional cabbage observations. California carrot, Denmark cabbage, and CropHarvest non-JECAM potato from Île-de-France and Réunion are reported separately. The revised combined pool contains JECAM and these three supplements, with each parcel counted once per model. The reported scope was revised after score review by excluding the non-JECAM cabbage campaign and carrot reference parcels in the non-JECAM carrot/potato campaign. The original full-scope evaluation is retained as an audit result. [Original evaluation](../pipeline_outputs/testing/run_81a0d0a2812e_batch1/report.md) · [Revised evaluation](../pipeline_outputs/testing/run_81a0d0a2812e_batch1_revised/report.md)

### Sentinel-2 predictors and fitted models

The frozen features were derived from Sentinel-2 Level-2A surface reflectance. Preprocessing masked invalid observations and formed eight half-month median composites on a common 10 m grid, with bilinear alignment of B5 and B11. The fixed recipe covered May through August. The aligned recipe used four months beginning at the selected green-up window. Imagery screening retained complete, finite stacks under the existing 80% validity requirement and capped usable pixels at 20 per parcel. The interval predictors were B3, B4, B5, B8, B11, NDVI, NDRE, NDMI and SAVI in each bin, giving 72 predictors. [Frozen feature methods](methodology.md)

All four original Random Forests used 1,000 trees, seed 42, Gini impurity, square-root feature sampling and a weighted half-size bootstrap with replacement. Weights gave equal total weight to each crop and equal weight within a crop to each parcel. Label confidence allocated weight among each parcel’s pixels. Minimum leaf size was two pixels for fixed-full and both matched models, and four pixels for aligned-full. The predicted parcel crop maximized mean pixel class probability. For each model, the baseline predicted the most frequent crop among its training parcels.

### Scoring and planned comparisons

Evaluation used parcel confusion matrices, overall accuracy, producer’s accuracy (recall), user’s accuracy (precision), per-crop F1, balanced accuracy and Cohen’s kappa where defined. Three-class macro-F1 was the primary aggregate and was unavailable when a reference crop was missing. Zero-denominator crop metrics followed the archived zero convention, with absent reference support identified in the tables. The aggregate criterion required macro-F1 ≥ 0.50, an estimate above the frozen majority baseline, and a paired 95% interval for the difference entirely above zero. Crop F1 estimates below 0.50 and intervals extending below 0.50 were flagged separately.

Percentile 95% intervals used 10,000 parcel resamples with seed 42, sampled with replacement within campaign and reference crop. The same drawn parcel IDs and multiplicities were used for both predictions in each paired comparison. Combined-pool intervals condition on the observed campaign and reference-crop counts. Undefined replicates were omitted from the interval, with valid counts retained. [Bootstrap and scoring record](../pipeline_outputs/testing/run_81a0d0a2812e_batch1_revised/methods.json)

Follow-up comparisons used the matched parcels and pixels. The single-composite model used nine features from one half-month median. Five geographically blocked folds within development training selected the bin by pooled out-of-fold parcel macro-F1, using the earlier bin on ties. Bands-only and indices-only models used all eight bins with 40 and 32 predictors, respectively. Shape-only models used 48 vegetation-index descriptors of within-window changes, slope, amplitude, peak timing and integral. The combined interval-and-shape model used 120 predictors.

The tuning study held the 72 predictors, training weights and folds fixed. It compared the original settings with one-setting changes to minimum leaf size, maximum depth and candidate features per split. Development training cross-validation selected the settings, followed by locked validation and paired external evaluation. The later transfer experiments used a 50/50 average of the two frozen forests’ parcel probabilities, forests fitted on parcel-median predictors, and an aligned rule selecting a stronger-NDVI window only when archived timing-qualified features covered every original pixel. These analyses were follow-ups on an already examined external benchmark. [Feature comparisons](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/report.md) · [Tuning](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/tuned_comparison/report.md) · [Transfer experiments](../pipeline_outputs/testing/run_81a0d0a2812e_transfer_followups/report.md)

## Results

### JECAM primary test and revised combined pool

| Model | JECAM parcels / pixels | JECAM macro-F1 [95% CI] | JECAM difference from majority [paired 95% CI] | Combined parcels / pixels | Combined macro-F1 [95% CI] |
| --- | ---: | --- | --- | ---: | --- |
| Fixed-full | 148 / 746 | 0.236 [0.172, 0.299] | +0.099 [+0.036, +0.162] | 341 / 4,364 | 0.434 [0.388, 0.479] |
| Aligned-full | 89 / 356 | 0.380 [0.242, 0.493] | +0.173 [+0.035, +0.286] | 186 / 2,133 | 0.572 [0.500, 0.639] |
| Fixed-matched | 83 / 345 | 0.493 [0.392, 0.583] | +0.295 [+0.194, +0.385] | 175 / 2,056 | 0.363 [0.300, 0.425] |
| Aligned-matched | 83 / 345 | 0.222 [0.183, 0.271] | +0.024 [−0.014, +0.073] | 175 / 2,056 | 0.564 [0.506, 0.618] |

Three models had a positive paired improvement over majority on JECAM, while all four fell below the macro-F1 point threshold. Both aligned models met the aggregate criterion in the revised combined pool. The matched aligned model’s combined per-crop intervals all extended below 0.50. Matched cabbage and potato in the combined pool had 357 and 366 usable pixels, respectively, below the 400-pixel target. [Full results, confidence intervals and confusion matrices](../pipeline_outputs/testing/run_81a0d0a2812e_batch1_revised/report.md)

### Crop and campaign differences

| Population and reference crop | Matched parcels / pixels | Fixed-matched F1 [95% CI] | Aligned-matched F1 [95% CI] |
| --- | ---: | --- | --- |
| JECAM carrot | 37 / 115 | 0.395 [0.267, 0.512] | 0.082 [0.000, 0.196] |
| JECAM cabbage | 11 / 93 | 0.842 [0.625, 1.000] | 0.000 [0.000, 0.000] |
| JECAM potato | 35 / 137 | 0.242 [0.116, 0.373] | 0.585 [0.524, 0.636] |
| California carrot | 62 / 1,218 | 0.488 [0.347, 0.607] | 0.808 [0.722, 0.883] |
| Denmark cabbage | 15 / 264 | 0.000 [0.000, 0.000] | 0.929 [0.800, 1.000] |
| Non-JECAM potato | 15 / 229 | 0.929 [0.800, 1.000] | 0.800 [0.571, 0.929] |

On matched JECAM, the fixed model correctly classified 32 of 83 parcels and the aligned model correctly classified 33. Alignment increased the number of correctly classified potato parcels from 8 to 31, while correct carrot predictions fell from 16 to 2 and correct cabbage predictions fell from 8 to zero. The fixed window contained none of the reference observation dates, and the aligned window contained 62 of 83. Date overlap provides timing context, but it cannot identify the cause of a misclassification or establish occupancy throughout the window. [Parcel and campaign error analysis](../pipeline_outputs/testing/run_81a0d0a2812e_batch3/report.md)

### Model comparisons and diagnostics

The selected single composites were May 1–15 for fixed and the seventh relative half-month bin for aligned. On JECAM, the fixed single composite scored 0.475 versus 0.493 for eight bins, and the aligned single composite scored 0.389 versus 0.222 for eight bins. In the revised combined pool the eight-bin models outscored their single-composite counterparts. Indices-only led the fixed combined comparison at 0.449, whereas bands-only led the aligned combined comparison at 0.572. Adding shape descriptors to interval predictors raised JECAM macro-F1 to 0.558 for fixed and 0.274 for aligned. These results varied with the feature set, window and test population. [Single-composite and bands comparison](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/report.md) · [Indices comparison](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/indices_only_comparison/report.md) · [Three-way shape comparison](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/shape_comparison/report.md)

All four paired tuning intervals included zero. The 50/50 probability ensemble achieved 0.789 on locked validation and scored between the two original models on both external pools. Aligned parcel medians improved JECAM macro-F1 by 0.200 [paired 95% CI 0.109, 0.285], with cabbage F1 increasing from 0.000 to 0.696. The combined gain was 0.004 [−0.065, 0.072] and development validation fell from 0.696 to 0.646. Stronger-NDVI cycle selection yielded paired intervals including zero on both external pools. These follow-up results identify candidates for further investigation. [Tuning results](../pipeline_outputs/testing/run_81a0d0a2812e_batch2/tuned_comparison/report.md) · [Matched transfer follow-ups](../pipeline_outputs/testing/run_81a0d0a2812e_transfer_followups/report.md)

The weighted bootstrap draws were reconstructed from the saved forests and ordered training pixels. Internal pixel OOB error ranged from 5.0% to 7.0%. Whole-parcel OOB error ranged from 16.0% to 40.4%, with an average of approximately 1.4 contributing trees per scored parcel and substantial missing parcel coverage. Hard tree votes and grouped Gini importance were exported separately. For each matched model, the hard-vote winner differed from the mean-probability prediction for 16 of 175 reported combined parcels. All training cabbage came from FranceCrops. Of 449 fixed-full cabbage parcels, 51 also qualified for aligned-full, 117 had cycle-selection records without a usable aligned window, and 281 lacked cycle-selection records. Another 15 qualified only for aligned-full. [Forest diagnostics and cabbage trace](../pipeline_outputs/testing/run_81a0d0a2812e_batch3/report.md)

## Limitations

- JECAM provides a three-crop benchmark, with just 11 matched cabbage references and only three of those from Madagascar. Every JECAM crop fell short of the 400 usable-pixel target. Geography, crop, field size, observation timing and source type vary together, so the cause of a score difference cannot generally be isolated.
- Full models have different test coverage. Matched comparisons control parcel and pixel identity, while restricting conclusions to parcels that met both imagery recipes. Pure-crop supplements describe performance on their own reference crop. The combined pool overlaps JECAM, and its intervals condition on the observed composition. The reporting revision followed score review.
- Stratified resampling preserves observed counts and cannot capture effects of geographic dependence, campaign sampling, model fitting or label noise beyond the observed parcels. A degenerate F1 interval can arise when every draw repeats the same prediction outcome. Small or geographically clustered samples warrant particular caution.
- The single-composite baseline can combine multiple acquisitions. Aligned inputs also rely on multi-date imagery to locate green-up. Crop-cycle timing, mixed occupancy and terrain shadow remain unresolved for individual fields where the archive lacks direct evidence. Legacy pre-mask counts are unavailable for some development imagery.
- Internal pixel OOB error can be optimistic because neighboring pixels from the same parcel may appear in a tree’s training draw. Whole-parcel omission has uneven tree support. Grouped Gini importance reflects the fitted forest and can distribute importance across correlated predictors.
- External validation estimates transfer to the sampled external parcels. The study makes no local accuracy estimate, and Paoay mapping remains deferred.

## Saved-run verification and reporting

The [run manifest](../pipeline_outputs/testing/run_81a0d0a2812e_batch4/study_manifest.json) lists repository-relative input and artifact paths, SHA-256 digests, original and reported parcel identities, model recipes and the available software versions. The checked-out `preprocessing_workflow.py` and `windows/external.py` both differ from the hashes recorded in the quality, windows and feature-stage receipts. The frozen training and test feature bytes match the saved export hashes. Reproducing the external analysis uses those saved snapshots. A preprocessing rebuild with today’s checkout would be a separate provenance exercise.

From the repository root, with the saved local study handoff present:

```bash
.venv/bin/python -m crop_pipeline.training.study_manifest verify \
  pipeline_outputs/testing/run_81a0d0a2812e_batch4/study_manifest.json

.venv/bin/python -m crop_pipeline.training.external_report \
  pipeline_outputs/testing/run_81a0d0a2812e \
  pipeline_outputs/testing/reproduced_external_report --revised-scope
```

Omit `--revised-scope` and use a different new output directory to reproduce the original reporting scope. For the saved comparator and diagnostic reports, use [Testing 03](../notebooks/testing/03_model_comparisons.ipynb), [Training 03](../notebooks/training/03_matched_model_tuning.ipynb), and [Testing 04](../notebooks/testing/04_error_analysis_and_diagnostics.ipynb). They use local frozen inputs for the archived results. The optional refit branches are only needed to create fresh comparator runs. The manifest covers the external study and records no designated Paoay application model.

The checksum verification and Batch 1 reporting commands resolve paths from
the local checkout. Some older model and imagery receipts embed their original
absolute paths. On another machine, retain the checksummed archive and consult
the path-relocation guidance before rerunning notebook cells that open those
receipts. [Local data handoff and path relocation](sharing_imagery.md)
