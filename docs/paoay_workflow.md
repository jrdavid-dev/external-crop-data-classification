# Paoay application

Notebooks 01–06 remain the external preprocessing/training workflow. The **five Paoay notebooks 07–11 are implemented**. They apply the fixed model, retain uncertainty and prepare field-review records. Software checks do not establish local crop accuracy.

## Run order

| Notebook | Responsibility |
| --- | --- |
| [01 Boundaries and references](../notebooks/paoay/01_reference_review.ipynb) | Read KMZ notes, audit geometry and review reference suitability |
| [02 Satellite inputs](../notebooks/paoay/02_satellite_inputs.ipynb) | Prepare matching Sentinel-2 and Dynamic World inputs |
| [03 Model application](../notebooks/paoay/03_model_application.ipynb) | Calculate exact saved-model features and preserve raw crop predictions |
| [04 Screening and maps](../notebooks/paoay/04_maps.ipynb) | Create boundary-limited maps and separate screening flags |
| [05 Reference and field review](../notebooks/paoay/05_reference_field_review.ipynb) | Summarize notes, freeze the preliminary map and prepare January records |

Use the existing project kernel; **Restart Kernel → Run All** for each notebook in order. Nothing runs just by opening a notebook. The original eight entry points, outputs and metadata remain in the [checksum-verified archive](../notebooks/paoay/archive/eight_notebook_workflow/README.md). Do not run archived entry points in the active sequence.

## 07: references

Put `00 - Region of Interest-2.kmz` and `Crop Area.kmz` in ignored `data/paoay/`, or configure `paoay_kmz_root` in local settings. The default boundary name is `Region of Interest`; override with `paoay_roi_name` if needed. The current inputs have seven reference areas. Geometry, holes, notes and overlaps are preserved; invalid areas remain flagged rather than repaired.

A separate `<bundle id>_suitability.csv` in `references/reviews/` records manual context. Existing edits are never overwritten:

- `area_type`: `unreviewed`, `single_crop`, `rotation_or_mixed`, `seedlings_or_flowers`, `non_target_crop`.
- `temporal_evidence`: `unreviewed`, `dated_evidence_available`, `calendar_only`, `timing_unknown`.
- `evidence_source`, `review_notes`: supporting evidence and caveats.

Everything starts unreviewed. These entries do not assign crop labels, establish ground truth or exclude pixels.

## 08: satellite inputs

The default **January–April 2024** window ends May 1 exclusive and remains a historical baseline. Clear imagery does not establish compatible growth stages. A later January visit cannot directly verify 2024 occupancy without historical evidence. `period_purpose` records intent in the run log; changing dates selects a separate request. Future/incomplete windows are rejected.

Both source modes are currently set to `reuse` for local-only execution. For new collection, choose `download_missing`: verified matching files are reused and only missing pairs are downloaded. Downloads use the existing Earth Engine project/sign-in and create no export jobs or cloud assets. Corrupt/unpaired files are never automatically replaced. Both stage selections stay pending until both sources verify.

Sentinel-2 uses the training recipe: SR Harmonized, exact cloud-probability matches, probability <40, SCL/edge masks, same-date mosaics and eight half-month medians. Bands are B3/B4/B5/B8/B11/B2/clear_count. Reflectance scale is 0.0001; counts are unscaled. The study grid is 10 m with no extra ROI buffer. B5/B11 are natively 20 m and bilinearly resampled; this creates no new spatial detail. Coverage uses ROI pixel centres.

[Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) supplies broad-cover predictions, not crop species or ground truth. Preserve its masks, nine probabilities, daily observation counts, attribution and receipts. It inherits the exact Sentinel-2 dates/grid; coverage can differ. Empty bins stay explicit.

## 09: fixed model

Use the unique selected-model package or set an explicit manifest; nothing is selected by age or local prediction scores. Retain notebook 05’s feature handoff for formula provenance. Keep exact feature order: 72 interval, 48 temporal-shape or 120 combined predictors. Existing NDVI/NDRE/NDMI/SAVI formulas and temporal definitions are shared with training.

All eight bins must be complete and features finite. The saved coverage threshold (currently 0.80) applies to each composite over the whole study ROI; no imputation or relaxed thresholds. Zero usable rows leave stages pending and retain audits. For cache compatibility, 09 builds/reuses the existing guidance dependency internally; it never filters model inputs. Its illustrative rules remain two dates, top probability ≥0.60 and margin ≥0.10.

Apply the fitted classifier without refitting. Map probabilities by classifier class names, not assumed order. Raw output codes are carrot=1, cabbage=2, potato=3. These are forced choices even on other crops/noncrop surfaces. Their areas are not verified planted area. Preserve raw probabilities before screening.

## 10: maps and screening

The primary map covers only the **union of valid reference polygons within the study boundary**. Holes remain outside; overlaps count once overall. Outside pixels are masked, not Unknown or missing. Keep carrot/cabbage/potato predictions with top probability ≥0.60 and positive margin ≥0.10; weaker/tied scores become Unknown crop type. Missing inputs stay separate. Unknown is not other crops; unmodelled species may still receive confident target labels.

Screening is **flags only**, without hard exclusions or score changes. The 18-band diagnostic TIFF has minimum/maximum/mean/amplitude for NDVI, NDRE, NDMI and SAVI, plus valid-bin count and bitwise flags:

| Bit | Review flag |
| --- | --- |
| 1 | Maximum NDVI <0.20 across all eight valid bins; illustrative weak-vegetation evidence |
| 2 / 4 / 8 / 32 | Repeated supported water / built / woody / bare context |
| 16 | Incomplete spectral evidence, not proof of noncrop cover |

Flags can overlap; do not sum them as exclusive areas. Bare early and green later does not alone trigger weak-vegetation screening. NDRE/NDMI/SAVI support interpretation without separate arbitrary cutoffs. Water needs explicit water support; bare does not confirm fallow. Hard exclusions require evaluation against reliable crop and noncrop labels.

The separate conservative seven-class Dynamic World context maps remain available. Other cultivated vegetation is unassessed; water/grass/flooded vegetation/snow remain out-of-scheme cover. Period context needs at least four supported bins and group agreement. Crop identity still uses the whole four-month window; context maps are not dated crop observations.

## 11: reference and field review

Review per-boundary predictions, overlapping screening flags, suitability and original KMZ notes. Optional calendar entries retain explicit source excerpts and month lists; blank entries stay unreviewed. Calendar overlap is not correctness. Mixed/seedling areas and unmodelled crops are not promoted to verified crop-field labels.

Notebook 11 freezes a preliminary-map TIFF and snapshots of the reference/calendar/field records in a new immutable review bundle. Editable `observations.csv` and `corrections.csv` live separately and are created once. Record actual dates, GPS, crop/cover, growth stage, evidence and observer. A point does not label a whole polygon. January observations and historical evidence remain distinct.

Follow the [field-check guide](paoay_field_checks.md) for columns and evidence rules. Rerunning after edits creates a new report; old snapshots remain readable. Correction proposals do not alter the model map. A later corrected raster needs compatible dates and explicit spatial scope. Neither corrected-map agreement nor these convenience records produce an accuracy score. Independent evaluation needs a separate evidence/sampling protocol.

## Preservation and outputs

Outputs remain under the parent of `report_root`, normally `sen4agrinet_work/paoay/`: `references`, `sentinel2`, `dynamic_world`, `land_cover`, `features`, `predictions`, `crop_baseline`, `screening`, `maps`, `review` and `field_review`.

Existing stage-pointer names remain compatible; 10 adds crop-baseline/screening pointers and 11 adds `field_review/field_review_stage.json`. Changed inputs, rules or code select new keyed bundles. Failed stages remain pending, preventing silent fallback to older successes. Existing imagery, archives, model packages, partitions, jobs, mutable reviews and historical outputs stay intact. Local settings, raw/generated files and field records remain outside Git.

TODOs only: pooled other-crop training, and individual-crop training with grouped reporting. Neither method is trained or launched. Old experiments remain notebooks 12–15.
