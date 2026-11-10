# Project handoff and future TODOs

Updated: 10 November 2026. All three Paoay implementation batches are complete. The future experiments below are **planned, not implemented**.

## Current workflow

| Notebooks | What they do |
| --- | --- |
| 01–03 | Collect labelled external parcels and Sentinel-2 imagery from Sen4AgriNet, FranceCrops and eligible CropHarvest records |
| 04 | Prepare usable four-month crop windows and preserve locked parcel splits |
| 05 | Calculate spectral bands, NDVI/NDRE/NDMI/SAVI and temporal features |
| 06 | Train/evaluate models and save the validation-selected model with its exact recipe |
| 07 | Review Paoay KMZ boundaries, original notes and reference suitability |
| 08 | Prepare matching Sentinel-2 and Dynamic World inputs; currently set to local-only `reuse` |
| 09 | Apply the saved model without retraining; preserve raw crop probabilities |
| 10 | Map reference boundaries, show Unknown/missing inputs, add screening flags and zoomed previews |
| 11 | Review results, freeze the preliminary map and prepare separate field observations/correction proposals |

The saved model uses **Sen4AgriNet + FranceCrops**; CropHarvest contributes no groups to its saved split. It predicts carrot, cabbage and potato. Other crops can receive incorrect target labels; Unknown is not an other-crops detector.

The current Paoay baseline is **January–April 2024**. All five Paoay notebooks ran successfully using existing downloads/model. Maps and reports saved; existing data/model were preserved. The implementation suite passed 426 tests; later local-reuse checks also passed. Software tests do not establish crop accuracy.

Current mapping rules:

- Pixels need complete observations in all eight half-months and valid features. Each composite must currently pass the whole-study-area 80% clear-coverage gate.
- Primary crop maps cover the KMZ reference-boundary union. Outside is masked; missing inputs and Unknown are separate.
- Crop predictions need top score ≥0.60 and positive top-two margin ≥0.10. These are illustrative, not locally calibrated accuracy guarantees.
- Dynamic World and spectral screening are **review flags only**. No hard exclusions or automatic correction of the primary crop map.
- Field templates currently contain no completed observations. Correction proposals never overwrite original predictions.

## Next TODOs, in order

### 1. Audit CropHarvest for an independent external test

- [ ] Check explicit carrot/cabbage/potato labels, independent field counts, dates and usable geometry.
- [ ] Check underlying source overlap with training data; a different dataset name alone does not guarantee independence.
- [ ] Plan training/internal validation on Sen4AgriNet + FranceCrops, with eligible CropHarvest examples reserved for external testing.
- [ ] Use the same imagery/features and keep whole parcels separated. Do not tune on the external test.
- [ ] If only some target crops qualify, report that limited scope. Binary crop/noncrop labels cannot test crop species; use another source if necessary.

### 2. Add a separate partial-coverage Paoay experiment

- [ ] Keep the 2024 baseline and strict mode unchanged.
- [ ] Add an explicit application mode that accepts only complete pixels instead of blocking every pixel because the whole ROI fails coverage.
- [ ] Keep cloud masks and eight-bin feature requirements unchanged; no imputation or threshold relaxation.
- [ ] Report full boundary area, usable area and missing area separately. Do not extrapolate clear-area percentages into cloudy portions.
- [ ] Select any boundary-specific periods using coverage and dated crop evidence, not attractive prediction scores. Different periods are separate historical cases.

### 3. Test field-level estimates from clear portions

- [ ] Confirm actual single-crop field boundaries and crop-period homogeneity. Mixed KMZ areas cannot automatically receive one label.
- [ ] On labelled external fields, hide spatial portions and test whether remaining clear pixels support the correct field label. Compare with observed-only results.
- [ ] Choose acceptance rules on training/validation data: clear coverage, spatial spread, prediction agreement and temporal evidence. Nearby pixels are not independent fields.
- [ ] Keep observed predictions, inferred field portions and unresolved areas visually separate. Do not present inferred cloudy pixels as satellite observations.
- [ ] Freeze these rules before the independent external test. This is proposed field inference, not implemented gap filling.

### 4. Prepare January evidence collection

- [ ] Prioritize B1, Area 2 and Area 3 for dated crop/rotation records from 2025–2026.
- [ ] Record actual visit date, GPS, crop/cover, growth stage, photos/document references and observer.
- [ ] Keep direct field visits separate from farmer recall/historical records. A later January visit alone cannot verify 2024 or 2025 occupancy.
- [ ] Preserve the preliminary map. Record corrections separately; a corrected raster later needs explicit spatial scope and compatible dates.

## Findings to retain

The per-boundary audit checked **7 boundaries × 19 completed four-month windows in 2025–2026**. Only **B1, January–April 2025** passes the separately audited boundary-level rule: 324/402 pixels (80.6%) have complete inputs. The current whole-ROI gate would still block that period; boundary-only processing is not implemented.

January–April 2025 was the strongest window by weakest-half-month coverage for every boundary. Complete usable portions were Area 1: 29.3%; Area 2: 61.6%; Area 3: 10.3%; Area 4: 13.4%; Area 5: 48.0%; B2: 43.5%. Small fragments are exploratory, not representative full-field samples.

B1 notes describe cabbage January–February then carrot March–June; Area 2 also describes rotations. These calendars do not verify one crop across a historical four-month window. Area 3 is a cabbage-reference candidate pending dated evidence. B2 is lettuce; Area 5 contains flowers/seedlings. Do not treat them as ordinary target-crop labels.

Current model training is small: carrot 6 parcels/103 pixels; cabbage 9/168; potato 33/521. Prioritize independent carrot/cabbage fields and timing/label checks. Overall accuracy hides carrot weakness; report per-crop performance. More pixels from the same field do not replace more independent fields.

## Deferred

- [ ] Train an explicit pooled other-crops class, or train individual species and group non-target species for reporting.
- [ ] Validate any hard noncrop exclusions before enabling them.
- [ ] Design independent local accuracy assessment if suitable dated evidence becomes available. Corrected-map agreement is not original-model accuracy.

## Resume safely

Work solo. Preserve existing downloads, models, partitions, jobs, mutable reviews and historical outputs. Keep new experiments in separate keyed folders. Inspect `git status` before edits; existing uncommitted work spans multiple earlier tasks. Nothing in this handoff authorizes new downloads, training or publishing by itself.

Start with the **CropHarvest feasibility audit**, then agree on experiment scope before implementation. Do not rerun collection/training just to review current results.

Useful files:

- [Current Paoay workflow](paoay_workflow.md)
- [January field-check instructions](paoay_field_checks.md)
- [All-boundary imagery audit](../sen4agrinet_work/paoay/period_search/by_boundary_20261110T064722Z/report.md) — local generated output, outside Git
- Current artifacts: `sen4agrinet_work/paoay/`; use stage-pointer JSONs rather than selecting folders by modification date.
- Original Paoay entry points: `notebooks/paoay/archive/eight_notebook_workflow/`. Active Paoay entry points are 07–11; older experiments remain 12–15.
