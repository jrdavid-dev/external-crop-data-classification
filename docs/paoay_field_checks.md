# Paoay field checks

Notebook 11 freezes the preliminary map and creates editable records under `paoay/field_review/records/<reference bundle id>/`. Reruns preserve edits and save new immutable reports. No observations, corrected maps or accuracy scores are invented.

## Before January

Run 07 → 08 → 09 → 10 → 11. Keep the map dates visible. The current baseline is January–April 2024; a later January visit does not directly verify those crops. For historical comparisons, obtain farmer recall or dated records covering the relevant period.

Use `field_check_plan.csv` for boundary names, notes and suitability. It omits predictions to help avoid observation bias. Visit a practical mix of accepted and uncertain areas, not only apparent errors. This checklist is not a statistical accuracy-sampling design.

## During the visit

Add rows to `observations.csv`; keep a unique `observation_id` for each observation. Multiple observations per boundary are allowed. Record the actual observation before checking the prediction where practical.

| Fields | What to enter |
| --- | --- |
| `observation_id`, `reference_id` | Your unique observation ID and the existing boundary ID |
| `observation_date` | Actual YYYY-MM-DD; no visit year is assumed |
| `latitude`, `longitude` | Both WGS84 decimal-degree coordinates, or both blank; located points must be inside the named polygon |
| `observed_crop` | Actual crop name, including lettuce/radish/other species; required when surface is crop |
| `surface_state` | `crop`, `bare`, `built`, `water`, `woody`, `mixed` or `unknown` |
| `crop_stage` | Useful description such as seedlings, growing, recently harvested |
| `evidence_type` | `field_visit`, `farmer_recall` or `historical_record` |
| `evidence_reference`, `observer` | Photo/document/interview reference and observer; both required |
| `evidence_start`, `evidence_end` | Optional inclusive dates describing the evidence period; enter both or neither |
| `review_status` | `unreviewed`, `context_only` or `supports_map_window` |
| `notes` | Rotations, mixtures, GPS limitations, source reliability and other caveats |

Blank template rows are ignored. A field visit documents its visit date; record historical accounts separately. `supports_map_window` requires explicit evidence dates covering the entire imagery window. It remains a manual evidence claim, not proof of correctness. One GPS point never establishes a whole-polygon crop label. Invalid/outside/tiny geometries remain unsuitable for automatic pixel comparisons.

## After the visit

Rerun 11. It snapshots your records and reports whether their evidence periods overlap the map. Located observations show the original prediction for the corresponding raster cell in `observation_review.csv`. This is point/context review, not an automatic correctness score or a confusion matrix.

Keep correction proposals in `corrections.csv`:

| Field | Meaning |
| --- | --- |
| `correction_id` | Unique proposal ID |
| `observation_id` | Existing observation supporting the proposal |
| `proposed_label` | Actual proposed crop or cover, including unmodelled classes |
| `decision` | `proposed`, `reviewed` or `rejected` |
| `reason` | Evidence and explanation; note any temporal mismatch |

These proposals never rewrite the model raster, relabel training data or imply whole-polygon corrections. To create a corrected raster later, specify the affected geometry/pixels and compatible dates explicitly, and save a separate product. Agreement between corrections and their source observations is not model accuracy. Quantitative local evaluation needs a separate sampling and evidence protocol applied to frozen original predictions.

## Saved review

Each report bundle contains the frozen preliminary TIFF, per-boundary crop and screening tables, original calendar context, field checklist and immutable copies of all editable records. Old reports remain readable after source CSV edits. Keep photographs/documents separately and retain stable references to them. All local records and generated artifacts remain outside Git.
