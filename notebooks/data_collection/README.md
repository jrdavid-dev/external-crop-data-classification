# Data collection

| Notebook | Responsibility |
| --- | --- |
| 01 External reference audit | Inspect local reference availability and saved selection coverage; record source roles |
| 02 Sen4AgriNet collection | Development references and imagery, mostly potato with some carrot |
| 03 FranceCrops collection | Potato/cabbage development references and imagery |
| 04 EuroCrops collection | Additional Dutch carrot development references and imagery |
| 05 CropHarvest test collection | Primary JECAM test imagery or separate non-JECAM carrot/potato and cabbage supplements |
| 06 California carrot test supplement | Supplemental California DWR carrot parcels |
| 07 Denmark cabbage test supplement | Supplemental Denmark Marker cabbage parcels |
| 08 Dataset summary | Read saved batches, report selected groups and status, save a versioned local manifest |

Run 01 when reviewing references, then only the source notebooks needed for top-ups.
Run 08 after collection or whenever a local progress snapshot is useful. Continue
with [preprocessing 01](../preprocessing/01_imagery_quality.ipynb), then
preprocessing 02–03. Collection and preprocessing numbering are separate.

## Matched development top-up

Use **Run All** in 03 for FranceCrops cabbage training and validation, then in
02 for Sen4AgriNet potato training. Each notebook plans
one named batch and resumes it on reruns. The existing locked split rule selects
new candidates. Use **Run All** in 04 for EuroCrops carrot validation. It uses
the [Dutch 2020 release from Zenodo](https://zenodo.org/records/10118572), caches
the archive locally, and screens candidate footprints against saved development
and test parcels. Keep notebooks 05–07 out of this top-up.

If a run reports pending exports, rerun the same notebook until it reports
completion. Then run preprocessing 01–03 in order. Check the matched rows in
`usable_samples.csv` before running Training 01 and, if you need frozen models,
Training 02. A top-up can still leave shortfalls, especially for carrot validation.

A confirmed match needs the same locations to pass both window recipes. For
archive triage, start with fixed-window parcels that have a detected green-up
candidate. Cross-reference the saved development features with
`windows/greenup/candidate_greenup.csv`, then check the aligned-window quality
and complete pixel overlap. A detected candidate only identifies parcels to
review. Newly queried parcels become confirmed matches during preprocessing.

## Outputs and safety

Each notebook refreshes `pipeline_outputs/collection/` when it finishes. References,
imagery and source state are linked to their original locations. Notebook 01 saves
new audit folders under `collection/reports/reference_audit_*`; notebook 08 saves
content-keyed `summary_*` folders there. Source downloads, selections and jobs are
never moved by output refresh or summary. New summaries do not replace old ones.

Notebook 08 is local-only and supports incomplete batches. Its status is what was
last saved, not a live Earth Engine query. Registered tiles are not newly verified
imagery. Selection counts are not usable-feature counts. Overlap and imagery quality
remain checked in preprocessing 01; this report does not replace existing handoffs
or declare the data scientifically ready. Metadata issues are recorded explicitly.

The existing source collection controls still apply: `reuse` concerns references,
not satellite downloads. Use `offline` in 02, 03 and 05 for local preparation. In 04,
set `run_satellite_collection = False` to disable setup downloads and satellite calls.
These preparation modes may still create saved selections; use 08 for a metadata-only
review that never selects fields or changes job state.

Original notebooks, outputs and metadata are in the [collection archive](https://github.com/jrdavid-dev/external-crop-data-classification/tree/c1fe80c/notebooks/archive/collection_before_reorganization). Saved output in the active notebooks reflects earlier runs.

See [output layout](../../docs/storage_layout.md#organized-outputs) for link behavior and backup details.

Notebook 05 is set to `jecam` for primary test collection. Select
`carrot_potato_supplement` for the separate non-JECAM carrot and potato campaign. Set
`collection_cohort = "cabbage_supplement"` for separate non-JECAM cabbage after
development and primary JECAM collection complete. Preprocessing retains primary JECAM and adds completed supplemental batches.

Optional supplemental test collection:

- [06 California carrot](06_california_carrot_test_supplement.ipynb) collects a
  pilot from DWR 2023 single-crop carrot polygons, at most two batches of 50.
- [07 Denmark cabbage](07_denmark_cabbage_test_supplement.ipynb) downloads
  the 2023 Marker declarations once, counts distinct white and red cabbage
  polygons, and collects them in up to two batches of 50. Run All to resume
  pending imagery and report usable green-up parcels from completed batches.

These supplements check existing development and CropHarvest test footprints and use separate
test campaigns. Preprocessing reports their usable windows by campaign alongside primary JECAM.

For the complete notebook order and handoffs, use [the workflow guide](../../docs/workflow.md). To use shared TIFFs, [import the package](../../docs/sharing_imagery.md) and start preprocessing 01.
