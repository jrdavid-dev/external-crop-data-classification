# Data collection — deliverables 1 and 2

| Notebook | Responsibility |
| --- | --- |
| 01 External reference audit | Inspect local reference availability and saved selection coverage; record source roles |
| 02 Sen4AgriNet collection | Development references and imagery, mostly potato with some carrot |
| 03 FranceCrops collection | Potato/cabbage development references and imagery |
| 04 EuroCrops collection | Additional Dutch carrot development references and imagery |
| 05 CropHarvest test collection | Eligible JECAM references and imagery for external testing of all three crops |
| 06 Dataset summary | Read saved batches, report selected groups and status, save a versioned local manifest |

Run 01 when reviewing references, then only the source notebooks needed for top-ups.
Run 06 after collection or whenever a local progress snapshot is useful. Continue
with [preprocessing 01](../preprocessing/01_imagery_quality.ipynb), then
06–07. Collection and preprocessing numbering are separate.

The source notebooks retain the exact pre-reorganization controls: new Sen4AgriNet
batches currently request potato, FranceCrops and CropHarvest request cabbage, and
EuroCrops requests carrot. These are deliberate top-ups; earlier batches may contain
other crops. Review the existing `extra_batches` and collection flags before Run All.
No operational settings, buffers, quality rules or campaign IDs were reset.

## Outputs and safety

Each notebook refreshes `pipeline_outputs/collection/` when it finishes. References,
imagery and source state are linked to their original locations. Notebook 01 saves
new audit folders under `collection/reports/reference_audit_*`; notebook 06 saves
content-keyed `summary_*` folders there. Source downloads, selections and jobs are
never moved by output refresh or summary. New summaries do not replace old ones.

Notebook 06 is local-only and supports incomplete batches. Its status is what was
last saved, not a live Earth Engine query. Registered tiles are not newly verified
imagery. Selection counts are not usable-feature counts. Overlap and imagery quality
remain checked in preprocessing 05; this report does not replace existing handoffs
or declare the data scientifically ready. Metadata issues are recorded explicitly.

The existing source collection controls still apply: `reuse` concerns references,
not satellite downloads. Use `offline` in 02, 03 and 05 for local preparation. In 04,
set `run_satellite_collection = False` to disable setup downloads and satellite calls.
These preparation modes may still create saved selections; use 06 for a metadata-only
review that never selects fields or changes job state.

Original notebooks, outputs and metadata are in the [collection archive](../archive/collection_before_reorganization). Saved output in the active notebooks reflects earlier runs.

See [output layout](../../docs/output_layout.md) for link behavior and backup details.

Notebook 05 currently defaults to the separate non-JECAM cabbage supplement.
Finish pending development and primary JECAM batches before running it. Set
`collection_cohort = "jecam"` to resume primary JECAM collection. The standard
preprocessing workflow reads only the primary JECAM test campaign.
