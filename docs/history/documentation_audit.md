# Documentation audit — 19 November 2026

> Historical record. Instructions and test totals describe their original checkpoint; do not use old notebook sequences as the current run guide. See the [current workflow](../workflow.md) and [history index](README.md). Links have been adjusted for this documentation move.

Assessment of the current working tree, including existing local changes. Scope:
all **24 existing files in `docs/`** (23 Markdown files and one CSV), the root
README, notebook folder guides, active notebook Markdown/configuration, and the
package/tool references relevant to their instructions. No documentation or data
was deleted. This audit is a maintenance record, not another workflow to follow.

## Recommendation

Keep a short reading path: **project design → workflow → methodology**. Use setup,
research, sharing and storage guides when needed. The remaining documents serve
source-specific, presentation, specialist, deferred-application or historical
purposes. The problem is mixed generations of instructions and duplication,
not that every reference file is unnecessary.

Seven guides are core, seven are supporting references, two cover deferred Paoay
work, two are specialist transport/catalogue references, five are historical
records, and one is the index. Several useful files need corrections before they
can be treated as authoritative current instructions.

## Findings to fix first

| Priority | Finding and evidence | Recommended correction |
| --- | --- | --- |
| High | `cropharvest.md` ends by telling receivers to set `jecam_import_folder` in Preprocessing 01. That notebook has no such setting or import call. | Replace that paragraph with the actual `crop_pipeline.share import` instructions in `sharing_imagery.md`. Retain the optional sender export helper as a separate route. |
| High | Root README advertises a default pilot/strict analysis switch. The active preprocessing and training entry points expose no `analysis_mode`; they report shortfalls and require qualifying class coverage. `training.md` mixes those legacy requirements with current instructions. | Describe active execution checks accurately; move legacy pilot/strict rules behind a historical label. Do not introduce or change a scientific gate to match the prose. |
| High | `full_year_collection.md` and `pixel_tables.md` refer to source notebooks 01–03, FranceCrops notebook 02, windows 04, and 5,000-tile controls. These are not the current collection/preprocessing entry points. | Preserve technical/transport evidence; remove obsolete notebook run instructions from normal navigation. Any deliberate full-catalogue procedure needs explicit current API/CLI entry points before being advertised as runnable. |
| High | `methodology.md` lists only Sen4AgriNet and FranceCrops, says “all three sources,” and describes a universal 50-parcel batch. Current sources also include EuroCrops, JECAM and registered supplements; FranceCrops and EuroCrops notebooks currently specify 100. | Complete the source/role register and distinguish per-source notebook workload settings from fixed scientific rules. Keep the 64-base/72-classifier distinction explicit. |
| Medium | `paoay_workflow.md` has a correct 01–06 table but headings/references still say 07–12. `paoay_field_checks.md` says run 07→11 and rerun 11. The root README still calls it five application notebooks in one place. | Use 01–05 plus optional 06 consistently; keep historical numbering only in explicitly historical attribution. Paoay is deferred, not disposable. |
| Medium | `progress.md` maps windows/features/training to old 04/05/06 notebooks and implies one progress scheme across all notebooks. | Keep progress semantics, log paths and pending/retry interpretation; replace the old table with current instrumented stages or merge those details into workflow troubleshooting. |
| Medium | `adaptive_training.md` describes every annual TIFF as 168 bands. The current exporter can add three observation diagnostics per interval (240 bands); root README already distinguishes the formats. | Describe both schemas and point to actual band descriptions/receipts. Keep B2/clear counts distinct from the five classifier bands and four indices. |
| Medium | `adaptive_training.md` calls the current Collection 05 top-up the carrot/potato supplement, while its saved notebook setting is `jecam`. `cropharvest.md` hardcodes six slots while the notebook now has `extra_batches = 12`. | Avoid undated operational “current” claims; use notebook configuration and saved run receipts for campaign limits and status. These observations do not authorize changing the settings. |
| Medium | `output_layout.md` presents `pipeline_outputs/` broadly as symlink navigation and discusses physical migration as future work. Current training/testing outputs are native folders there, and physical migration is completed in `storage_layout.md`. | Merge the output map into storage documentation, separating native outputs from navigation links. Keep live-link deletion warnings and the actual refresh behavior. |
| Medium | Workflow, storage and cleanup documents retain test totals from different milestones (482, 485, 490, etc.). `repository_cleanup_audit.md` starts with a recommendation that has since been executed. | Label dated execution evidence as history and give one current verification summary. Do not overwrite historical results with a newer test count. |
| Medium | `repository_python_inventory.csv` is a pre-cleanup snapshot: 95 module rows, 15 paths now removed. | Keep it with the historical cleanup audit. Do not use it to decide whether current modules are unused. |
| Low | Setup, budgets, run order, model settings and preservation rules are repeated across the root README and several guides. The docs index had no inbound local Markdown link from the reviewed entry points. | Point the root README to the index; keep one authoritative home per topic and link to it instead of duplicating procedures. |

## File-by-file disposition

“Merge” means preserve useful content and update incoming links before removing
its old file. “History” means retain evidence outside the normal reading path;
it is not an instruction to delete it now.

| Existing file | Role | Importance and disposition |
| --- | --- | --- |
| [README.md](../README.md) | Index | Keep as the entry page; organize by what the reader needs. |
| [project_design.md](../project_design.md) | Core | Keep. Experiment scope, data roles, comparisons and invariants. |
| [workflow.md](../workflow.md) | Core | Keep as the canonical notebook order/handoff/retry guide. Can absorb progress semantics and training handoff details. |
| [fresh_clone.md](../fresh_clone.md) | Core | Keep one concise environment/setup guide with collect/import routes. Reduce repeated storage/workflow text. |
| [methodology.md](../methodology.md) | Core | Keep and repair source/budget/schema descriptions. Scientific rules belong here. |
| [research_basis.md](../research_basis.md) | Core | Keep. Claim-to-source mapping and limits; linked from 27 reviewed local Markdown locations and copied into imagery packages. |
| [sharing_imagery.md](../sharing_imagery.md) | Core | Keep. Actual groupmate export/import procedure; copied into packages by `share.py`. |
| [storage_layout.md](../storage_layout.md) | Core | Keep. Canonical storage and recovery explanation; absorb output layout, move detailed migration milestones to history if consolidating. |
| [adaptive_training.md](../collection.md) | Reference | Keep its quotas, locked partitions and resume rules; combine source-control explanations with Sen4AgriNet/CropHarvest into one collection reference if reducing files. |
| [sen4agrinet.md](../collection.md#sen4agrinet-references) | Reference | Useful for selective reference scans, pool expansion, source-specific edge rules and settings. Merge into collection reference, preserving these details. |
| [cropharvest.md](../collection.md#cropharvest-labels-and-cohorts) | Reference | Useful label/date/cohort rules. Fix receiver instructions and stale slot count, then merge into collection reference if desired. |
| [external_training_notebooks.md](../workflow.md#training-and-external-testing) | Reference | Useful current model settings, selection and baseline rules. Merge handoffs into workflow and scientific settings into methodology; avoid a third competing run guide. |
| [notebook_demo.md](../notebook_demo.md) | Reference | Keep. Short presentation checklist for the current demo goal. |
| [output_layout.md](../storage_layout.md#organized-outputs) | Reference | Merge into storage guide after correcting native-output versus link behavior. Code currently links this file as `pipeline_outputs/README.md`, so update `output_layout.py` before removing it. |
| [progress.md](../workflow.md#progress-and-troubleshooting) | Reference | Preserve progress/ETA/log/retry semantics; merge into workflow troubleshooting after fixing old numbering. |
| [paoay_workflow.md](../paoay_workflow.md) | Deferred | Keep for the broader project and active application code. Fix numbering; do not treat deferred science as unused implementation. |
| [paoay_field_checks.md](../paoay_workflow.md#field-observations) | Deferred | Keep the evidence/observation schema and bias cautions. Can become the field-review section of one Paoay guide. |
| [pixel_tables.md](pixel_tables.md) | Specialist | Keep the transport schema and reusable-cache details. Separate those from retired notebook controls and the dated benchmark procedure. |
| [full_year_collection.md](full_year_collection.md) | Specialist | Retire its old notebook instructions from the normal path. Preserve still-supported API details in a maintainer transport appendix; full-catalogue operation is not the default workflow. |
| [feature_experiments.md](feature_experiments.md) | History | Preserve the old 48/72/120-predictor experiment design; it is not the active 72-feature run procedure. Formula details still explain supported historical model recipes. |
| [training.md](training.md) | History | Current paragraphs duplicate the dedicated guide; keep the remaining pooled-pilot/production history separately after extracting unique useful explanations. |
| [live_batch_validation.md](live_batch_validation.md) | History | Keep the dated small live transport comparison and mask-bug correction evidence. It does not certify dataset-wide throughput or accuracy. |
| [repository_cleanup_audit.md](repository_cleanup_audit.md) | History | Keep the cleanup rationale, recovery commit and executed verification as a maintenance record. Its opening recommendation is historical. |
| [repository_python_inventory.csv](repository_python_inventory.csv) | History | Keep with the cleanup audit, explicitly dated; 15 listed module paths have been retired. |

## A smaller documentation structure

A reasonable consolidation target is roughly ten reader-facing guides, plus
specialist/history material behind separate headings:

- Design, workflow, setup, methodology, research basis, sharing, storage.
- One collection reference instead of three source/collection guides.
- Demo notes.
- One optional Paoay guide containing field checks.

Merge `external_training_notebooks.md` into workflow/methodology, `progress.md`
into workflow, and `output_layout.md` into storage. Keep a specialist transport
reference and dated history accessible without making them onboarding steps.
This target is a proposal, not a completed merge or a requirement to create more
files. No existing file was moved or deleted for this audit.

## Checks and limits

Read the documents, compared active notebook settings and entry points, inspected
local incoming links, checked package/tool documentation dependencies and
verified local Markdown targets. No broken local path links were found. Checked
that the CSV snapshot's missing paths are the retired module set. Low inbound
link count alone does not establish that a document is unimportant.

The audit does not revalidate published research, remote URL availability, data
licenses, current scientific results or every legacy API. External URLs were not
fetched. No notebooks, cloud collection, data migration or training runs were
executed. No software tests were rerun for this documentation-only assessment.

The only presentation changes made with this audit are the docs index grouping
and a root README link to it. Content corrections and consolidation above remain
explicit follow-up work. This audit itself is a maintenance reference, not an
additional document the groupmates must read.

## Consolidation execution — 19 November 2026

The consolidation was completed after this assessment. The main index now points
to ten reader-facing guides. Collection/source instructions were combined;
training/progress details were merged into workflow/methodology; output navigation
was merged into storage; field checks were merged into the optional Paoay guide.
Historical procedures and dated evidence were moved here with explicit banners.
A separate transport reference retains technical format and maintenance details.
Incoming local links, notebook Markdown and the code-owned output README target
were updated. Notebook code, outputs and metadata were preserved. See the
[current documentation index](../README.md) for supported instructions.

Consolidation checks: **491 tests passed**; active notebook code compiles; local
Markdown paths and section anchors resolve. Notebook code, outputs and metadata
were compared with the pre-consolidation backup and remained unchanged. The
output README was refreshed to the storage guide. No imagery collection, data
migration or full-data model run was performed for this documentation change.
