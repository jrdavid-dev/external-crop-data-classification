# Storage and organized outputs

All sources share **`data/pipeline_work/`**. Upstream reference downloads remain
under `data/external/`. Local settings, raw data and generated artifacts stay out
of Git. Collection notebooks resolve relative archive paths against the repository.

| Path under `data/pipeline_work/` | Contents |
| --- | --- |
| `annual_archive/` | Reference snapshots, masks, transport caches and acquisition metadata |
| `annual_tifs/` | Annual TIFFs, named bands and checksum indexes |
| `adaptive_training/` | Campaign policies, locked selections, plans, states and completion receipts |
| `inventory/`, `progress/` | Saved collection reports and progress logs |
| `analysis/` | Quality/window/feature snapshots and older experiments |
| `paoay/` | Local application bundles where present |

`report_root` locates the inventory and its sibling campaign/analysis directories.
`annual_archive_root` can select an external drive; `annual_tifs` is its sibling.
Absolute custom paths remain supported. Keep the drive connected while using its data.

## Organized outputs

`pipeline_outputs/` combines **navigation links** to collection/preprocessing
files with **native training/testing run folders**. It is not all symlinks.

```text
pipeline_outputs/
  collection/<source>/{references,imagery,state}/
  collection/reports/
  preprocessing/<run>/{quality,windows,features}/
  preprocessing/current_<stage>.json
  training/run_<id>/{validation,models}/
  testing/run_<id>/{predictions,reports}/
```

Collection shortcuts cover Sen4AgriNet, FranceCrops, EuroCrops and CropHarvest.
Supplemental data remains accessible in the canonical archive/TIFF folders and
shared campaign view. Collection links expose references, annual imagery and source state.
Policies, feedback and locked partitions stay together under
`collection/reports/campaigns`. Collection 01/08 save reference audits and
versioned metadata summaries; these do not replace scientific stage handoffs.

Preprocessing views point to authoritative analysis files. Native training/testing
runs retain copied inputs, fitted packages, predictions and receipts. Older model
views can be labelled `legacy_...`; keep separate experiments separate. Empty
folders do not establish that an evaluation ran. Read the current verified stage
pointer or select an exact saved handoff, as described in [workflow](workflow.md).

Refresh after new batches/runs or individually linked output files appear:

```bash
.venv/bin/python -m crop_pipeline.output_layout
```

Files added within an already linked directory are immediately visible. Refresh
creates navigation, not collection or analysis. Existing user-file collisions
stop refresh. To hash linked source files before/after refresh, stop collection
writers and run the same command with `--verify`. Dated integrity receipts and
the layout manifest record its checks; native output folders remain preserved.

## Preservation and recovery

A navigation link is not a backup: deleting/editing files inside a linked folder
affects the originals. Use [the portable export](sharing_imagery.md) to share
imagery; copying links alone does not copy their contents. Back up authoritative
data and your private settings separately, without distributing credentials.

Both former `sen4agrinet_work` locations have been removed. Saved runtime paths
and dependent metadata checksums now use `data/pipeline_work/` directly. Original
metadata and before/after checksums are retained locally under
`data/storage_migrations/`. Do not rewrite receipts
to pretend a historical run used the current implementation.

Relocated pixel caches retain their original filenames. Exact-contract matching
can reuse them when a path change alters the computed name; normal checksum and
quality checks still apply. Changed settings or corrupt data are rejected.

If an old preprocessing receipt rejects its recorded implementation, preserve it
and start a new Preprocessing 01–03 run. Complete older feature snapshots can still
be used explicitly where their model/test contracts match. Storage maintenance
does not create new scientific results.
