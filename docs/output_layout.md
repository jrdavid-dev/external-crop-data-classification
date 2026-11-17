# Organized pipeline outputs

Collection notebooks in `notebooks/data_collection/` now refresh this layout on completion. Notebook 06 creates versioned dataset summaries under `collection/reports/summary_<id>/`; new reference audits from notebook 01 are saved under `collection/reports/reference_audit_*`. Downloads and jobs continue using their original paths.

`pipeline_outputs/` is a shallow navigation layout for the existing data. It uses
relative filesystem symbolic links: there is one authoritative copy of each file,
and the original locations remain intact. Nothing is downloaded, moved, relabelled,
or retrained. Saved manifests, checksums, partitions, and collection jobs retain
their original paths and bytes.

```text
pipeline_outputs/
  collection/
    sen4agrinet/{references,imagery,state}/
    francecrops/{references,imagery,state}/
    eurocrops/{references,imagery,state}/
    cropharvest/{references,imagery,state}/
    reports/
  preprocessing/<run>/{quality,windows,features}/
  training/<run>/{validation,models}/
  testing/<run>/{predictions,reports}/
```

Collection references link to downloaded sources and existing preparation runs.
Imagery links expose annual TIFFs, retaining their existing year/campaign structure.
State links expose each source batch and its collection archive. Shared campaign
policies, locked partitions and feedback remain together under
`collection/reports/campaigns`; they are never copied independently per source.

Preprocessing groups the current external experiments by quality, windows and
features. Model packages appear under training, with their embedded validation
records accessible under validation. External test predictions and metrics appear
under testing when they exist. Older pilot models are labelled `legacy_...`.
An empty testing folder means no current external test outputs were found; the
layout does not manufacture results. Historical experiments and Paoay stay in place.

## Create or refresh

Run from the repository root using the project environment:

```bash
.venv/bin/python -m crop_pipeline.output_layout
```

Refresh after creating new batches, preprocessing runs, models or test results.
Files added inside an already linked directory are visible immediately. Newly
created runs and individually linked feature files need a refresh. Notebook output
paths remain valid and continue to be the authoritative handoffs. This change does
not alter where notebook code writes; it organizes access to those outputs.

For a full before/after SHA-256 check of all files referenced by the planned layout:

```bash
.venv/bin/python -m crop_pipeline.output_layout --verify
```

Run verification with collection writers stopped; otherwise changing source files
will correctly fail the integrity check. The layout manifest lists link targets,
and dated integrity receipts record file hashes. The entire output folder is ignored
by Git. Existing files at conflicting destinations cause an error; nothing is
overwritten. Rerunning after an interruption reuses correct links.

These are live links, not backups: editing/deleting files **inside** a linked
directory affects the originals. Keep source directories in place. To transfer or
back up the dataset, include the original data and settings; copying only the links
does not copy their contents. A future physical migration requires a separate path
and manifest migration; this layout deliberately avoids that risk.

## Preprocessing batch 2

Preprocessing 01–03 refresh their views after successful stages. Verified
`current_quality.json`, `current_windows.json`, and `current_features.json` pointers
live directly under `pipeline_outputs/preprocessing/`. A failed stage leaves its
pointer pending. The features view includes `usable_samples.csv` and
`preprocessing_report.md`, reporting final counts and budget shortfalls separately
for training, validation and external testing. No live analysis is run during a
code reorganization; collection can continue independently.
