# Repository cleanup audit

> Historical record. Instructions and test totals describe their original checkpoint; do not use old notebook sequences as the current run guide. See the [current workflow](../workflow.md) and [history index](README.md). Links have been adjusted for this documentation move.

Audit date: 2026-11-18. Baseline: `c1fe80c`, including the existing uncommitted
collection, preprocessing, supplemental-source, training and documentation work.
This is an assessment, not a deletion change. Existing files were not edited.

## Recommendation

Remove historical notebook snapshots and their isolated experiment modules in a
dedicated cleanup change, after migrating the relevant integration tests. Do not
delete every directory or Python file containing the word `archive`: annual
imagery archives and their readers are essential to the current workflow.

The active scope includes external collection, preprocessing, training, testing,
all six Paoay notebooks, both uncommitted supplement notebooks, and the three
command-line tools. Paoay being deferred scientifically does not make its code
unused, especially given the planned prototype map.

## Checks performed

- Inventoried 249 tracked/nonignored files at audit start, including 95 package
  Python files. Parsed Python and notebook code cells for imports; no parse
  failures. Followed transitive package imports from 21 active notebooks, three
  tools, and the documented `crop_pipeline.output_layout` CLI.
- Found 77 package files reachable from those entry points and 18 outside that
  graph. The latter include three package initializers and 15 implementation
  files. These are retirement candidates, not proof of universally unused code.
- Checked reverse imports, textual references, dynamic-import mechanisms,
  archive paths in tests, active notebook cells, and local Markdown links in
  README/docs/notebook guides. No broken links were found in those Markdown
  files before cleanup; archived-notebook links will need replacement.
- Ran the existing full suite: **515 tests passed**, in 52.826 seconds.
- Copied the tracked/nonignored working tree to a disposable directory, omitted
  both notebook archive trees, and ran the suite against that copy's package:
  **515 tests ran; 16 errors, no assertion failures**, in 58.296 seconds. All
  errors were missing archived notebook/inventory files; 499 tests passed.
- Subsequently removed all 18 Python candidates from that disposable copy.
  **All 77 remaining active-reachable package modules imported successfully.**
  This was an import smoke check, not an end-to-end workflow run or a full
  passing suite after Python deletion.
- Verified the 35 tracked archive files are byte-identical to `origin/main` at
  `c1fe80c`. Verified all 3,058 payload checksums in the ignored acquisition
  archive's cleanup manifest. Found no symlinks from `pipeline_outputs` or
  `sen4agrinet_work` into the root `archive` directory.

The full per-module classification and direct importers are in
[repository_python_inventory.csv](repository_python_inventory.csv). This is a
snapshot: regenerate the analysis if entry points or imports change.

## Archive inventory and disposition

| Location | Contents and size | Disposition |
| --- | --- | --- |
| `notebooks/archive/` | 22 tracked files, including 17 notebooks; 2,692,616 bytes | Retire after test migration and provenance/link updates. Fifty cells retain outputs; preserve recovery via the recorded Git commit. |
| `notebooks/paoay/archive/` | 10 tracked files, including eight notebooks; 81,107 bytes | Retire after moving notebook integration checks to the active six-notebook workflow. |
| `src/crop_pipeline/archive/` | Three tracked Python files; 36,505 bytes | Retire together with the dependent legacy experiment modules below. Deleting this package alone breaks their imports. |
| `archive/legacy_full_acquisition/` | 3,060 ignored files; 252,229,675 bytes, about 240.5 MiB | Separate local-data cleanup. Contains acquisition inventories, download-progress records, CSVs, README and manifest. Back up and verify externally before any removal; Git cannot restore it. |
| `.notebook_backups/` | 20 ignored files; 5,372,710 bytes | Local backups, not part of Git recovery. Review for unique work and back up separately before removal. |
| `sen4agrinet_work/annual_archive/`, `annual_tifs/`, campaign state and saved outputs | Live scientific data and resumability records | Preserve. Outside the historical-code cleanup scope. |

Tracked archive removal only removes about 2.68 MiB from the checked-out tree.
It does not purge old blobs from Git history or reclaim the live imagery storage.
No history rewrite is needed or recommended for this cleanup.

## Python retirement candidates

| Candidate | Current consumers | Required companion change |
| --- | --- | --- |
| `archive/external_combined.py` and `archive/transfer_demo.py` | Overfitting/transferability experiments and legacy tests | Retire the dependent experiment group together, or relocate the helpers if those experiments must remain supported. |
| `training/overfitting.py` | `tests/test_overfitting.py`; historical experiment context | Remove only when retiring the old overfitting experiment, with its dedicated tests. |
| `transferability/` (10 Python files) | Archived `12_external_training.ipynb`, internal imports and `tests/test_transferability.py` | Retire the whole legacy package if that notebook is retired. This is distinct from current external evaluation in `training/source_holdout.py`. |
| `imagery/observation_quality.py` | `archive/transfer_demo.py` and `tests/test_observation_quality.py` | Candidate after retiring transfer demo. It is not the current `imagery/quality.py`; preserve the latter. |
| `visualization/pipeline.py`, `visualization/preprocessing.py`, initializer | Archived notebooks and visualization tests; preprocessing previews are also tested in live TIFF/table tests | Optional second cleanup. Keep for now, or deliberately retire the preview API and adapt the affected tests without losing TIFF/table transport coverage. |

Paths in this table are relative to `src/crop_pipeline/`. The 18-file total also
includes the `archive` and `transferability` initializers. No implementation file
was found to have literally no consumers across active, archived and test code.
The useful distinction is active workflow versus legacy/test-only use.

Dedicated tests associated with retirement are `test_external_combined.py`,
`test_overfitting.py`, `test_transferability.py`, and, if that helper is removed,
`test_observation_quality.py`. `tests/transfer_fixtures.py` supports the old
overfitting tests. Optional visualization retirement additionally affects
`test_pipeline_visuals.py`, `test_preprocessing_visuals.py`, and preview-specific
assertions in `test_annual_tif.py` and `test_pixel_tables.py`.

## Files that look old but must stay

- `analysis_workflow.py`: imported by active Paoay entry points for shared
  workflow functions. It also reaches shared training helpers.
- `training/exploratory.py`: current `training/source_holdout.py` uses its feature
  matrix, input checks and crop/parcel weighting.
- `training/earth_engine.py`: still imported by `training/notebook_workflow.py`.
  Its name does not establish that the file is unused by local training.
- `windows/annual_archive.py`, `windows/workflow.py`, `windows/seasons.py`, and
  `sources/annual.py`: active annual-data/window/collection dependencies.
- `paoay/maps.py` and the other Paoay modules: active prototype dependencies,
  including the newly pulled whole-boundary/application work.
- `sources/supplemental.py`: untracked but imported by the new collection 07/08
  notebooks. Preserve it and its regression tests with the existing local work.
- All three `tools/*.py` scripts: deliberate CLI entry points, even though they
  need not be imported by another Python file.

Retiring old functions inside a retained module would require a separate
function-level audit. Do not delete these entire modules based on notebook age.

## Test migrations required before notebook deletion

The disposable-copy run identifies these 16 errors across 12 test files:

| Test file | Errors | Action |
| --- | ---: | --- |
| `test_analysis_notebooks.py` | 2 | Replace legacy 04→05→06 notebook execution with active preprocessing/training handoff coverage, or retire specifically legacy behavior after confirming current coverage. |
| `test_collection_summary.py` | 1 | Replace archive-byte inventory assertion with active notebook/path checks; retain offline summary and settings-redaction tests. |
| `test_feature_experiments.py` | 1 | Move the pending-input/no-fit assertion off archived training notebook 06; keep feature and partition checks. |
| `test_paoay_batch1.py` | 1 | Retire archive-inventory checksum assertion; retain active reference/geometry/notebook checks. |
| `test_paoay_imagery.py` | 1 | Move cached-imagery and pending-stage notebook checks to active `02_satellite_inputs.ipynb`. |
| `test_paoay_dynamic_world.py` | 1 | Move standalone Dynamic World notebook checks into active satellite-input coverage. |
| `test_paoay_features.py` | 1 | Move feature-stage notebook checks to active `03_model_application.ipynb`. |
| `test_paoay_predictions.py` | 1 | Move no-refit/prediction checks to active `03_model_application.ipynb`. |
| `test_paoay_land_cover.py` | 1 | Exercise context/guidance through the active orchestration or direct module API; keep underlying cover tests. |
| `test_paoay_final_workflow.py` | 1 | Replace archived 13/14 execution with active map/review 04/05; retain pending-input and no-network/no-fitting assertions. |
| `test_preprocessing_workflow.py` | 1 | Retire archived-byte inventory assertion; retain active preprocessing handoff tests. |
| `test_season_workflow.py` | 4 | Retire or replace `LegacyNotebookIntegrationTests`; keep shared season/window/partition unit tests. |

Do not remove whole mixed-purpose test files merely to get a passing suite.
Preserve the behavioral protections and make notebook integration coverage match
the actual supported entry points.

## Provenance and documentation changes

1. Replace archive links in README, collection/preprocessing guides and
   `docs/paoay_workflow.md` with a historical commit reference or a brief retired
   workflow note. Update `docs/feature_experiments.md` and the historical sections
   of `docs/training.md` to stop advertising removed import paths.
2. `windows/workflow.py:213` includes the archived windows notebook in a packaged
   provenance input list. It also names already-moved source-preparation
   notebooks. Update new-run provenance to current paths. The packager records
   missing files rather than always failing, so passing tests alone would not
   establish complete provenance.
3. `paoay/land_activity.py` names archived notebook 14 in its docstring and
   `method_source` receipt. Preserve its historical attribution with a pinned
   Git reference when the local notebook disappears. This is attribution, not
   a runtime read dependency.
4. Update notebook 06's archived-method link and historical rule caption. Review
   stale Paoay numbering: active notebook 05 still says to finish 10, and active
   notebook 02 points to notebook 09. Use current notebook names/numbers.
5. Do not edit old result manifests to pretend they were generated by new code.
   Preserve existing receipts, model packages and outputs. Changed code hashes
   may require new bundles for new runs; retain compatible old run inputs.

## Safe execution order

1. Preserve the existing 32 modified/untracked files with a reviewed local
   checkpoint or verified backup. Do not reset them. Record `c1fe80c` as the
   historical archive recovery commit; no push or remote mutation is required.
2. Migrate active integration tests and update new-run provenance/doc references.
3. Remove the two tracked notebook archive trees and the explicitly retired
   legacy experiment group. Keep visualization helpers for a separate decision
   unless their preview API is intentionally retired too.
4. Run focused tests followed by the full suite. Check active notebook code
   syntax, importability, links and remaining references to removed paths.
   Validate with local fixtures; no collection jobs or real training runs are
   needed just for cleanup.
5. Check `git diff --check`, inspect the complete diff, and confirm saved data,
   campaign state, locked partitions, local settings and existing local edits
   are untouched.
6. Only then consider the ignored acquisition archive and notebook backups as a
   separate local-data operation, after verifying a recoverable external copy.

## Limits of this audit

Static reachability conservatively includes imports inside functions; it does
not prove every function in a retained module executes. Conversely, untracked
scripts outside the inspected entry points, other machines, external users and
serialized historical artifacts may still rely on retired APIs. No large data
scan, saved-model deserialization, cloud collection, or notebook training run
was performed. The ignored acquisition archive's checksums establish internal
integrity, not that a separate backup already exists. Passing software tests
does not establish scientific accuracy.

## Cleanup execution — 2026-11-19

Retired both historical notebook trees and the isolated archive/transferability,
overfitting and observation-quality experiment group, with dedicated tests.
Kept visualization helpers and their TIFF/table coverage. Ignored acquisition
archives, notebook backups, live TIFFs and campaign state were not removed.
Historical recovery is Git commit `c1fe80c`; the pre-cleanup tracked/nonignored
working files were also backed up locally before editing.

Removed inventory checksum assertions and integrations for retired notebook
interfaces. Active preprocessing integration remains in
`test_preprocessing_workflow.py`; combined satellite/features/prediction coverage
and pending-input guards remain in `test_paoay_batch2.py`; field review coverage
remains in `test_paoay_batch3.py`. Map/review integration now uses active 04/05.
Shared season/window/partition and feature handoff unit tests remain. Added a
compile check across all active notebook code cells. The inventory CSV above
remains the historical pre-cleanup snapshot.

Verification after cleanup: **475 tests passed** in 50.247 seconds. The active
notebook compilation check also passed after the final wording edits.
`git diff --check` passed and README/docs/notebook-guide local links resolve.
These are local software checks; no cloud collection or scientific evaluation
was launched. Existing Shapely deprecation warnings remain.
