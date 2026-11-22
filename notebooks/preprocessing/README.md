# External preprocessing — batch 2

| Notebook | Responsibility | Main output |
| --- | --- | --- |
| 01_imagery_quality.ipynb | Verify completed development, primary JECAM and available supplements, exclude test overlaps, audit original imagery quality | Quality counts, collection snapshot, quality.json |
| 02_window_selection.ipynb | Apply fixed May–August and existing green-up-aligned windows | Complete-window counts before pixel cap, windows.json |
| 03_feature_engineering.ipynb | Select and cap matched pixels, lock the development split by parcel ID, and export the 72 interval predictors | development_split.json, separate development/test packages, usable_samples.csv, readable report |

Run 01 → 02 → 03 with a fresh kernel per notebook if desired. Notebooks 02 and 03
use verified stage pointers from the preceding notebook; no stale hard-coded run
folder or automatic newest-folder selection is used. Optional explicit paths must
refer to successful runs of these new notebooks.

## Running while collection is active

Developing this code does not affect collection. Running a new live audit still
requires pending launched batches to finish, exactly as before. Nothing here
submits jobs, downloads data, imports teammate batches, or changes
campaign state. Prepare/import collection inputs before starting 01.

The audit records campaign metadata hashes. If batches, assignments or saved job
states change during an audit or between stages, downstream preparation stops.
Finish collection and start a new 01 run. The earlier attempt remains saved.

A failed stage marks its pointer pending so it cannot silently fall back to an old
result. Successful unchanged stages are verified and reused; partial or edited
outputs are preserved and require a new audit. Every new audit uses its own run
folder. Quality masks, the 0.80 coverage requirement, eight-bin completeness,
timing rules, and the 20-pixel cap apply to both development and test.

For the post-matching split, start a fresh run at Preprocessing 01 using the
existing completed imagery. Then run Preprocessing 02–03, Training 01–02,
and Testing 01–02. Collection can be skipped when the imagery is already local.

## Outputs and interpretation

Preprocessing 03 indexes pixel identities and file offsets from the verified
`windows.json` handoff, then reads selected rows and writes feature packages
incrementally. A completed Preprocessing 02 handoff can be reused with this export.

The canonical artifacts remain under the configured report root's sibling
`analysis/external/external_<id>/`. Organized views are refreshed under
`pipeline_outputs/preprocessing/<run>/{quality,windows,features}/`.

- Quality counts describe input parcel groups and tile/interval outcomes, not usable training rows.
- Window counts describe complete pixel windows before capping.
- Feature export excludes ground coordinates shared by different development parcel groups within each window recipe. It removes the shared pixels from both records before sampling and matching. See `development_overlap_exclusions.json` for the affected groups and counts.
- `usable_samples.csv` reports final usable pixels and parcel groups for **each crop and partition**, including zeros, targets and shortfalls.
- Preprocessing 03 saves one assignment per usable development parcel group in `development_split.json`. Assignments are deterministic and stratified by crop and recipe availability, with the matched cohort prioritized. The split aims for 75% training and 25% validation within strata, with at least one parcel on each side when a stratum has two or more parcels. The same assignments apply across all development exports.
- Fixed/aligned **full** cohorts have separate coverage; **matched** cohorts use the same locations. Never add cohort counts together.
- Budgets are collection targets, not scientific accuracy criteria. Green-up estimates are not verified crop dates. Legacy cloud-loss diagnostics remain unavailable where counts were not collected.

The test feature export carries cohort labels. `usable_samples.csv` reports development and primary JECAM. `test_population_coverage.csv` also reports each registered supplement and the combined pool. Only completed supplemental batches contribute imagery. Earlier cohorts take precedence over overlapping supplements, with JECAM first among test cohorts.

Continue with [training](../training) and [testing](../testing) after feature export.

Original notebooks and saved outputs are available in the [preprocessing archive](https://github.com/jrdavid-dev/external-crop-data-classification/tree/c1fe80c/notebooks/archive/preprocessing_before_reorganization) and [collection archive](https://github.com/jrdavid-dev/external-crop-data-classification/tree/c1fe80c/notebooks/archive/collection_before_reorganization).

To check only the new carrot and cabbage top-ups, open
[05_matched_coverage.ipynb](05_matched_coverage.ipynb). It processes the locally
completed top-up batches and checks them against the saved development windows.
The report shows new matched candidates before train/validation assignment.
