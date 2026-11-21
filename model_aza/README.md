# model_aza — Aza's experiments and improved pipeline

Kept apart from the main pipeline (`src/crop_pipeline/`, `notebooks/`, `tests/`) so the two can
change without clashing. Nothing here edits the main pipeline's code, notebooks, stage pointers or
frozen models; it only reads them (for example the verified Preprocessing 01 handoff and
`crop_pipeline` helpers).

| Folder | Contents |
|---|---|
| `improved/` | Improved pipeline code: `workflow.py` (inputs → selection → models → test → Paoay / Atok maps), `field_plan.py`, `mixing.py` |
| `pilots/` | Harmonic features, Sentinel-1 / ERA5 / WorldCover requests, evaluation protocol (used by `improved/`) |
| `training/` | Tuning, transfer and start-month experiment code |
| `walkthrough/` | Loaders, summaries and figures for the walkthrough notebook (reads current outputs only) |
| `notebooks/walkthrough/` | `00_pipeline_walkthrough`: glossary, references, data, preprocessing, features, baseline vs improved, maps, limits |
| `notebooks/improved/` | 01 inputs · 02 model selection · 03 freeze and test · 04 Paoay map · 05 field-visit plan · 06 Atok boundary map · 07 dataset variants |
| `notebooks/mixed_training/` | Part of the test places moved into training; "new place" check |
| `notebooks/transfer/` | 06 start-month sweep · 13 Paoay start-month maps |
| `notebooks/tuning/` | 01 grid · 02 spatial CV · 03 seed stability |
| `notebooks/archive/` | Superseded experiments with their findings |
| `tests/` | Tests for everything above |
| `patches/` | My earlier edits to main-pipeline files, saved before they were reverted (see below) |

Outputs still go to `pipeline_outputs/` (`improved/`, `mixed_training/`, `transfer/`, `tuning/`,
`pilots/`, `reports/`), which Git ignores. Slides: `pipeline_outputs/reports/slides_20261121/slides.md`.

## Running

Notebooks find the repository root and add both `src/` (main pipeline) and the root (this package) to
`sys.path`. Tests:

```bash
.venv/bin/python -m unittest discover -s model_aza/tests -t . -v
```

Paired comparison of the improved model with the original frozen models, on the parcels each
original model scored (same parcel bootstrap as `crop_pipeline.training.external_report`; add
`--revised-scope` for Jose's revised reporting scope):

```bash
.venv/bin/python -m model_aza.improved.paired_comparison \
    pipeline_outputs/improved/test/run_<id>/test_receipt.json \
    pipeline_outputs/testing/run_<id> pipeline_outputs/reports/<new_folder>
```

## Main-pipeline changes

Only the **streaming fix** stays in the main pipeline (`src/crop_pipeline/common.py`,
`preprocessing_workflow.py`, `windows/external.py`, with `tests/test_streaming_windows.py`). Every
preprocessing receipt hashes `preprocessing_workflow.py` and `windows/external.py`, so reverting them
would make the current preprocessing runs fail verification, and May–August preprocessing runs out of
memory on a 16 GB laptop without it.

Reverted on 21 November 2026 and saved in `patches/`:

| Patch | What it was |
|---|---|
| `removed_docs_and_caption.patch` | README experiment table and Paoay note, Paoay workflow section, dataset-register re-check line, Preprocessing 01 caption |
| `removed_test_cohort_supplement_matching.patch` | Carrot/potato supplement overlap check matching saved selections by source + parcel ID, with its regression test. Without it, rerunning that supplement's collection fails again with "Current references lack saved test selections" |
| `kept_streaming_fix.patch`, `kept_streaming_fix_test.py` | Copy of the streaming fix that is still applied |
| `all_main_pipeline_changes_20261121.patch` | Everything above in one file |

Re-apply a patch from the repository root with `git apply model_aza/patches/<name>.patch`.
