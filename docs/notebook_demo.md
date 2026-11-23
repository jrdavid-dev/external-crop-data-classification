# Present the active notebooks

## Offline final demo

Open [SakAI Final Demo](../notebooks/SakAI_final_demo.ipynb) for the saved
presentation. The seven linked sections run from the crop-reference problem
in Paoay through prototype construction, training and validation, external
testing, regional adaptation, the preliminary map, and next steps. The
detailed methods, tables, and figures appear in the relevant sections. The
additional model comparisons follow the takeaway in a technical appendix.
All cells are Markdown with saved tables and embedded figures.
The offline handoff also includes a browser-ready HTML export, the standalone
Paoay study-area map, cached tables and predictions, and the frozen model files.
Open the HTML from the extracted folder to keep the relative links working.
The linked workflow notebooks contain the executable steps.

| Evidence | Implementation |
| --- | --- |
| Frozen matched models and field weights | [Training 01](../notebooks/training/01_train_and_validate.ipynb), [Training 02](../notebooks/training/02_freeze_models.ipynb), [Random Forest weights](../src/crop_pipeline/training/exploratory.py) |
| Primary 111-field JECAM test and supplemental predictions | [Testing 01](../notebooks/testing/01_external_test_evaluation.ipynb), [external scoring](../src/crop_pipeline/training/notebook_workflow.py) |
| Adaptation split, 1×–4× sample weights, and held-out predictions | [JECAM adaptation verifier](../tools/verify_jecam_adaptation.py) |
| Exploratory January–April 2024 Paoay map | [Paoay 08](../notebooks/paoay/08_whole_barangay_farmland.ipynb), [map workflow](paoay_workflow.md#whole-barangay-farmland-first-demonstration) |

The main adaptation comparison is the frozen 0-local-field forest versus the
23-local-field refit at 1×, with macro-F1 increasing from 0.261 to 0.657 on the
same 88 JECAM fields. The 3× result of 0.702 is exploratory. The verifier checks
original boundaries, parcel identities independent of year, sampled pixel
locations, and the 500 m geographic blocks. It checks the sample weights and
reproduces each held-out prediction with the archived feature snapshots and
scikit-learn 1.9.1.

Run the verifier from the original workspace or after relocating the paths in
the saved adaptation and quality handoffs. It needs the archived feature
snapshots and JECAM source footprints from the full local study data.

```bash
.venv/bin/python tools/verify_jecam_adaptation.py \
  pipeline_outputs/testing/run_ca5c900c415b \
  data/pipeline_work/analysis/external/external_t4tyh93t/quality.json
```

## Present the workflow notebooks

Use the project Python kernel. Saved notebook outputs can reflect earlier data;
show the run path and receipt when discussing a result. Review collection settings
before Run All: collection notebooks can submit jobs. Opening a notebook is inert.

| Demo step | What to show | Explain |
| --- | --- | --- |
| Collection 01 and 08 | Source roles, selected parcels and batch progress | Candidate counts precede imagery quality checks. Source top-up notebooks 02–07 collect or resume data. |
| Preprocessing 01–02 | Quality losses and qualifying fixed/aligned windows | Green-up is estimated vegetation timing. Missing observations remain excluded. |
| Preprocessing 03 | Usable parcels/pixels chart and export paths | Full cohorts have separate coverage; matched cohorts hold locations constant. |
| Training 01 | Saved validation candidates | Blue marks the saved leaf-size selection; test scores are not involved. |
| Training 02 | Frozen package paths | These packages copy fitted models and preserve their recipes. |
| Testing 01–02 | Scores, per-crop support and parcel confusion matrices | Rows are reference crops; columns are predictions. Compare against the training-majority baseline. |

Choose a test population explicitly in the diagnostics chart; show JECAM and
supplements separately before discussing a combined score. Missing reference
crops prevent a three-class aggregate. The reporting F1 criteria are project
targets, not a published guarantee of usefulness.

Paoay 01–05 with optional 06–07 is a separate application demo. Reference notes, screening maps,
applicability and seasonal comparisons are diagnostics, not local accuracy
validation. Keep dates visible and distinguish estimated crop choices from dated
field observations.

See [research basis and limits](research_basis.md) for the claim-to-source table.
The new chart helpers only read saved tables. They do not fit models, change
partitions, alter quality thresholds or edit input records.
