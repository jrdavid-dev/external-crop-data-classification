# Present the active notebooks

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
