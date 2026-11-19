# External testing

Run `01_external_test_evaluation.ipynb`, then `02_results_and_diagnostics.ipynb` after freezing models. Outputs go to `pipeline_outputs/testing/run_<id>/`. See [handoffs and safeguards](../../docs/workflow.md#training-and-external-testing). Each notebook runs in a fresh kernel. The report gives separate rows for primary JECAM, each registered supplement, and the combined pool. Single-crop supplements report per-crop metrics and coverage without three-class macro-F1.
