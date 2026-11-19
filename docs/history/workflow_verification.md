# Workflow verification checkpoint — 19 November 2026

> Historical record. Instructions and test totals describe their original checkpoint; do not use old notebook sequences as the current run guide. See the [current workflow](../workflow.md) and [history index](README.md). Links have been adjusted for this documentation move.

Verified on 2026-11-19: **482 tests passed**. Active notebook code compiles;
local links resolve and the diff passes whitespace checks.

`test_complete_notebook_workflow.py` exports/imports tiny real TIFF fixtures,
runs Collection 08 and all seven downstream notebooks in fresh namespaces using
default stage pointers, and checks report publication, unchanged collection
files, no refitting after training and rejection of mismatched test snapshots.
Imagery audit/window selection are mocked in that integration test and forests
use three trees for speed. Their processing and transfer behavior have separate
fixture tests. No live collection, full-data preprocessing or scientific model
run was performed for this verification.
