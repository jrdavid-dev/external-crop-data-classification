# Progress bars and notebook estimates

All six notebooks show **one overall bar** after setup, with elapsed time and notebook ETA. The same bar switches from stage counts to the active stage’s total work:

| Notebook | Main progress units |
|---|---|
| 01 Sen4AgriNet | Repository entries, reference patches, consolidation, parcels, satellite tiles and half-month files |
| 02 FranceCrops | Missing reference tables, parcels, satellite tiles and half-month files |
| 03 CropHarvest | Label-processing stages, parcels, satellite tiles and half-month files |
| 04 Four-month windows | Audited tiles, timelines, candidate cycles and complete-feature tiles |
| 05 Feature engineering | Pixel windows enhanced and feature-handoff stages |
| 06 Training | Completed Random Forest designs and evaluation/package stages |

During collection, the total counts **tiles attempted**, including retries that remain pending. It does not count individual filenames or imply that all images passed. Between work stages, the bar counts completed notebook stages, **not a percentage of total running time**. A model unit advances after a whole forest finishes, not after each tree. Elapsed time continues updating during long calls. Unknown-length discovery or remote jobs do not show a fabricated completion percentage.

## How estimates work

- Active-stage ETAs use measured item throughput. They become useful after some work finishes and adjust as rates change.
- The first whole-notebook ETA says **learning** when future stages have not been timed. The active stage still has its own bar and estimate.
- A comparable completed run supplies rough timings for remaining notebook stages. Different input/run identities are kept separate.
- During measurable final-stage work, such as annual tile collection, current throughput can supply a rough remaining notebook estimate. Authentication, network throttling, tile size, cache reuse and final export overhead can change it.
- Optional or disabled stages are marked skipped. Pending training and failed runs do not become completed timing profiles.

Bars use notebook widgets when available and text progress otherwise. Bounded reference scans and annual batch collection reuse this same bar: they do not create a row for each patch, Earth Engine task or download page. Gathering counts completed inventory calls; these are not verified imagery completion or elapsed-time percentages. When a bounded scan can stop early or remote work has no measurable total, the bar shows counts without a percentage. Final collection summaries still report verified local completion and cloud states.

Verbose per-file collector messages are saved in a `.log` file beside the timing report rather than printed in the cell. Authentication prompts, final summaries and error tracebacks remain visible. Progress observes successful calls without changing arguments, selected windows, model parameters or parcel splits.

## Saved timing reports

Each run writes a separate JSON under the parent of `report_root`:

```text
progress/<notebook-name>/runs/<timestamp-and-run-id>.json
```

The final monitored cell prints the total elapsed time and report path. Reports contain stage durations, statuses and the workload identity. Earlier reports and datasets/results remain untouched. Estimates refer to pipeline work after environment setup; initial package installation is excluded.

Progress reporting lives outside the fingerprinted processing helpers. CropHarvest's processing fingerprint excludes only explicitly identified progress UI statements; scientific code edits still change it. Adding bars therefore does not create a new collection revision solely for the UI change.
