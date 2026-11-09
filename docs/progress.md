# Progress and saved reports

Notebooks 01–06 reuse **one progress display for the active step**, with elapsed time and an estimate when measurable. Nested patches, filenames, cloud tasks and download pages do not create separate progress rows; verbose details go to a saved log.

| Notebook | Main measured work |
| --- | --- |
| 01–03 Sources | Reference entries/patches, parcel masks and imagery inventories |
| 04 Windows | Audited tiles, timelines, candidate cycles and complete windows |
| 05 Features | Pixel windows enhanced and feature export |
| 06 Training | Completed model designs, evaluation and packaging |

Counts describe their named units, not elapsed processing time or scientific validity. Inventory calls are not verified imagery downloads. A model advances after a whole forest finishes, not after every tree. Unknown-length work shows counts without an invented percentage.

ETAs use observed throughput or a comparable completed run. They remain **learning** when future stages have not been timed. Cache reuse, authentication, retries and cloud queues can change runtime. Pending and failed stages are not recorded as successful timing profiles.

Each run saves timing JSON and a detail log under:

```text
<parent-of-report_root>/progress/<notebook-name>/runs/
```

The final summary prints exact output/report paths and the next action. Authentication prompts, summaries and errors remain visible. Earlier reports are preserved; progress reporting does not alter imagery rules, selected windows, model settings or locked partitions.

In sources, `pending_retry` means inspect task states/errors and rerun existing work. In analysis, distinguish **collection targets unmet**, **pilot inputs ready**, and **production requirements pending**. A pilot can complete with low scores; incomplete inputs or overlapping partitions still block it.
