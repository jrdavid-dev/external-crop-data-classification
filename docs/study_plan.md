# Study plan

Test whether external crop-reference data improve crop classification in Benguet when verified local labels are limited.

## Current scope

The intended crops are cabbage, carrot and potato. The implemented Sen4AgriNet pilot contains **carrot and potato only**. No cabbage was found in the inspected local 2020 subset; that does not establish absence across the full source dataset. No verified cabbage source is configured, so a complete three-crop experiment is not ready.

Use S4A 2020 now and leave 2019 for future work. Exact Benguet boundaries, map reference dates and independent local training/test budgets still need agreement.

## Source labels and imagery

S4A provides dated observations, annual crop masks and parcel IDs. Labels come from LPIS farmer declarations rather than classifier predictions. They are useful reference labels, but are not automatically independently field-verified.

Original S4A imagery is L1C. The notebook uses its labels and locations to retrieve matching Sentinel-2 L2A surface reflectance. Do not combine processing levels as if they were equivalent.

| Crop | Original S4A code |
|---|---:|
| Cabbage | 213 |
| Carrot | 231 |
| Potato | 510 |

Confirm the actual encoding before interpreting counts. Check the declaration reference period and its relevance to each selected cycle.

## Model input design

| Setting | Design |
|---|---|
| Classifier | Random Forest |
| Window | Four consecutive calendar months |
| Bins | Days 1–15 and 16–month-end: eight bins |
| Bands | B3, B4, B5, B8 and B11 |
| Indices | NDVI, NDRE and NDMI |
| Primary features | Eight variables × eight bins = 64 |
| Comparison features | Five bands × eight bins = 40 |

The proposed window starts on the first day of the month containing accepted green-up and includes the following three months. Validate and freeze the rule before final testing. Onset uncertainty crossing months may leave the window unresolved.

## Development pilot

Select 1,000 unique labeled pixels per crop across 20 conservative parcel groups each, using 50 pixels per group, geographic spread and a fixed seed. The result is 40 retrieval regions and 960 annual composites: 24 half-month intervals per region.

Sampling masks distinguish selected pixels from surrounding image context. The selection favors groups large enough to supply the quota and excludes the smallest fragments. Equal initial pixel budgets do not guarantee equal usable counts after quality checks. Parcel groups are not yet verified independent-field identities.

## Main experiment

| Training set | Question |
|---|---|
| External only | How well do external samples transfer to Benguet? |
| Limited verified Benguet samples only | What does local-only training achieve? |
| External plus the same local training fields | Does adding external data help? |

Train a new forest for each comparison and evaluate all on the same independent Benguet test fields. Keep every fragment, pixel, season and overlapping window of a field in one partition.

Report per-crop precision, recall and F1, macro-F1, balanced accuracy and a confusion matrix. Report independent-field counts, pixel counts, detector failures, missing-data losses and mapping coverage. A Paoay-only test supports local conclusions; broader Benguet claims require broader validation.

## Before final training

1. Record source revisions, class definitions and declaration periods.
2. Resolve field duplication and independent-field identities.
3. Calibrate cloud/coverage settings on development data.
4. Validate green-up estimates and crop-cycle relevance.
5. Confirm a cabbage source before three-class classification.
6. Reserve independent local test fields before model selection.

The detector is retrospective and may use more observations than the classifier's four-month window. Explain this distinction. Green-up is not a verified planting date.

## Sources and related guides

- [S4A dataset](https://huggingface.co/datasets/orion-ai-lab/S4A)
- [S4A repository](https://github.com/Orion-AI-Lab/S4A)
- [S4A paper](https://arxiv.org/abs/2204.00951)
- [Workflow](workflow.md)
- [Label evidence](label_quality.md)
