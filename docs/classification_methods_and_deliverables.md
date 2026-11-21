# Project SakAI classification methods and deliverables

This document records the classification methods stated in the thesis proposal and the revised roles of external datasets and Paoay. The methods below describe the proposed study design.

## Revised Method 1 and study design

**Method 1 replaces the proposal’s original Paoay-trained, Paoay-validated classification design with an external-to-external evaluation and an external-to-local application.**

1. **External-to-external evaluation.** Train and select the classifier using labelled external development parcels. Evaluate the frozen model on spatially independent, labelled external parcels. Report performance for the named test campaigns and describe the population each result represents.
2. **External-to-local application.** Apply the frozen, externally trained classifier to Sentinel-2 imagery in Paoay, Atok, Benguet. Present the output as a preliminary crop prediction map with imagery coverage and model confidence diagnostics. The available Paoay references consist of area-level crop-calendar guidance and possible rotations. They lack verified observations that identify which crop occupied a specific field during the mapped cycle. Paoay therefore serves as an application area, with ground-truth validation and local accuracy measurement deferred until suitable dated field observations are available.

## Classification methods from the original proposal

### Study area and spatial transfer

The original proposal uses Paoay for training and independent field validation. A single fitted Random Forest is then applied to the rest of Atok and the remaining municipalities of Benguet. The proposed comparison across extents covers valid imagery coverage, predictor-range coverage, predicted class proportions, model vote strength and computational requirements. Quantitative validation was planned for Paoay. The revised design above supersedes these original training, validation and application roles.

### Reference data and sample selection

- Record the crop or land-cover class, field boundary or coordinates, date, label source and verification evidence. Use drone imagery for visual context and verify its date, location and crop identity before using it as independent reference evidence.
- The original scheme proposes seven classes: carrot, cabbage, potato, other cultivated vegetation, forest or woody vegetation, built-up area, and bare soil or fallow land. The revised external experiment focuses on carrot, cabbage and potato.
- Match each reference label to imagery from its documented crop cycle. Flag potential harvest, replanting and within-window crop transitions.
- Sample homogeneous field interiors where feasible. Exclude uncertain labels and avoid mixed beds, boundaries, roads, structures and strong shadows. Retain a mixed validation pixel only when the dominant class can be established. Where footprint evidence permits, require more than 50% occupancy by that class and flag the pixel as mixed.
- Assign whole fields to partitions using class-stratified random allocation and a fixed seed in the original design. Keep pixels from a single field together and describe how sparsely sampled classes are handled. In the revised design, preserve field-level separation within development and spatial independence between development and the external test campaigns.

### Sentinel-2 preprocessing and predictors

- Use Sentinel-2 Level-2A surface reflectance in Google Earth Engine. Filter by area and crop cycle. Apply consistent masking rules to clouds, shadows, cirrus, saturation and other unusable observations.
- Create per-pixel median composites for the first and second halves of each month. Place predictors on a common 10 m grid, including resampling B5 and B11 from their native 20 m resolution. Mark incomplete retained predictor stacks as no-data.
- Retain B3, B4, B5, B8, B11, NDVI, NDRE and NDMI for each interval. The feature-engineering section also proposes SAVI to address soil-background effects.
- Derive temporal differences, green-up and senescence slopes, peak interval, amplitude and temporal integral from vegetation-index trajectories.
- Compare interval values only, interval values plus temporal-shape features, and temporal-shape features only using the same reference data. Report performance by class and in aggregate.
- Use Sentinel-2 spectral bands and derived indices for classification. Soil, terrain, climate and the NAMRIA land-use constraint serve the separate suitability workflow.

### Random Forest training and diagnostics

- The proposal specifies 1,000 trees, seed 42, the square-root rule for candidate predictors at each node and Gini impurity for selecting splits. Document the minimum leaf population, bag fraction and any class-balancing procedure.
- **Bootstrap aggregating.** Draw a training sample with replacement for each tree and combine the predictions. With a full-size bootstrap, about 63.2% of distinct training samples appear at least once, and about 36.8% are left out. This corrects the reversed wording in proposal Section 3.3.1.
- **Out-of-bag error.** Aggregate each sample’s predictions from trees that left it out. Treat the result as an internal diagnostic. Spatial dependence between pixels from the same field can make it optimistic.
- **Prediction and feature importance.** The proposal describes selecting the class with the most tree votes, recording the highest vote proportion, and reviewing mean decrease in Gini impurity by band, index and time interval.

### Independent evaluation and uncertainty

- Compare against a majority-class classifier, a single-date Random Forest and a spectral-bands-only Random Forest on the same evaluation data.
- The original proposal plans 112 independent Paoay plots of 50 × 50 m, each represented by 25 grid cells. The revised accuracy assessment uses independent labelled external parcels. Paoay remains a preliminary application.
- Report a confusion matrix, overall accuracy, producer’s accuracy, user’s accuracy, per-class F1, macro-F1 and Cohen’s kappa. Use macro-F1 as the primary aggregate metric.
- Estimate 95% confidence intervals by resampling independent reference plots as groups, keeping their pixels together. For the revised external analysis, the corresponding resampling unit is the independent reference parcel. Use a paired bootstrap for the difference from the majority baseline. This evaluation bootstrap serves a different purpose from the training bootstrap used to build the forest.
- The proposal’s aggregate interpretation criterion requires macro-F1 of at least 0.50, performance above the majority baseline, and a paired 95% confidence interval for the difference that excludes zero. For each target crop, use F1 ≥ 0.50 as the interpretation floor and flag substantial uncertainty if its interval extends below that floor.

### Maps, coverage and error analysis

- Produce 10 m thematic rasters with explicit no-data and report the mapped, valid-prediction and no-data areas. For the revised Paoay application, distinguish withheld predictions from missing imagery.
- Report clear-observation counts by semi-monthly interval, maximum class vote proportion and predictor-range coverage. The proposal uses the training 5th–95th percentile ranges and flags pixels with more than 20% of predictors outside those ranges. A vote proportion below 0.50 indicates lower vote strength.
- Review mistakes alongside mixed pixels, small field interiors, boundary proximity, terrain shadow and possible crop transitions. Report imagery losses and uncertainty diagnostics alongside classifier performance.
- Where comparing application extents, record processed pixels, feature-stack size, runtime, resource use and export or memory failures.

## Classification deliverables

1. **Prepare and assess the classification datasets.** Document the sources and selection rationale, harmonize crop codes, and profile parcels by class, split, geography, year and temporal correspondence. Report class imbalance and exclusions. Record Paoay boundaries and crop-calendar notes as application context.
2. **Implement Sentinel-2 preprocessing and feature engineering.** Mask unusable observations, build semi-monthly composites, derive the spectral and vegetation-index predictors, and quantify missing intervals, cloud-related loss and usable pixel and parcel coverage. Describe the fixed and green-up-aligned windows used in the revised experiment.
3. **Implement and evaluate the classification prototype.** Define the classes, predictors, temporal matching, parcel-level splits and exclusions. Train and freeze a Random Forest, then report the independent external evaluation. Produce a preliminary Paoay application map with prediction and coverage diagnostics.
5. **Assess limitations and requirements for improvement.** Analyze external crop confusions, geographic and source differences, temporal correspondence, mixed pixels and coverage losses. Specify the reference data and validation design needed to measure local performance in Paoay.
6. **Submit a journal or conference paper.** Present the classification methods, external evaluation, preliminary application and error analysis. Paper submission is a THSST2 completion requirement.

Deliverable 4 concerns crop-suitability mapping and sits outside this classification scope.
