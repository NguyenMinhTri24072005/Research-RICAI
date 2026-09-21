# Pipeline review — 2026-09-21

## Scope and verdict

This is a read-only CPU review of `CODE/RICE_VISION_MAIN_PIPELINE.ipynb`,
`CODE/modules`, the raw images, and the current final regression dataset.
No pipeline implementation, model artifact, dataset, or notebook was changed.

**Verdict:** the pipeline is a useful research prototype, and its saved
ExtraTrees regression artifact is reproducible on the stored feature table.
It is **not yet validated as an end-to-end physical counting system**.  The
physical branch has large error on a stored-feature surrogate, and the current
geometry implementation does not reproduce the scale recorded for several
dataset images.  The full CPU image-to-count E2E test is **NOT VERIFIED**
because TensorFlow in `AI_SERVICES/.venv` cannot import
`tensorflow.python`.

## What was verified

### Stored regression artifact

`review_cpu.py --mode tabular` loaded `model.joblib`, `scaler.joblib`, and
`pipeline.joblib` from `LINEAR_REGRESSION_MODEL/models/extra_trees` and used
all 254 rows with finite 31 features plus `Actual_Count`.

| Check | Result |
| --- | --- |
| Feature count | 31 |
| Current `model.predict(scaler.transform(X))` vs saved pipeline | maximum difference `0.0` |
| Current predictions vs `predictions_test.csv` on 51 saved holdout rows | maximum difference `0.0` |
| Saved group split | 45 train groups, 12 test groups, no overlapping `M###` groups |
| Mean warm CPU inference, one batch of 254 stored feature rows | 76.6 ms |

On the **saved 51-row group holdout**, regression has MAE `4.93` seeds,
RMSE `7.75`, MAPE `4.59%`, and bias `+2.71`.  These are valid only for the
stored features, the two known cup families, and the historical split.  The
near-perfect score over all 254 rows is in-sample and is not a generalization
metric.

### Physical-count surrogate on stored features

This is not a full current-notebook E2E run.  It applies the core physical
formula `round(V_bulk * packing_fraction / stored_mean_grain_volume)` to the
same stored 51-row holdout.  It shows the scale of the existing physics
problem:

| Packing fraction | MAE | RMSE | Bias | MAPE |
| --- | ---: | ---: | ---: | ---: |
| 0.82, the current physical notebook setting | 77.14 | 86.85 | +44.86 | 57.01% |
| 0.62, historical dataset hybrid setting | 47.24 | 64.37 | -6.45 | 32.20% |

`0.82` remains a legitimate *intended physical assumption*.  The result above
means the current measured volumes/scale and that assumption are not jointly
calibrated on this dataset.  It does not justify choosing 0.62 merely because
it is numerically closer on this contaminated historical table.

### Container detector on original CPU images

`check_geometry.py --mode geometry` ran only `detect_container_and_scale` on
six original 8160x6120 images.  Each call took 6.7–11.5 seconds on CPU.  The
orange/green diagnostic overlays are in `container_overlays.jpg`.

| Sample | Stored px/mm | Current px/mm | Ratio current/stored |
| --- | ---: | ---: | ---: |
| M014a | 70.71 | 83.95 | 1.19 |
| M031a | 72.71 | 74.21 | 1.02 |
| M056a | 70.31 | 79.92 | 1.14 |
| M029c | 7.83 | 74.31 | 9.49 |
| M037e | 7.22 | 75.16 | 10.41 |
| M038e | 7.18 | 80.42 | 11.20 |

The last three historical scales make mean grain lengths 40.7–62.8 mm in a
22.8-mm cup.  They are invalid physical measurements, not ordinary noise.
Because volume is divided by `pixels_per_mm ** 3`, even a 10% scale increase
changes calculated volume by about 25% and count at fixed bulk volume by about
33% (the controlled synthetic calculation is in `geometry_synthetic.json`).

## Root causes and risks

1. **Scale/data-version drift — critical.**  The currently running container
   detector returns a materially different scale from the historical dataset
   for multiple images.  Regression was trained on the historical feature
   distribution, while the notebook will generate current features.  This can
   invalidate both physical count and regression before model quality is even
   assessed.

2. **Ellipsoid thickness is an uncalibrated dominant assumption.**
   `ellipsoid_geometry.py` defines `c_px = a_px * 0.85` for a whole grain,
   i.e. thickness is 85% of the *semi-major (length) axis*.  In all 254 valid
   historical rows, calculated thickness exceeds width.  This is a structural
   red flag, not a fact measured by the camera.  It must be calibrated against
   direct physical measurements; do not tune it from the target count alone.

3. **The regression signal is largely manual/container metadata.**  ExtraTrees
   impurity importances allocate 99.89% to container height, inner diameter,
   weight, bulk volume, rice height, and empty height.  `Weight_g` and
   `Actual_Count` correlate at 0.9986.  This importance is not causal proof,
   but it is enough evidence that the apparent held-out success may come mostly
   from cup geometry and weight rather than vision-derived grain morphology.
   The dataset has only two cup configurations, with disjoint count ranges.
   It has not demonstrated transfer to a new cup, camera height, lighting
   setup, or density range.

4. **Dataset quality gates are incomplete.**  Of 285 rows, 31 are incomplete
   because their images are `MISSING`; only 254 can enter regression.  M017d
   appears twice, including twice in the saved training predictions.  101/254
   valid rows have fewer than eight whole grains, so the current area-IQR size
   filter intentionally does not filter them.  The filter can reduce small
   false detections but cannot correct scale, segmentation, or classifier
   error.

5. **Current CPU pipeline is not performance-profiled end to end.**  SAHI is
   applied to the full high-resolution image with 640px slices and 25% overlap.
   The segmenter allocates a full-image mask and scans it for every detection,
   which is costly at 8160x6120.  The detector alone already costs up to 11.5s;
   full YOLO/SAHI plus CNN will be slower.  The notebook should be profiled by
   stage after environment repair, not optimized blindly.

6. **Reproducibility issues.**  Artifacts were created with scikit-learn 1.6.1
   but this review runtime is 1.9.1 and emits `InconsistentVersionWarning`.
   Current stored-feature outputs match exactly, but this remains unsupported
   version drift.  The container RANSAC path initializes an unseeded random
   generator, so repeated scale results may vary.

7. **Notebook configuration ordering is misleading.**  The size-filter cell
   runs before `REGRESSION_CONFIG`, where `ENABLE_SIZE_FILTER`, `SIZE_FILTER_K`,
   and `SIZE_FILTER_MIN_SAMPLES` are displayed.  A fresh Run All uses fallback
   values (True, 1.5, 8), then defines the same settings later.  Editing the
   lower configuration cell does not affect that preceding filter pass.

## Strengths

- Clear staged flow: container scale, SAHI segmentation, crop cleaning, CNN,
  geometry, physical estimate, then a schema-checked regression bundle.
- The regression bundle has an ordered 31-feature schema, model/scaler files,
  manifest hashes, and a saved group split; direct model/scaler output equals
  the saved pipeline on stored inputs.
- Missing weight and no-whole-grain cases fail explicitly instead of silently
  producing a count.
- The current container detector creates overlays and exposes detection type,
  which makes visual quality control possible.
- IQR filtering is applied after CNN, so it does not silently redefine the raw
  regression feature contract.

## Tests executed for this review

| Command | Result |
| --- | --- |
| `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_review_20260921/review_cpu.py --mode tabular` | PASS; results above |
| `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_review_20260921/check_geometry.py --mode data` | PASS; 285-row data audit |
| `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_review_20260921/check_geometry.py --mode geometry` | PASS; container-only CPU check on six images |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest CODE.tests.test_grain_size_filter CODE.tests.test_notebook_measurement_contract CODE.tests.test_main_pipeline_size_filter -v` | PASS; 11 tests |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest CODE.tests.test_main_pipeline_regression -v` in an isolated temp directory | PASS; 7 tests |
| `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_review_20260921/review_cpu.py --mode vision` | NOT VERIFIED; `ModuleNotFoundError: tensorflow.python` during TensorFlow import |

## Recommended improvement order

1. **Freeze an acquisition/calibration protocol and repair the data first.**
   Create a small manually measured calibration set across the two cups: known
   inner diameter in pixels, a physical ruler/reference in the imaging plane,
   direct grain length/width/thickness, actual count, weight, and recorded
   packing state.  Review all scale outliers, remove/deduplicate invalid rows
   with an audit reason, then regenerate all 31 features with one frozen module
   version.  Keep raw data untouched and write a separate cleaned manifest.

2. **Validate the physics branch before retraining regression.**
   Calibrate the ellipsoid thickness relation independently from count labels,
   then estimate a packing-fraction distribution from quality-controlled
   samples.  Report MAE, bias, and intervals by cup/configuration.  Retain
   `0.82` only if this experiment supports it for the intended capture state.

3. **Repair the local CPU environment and run a bounded E2E evaluation.**
   Install a consistent CPU TensorFlow/Keras environment, pin scikit-learn to
   1.6.1 for this artifact or retrain/re-export under the pinned project
   environment.  Run a fixed, labelled subset through every stage; save
   detection overlays, accepted/rejected crops, feature drift, stage time, and
   final errors.  Do not claim performance from stored CSV inference alone.

4. **Evaluate model transfer honestly.**  After data regeneration, use group
   split by physical sample plus leave-one-cup/configuration-out validation.
   Compare regression with a weight-only baseline and a no-manual-metadata
   vision baseline.  This tells whether vision contributes useful information.

5. **Only then optimize CPU work.**  Profile actual stage times.  First likely
   wins are avoiding full-resolution-mask scans per detection, testing SAHI
   slice/overlap on labelled images, and rejecting detections outside the
   validated inner-rim region.  Preserve accuracy tests before changing the
   current full-image approach.

## Files in this review directory

- `tabular_summary.json`, `tabular_predictions.csv`: reproducible stored-table
  regression and physical-surrogate metrics.
- `data_quality.json`: missing rows, duplicate ID, cup coverage, and impossible
  scale-derived measurements.
- `geometry_summary.json`, `geometry_synthetic.json`, `container_overlays.jpg`:
  bounded container-stage CPU evidence.
- `review_cpu.py`, `check_geometry.py`: diagnostic scripts only.  They are not
  production pipeline modules.
