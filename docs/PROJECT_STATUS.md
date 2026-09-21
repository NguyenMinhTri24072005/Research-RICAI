# Project Status

Updated: 2026-09-20 +07:00

## Repository Snapshot

Branch: `main`
Latest implementation commit before this status publication: `fb686a2 refactor: modularize AI Services inference runtime` (with uncommitted changes: Grain Size Filter Implementation).
Working tree reviewed: AI Services modularization, Colab-notebook relocation, tests, artifact layout, reports, plans, the approved cleanup of unused assets, and the recent integration of the Grain Size Filter (Area IQR) into the main pipeline.

## Current Phase

The project has achieved a modular AI Services runtime under `AI_SERVICES/src/rice_ai` and successfully integrated a robust one-sided Area IQR filter into the `CODE/RICE_VISION_MAIN_PIPELINE.ipynb` pipeline. The filter segregates physical volume estimation from regression features, strictly maintaining backward compatibility with the deployed 31-feature Extra Trees bundle. The next scientific gates are field acceptance of the capture protocol, evaluating the efficacy of the size filter, and a QC-approved official dataset.

## Task Overview

| Task | Status | Evidence | Remaining |
|---|---|---|---|
| TASK-00 | IN PROGRESS — technical implementation complete; field acceptance pending | Capture app, protocol documents, metadata support | Record real capture/QC acceptance. |
| TASK-01 | VERIFIED COMPLETE — local runtime | 75 focused tests, artifact preflight, inherited M001A E2E | Colab operational acceptance and environment alignment are follow-up safeguards. |
| TASK-02 | READY | TASK-01 local dependency is verified | Separate implementation authorization. |
| TASK-03 | IN PROGRESS — pilot only | Capture and pilot workflow | QC-approved official dataset. |
| TASK-04 | IN PROGRESS — PROVISIONAL benchmark | Grouped-split pilot reports | Frozen dataset and accepted protocol. |
| TASK-05 | NOT STARTED | Project task list | Backlog. |
| GRAIN-FILTER | VERIFIED COMPLETE — local runtime | 9 filter tests, `CODE/reports/grain_size_filter/verification.md`, regression parity maintained | E2E Colab evaluation on real images to confirm filter efficacy (adjust `SIZE_FILTER_K` as needed). |

## GRAIN_SIZE_FILTER Implementation (New)

Status: **VERIFIED COMPLETE — local Windows runtime.**

Completed:
- Created `CODE/modules/grain_size_filter.py` implementing a one-sided IQR filter (`lower_bound = Q1 - k * IQR`) based on 2D area (`area_mm2`) to strictly remove small outliers without penalizing large grains.
- Separated notebook logic in `CODE/RICE_VISION_MAIN_PIPELINE.ipynb` into modular cells: `GRAIN_MEASUREMENTS`, `GRAIN_SIZE_FILTER`, and `GRAIN_PHYSICAL_STATISTICS`.
- Guaranteed regression backward compatibility: The legacy `REGRESSION_FEATURES` explicitly extracts the unfiltered `raw_valid_cnn_whole` list, while the explicit physical math step utilizes the `filtered` population.
- Extensive test coverage added (`test_grain_size_filter.py`, `test_main_pipeline_size_filter.py`) and executed regression tests (`test_main_pipeline_regression.py`) successfully to prove no schema breakage.

Pending:
- Colab deployment and human visual review of the rejected grain gallery to adjust the default `SIZE_FILTER_K = 1.5` threshold if necessary.

## TASK-00 — Capture Protocol / Data Governance

Status: **IN PROGRESS — technical implementation complete; field acceptance pending.**

Completed:

- `CAPTURE_APP` supports phone-camera-node capture, QR/HTTPS connection, flat image identifiers, and Excel/SQLite recording.
- The documented capture workflow covers top-down setup, lighting, scale, `Sample_ID`, `Actual_Count`, `QC_Status`, and `QC_Reason`. `Capture_Batch` and `Device_ID` are intentionally not required.

Pending:

- Execute the protocol in the field and record acceptance/QC evidence before calling this task verified.

## TASK-01 — 31-Feature Regression Integration

Status: **VERIFIED COMPLETE — local Windows runtime.**

### Verified Contract

- `AI_SERVICES/src/rice_ai/regression/feature_schema.py` defines exactly 31 ordered features; runtime assembly follows that deterministic contract.
- `Actual_Count` is a training target and is not accepted as a runtime regression input.
- `Estimated_Total_Seeds_Hybrid` uses the audited training-equivalent formula, `round(Bulk_Rice_Volume_mm3 * 0.62 / mean(Grain_Volume_mm3))`.
- Required missing or non-finite feature values are rejected before `scaler.transform()` and `model.predict()`; a valid numerical zero remains valid.
- The active bundle loads an `ExtraTreesRegressor` and `StandardScaler`; loader checks feature count/order, metadata, fitted state, and declared artifact integrity.
- Runtime paths are settings-driven (`YOLO_MODEL_PATH`, `CNN_MODEL_PATH`, `REGRESSION_MODEL_DIR`, `YOLO_DEVICE`) and do not require `/content/drive` on local Windows.

### Remaining Safeguards / Limitations

- A real Google Colab launch is **NOT VERIFIED** in this review; the notebook has static-structure tests and its manual procedure is in `AI_SERVICES/src/notebooks/README.md`.
- Artifacts saved with scikit-learn 1.6.1 emit `InconsistentVersionWarning` under local scikit-learn 1.9.1. Preflight and tests pass, but pin the training version or re-export the selected model before claiming cross-version reproducibility.

## TASK-02 — Explainability

Status: **READY.** TASK-01's local dependency is verified; no TASK-02 implementation has started.

## TASK-03 — Dataset Pilot / QC

Status: **IN PROGRESS — pilot only.** The existing pilot/demo material is not an official benchmark dataset. Next step: collect under the accepted capture protocol, apply QC, and freeze the official dataset.

## TASK-04 — Regression Benchmark

Status: **IN PROGRESS — PROVISIONAL benchmark.** `LINEAR_REGRESSION_MODEL/results/` contains grouped-split pilot comparisons; these results remain provisional until the dataset and evaluation protocol are locked.

## TASK-05 — Few-shot / Transfer

Status: **NOT STARTED.**

## Tests and Verification Evidence

- Grain Size Filter Evidence: `CODE/reports/grain_size_filter/verification.md`, `CODE/tests/test_grain_size_filter.py`, `CODE/tests/test_main_pipeline_size_filter.py`.
- Regression Integration Parity (Passed with Grain Filter): `CODE/tests/test_main_pipeline_regression.py`.
- Current modular-runtime evidence: `AI_SERVICES/tests/`, `AI_SERVICES/scripts/verify_artifacts.py`.
- Current restructure/remediation evidence: `reports/ai_services_restructure/remediation_verification.md`.

## Blockers

None currently identified for the locally verified TASK-01 contract and Grain Size Filter. Project-level gates are TASK-00 field acceptance, TASK-03 dataset QC, assessing the real-world efficiency of the new size filter via Colab runs, and a real Colab operational check before relying on Colab execution.

## Next Actions

1. Review the Grain Size Filter on Colab using real images and tune `SIZE_FILTER_K` as appropriate based on visual reports.
2. Perform and record TASK-00 field-protocol/QC acceptance.
3. Run the documented Colab notebook once with actual Drive/model paths; record the outcome and align the scikit-learn version if needed.
4. Build and QC the official dataset before treating TASK-04 metrics as official.
5. Start TASK-02 only under separately authorized scope.

## Important Files

- `CODE/RICE_VISION_MAIN_PIPELINE.ipynb`
- `CODE/modules/grain_size_filter.py`
- `CODE/reports/grain_size_filter/verification.md`
- `AI_SERVICES/src/rice_ai/settings.py`
- `AI_SERVICES/src/rice_ai/api/application.py`
- `AI_SERVICES/src/rice_ai/estimation/feature_schema.py`
- `AI_SERVICES/src/rice_ai/models/regression_loader.py`
- `AI_SERVICES/src/notebooks/API_Server.ipynb`
- `AI_SERVICES/scripts/verify_artifacts.py`
- `LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb`

## Git Snapshot

Branch: `main`
Implementation commit: `fb686a2 refactor: modularize AI Services inference runtime`
Status publication: pending commit on `main`.
