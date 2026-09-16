# Project Status

Updated: 2026-09-17 02:47:47 +07:00

## Repository Snapshot

Branch: `main`
Base commit before TASK-01 finalization: `1f1bb70 chore: checkpoint TASK-01 regression verification`
Working tree before finalization: TASK-01 source, tests, reports, plan, and documentation changes were reviewed and preserved.

## Current Phase

TASK-01 is verified as the deployed 31-feature Extra Trees inference contract on Windows local. TASK-00 technical implementation is complete, but field protocol acceptance is still pending. TASK-02 is ready by dependency only and has not started.

## Task Overview

| Task | Status | Evidence | Remaining |
|---|---|---|---|
| TASK-00 | IN PROGRESS — technical implementation complete; field acceptance pending | Capture app and protocol docs | Record field-protocol/QC acceptance. |
| TASK-01 | VERIFIED COMPLETE | Final verification below | None for TASK-01. |
| TASK-02 | READY | TASK-01 dependency verified | Separate TASK-02 authorization required. |
| TASK-03 | IN PROGRESS — pilot only | Capture tooling and pilot data | QC-approved official dataset. |
| TASK-04 | IN PROGRESS — PROVISIONAL benchmark | Exploratory results | Frozen dataset and grouped splits. |
| TASK-05 | NOT STARTED | Task specification | Backlog. |

## TASK-00 — Capture Protocol / Data Governance

Status: IN PROGRESS — technical implementation complete; field acceptance pending.

- `CAPTURE_APP` implements phone-camera node capture, QR/HTTPS connection, timestamped capture, Excel/SQLite storage, and flat image identifiers.
- Protocol documents cover top-down setup, lighting, scale handling, `Sample_ID`, `Actual_Count`, `QC_Status`, and `QC_Reason`.
- `Capture_Batch` and `Device_ID` are intentionally not required.
- The research group must execute and record field-protocol acceptance before TASK-00 is called verified.

## TASK-01 — 31-Feature Regression Integration

Status: VERIFIED COMPLETE.

### Feature Contract

- `AI_SERVICES/feature_schema.py` centralizes exactly 31 deterministic inputs; its schema hash is verified at bundle load.
- `Actual_Count` is a target only and is absent from runtime feature inputs.
- `Estimated_Total_Seeds_Hybrid` is `round(Bulk_Rice_Volume_mm3 * 0.62 / mean(Grain_Volume_mm3))`, independent of weight/label. The post-fix CSV-parity test passed for 254/254 `Image_Status == FOUND` rows.
- Required missing/non-finite values are rejected before `scaler.transform()` or `model.predict()`; permitted numeric zero values remain valid.

### Artifact Contract

- `ExtraTreesRegressor` model and `StandardScaler` both load with 31 inputs.
- Manifest verification checks SHA-256 values for model/scaler/metadata, schema hash, type, dimensions, scaler arrays, and model importances.
- Deployment bundle provenance is verified. `historical_run_verified` remains `false`: historical training was not falsely represented as reproduced.

### Final Review Tests

| Command | Result |
|---|---|
| `AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py` with UTF-8 console | PASS — 31/31 features, model/scaler load, SHA-256/metadata checks, `scaler.transform()`, and finite `model.predict()`. |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_model_registry AI_SERVICES.tests.test_api_endpoints -v` | PASS — 33 tests in 15.830s. Includes feature schema/parity, validation, registry, `/api/status`, and invalid `/predict`. |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_real_pipeline -v` | PASS — M001A E2E in 440.777s: ground truth 85, final 85, raw Extra Trees 85.2, Feature 11 = 167.0, absolute error 0. |

API status: PASS through the API test.
Predict API: PASS for validation rejection and valid M001A E2E.
Windows/local: PASS; local relative paths are used and `/content/drive` is not required.

### Known Limitation

Artifacts saved by scikit-learn 1.6.1 emit `InconsistentVersionWarning` when loaded by local 1.9.1. All current preflight, 33 fast tests, and E2E passed; pin or re-export under the training version before making a cross-version reproducibility claim.

## TASK-02 — Explainability

Status: READY. TASK-01 is verified, but no TASK-02 implementation has begun.

## TASK-03 — Dataset Pilot / QC

Status: IN PROGRESS — pilot only. The pilot/demo dataset is not an official benchmark dataset; collection must follow accepted protocol and QC.

## TASK-04 — Regression Benchmark

Status: IN PROGRESS — PROVISIONAL benchmark. Existing random-split results are exploratory; official results require a locked QC-approved dataset and grouped physical-sample validation.

## TASK-05 — Few-shot / Transfer

Status: NOT STARTED.

## Blockers

None for TASK-01 finalization. Project-level gates are TASK-00 field acceptance and TASK-03 official dataset/QC.

## Next Actions

1. Execute and record TASK-00 field-protocol/QC acceptance.
2. Collect and QC TASK-03 pilot data; keep it distinct from an official benchmark dataset.
3. Begin TASK-02 only under a separate authorized scope.
4. Freeze dataset and use grouped physical-sample splits before TASK-04 official metrics.

## Important Files

- `PROJECT_TASKS/TASK_01_COMPLETION_PLAN.md`
- `AI_SERVICES/feature_schema.py`
- `AI_SERVICES/regression_engine.py`
- `AI_SERVICES/model_registry.py`
- `AI_SERVICES/artifacts/manifest.json`
- `AI_SERVICES/scripts/verify_artifacts.py`
- `AI_SERVICES/tests/`
- `docs/TASK_01_AUDIT.md`
- `reports/task_01/verification.md`
- `reports/task_01/test_report.json`
