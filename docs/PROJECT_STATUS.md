# Project Status

Updated: 2026-09-18 +07:00

## Repository Snapshot

Branch: `main`
Latest implementation commit before this status publication: `fb686a2 refactor: modularize AI Services inference runtime`
Working tree reviewed: AI Services modularization, Colab-notebook relocation, tests, artifact layout, reports, plans, and the approved cleanup of unused assets.

## Current Phase

The project now has a modular AI Services runtime under `AI_SERVICES/src/rice_ai`, with explicit settings for vision and regression artifacts. TASK-01 is locally verified against its deployed 31-feature Extra Trees bundle. The next scientific gates are field acceptance of the capture protocol and a QC-approved official dataset; no later research task is treated as complete merely because the software path exists.

## Task Overview

| Task | Status | Evidence | Remaining |
|---|---|---|---|
| TASK-00 | IN PROGRESS — technical implementation complete; field acceptance pending | Capture app, protocol documents, metadata support | Record real capture/QC acceptance. |
| TASK-01 | VERIFIED COMPLETE — local runtime | 75 focused tests, artifact preflight, inherited M001A E2E | Colab operational acceptance and environment alignment are follow-up safeguards. |
| TASK-02 | READY | TASK-01 local dependency is verified | Separate implementation authorization. |
| TASK-03 | IN PROGRESS — pilot only | Capture and pilot workflow | QC-approved official dataset. |
| TASK-04 | IN PROGRESS — PROVISIONAL benchmark | Grouped-split pilot reports | Frozen dataset and accepted protocol. |
| TASK-05 | NOT STARTED | Project task list | Backlog. |

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

### Final Review Verification — 2026-09-18

| Command | Result |
|---|---|
| `AI_SERVICES/.venv/Scripts/python.exe -m compileall -q AI_SERVICES/src/rice_ai AI_SERVICES/app.py AI_SERVICES/scripts AI_SERVICES/tests` | PASS |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_settings AI_SERVICES.tests.test_regression_loader AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_estimators AI_SERVICES.tests.test_pipeline AI_SERVICES.tests.test_api_endpoints AI_SERVICES.tests.test_vision_models AI_SERVICES.tests.test_api_lifecycle AI_SERVICES.tests.test_colab_notebook -v` | PASS — 75 tests in 11.252 s |
| `AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py` | PASS — active Extra Trees/StandardScaler bundle; 31 features, metadata/integrity checks, transform and finite prediction |
| `AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py --model-dir ../LINEAR_REGRESSION_MODEL/models/ard` | PASS — canonical ARD training bundle preflight |
| `AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_api_endpoints AI_SERVICES.tests.test_colab_notebook -q` after unused-file cleanup | PASS — 12 tests |

### Inherited E2E Evidence

The strict M001A E2E was executed during the immediately preceding remediation before the approved unused-file cleanup; no runtime source changed afterwards. It passed with ground truth 85, raw regression 85.2, final result 85, geometry result 282, and Feature 11 = 167. It is inherited evidence, not a new E2E run in this publication pass. See `reports/ai_services_restructure/remediation_verification.md`.

### Remaining Safeguards / Limitations

- A real Google Colab launch is **NOT VERIFIED** in this review; the notebook has static-structure tests and its manual procedure is in `AI_SERVICES/src/notebooks/README.md`.
- Artifacts saved with scikit-learn 1.6.1 emit `InconsistentVersionWarning` under local scikit-learn 1.9.1. Preflight and tests pass, but pin the training version or re-export the selected model before claiming cross-version reproducibility.
- `reports/task_01/` documents the earlier registry implementation; use the current modular source and remediation report for the present runtime architecture.

## TASK-02 — Explainability

Status: **READY.** TASK-01's local dependency is verified; no TASK-02 implementation has started.

## TASK-03 — Dataset Pilot / QC

Status: **IN PROGRESS — pilot only.** The existing pilot/demo material is not an official benchmark dataset. Next step: collect under the accepted capture protocol, apply QC, and freeze the official dataset.

## TASK-04 — Regression Benchmark

Status: **IN PROGRESS — PROVISIONAL benchmark.** `LINEAR_REGRESSION_MODEL/results/` contains grouped-split pilot comparisons; these results remain provisional until the dataset and evaluation protocol are locked.

## TASK-05 — Few-shot / Transfer

Status: **NOT STARTED.**

## Tests and Verification Evidence

- Current modular-runtime evidence: `AI_SERVICES/tests/`, `AI_SERVICES/scripts/verify_artifacts.py`, and the commands recorded above.
- Current restructure/remediation evidence: `reports/ai_services_restructure/remediation_verification.md`.
- Historical TASK-01 evidence: `reports/task_01/verification.md` and `reports/task_01/test_report.json`; it applies to the former registry layout and is not presented as a fresh test of the modular runtime.

## Blockers

None currently identified for the locally verified TASK-01 contract. Project-level gates are TASK-00 field acceptance, TASK-03 dataset QC, and a real Colab operational check before relying on Colab execution.

## Next Actions

1. Perform and record TASK-00 field-protocol/QC acceptance.
2. Run the documented Colab notebook once with actual Drive/model paths; record the outcome and align the scikit-learn version if needed.
3. Build and QC the official dataset before treating TASK-04 metrics as official.
4. Start TASK-02 only under separately authorized scope.

## Important Files

- `AI_SERVICES/src/rice_ai/settings.py`
- `AI_SERVICES/src/rice_ai/api/application.py`
- `AI_SERVICES/src/rice_ai/estimation/feature_schema.py`
- `AI_SERVICES/src/rice_ai/models/regression_loader.py`
- `AI_SERVICES/src/rice_ai/pipeline/estimators.py`
- `AI_SERVICES/src/notebooks/API_Server.ipynb`
- `AI_SERVICES/src/notebooks/README.md`
- `AI_SERVICES/artifacts/README.md`
- `AI_SERVICES/scripts/verify_artifacts.py`
- `AI_SERVICES/tests/`
- `reports/ai_services_restructure/remediation_verification.md`
- `LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb`
- `LINEAR_REGRESSION_MODEL/results/comparison.csv`

## Git Snapshot

Branch: `main`
Implementation commit: `fb686a2 refactor: modularize AI Services inference runtime`
Status publication: pending commit on `main`.
