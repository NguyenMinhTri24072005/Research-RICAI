# Project Status

Updated: 2026-09-17 16:54:10 +07:00

## Repository Snapshot

Branch: `main`
Base commit before TASK-01 finalization: `1f1bb70 chore: checkpoint TASK-01 regression verification`
Working tree before finalization: TASK-01 source, tests, reports, plan, and documentation changes were reviewed and preserved.
Latest commit before this publication checkpoint: `d0c0eb6 fix: verify TASK-01 training-inference contract`.
Working tree before publication: DIRTY — notebook consolidation, approved legacy deletions, new training bundles/results, and pre-existing `scratch/`.
Publication scope: training source, documentation, tests, approved legacy deletions, global textual reports, and per-model configuration/schema/metrics/manifests/CV summaries. Serialized model/scaler artifacts, detailed predictions, plots, venv, and `scratch/` remain local and are excluded from this checkpoint.

## Current Phase

TASK-01 is verified as the deployed 31-feature Extra Trees inference contract on Windows local. TASK-00 technical implementation is complete, but field protocol acceptance is still pending. TASK-02 is ready by dependency only and has not started.

## Work Recorded Today — 2026-09-17

### Completed Work

- TASK-01 deployment verification was finalized in commit `d0c0eb6`; its earlier test evidence is preserved below.
- `LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb` is the single training notebook, organized into environment/configuration, data loading/cleaning, feature contract, grouped splits, preprocessing, separate model configuration cells, training, evaluation/comparison, export, and reload checks. The inventory contains 19 candidate configurations; current saved outputs contain 18 successful models, with Stacking disabled.
- The notebook's `FEATURE_CONTRACT` declares the ordered 31-feature schema locally without importing AI Services. `Actual_Count` is the target; sample/image-status metadata are excluded from model inputs.
- `StandardScaler` is inside each sklearn Pipeline and is fitted within CV training folds. Each exported model/scaler pair comes from the same final pipeline fitted on outer-training data.
- `EXPORT_ALL` and `RELOAD_SMOKE` now use `models/<model_id>/` directly. Future exports overwrite that model's named artifacts and global reports directly in `results/`. Run IDs remain metadata only.
- All 18 existing model bundles and the global reports were moved up one level. No timestamp/random-ID output folders remain. SHA-256 checks passed for all 18 bundle manifests after the move.
- Removed the obsolete Decision Tree trainer, both old Linear Regression trainer notebooks, and `Untitled0.ipynb`; removed earlier legacy training reports/plots and unused model artifacts with user approval.
- Retained the four active root deployment files: `best_tree_ensemble_model.joblib`, `scaler.joblib`, `best_tree_model_info.json`, and `scaler_params.json`. Their protected SHA-256 checks and the AI Services manifest check passed.
- Aligned README, consolidation plan, and tests with the direct model-folder layout. Removed test assertions for the legacy model/equation already deleted by request.
- Cleared stale notebook execution outputs/counts for rerunning. Cleaned Windows `desktop.ini` files from models/results and the training venv's schema package, where they prevented `nbformat` imports.

### Saved Training Results — PROVISIONAL

Evidence: `LINEAR_REGRESSION_MODEL/results/run_metadata.json`, `data_quality_report.json`, `comparison.csv`, `split_membership.csv`, and `models/ard/`.

- Available training run: `20260917_093236_436fb9`, seed 42, GROUPED split, 5 CV folds, 18 trained models; ARD selected by CV.
- Dataset: 285 input rows, 254 FOUND/clean rows, exactly 31 features; SHA-256 `a9fddb014fce8899ceda9fc392cc1ed1ad602fa7c5d30330978b00176c3bf78b`.
- ARD CV: MAE 2.063336 seeds, RMSE 3.001832, R2 0.998507. Holdout: MAE 2.054241 seeds, RMSE 3.041404, R2 0.998466. These are pilot metrics, not official benchmark results or evidence of external generalization.
- Saved ARD configuration records Colab Python 3.13.15 / scikit-learn 1.6.1; local training requirements specify Python 3.12.10 / scikit-learn 1.9.1. These environments are not identical.
- No full training execution occurred after the export-layout change. Available metrics belong to the earlier run; moving bundles did not retrain or deploy them. AI Services still uses its existing Extra Trees bundle, not ARD.

### Verification Executed Earlier Today

Executed in the preceding Codex work; not rerun during this documentation-only update:

| Command/check | Result |
|---|---|
| `LINEAR_REGRESSION_MODEL/.venv/Scripts/python.exe -m unittest discover -s LINEAR_REGRESSION_MODEL/tests -p test_notebook_consolidation.py -v` | PASS — 7/7 tests; final run 0.465s. Notebook syntax/structure, 19-model inventory, feature isolation, grouped splitting/fold-local scaling, metric guards, synthetic export/reload parity, protected deployment hashes. |
| Read-only Python SHA-256 check against each direct bundle's `manifest.json` | PASS — 18/18 bundles. Verifies declared integrity; does not reproduce training. |
| Notebook-source/output-directory structural checks | PASS — export/reload use direct model paths; no run-ID directories remain. |

### Pending Training Follow-up

- Per-model editable scaler configuration is still pending: `PREPROCESSING.build_pipeline()` currently uses `StandardScaler()` centrally. Each model does have its own fitted scaler file.
- `RUN_CONFIG` assigns a Colab `DATA_PATH_OVERRIDE` and immediately resets it to `None`. Its default root still uses `GROUP_MEMBERS`, while the user also supplied a `MEMBERS` root earlier. Resolve the intended path/configuration before the next Colab run.
- Legacy CLI helpers remain. `inference_linear_regression.py` still defaults to the deleted `models/regression_equation.json`; review these helpers before using them. The consolidated notebook is the current training entrypoint.
- Full notebook export/reload execution with the new layout is pending; current verification is static/synthetic plus integrity checks of moved bundles.
- Published manifests describe locally retained artifacts; `.joblib` files are excluded by Git policy. A GitHub checkout alone does not contain the trained model/scaler binaries.

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

### Inherited TASK-01 Finalization Evidence

These tests ran during the earlier TASK-01 finalization today. They were not rerun during this status update or the later notebook-storage changes.

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

Status: IN PROGRESS — PROVISIONAL benchmark. Current consolidated-notebook results use grouped physical-sample validation; older random-split results were exploratory. Official results still require a locked QC-approved dataset and an accepted evaluation protocol.

Current evidence: `LINEAR_REGRESSION_MODEL/results/comparison.csv`, `split_membership.csv`, and per-model `metrics.json`/`manifest.json`. ARD is the current CV-selected pilot candidate; the deployed Extra Trees model is unchanged.

## TASK-05 — Few-shot / Transfer

Status: NOT STARTED.

## Blockers

None newly identified for the existing TASK-01 deployment contract; it was not reverified end-to-end during this documentation update. Project gates remain TASK-00 field acceptance and TASK-03 official dataset/QC. Training follow-up items are listed above.

## Next Actions

1. Finish the requested per-model scaler configuration and resolve data-path overrides before rerunning training.
2. Rerun the notebook with direct output folders under a recorded environment; verify export/reload integrity and prediction parity.
3. Audit the remaining legacy CLI helpers and align recorded local/Colab environments before relying on reproducible comparisons.
4. Record TASK-00 field acceptance; collect/QC TASK-03 data and lock the official evaluation dataset/protocol.
5. Begin TASK-02 only under a separate authorized scope; keep TASK-04 results provisional until official evaluation gates pass.

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
- `LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb`
- `LINEAR_REGRESSION_MODEL/NOTEBOOK_CONSOLIDATION_PLAN.md`
- `LINEAR_REGRESSION_MODEL/README.md`
- `LINEAR_REGRESSION_MODEL/requirements_training.txt`
- `LINEAR_REGRESSION_MODEL/tests/test_notebook_consolidation.py`
- `LINEAR_REGRESSION_MODEL/results/run_metadata.json`
- `LINEAR_REGRESSION_MODEL/results/comparison.csv`
- `LINEAR_REGRESSION_MODEL/results/split_membership.csv`
- `LINEAR_REGRESSION_MODEL/models/<model_id>/manifest.json`
