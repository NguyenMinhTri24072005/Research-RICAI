# TASK-01 Completion Plan

Status: Ready for execution
Created: 2026-09-17
Planner: Codex 5.6 Sol High
Target task: TASK-01 — Regression 31-feature integration

Purpose:
Provide an evidence-based execution plan to bring TASK-01 from IN PROGRESS to
VERIFIED without expanding scope into TASK-02/03/04.

## Current Diagnosis

### Root cause 1: Hybrid feature training/inference mismatch

Estimated_Total_Seeds_Hybrid does not currently have the same semantic in
training and runtime.

- CODE/modules/dataset_extractor.py creates it from bulk rice volume,
  packing_fraction, and mean grain volume.
- DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb instantiates the
  extractor with packing_fraction=0.80.
- AI_SERVICES/app.py supplies estimates.final to the regression feature. That
  can combine geometry and weight estimates and is not the training calculation.
- docs/TASK_01_AUDIT.md records this mismatch explicitly.

Candidate training formula:

    int(round(Bulk_Rice_Volume_mm3 * 0.80 / Grain_Volume_mm3_Mean))

This becomes canonical only after row-by-row verification against the CSV used
to train the deployed model.

### Root cause 2: Model/scaler provenance is incomplete

The deployed paths are:

- Model: LINEAR_REGRESSION_MODEL/models/best_tree_ensemble_model.joblib
- Scaler: LINEAR_REGRESSION_MODEL/models/scaler.joblib
- Scaler metadata: LINEAR_REGRESSION_MODEL/models/scaler_params.json
- Model metadata: LINEAR_REGRESSION_MODEL/models/best_tree_model_info.json

AI_SERVICES/artifacts/manifest.json has provenance.verified=false. Timestamps
are supporting evidence only. model_registry.py contains a SHA-256 helper but
does not enforce hashes declared by the manifest.

### Root cause 3: Missing regression features may become zero

AI_SERVICES/app.py validates a feature dictionary, then changes None into 0.0
before regression prediction. A required unavailable feature can therefore
reach scaler.transform() and model.predict().

Valid numerical zero and missing data must remain distinct. Missing data may
not be imputed with zero unless the verified training contract defines it.

### Scope and baseline

- Latest reviewed commit: 1f1bb70 chore: checkpoint TASK-01 regression verification.
- Prior review found the worktree clean.
- Existing reports state 22/22 tests passed and M001A E2E passed, but those
  results predate the Hybrid correction and must be rerun afterward.
- Do not modify CAPTURE_APP, TASK-02/03/04/05, datasets, legacy model binaries,
  .env, tokens, or unrelated code.

## Source of Truth

| Concern | Source of truth | Required interpretation |
| --- | --- | --- |
| Training CSV | DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv | Stored Hybrid values are authoritative for deployed-model semantics. |
| Dataset generation | CODE/modules/dataset_extractor.py and AI_DATASET_EXTRACTION_PIPELINE.ipynb | Verify extraction configuration against actual CSV rows. |
| Tree training | LINEAR_REGRESSION_MODEL/RICE_SEED_DECISION_TREE_TRAINER.ipynb | Confirm target, order, split, scaler fit, Extra Trees settings, and artifact export paths. |
| Deployed feature contract | scaler_params.json | Ordered feature list must match runtime exactly. |
| Runtime schema | AI_SERVICES/feature_schema.py | Must be one runtime schema and match deployed contract. |
| Model metadata | best_tree_model_info.json | Validate target, count, estimator family, and importances. |
| Environment | AI_SERVICES requirements.txt, setup_ai_service.bat, run_ai_service.bat | Use AI_SERVICES/.venv/Scripts/python.exe with Python 3.10 to 3.12. |

Actual_Count is a training target and must never enter a runtime feature vector.
Hybrid may remain an engineered feature only if it is independent from
Actual_Count and exactly matches the training CSV calculation.

## Definition of Done

TASK-01 may be marked Completed or Verified only when all items below have
post-fix evidence.

- [ ] Exactly 31 input features with deterministic order.
- [ ] Runtime schema equals tree trainer and scaler_params.json.
- [ ] Training/inference semantic parity, including Hybrid, is proven.
- [ ] Actual_Count is absent from runtime features and Hybrid proxy is audited.
- [ ] Model and scaler load in the supported project environment.
- [ ] Bundle paths, hashes, schema, scaler metadata, and model metadata validate.
- [ ] Historical provenance remains explicitly unverified unless direct evidence exists.
- [ ] scaler.transform() and model.predict() return finite output.
- [ ] Required missing/non-finite features never silently become zero.
- [ ] API status, predict, unit/API/registry tests, and M001A E2E pass.
- [ ] Windows/local uses relative paths and does not require /content/drive.
- [ ] Audit, verification report, task board, and project status make only proven claims.

## Files Expected to Change

- AI_SERVICES/feature_schema.py: canonical Hybrid semantic and helper.
- AI_SERVICES/regression_engine.py: assemble feature 11 from training semantics.
- AI_SERVICES/app.py: remove zero-imputation and validate before prediction.
- AI_SERVICES/model_registry.py: enforce hash/schema/type/count checks.
- AI_SERVICES/artifacts/manifest.json: bind artifact and schema provenance.
- AI_SERVICES/scripts/verify_artifacts.py: verify compatibility and smoke prediction.
- AI_SERVICES/tests/test_feature_schema.py: Hybrid formula and order tests.
- AI_SERVICES/tests/test_regression_engine.py: missing-vs-zero and vector tests.
- AI_SERVICES/tests/test_model_registry.py: hash/schema mismatch tests.
- AI_SERVICES/tests/test_api_endpoints.py: explicit validation behavior.
- AI_SERVICES/tests/test_real_pipeline.py: post-fix M001A feature-vector check.
- docs/TASK_01_AUDIT.md, reports/task_01/verification.md,
  docs/PROJECT_STATUS.md, PROJECT_TASKS/TASK_01_AI_SERVICES_INTEGRATION.md:
  update only after verification passes.

## Phase 0 — Baseline

### Goal

Establish the actual training and environment contract before changing runtime.

### Files

- LINEAR_REGRESSION_MODEL/RICE_SEED_DECISION_TREE_TRAINER.ipynb
- DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv
- CODE/modules/dataset_extractor.py
- DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb
- AI_SERVICES requirements.txt, setup_ai_service.bat, run_ai_service.bat

### Exact actions

1. Record git status --short, git diff --stat, and the latest five commits.
2. Run py -0p, then identify the interpreter behind
   AI_SERVICES/.venv/Scripts/python.exe.
3. Verify this interpreter is Python 3.10, 3.11, or 3.12 and imports fastapi,
   joblib, sklearn, and numpy.
4. Inspect tree trainer code cells. Record target, ordered features,
   train_test_split settings, StandardScaler.fit location, Extra Trees settings,
   and artifact export paths.
5. Create a read-only CSV parity check. For each Image_Status == FOUND row with
   Hybrid, bulk volume, and mean grain volume, compare stored Hybrid against
   int(round(bulk * 0.80 / mean_volume)).
6. Compare ordered names from tree trainer, scaler_params.json,
   best_tree_model_info.json, and feature_schema.py.

### Commands

    cd AI_SERVICES
    $py = ".\.venv\Scripts\python.exe"
    & $py -c "import sys, fastapi, joblib, sklearn, numpy; print(sys.version); print(fastapi.__version__); print(joblib.__version__); print(sklearn.__version__)"
    & $py scripts\verify_artifacts.py --manifest artifacts\manifest.json

### Tests

The CSV parity check is baseline evidence and must later become a regression test.

### Pass condition

The supported virtual environment works, all sources use the same 31 feature
names/order, and the CSV confirms the formula or produces an exact mismatch list.

### Rollback / stop condition

If the CSV does not match 0.80, do not implement that formula in runtime.
Identify the true extraction contract or move to Phase 3 Plan B.

## Phase 1 — Training/Inference Feature Parity

### Goal

Make runtime feature 11 exactly equal to training feature semantics.

### Files

- AI_SERVICES/feature_schema.py
- AI_SERVICES/regression_engine.py
- AI_SERVICES/app.py
- AI_SERVICES/tests/test_feature_schema.py
- AI_SERVICES/tests/test_regression_engine.py
- AI_SERVICES/tests/test_real_pipeline.py

### Exact implementation

1. Only after Phase 0 CSV parity passes, define
   TRAINED_HYBRID_PACKING_FRACTION = 0.80 in feature_schema.py.
2. Add one helper, for example compute_trained_hybrid_feature, accepting raw
   bulk_volume_mm3 and whole-grain volume_mm3 values. It must use full precision
   mean volume and return int(round(bulk_volume_mm3 * 0.80 / mean_volume_mm3)).
3. The helper returns None for invalid inputs. It must not use weight, sample
   count, median px3, estimates.final, or display factor 0.82.
4. In assemble_31_features(), set Estimated_Total_Seeds_Hybrid only through the
   helper using container_res bulk_rice_volume_mm3 and volume_mm3 values in
   whole_grains.
5. Remove the app.py path that supplies estimates.final to Hybrid. Remove the
   obsolete internal parameter if it has no valid callers.
6. Keep compute_final_estimates() separate. Its 0.82 geometry display behavior
   may remain, but it must not enter the tree feature vector.
7. Retain Hybrid because the tree expects it. Document it as a deterministic
   engineered proxy, never ground truth.

### Tests

- Synthetic reference: bulk 8000, volumes [10, 20, 30] returns 320.
- CSV parity test covers every complete training row.
- Assembly test asserts feature index 10 equals the helper output.
- E2E debug vector asserts index 10 equals canonical calculation.
- Changing sample weight/count does not change feature index 10.

### Pass condition

No path supplies estimates.final to feature 11. CSV parity and runtime assembly
tests pass.

### Rollback / stop condition

If runtime metrics lack reliable volume_mm3, do not substitute pixel or median
volume. Stop and verify geometry units.

## Phase 2 — Required Feature Validation

### Goal

Prevent invalid regression vectors from reaching scaler.transform() or
model.predict().

### Files

- AI_SERVICES/feature_schema.py
- AI_SERVICES/regression_engine.py
- AI_SERVICES/app.py
- AI_SERVICES/tests/test_feature_schema.py
- AI_SERVICES/tests/test_regression_engine.py
- AI_SERVICES/tests/test_api_endpoints.py

### Exact implementation

1. Make feature_vector_to_ordered_list() and predict_from_tree() reject missing
   required inputs; neither may use get(name, 0.0) for model input.
2. Remove the None -> 0.0 comprehension in app.py.
3. Run validate_feature_vector() before registry load, scaler transform, and
   model prediction.
4. For estimator_mode=regression, return HTTP 422 with MISSING_FEATURE,
   NON_FINITE_FEATURE, or INVALID_INPUT before model inference.
5. For estimator_mode=auto, skip regression only when geometry or weight fallback
   is valid. Return regression_est=null, the actual method, and a machine-readable
   warning; otherwise return HTTP 422.
6. Preserve valid zero permitted by schema, such as one-grain standard deviation.
   Treat None as unavailable.
7. Treat Weight_g=0.0 as valid only if Phase 0 verifies explicit training policy;
   do not turn absent weight into zero implicitly.
8. Preserve ai_est as the geometry compatibility alias.

### Tests

- Missing Grain_Volume_mm3_Mean in regression mode returns HTTP 422 and does not
  call the model.
- NaN and Inf return HTTP 422.
- One valid grain with every standard deviation equal to 0.0 remains valid.
- Explicit zero weight follows verified policy.
- A mock proves no prediction after validation failure.

### Pass condition

No required missing or non-finite feature is silently transformed into zero
before prediction.

### Rollback / stop condition

Do not make every zero invalid. Keep only explicit valid-zero rules from the
verified training contract.

## Phase 3 — Artifact Provenance

### Goal

Verify the deployed bundle contract without fabricating historical provenance.

### Files

- AI_SERVICES/artifacts/manifest.json
- AI_SERVICES/model_registry.py
- AI_SERVICES/scripts/verify_artifacts.py
- AI_SERVICES/tests/test_model_registry.py
- optionally reports/task_01/artifact_verification.json

### Exact implementation

1. Recompute SHA-256 from bytes for model, scaler, scaler parameters, model
   metadata, and canonical JSON serialization of schema version plus ordered features.
2. Add meaningful manifest fields: relative paths, hashes, expected class,
   expected count, schema hash, target, training source, dataset path/hash, and
   scaler metadata path/hash.
3. Split provenance into deployment_contract_verified and historical_run_verified.
   The latter is true only with direct original-run evidence.
4. Update ModelRegistry.load_bundle() so a hash, type, schema, count, or scaler
   mismatch prevents verified status.
5. Update verify_artifacts.py to check ExtraTrees and StandardScaler class,
   31 inputs, scaler mean/scale versus scaler_params.json, schema order, model
   importances versus best_tree_model_info.json, and finite transform/predict.
6. Save measured verification output. Do not set historical provenance true from
   timestamps alone.

### Tests

- Valid current bundle verifies.
- Wrong model/scaler hash fails before verified status.
- Wrong schema hash/order fails.
- Missing artifact fails explicitly.
- Scaler/model compatibility and smoke prediction are finite.

### Pass condition

Deployed artifact bytes are pinned and cross-validated. Reports state the exact
provenance level achieved.

### Plan B if provenance cannot be proven

1. Never overwrite old artifacts.
2. Create a deterministic export script from the actual tree trainer contract.
3. Read the same CSV, preserve feature order, split seed, scaler-fit-on-train
   behavior, and Extra Trees settings.
4. Export a new model/scaler pair into a new bundle directory.
5. Record dataset/source hashes, seed, package versions, schema hash, artifact
   hashes, and metrics at export time.
6. Point the manifest to the new bundle only after Phase 4 passes.
7. Do not claim old historical provenance or force old M001A output.

## Phase 4 — Full Verification

### Goal

Run post-fix verification once in the intended environment. M001A E2E was
previously about 272.7 seconds on CPU, so run it only after fast tests pass.

### Commands in exact recommended order

    cd AI_SERVICES
    $py = ".\.venv\Scripts\python.exe"
    & $py -m compileall -q .
    & $py scripts\verify_artifacts.py --manifest artifacts\manifest.json
    & $py -m unittest tests.test_feature_schema tests.test_regression_engine tests.test_model_registry tests.test_api_endpoints -v
    & $py -m unittest tests.test_real_pipeline -v
    & $py scripts\run_tests.py
    git diff --check
    git status --short

### Expected evidence

- Artifact verifier reports matching hashes and compatible model/scaler.
- Fast feature, regression, registry, and API tests pass in the project venv.
- API status test passes.
- M001A runs real weights and records actual final/regression estimate, feature
  11, method, duration, and error against ground truth 85.
- E2E must not require output 85 after correction. Any change requires
  investigation and documentation.

### Pass condition

Every technical Definition of Done gate has current post-fix evidence.

### Rollback / stop condition

If M001A fails or changes, do not modify expected output to pass. Preserve the
observed result and inspect feature vector and bundle evidence.

## Phase 5 — Documentation and Status

### Goal

Make documentation state only measured post-fix results.

### Files

- docs/TASK_01_AUDIT.md
- reports/task_01/verification.md
- docs/PROJECT_STATUS.md
- PROJECT_TASKS/TASK_01_AI_SERVICES_INTEGRATION.md
- PROJECT_TASKS/README.md
- docs/EXPERIMENTS.md only if it is the established location for this evidence.

### Exact changes

1. Replace the Hybrid mismatch entry only after CSV and runtime parity tests pass.
   Record formula, units, factor, rounding, and target independence.
2. Record interpreter/package versions, commands, hashes, provenance level,
   current test results, and actual M001A result.
3. Remove stale references to get_regression_artifacts() or implicit OLS fallback
   if those symbols are no longer live.
4. Mark Task-01 Complete only when every Definition of Done item is evidenced.
5. Otherwise keep it IN PROGRESS and name the blocker.

### Pass condition

No document makes a stronger claim than its verification artifacts support.

## Test Matrix

| Layer | Evidence required | Pass condition |
| --- | --- | --- |
| Static/import | compileall in project venv | Exit code 0. |
| Feature contract | 31 count/order; synthetic and CSV Hybrid parity | No mismatch. |
| Validation | Missing vs zero; NaN/Inf; no-model-call | Invalid vector never predicts. |
| Artifact | Hashes, schema, scaler params, model metadata, smoke inference | Verifier reports valid contract. |
| API | Health, status, invalid predict | Explicit error and status schema pass. |
| E2E | M001A with real weights | Canonical debug vector and valid response. |
| Repository | git diff --check | No whitespace errors. |

## Risks

| Risk | Mitigation |
| --- | --- |
| CSV does not use 0.80 | Stop after parity check and identify true extraction contract. |
| Runtime lacks volume_mm3 | Do not substitute pixel median; verify geometry units. |
| Historical run cannot be proven | Keep it unverified; make a new reproducible bundle if strict proof is needed. |
| M001A output changes | Record and investigate; never hardcode 85. |
| Global Python 3.14 lacks packages | Use project venv only. |
| Large binaries are untracked | Commit only source, manifest, hashes, metadata, and reports. |

## Stop Conditions

Stop and report rather than guess or fake evidence if:

1. CSV Hybrid values do not match the proposed formula.
2. Runtime lacks reliable mean volume_mm3 for the feature.
3. Model/scaler hashes, scaler params, or model metadata fail cross-validation.
4. Training source cannot recreate a valid bundle contract.
5. Existing venv is unsupported and repair requires deleting it without authorization.
6. M001A E2E fails with corrected semantics.
7. Hybrid is found to use Actual_Count or target-derived manual adjustment.

## Suggested Commit Message

    fix: enforce TASK-01 training inference contract

Use one atomic commit only after all gates pass. Do not commit datasets,
checkpoints, joblib model binaries, venv, secrets, or generated bulky files.

## Execution Checklist for Implementation Agent

- [ ] Read this file, Task-01, audit, manifest, reports, and current Git state.
- [ ] Preserve existing work; do not change CAPTURE_APP, datasets, old binaries,
      .env, TASK-02/03/04/05, or unrelated files.
- [ ] Use AI_SERVICES/.venv/Scripts/python.exe; verify Python 3.10–3.12 and
      imports fastapi, joblib, sklearn, and numpy.
- [ ] Inspect RICE_SEED_DECISION_TREE_TRAINER.ipynb for the actual target, split,
      scaler, Extra Trees, feature order, and export settings.
- [ ] Check all complete CSV rows against the 0.80 Hybrid formula.
- [ ] Stop and report exact mismatches; do not guess another formula.
- [ ] In feature_schema.py add one canonical Hybrid helper after CSV parity passes.
- [ ] In regression_engine.py make assemble_31_features() call that helper.
- [ ] Ensure feature 11 never uses weight, sample count, median px3,
      estimates.final, or display factor 0.82.
- [ ] In app.py remove the estimates.final caller path and preserve geometry output
      separately from tree features.
- [ ] Reject missing required values in ordered-vector and tree prediction code;
      never use missing-to-zero for model inputs.
- [ ] In app.py validate before registry/scaler/model and remove None to 0.0.
- [ ] Return HTTP 422 for invalid regression mode before inference.
- [ ] In auto mode use only an explicit valid fallback with regression_est=null
      and a warning; otherwise return HTTP 422.
- [ ] Preserve legitimate numerical zero only where verified schema permits it.
- [ ] Add synthetic Hybrid, CSV parity, ordered vector, missing-vs-zero, NaN/Inf,
      and no-model-call tests.
- [ ] Recompute model/scaler/scaler-params/model-info/schema SHA-256 values;
      place only recomputed values in manifest.json.
- [ ] In model_registry.py make hash/type/order/count mismatch prevent verified status.
- [ ] In verify_artifacts.py compare scaler object/params and model/importances,
      then run finite transform/predict smoke inference.
- [ ] Keep deployment_contract_verified separate from historical_run_verified;
      never fake historical provenance.
- [ ] If strict pairing cannot be evidenced, export a new deterministic bundle
      from the actual trainer into a new directory without overwriting old artifacts.
- [ ] Run compileall, verifier, fast tests, and M001A E2E once in that order.
- [ ] Do not force M001A to predict 85; record and explain the measured result.
- [ ] Update audit, report, task board, and project status only after every gate passes.
- [ ] Keep Task-01 IN PROGRESS if any stop condition remains.
- [ ] Run git diff --check and git status --short before handoff.
- [ ] Do not commit or push unless separately requested.
