# VERIFICATION AND CLEANUP REPORT: DATASET BUILDER PIPELINE

**Generated:** 2026-09-22  
**Role:** Implementation Verifier + Cleanup Agent  
**Standard:** `DATASET_BUILDER/POST_IMPLEMENTATION_VERIFICATION_AND_CLEANUP_PLAN.md`  

---

## 1. Executive Summary

This report provides complete verification and cleanup documentation for the `DATASET_BUILDER` extraction pipeline and sample ID migration tooling in `Research-RICAI`.

- **Orchestration & Static Contracts:** Fully verified (100% pass across all 8 contract checks, 14 fake orchestration smoke tests, and temporary fixture migration & rollback tests).
- **Vendor Module Parity:** Byte-for-byte identical across all 7 modules with `CODE/modules/`.
- **Real Data Safety:** 100% preserved. Real workbook, canonical CSVs, and all 261 raw images maintained exact SHA256 integrity. Real ID migration dry-run executed with `--dry-run` only and halted safely on detected dataset conflicts.
- **Colab/T4 Behavioral Parity:** Marked `NOT VERIFIED (No GPU/T4 runtime available in this environment)`.
- **Cleanup Gate Status:** Category A (duplicate dry-run directories) and Category C (archived legacy tools in `archive_legacy/`) safely removed after verifying zero active runtime consumers and full git traceability. Category B (legacy extractor and extraction notebook in `CODE/`) retained under `REVIEW REQUIRED` pending Colab T4 GPU execution.

---

## 2. Environment & Repository Baseline

### 2.1 Python Runtime
- **Active Verification Interpreter:** `AI_SERVICES/.venv/Scripts/python.exe` (Python 3.12.10)
- **Key Installed Libraries:**
  - `torch`: 2.6.0+cu124
  - `ultralytics`: 8.3.82
  - `sahi`: 0.11.20
  - `opencv-python`: 4.11.0.86
  - `openpyxl`: 3.1.5
  - `pandas`: 2.2.3
  - `numpy`: 2.1.3
- **System Python (Not used):** Python 3.14.5 (lacks scientific packages)

### 2.2 Pre-Verification Data Baseline (SHA256 & Counts)
- `manual_data.xlsx`: `22035ceddbd10805065349a081c29f2c1ca9c086c928a0eeb18692fdfa1d88e0`
- `ai_extracted_dataset.csv`: `a9fddb014fce8899ceda9fc392cc1ed1ad602fa7c5d30330978b00176c3bf78b`
- `final_linear_regression_dataset.csv`: `091ef50e3f17aa0c8b33f793f22efe34c945b9a91bce852c28eff7de34d7692e`
- **Total Real Raw Images:** 261 images in `DATASET_BUILDER/1_Raw_Images/`
- **Git Commit Baseline:** `ab73998 (HEAD -> main, origin/main) asdad`

---

## 3. Defects Reproduced and Minimal Corrections

All fixes strictly preserved the vendored vision modules (`src/rice_dataset/vision/`) byte-for-byte identical to `CODE/modules/` and corrected only caller orchestration mismatches and edge-case contracts.

### Defect 1: SAHI Segmentation Keyword Mismatch (P0)
- **File:** `DATASET_BUILDER/src/rice_dataset/pipeline.py`
- **Evidence:** `segment_grains_sahi()` signature is `(detection_model, image_path, ...)` but pipeline caller was passing `img_input=img_bgr`, causing `TypeError: segment_grains_sahi() got an unexpected keyword argument 'img_input'`.
- **Fix:** Corrected caller parameters to `detection_model=self.sahi_model` and `image_path=img_bgr`.

### Defect 2: Two-Pass Cleaner Return Type Mismatch (P0)
- **File:** `DATASET_BUILDER/src/rice_dataset/pipeline.py`
- **Evidence:** `clean_single_grain_crop()` returns `Optional[np.ndarray]` (RGBA image), but pipeline caller called `cleaned.get("cleaned_rgba")`, causing `AttributeError: 'numpy.ndarray' object has no attribute 'get'`.
- **Fix:** Validated `cleaned` directly as `isinstance(cleaned, np.ndarray)` and `shape[2] == 4`, assigning `grain_copy["crop_rgba"] = cleaned2`.

### Defect 3: Ellipsoid Geometry Keyword Mismatch (P0)
- **File:** `DATASET_BUILDER/src/rice_dataset/pipeline.py`
- **Evidence:** `compute_single_grain_metrics()` signature is `(image_input, ...)` but pipeline caller passed `crop_input=crop_rgba`, causing `TypeError: compute_single_grain_metrics() got an unexpected keyword argument 'crop_input'`.
- **Fix:** Corrected caller parameter to `image_input=crop_rgba`.

### Defect 4: Canonical Column Duplicate (P1)
- **File:** `DATASET_BUILDER/src/rice_dataset/contracts.py` & `feature_builder.py`
- **Evidence:** `Container_Detected_Diam_px` was listed in both `ORDERED_FEATURES` and `DIAGNOSTIC_COLUMNS`, causing duplicate column names in canonical dataset exports (48 columns vs 47 unique).
- **Fix:** Removed duplicate `Container_Detected_Diam_px` from `DIAGNOSTIC_COLUMNS` and `build_physical_diagnostics()`. `CANONICAL_COLUMNS` now contains exactly 47 unique columns (`len(CANONICAL_COLUMNS) == len(set(CANONICAL_COLUMNS)) == 47`).

### Defect 5: Domain Validation on Grain Dimensions (P1)
- **File:** `DATASET_BUILDER/src/rice_dataset/feature_builder.py`
- **Evidence:** Only `length_mm` and `volume_mm3` were checked for positivity, allowing non-positive width or thickness to pass without error.
- **Fix:** Enforced strict checks that `length_mm > 0`, `width_mm > 0`, `thickness_mm > 0`, `area_mm2 > 0`, and `volume_mm3 > 0`.

### Defect 6: Manual Record Integrity & Lineage Preservation (P1)
- **File:** `DATASET_BUILDER/src/rice_dataset/io/manual_records.py` & `pipeline.py`
- **Evidence:**
  1. Float `Actual_Count` was silently converted with `int(round(val))`, risking data corruption.
  2. `Capture_Timestamp` was overwritten by pipeline execution timestamp instead of preserving manual source timestamp.
  3. Non-empty invalid manual record rows were dropped from batch processing instead of being recorded in the audit dataset.
- **Fix:**
  1. Enforced strict integer validation (`val.is_integer()`); non-integers are reported as loading errors.
  2. Preserved original `capture_timestamp` from workbook; default to run timestamp only if absent.
  3. `run_batch` now injects audit rows with `QC_Status = INVALID_MANUAL_DATA` for all invalid manual records so 100% of input rows are audited.

### Defect 7: Non-Atomic Feature Updates on Failure (P1)
- **File:** `DATASET_BUILDER/src/rice_dataset/pipeline.py`
- **Evidence:** If `build_physical_diagnostics` failed after `build_regression_features` succeeded, regression features remained partially attached to a failed sample row.
- **Fix:** Made feature and diagnostic dictionary updates atomic (`row.update(features)` and `row.update(diagnostics)` occur only when both succeed, otherwise row exits with `QC_Status = FEATURE_VALIDATION_FAILED` and no feature block).

### Defect 8: Sample ID Migrator Safety Hardening (P0)
- **File:** `DATASET_BUILDER/src/rice_dataset/migration/migrate_ids.py`
- **Evidence:**
  1. In theory, `--force` could have allowed applying migrations with fatal reconciliation errors.
  2. `--force-resequence` was documented in CLI help but not supported in the migration logic.
  3. Path traversal vulnerability: manifest entries pointing outside allowed directories were not validated before unlinking or copying during rollback.
- **Fix:**
  1. Hardened `apply()` so critical blockers (`duplicate_excel_id_count`, `duplicate_image_stem_count`, `excel_without_image_count`, `images_without_excel_count`) can NEVER be bypassed even with `--force`.
  2. Implemented `--force-resequence` cleanly to allow re-sequencing previously migrated datasets.
  3. Implemented path traversal checks in `rollback()` verifying that every path resolves strictly within the target directories.

---

## 4. Phase Verification Matrix

| Phase | Description | Result | Details |
|---|---|---|---|
| **Phase 0** | Baseline Preservation | **PASS** | Hashes recorded, 261 real images preserved, external files untouched |
| **Phase 1** | Defect Reproduction & Minimal Fixes | **PASS** | Orchestration caller defects A, B, C reproduced and resolved |
| **Phase 2** | Static Contract & Parity Verification | **PASS** | 8/8 static checks passed (schema, 31 features, parameters, isolation, byte parity) |
| **Phase 3** | Loader, Lineage & Reconciliation | **PASS** | Lineage preserved, invalid rows recorded in audit, 47 canonical columns |
| **Phase 4** | Migration Safety & Rollback | **PASS** | Fixture tests passed; rollback restored exact bytes; path traversal blocked |
| **Phase 5** | Deterministic Smoke / Orchestration | **PASS** | 14/14 test cases passed (1 success path + 13 failure branches) |
| **Phase 6** | Colab / T4 Behavioral Parity | **NOT VERIFIED** | No GPU/T4 runtime available in this environment |
| **Phase 7** | Notebook & Docs Verification | **PASS** | 10 cells, 0 embedded outputs, 0 legacy imports from `CODE` or `AI_SERVICES` |
| **Phase 8** | Cleanup Inventory & Deletion Gates | **PASS** | Duplicate dry-run sets and `archive_legacy` deleted; Category B retained |
| **Phase 9** | Post-Cleanup Verification | **PASS** | Compileall, static checks, smoke suite, fixture tests, and dry-run re-verified |

---

## 5. Migration Preflight Inspection (Real Data Dry-Run)

Command executed:
```powershell
& "AI_SERVICES/.venv/Scripts/python.exe" DATASET_BUILDER/scripts/migrate_sample_ids.py `
  --workbook DATASET_BUILDER/2_Manual_Records/manual_data.xlsx `
  --images DATASET_BUILDER/1_Raw_Images `
  --prefix M --digits 4 --start 1 --dry-run
```

**Output Summary:**
- **Total Excel Rows:** 1000
- **Non-empty Excel Rows:** 285
- **Unique Normalized IDs:** 284
- **Duplicate Excel IDs:** 1 (`M017D` on rows 85 and 86)
- **Total Images on Disk:** 261
- **Unique Image Stems:** 261
- **Duplicate Image Stems:** 0
- **Excel-only (No Image):** 31 (`M025C`, `M025D`, `M025E`, `M026C`, `M026D`, ...)
- **Image-only (No Excel):** 8 (`M005F`, `M006F`, `M011F`, `M017E`, `M017F`, `M021F`, `M055F`, `M056F`)
- **Apply Ready:** **NO (Blocked)**
- **Exit Code:** 1 (Safety halt)

---

## 6. Cleanup Inventory and Deletion Gate Evidence

### 6.1 Category A — Duplicate Generated Reports & Caches (DELETED)
- **Deleted Paths:**
  - `DATASET_BUILDER/migrations/dry_run_1790040235/` (duplicate report set)
  - `DATASET_BUILDER/migrations/dry_run_1790052370/` (duplicate report set)
- **Protection Added:** Created `DATASET_BUILDER/migrations/.gitignore` ignoring `dry_run_*`, `backup*`, and `staging*`.
- **Retained Canonical Report:** Root files in `DATASET_BUILDER/migrations/` (`id_mapping.csv`, `migration_report.json`, etc.).

### 6.2 Category B — Obsolete Extraction Implementation Candidates (RETAINED - REVIEW REQUIRED)
- **Paths:**
  - `CODE/AI_DATASET_EXTRACTION_PIPELINE.ipynb`
  - `CODE/modules/dataset_extractor.py`
  - `DatasetExtractor` export in `CODE/modules/__init__.py`
  - `DATASET_BUILDER/CAPTURE_APP/tests/test_dataset_extractor_compat.py`
- **Status:** **RETAINED (`REVIEW REQUIRED`)**
- **Rationale:** Under the strict Deletion Gate in Section 8 of the verification plan, Category B candidate deletion is permitted *only* after end-to-end Colab/T4 behavioral parity is proven. Furthermore, files in `CODE/` are outside the authorized working directory boundary. Deletion must wait until GPU behavioral parity verification is executed on Colab.

### 6.3 Category C — Archived Legacy Tooling Candidates (DELETED)
- **Deleted Paths:**
  - `DATASET_BUILDER/archive_legacy/raw_image_renaming/rename_raw_images.py`
  - `DATASET_BUILDER/archive_legacy/legacy_raw_survey/survey_raw_images.py`
  - `DATASET_BUILDER/archive_legacy/legacy_raw_survey/survey_report.md`
  - `DATASET_BUILDER/archive_legacy/legacy_raw_survey/survey_report.txt`
  - `DATASET_BUILDER/archive_legacy/legacy_raw_survey/survey_summary.csv`
  - `DATASET_BUILDER/archive_legacy/README.md`
- **Gate Evidence:**
  1. *Redundancy:* Replaced completely by `DATASET_BUILDER/scripts/migrate_sample_ids.py` which reproduces exact image surveys and conflict detection.
  2. *Replacement Path:* `DATASET_BUILDER/scripts/migrate_sample_ids.py`.
  3. *Reference Scan:* Zero active references in runtime code, tests, or notebooks across the entire repository.
  4. *Test Proof:* Proven in `verify_pipeline_parity.py --fixture-test` and real data dry-run reproducing identical duplicate and mismatch statistics.
  5. *Git Traceability:* Committed in Git history at commit `5a4db3ed8d14f84447029761e3fe587ca6678822` and fully recoverable.

### 6.4 Category D — Preserved Operational Assets (KEPT)
- `DATASET_BUILDER/check_filtered_grains.py` (referenced by diagnostic workflows)
- `DATASET_BUILDER/analyze_dataset.py`
- `DATASET_BUILDER/DATASET_EDA_AND_FEATURE_SELECTION.ipynb`
- `DATASET_BUILDER/scripts/verify_pipeline_parity.py` (permanent verification suite)
- `DATASET_BUILDER/CAPTURE_APP/` runtime and active tests

---

## 7. Integrity Verification After Cleanup

1. **Python Compilation:**
   ```powershell
   & "AI_SERVICES/.venv/Scripts/python.exe" -m compileall DATASET_BUILDER/src DATASET_BUILDER/scripts
   ```
   *Result:* 0 errors.

2. **Static & Orchestration Suite:**
   ```powershell
   & "AI_SERVICES/.venv/Scripts/python.exe" DATASET_BUILDER/scripts/verify_pipeline_parity.py
   ```
   *Result:* 8/8 static contract checks passed, 14/14 fake orchestration smoke checks passed, fixture migration/rollback passed 100%.

3. **Baseline File Hashes:**
   - `manual_data.xlsx`: Identical (`22035ceddbd10805065349a081c29f2c1ca9c086c928a0eeb18692fdfa1d88e0`)
   - `ai_extracted_dataset.csv`: Identical (`a9fddb014fce8899ceda9fc392cc1ed1ad602fa7c5d30330978b00176c3bf78b`)
   - `final_linear_regression_dataset.csv`: Identical (`091ef50e3f17aa0c8b33f793f22efe34c945b9a91bce852c28eff7de34d7692e`)
   - `Raw Images Count`: Exactly 261 files.

4. **Active Capture App Tests:**
   Ran 14 tests across `test_capture_app_core.py`, `test_mobile_api.py`, `test_mobile_capture_service.py`, and `test_desktop_mobile_integration.py`.
   *Result:* 14/14 passed (`OK`).

5. **`CODE/modules` Import Smoke Test:**
   Imported all modules used by `CODE/RICE_VISION_MAIN_PIPELINE.ipynb`.
   *Result:* All modules imported successfully without errors.

---

## 8. Remaining Blockers & Next Actions

1. **Colab / T4 Behavioral Parity Execution:**
   - Execute `DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb` on Google Colab with a T4 GPU runtime for the representative reference sample `M001A` and mini-batch.
   - Verify floating-point output values against legacy output.
2. **Category B Cleanup Gate:**
   - Once Colab behavioral parity is verified, remove `CODE/modules/dataset_extractor.py`, `CODE/AI_DATASET_EXTRACTION_PIPELINE.ipynb`, and update `CODE/modules/__init__.py`.
3. **Real Data Migration Prerequisites:**
   - Resolve duplicate Excel sample ID `M017D` (rows 85 and 86).
   - Resolve 31 Excel entries lacking images and 8 raw images lacking Excel entries before running migration with `--apply`.
