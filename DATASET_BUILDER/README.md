# DATASET_BUILDER: Standalone Rice Dataset Extraction Pipeline

## Overview

`DATASET_BUILDER` is an independent, reproducible pipeline package in `Research-RICAI` for:
1. Extracting 31-feature tabular datasets from raw images and manual measurements with 1:1 parity to `CODE/RICE_VISION_MAIN_PIPELINE.ipynb`.
2. Migrating legacy hierarchical sample IDs (`M001/M001A.jpg`) to flat, collision-free naming (`1_Raw_Images/M0001.jpg`).
3. Generating canonical audit datasets (`ai_extracted_dataset.csv`) and training-ready datasets (`final_regression_dataset.csv`, `final_regression_dataset.xlsx`).

This package operates completely standalone and does not import logic from `CODE` or `AI_SERVICES` at runtime.

The existing files under `3_AI_Extracted` and `4_Final_Dataset` still contain
legacy `M001A`-style IDs. Run extraction on the canonical `M####` inputs before
using those outputs as the new training dataset.

## Canonical Raw Input Contract

- Every data row uses a unique ID in the configured `M####` format.
- Every raw image is stored directly in `DATASET_BUILDER/1_Raw_Images`; nested sample folders are rejected.
- The image stem must equal `Sample_ID` exactly, for example `M0001.jpg`.
- Batch extraction requires a one-to-one match between workbook rows and flat image files.
- Legacy IDs and physical grouping are retained only as audit columns
  (`Original_Sample_ID` and `Physical_Sample_ID`), not as filenames or folders.

---

## Directory Layout

```text
DATASET_BUILDER/
├── AI_DATASET_EXTRACTION_PIPELINE.ipynb   # Thin Colab orchestrator (12 sections)
├── SOURCE_SNAPSHOT.md                     # Evidence record of vendored vision modules
├── README.md                              # This documentation
├── config/
│   └── extraction_config.json             # Centralized scientific & path parameters
├── src/
│   └── rice_dataset/                      # Standalone Python package
│       ├── __init__.py
│       ├── config.py                      # Validated configuration loader
│       ├── contracts.py                   # 31-feature contract, column schemas, QC statuses
│       ├── feature_builder.py             # Raw-valid feature calculation & physical diagnostics
│       ├── pipeline.py                    # 15-step batch extraction orchestrator
│       ├── io/
│       │   ├── manual_records.py          # XLSX/CSV loader and validator
│       │   ├── image_index.py             # Flat image resolver and inventory validator
│       │   └── dataset_writer.py          # Atomic CSV/XLSX dataset writer
│       ├── migration/
│       │   ├── id_mapping.py              # Natural sorting and deterministic M#### mapping
│       │   └── migrate_ids.py             # Migration engine with dry-run, staging, and rollback
│       └── vision/                        # Vendored vision & geometry modules
│           ├── container_detector.py
│           ├── grain_segmenter.py
│           ├── grain_crop_cleaner.py
│           ├── grain_classifier.py
│           ├── ellipsoid_geometry.py
│           ├── uniformity_evaluator.py
│           └── grain_size_filter.py
├── scripts/
│   ├── migrate_sample_ids.py              # Legacy ID migration CLI
│   ├── reconcile_flat_raw_data.py         # Reconciliation and rollback for legacy raw data
│   ├── extract_dataset.py                 # CLI for batch extraction
│   └── verify_pipeline_parity.py          # Parity and contract verification script
├── migrations/                            # Staging, manifests, and dry-run reports
├── 1_Raw_Images/                          # Raw sample images
├── 2_Manual_Records/                      # Manual measurement workbooks
├── 3_AI_Extracted/                        # Canonical audit dataset output
└── 4_Final_Dataset/                       # Canonical training-ready dataset output
```

---

## Scientific Parameters & Constants

All parameters are centralized in `config/extraction_config.json`:

| Parameter | Value | Scope |
|---|---|---|
| `yolo_model_load_confidence` | `0.70` | Model loading confidence threshold |
| `yolo_result_confidence` | `0.50` | Detection score filter |
| `sahi_slice_size` | `640` | SAHI tile slice height/width |
| `sahi_overlap_ratio` | `0.25` | Overlap ratio between adjacent slices |
| `sahi_min_area_px` | `50` | Minimum bounding polygon area |
| `clean_step1` | `k=5, neck=0.15, area=35, sever=False` | First-pass coarse cleaning |
| `clean_step2` | `k=3, neck=0.15, area=25, sever=False, fill=True` | Second-pass fine cleaning |
| `cnn_whole_confidence` | `0.90` | DenseNet121 threshold for `hat_nguyen` |
| `whole_grain_thickness_ratio` | `0.80` | $c = 0.80 \times b$ (caliper-verified) |
| `physical_packing_fraction` | `0.55` | Used exclusively for physical diagnostic estimate |
| `trained_hybrid_packing_fraction` | `0.62` | Used exclusively for feature 11 (`Estimated_Total_Seeds_Hybrid`) |
| `size_filter_k` | `0.1` | Size filter IQR fence for physical diagnostic subset |

---

## 31-Feature Output Contract

1. `Bulk_Rice_Volume_mm3`
2. `Rice_Height_mm`
3. `Weight_g`
4. `Empty_Height_mm`
5. `Pixels_Per_mm`
6. `Container_Detected_Diam_px`
7. `Inner_Diameter_mm`
8. `Container_Height_mm`
9. `Whole_Grains_Count`
10. `Uniformity_Rate_Pct`
11. `Estimated_Total_Seeds_Hybrid`
12–15. `Grain_Length_mm_Mean`, `_Min`, `_Max`, `_Std`
16–19. `Grain_Width_mm_Mean`, `_Min`, `_Max`, `_Std`
20–23. `Grain_Thickness_mm_Mean`, `_Min`, `_Max`, `_Std`
24–27. `Grain_Area_mm2_Mean`, `_Min`, `_Max`, `_Std`
28–31. `Grain_Volume_mm3_Mean`, `_Min`, `_Max`, `_Std`

> **Note on Target Isolation:** `Actual_Count` is stored as a target column after all diagnostics, and is NEVER included in feature vectors or vision processing.

---

## Command-Line Usage

### 1. Verification of Parity & Contracts

Run static contract checks and fixture tests:
```bash
python DATASET_BUILDER/scripts/verify_pipeline_parity.py --static
```

### 2. Sample ID Migration (Dry-Run & Reports)

The commands in sections 2-3 are for archived legacy inputs only. The current
261-row workbook and flat images are already reconciled; do not reapply an ID
migration to them.

Inspect an archived legacy workbook and image folder without changing them:
```bash
python DATASET_BUILDER/scripts/migrate_sample_ids.py \
  --workbook path/to/legacy_manual_data.xlsx \
  --images path/to/legacy_raw_images \
  --prefix M --digits 4 --start 1 --dry-run
```

This generates five diagnostic reports in `DATASET_BUILDER/migrations/`:
- `id_mapping.csv`
- `duplicate_excel_ids.csv`
- `excel_without_image.csv`
- `images_without_excel.csv`
- `migration_report.json`

### 3. Applying Migration (When Data Conflicts Are Resolved)

```bash
python DATASET_BUILDER/scripts/migrate_sample_ids.py \
  --workbook path/to/legacy_manual_data.xlsx \
  --images path/to/legacy_raw_images \
  --prefix M --digits 4 --start 1 --apply
```

To roll back an applied migration:
```bash
python DATASET_BUILDER/scripts/migrate_sample_ids.py \
  --workbook path/to/legacy_manual_data.xlsx \
  --rollback DATASET_BUILDER/migrations/migration_manifest_<timestamp>.json
```

For a legacy inventory with duplicate workbook IDs, workbook-only rows, or
image-only captures, first build a reconciliation plan from the images that
actually exist:

```bash
python DATASET_BUILDER/scripts/reconcile_flat_raw_data.py --dry-run
```

Apply the reviewed plan with staging, SHA-256 verification, backup and rollback
manifest:

```bash
python DATASET_BUILDER/scripts/reconcile_flat_raw_data.py --apply
```

The reconciliation tool copies measurements only within the same physical
sample and only when those manual measurements are identical. Workbook rows
without an image are exported to `excluded_manual_rows.csv` and remain
recoverable from the migration backup.

### 4. Running Batch Extraction

Validate the canonical flat inventory without loading YOLO or CNN models:

```bash
python DATASET_BUILDER/scripts/extract_dataset.py --validate-only
```

Process all rows:
```bash
python DATASET_BUILDER/scripts/extract_dataset.py \
  --config DATASET_BUILDER/config/extraction_config.json
```

Process single sample:
```bash
python DATASET_BUILDER/scripts/extract_dataset.py \
  --sample-id M0001
```

---

## Google Colab Usage

Open `DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb` in Google Colab (with a GPU/T4 runtime).
Run all cells sequentially:
- Section 1: Mounts Drive and sets `PROJECT_ROOT`.
- Section 2-3: Sets up `sys.path` and installs minimal dependencies.
- Section 4-5: Loads configuration and preflights models.
- Section 6-7: Validates the flat M#### workbook/image inventory.
- Section 8-10: Initializes models once and runs batch extraction with per-sample progress.
- Section 11-12: Exports datasets atomically and runs verification.
