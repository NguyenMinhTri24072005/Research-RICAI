# Commands and outcomes

Environment: Windows 11, PowerShell, CPU, `AI_SERVICES/.venv/Scripts/python.exe` unless otherwise noted.

1. `python -m py_compile CODE/reports/pipeline_investigation_20260921/investigate.py` — PASS.
2. `python CODE/reports/pipeline_investigation_20260921/investigate.py --mode tabular` — PASS after the audit script exposed stale `source_row_id` values and switched the evidence join to unique `Sample_ID`. Outputs: `tabular_audit.json`, `tabular_audit.log`, `grouped_baseline_predictions.csv`, `packing_holdout_predictions.csv`.
3. `python CODE/reports/pipeline_investigation_20260921/investigate.py --mode scale` — PASS. Six AI_SERVICES detector images plus five repeats. Outputs: `scale_audit.json`, `scale_audit.log`, `scale_comparison.csv`, `scale_*.jpg`.
4. `python CODE/reports/pipeline_investigation_20260921/audit_main_scale.py` — PASS. Six main-notebook detector images plus five M010a repeats. Outputs: `main_scale_audit.json`, `main_scale_audit.log`, `main_scale_runs.csv`, `main_scale_overlay_*.jpg`.
5. Bundled spreadsheet runtime read `DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.xlsx` — PASS; 285 rows / 254 FOUND and historical scale rows recovered. Output subset: `historical_removed_rows.csv`.
6. `python CODE/reports/pipeline_investigation_20260921/audit_removed_scale.py` — PASS. Rechecked M029c, M037e, M038e. Outputs: `removed_scale_audit.json`, `removed_scale_audit.log`, `removed_scale_overlay_*.jpg`.
7. `python CODE/reports/pipeline_investigation_20260921/audit_actual_case.py` — PASS. Parsed saved notebook outputs and independently recomputed the 450→802 arithmetic. Outputs: `actual_case_arithmetic.json`, `actual_case_arithmetic.log`.
8. `python -c "import tensorflow as tf; print(tf.__version__)"` — FAIL / NOT VERIFIED: `ModuleNotFoundError: No module named 'tensorflow.python'`. Therefore no local full YOLO+CNN E2E rerun was claimed.

Notable diagnostic failures retained in the narrative:

- Initial tabular audit found saved holdout `source_row_id` values 276–279 outside the current 276-row CSV. This demonstrated dataset drift; final audit joins by unique `Sample_ID` and reports that only 25/51 saved positions still match.
- Initial scale sample list included IDs removed from the current CSV. The old rows were subsequently recovered independently from the parallel 285-row XLSX and audited separately.
## Additional raw-fixture commands

9. `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_actual_raw.py` — PASS. Raw fixture SHA-256 `b51a23...1638`; notebook detector repeats `88.8384748 px/mm` 5/5 and reproduces saved `88.84`; AI_SERVICES detector returns wrong `28.0597232 px/mm` 5/5.
10. `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_actual_segmentation.py` — PASS on CPU. 221 SAHI slices, 143.43 s; raw SAHI `86`, cleaner `86`, geometry `86`. CNN inference intentionally NOT VERIFIED because TensorFlow import still fails. The script maps all 21 saved CNN-whole volume records back to the deterministic rerun and reproduces mean `6.618097 mm³` / physical `802`.
11. `AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_actual_geometry.py` — PASS. Cylinder `9655.37 mm³ → 802.41`; linear-frustum counterfactual `5440.85 mm³ → 452.16`. The latter is conditional on measuring/confirming a linear interior taper before deployment.