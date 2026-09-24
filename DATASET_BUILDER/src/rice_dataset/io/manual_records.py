"""Reader and schema validator for manual measurement records (Excel and CSV)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ManualRecord:
    sample_id: str
    original_sample_id: str
    physical_sample_id: str
    weight_g: float
    container_height_mm: float
    inner_diameter_mm: float
    empty_height_mm: float
    rice_height_mm: float
    actual_count: Optional[int] = None
    image_filename: Optional[str] = None
    capture_timestamp: Optional[str] = None
    row_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Sample_ID": self.sample_id,
            "Original_Sample_ID": self.original_sample_id,
            "Physical_Sample_ID": self.physical_sample_id,
            "Weight_g": self.weight_g,
            "Container_Height_mm": self.container_height_mm,
            "Inner_Diameter_mm": self.inner_diameter_mm,
            "Empty_Height_mm": self.empty_height_mm,
            "Rice_Height_mm": self.rice_height_mm,
            "Actual_Count": self.actual_count,
            "Image_Filename": self.image_filename,
            "Capture_Timestamp": self.capture_timestamp,
            "_row_index": self.row_index,
        }


def derive_legacy_physical_id(sample_id: str) -> str:
    """Derive physical sample ID from legacy pattern (e.g. M001A -> M001)."""
    clean_id = sample_id.strip().upper()
    match = re.match(r"^([A-Za-z]+\d+)[A-Za-z]*$", clean_id)
    if match:
        return match.group(1)
    return clean_id


def load_manual_records(
    file_path: Path | str,
    sheet_name: Optional[str | int] = 0,
) -> Tuple[List[ManualRecord], List[Dict[str, Any]]]:
    """Load manual records from XLSX or CSV with strict schema validation.

    Returns:
        Tuple of (valid_records, error_records)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Manual record file not found: {path}")

    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manual records format '{ext}'. Expected .xlsx or .csv")

    if df.empty:
        return [], []

    # Map column names case-insensitively
    col_map: Dict[str, str] = {}
    for col in df.columns:
        clean_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        col_map[clean_col] = col

    def get_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    sample_id_col = get_col(["sample_id", "sampleid", "id"])
    weight_col = get_col(["weight_g", "weight", "input_weight_g", "can_nang_g"])
    container_h_col = get_col(["container_height_mm", "container_h_mm", "container_height", "chieu_cao_ly_mm"])
    inner_d_col = get_col(["inner_diameter_mm", "inner_diam_mm", "inner_d_mm", "duong_kinh_trong_mm"])
    empty_h_col = get_col(["empty_height_mm", "empty_h_mm", "empty_height", "chieu_cao_hut_mm"])
    rice_h_col = get_col(["rice_height_mm", "rice_h_mm", "rice_height", "chieu_cao_lua_mm"])
    actual_count_col = get_col(["actual_count", "actual_seed_count", "count", "so_hat_that"])
    orig_id_col = get_col(["original_sample_id", "original_id", "legacy_sample_id"])
    phys_id_col = get_col(["physical_sample_id", "physical_id", "group_id"])
    image_fn_col = get_col(["image_filename", "filename", "image_name"])
    capture_ts_col = get_col(["capture_timestamp", "timestamp", "capture_time", "thoi_gian_chup"])

    required_missing = []
    if not sample_id_col:
        required_missing.append("Sample_ID")
    if not weight_col:
        required_missing.append("Weight_g")
    if not container_h_col:
        required_missing.append("Container_Height_mm")
    if not inner_d_col:
        required_missing.append("Inner_Diameter_mm")
    if not empty_h_col:
        required_missing.append("Empty_Height_mm")

    if required_missing:
        raise ValueError(f"Manual record workbook '{path.name}' is missing required columns: {required_missing}")

    records: List[ManualRecord] = []
    errors: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        raw_id = row[sample_id_col]
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            continue

        raw_id_str = str(raw_id).strip()
        normalized_id = raw_id_str.upper()

        orig_id = str(row[orig_id_col]).strip() if orig_id_col and pd.notna(row[orig_id_col]) else raw_id_str
        phys_id = str(row[phys_id_col]).strip() if phys_id_col and pd.notna(row[phys_id_col]) else derive_legacy_physical_id(orig_id)
        img_fn = str(row[image_fn_col]).strip() if image_fn_col and pd.notna(row[image_fn_col]) else None
        cap_ts = str(row[capture_ts_col]).strip() if capture_ts_col and pd.notna(row[capture_ts_col]) else None

        try:
            w_val = float(row[weight_col])
            c_h_val = float(row[container_h_col])
            in_d_val = float(row[inner_d_col])
            emp_h_val = float(row[empty_h_col])

            if rice_h_col and pd.notna(row[rice_h_col]):
                rice_h_val = float(row[rice_h_col])
            else:
                rice_h_val = max(0.0, c_h_val - emp_h_val)

            if w_val <= 0 or c_h_val <= 0 or in_d_val <= 0 or emp_h_val < 0 or rice_h_val <= 0:
                raise ValueError(
                    f"Physical dimensions must be positive: weight={w_val}, H={c_h_val}, "
                    f"D={in_d_val}, H_empty={emp_h_val}, H_rice={rice_h_val}"
                )

            act_cnt = None
            if actual_count_col and pd.notna(row[actual_count_col]):
                raw_act = row[actual_count_col]
                try:
                    act_f = float(raw_act)
                    if not act_f.is_integer():
                        raise ValueError(f"Actual_Count must be an integer, got: {raw_act}")
                    act_cnt = int(act_f)
                except ValueError as ve:
                    raise ValueError(f"Invalid Actual_Count '{raw_act}': {ve}")

            records.append(
                ManualRecord(
                    sample_id=normalized_id,
                    original_sample_id=orig_id,
                    physical_sample_id=phys_id,
                    weight_g=w_val,
                    container_height_mm=c_h_val,
                    inner_diameter_mm=in_d_val,
                    empty_height_mm=emp_h_val,
                    rice_height_mm=rice_h_val,
                    actual_count=act_cnt,
                    image_filename=img_fn,
                    capture_timestamp=cap_ts,
                    row_index=int(idx),
                )
            )
        except Exception as e:
            errors.append({
                "row_index": int(idx),
                "sample_id": normalized_id,
                "original_sample_id": orig_id,
                "physical_sample_id": phys_id,
                "image_filename": img_fn,
                "capture_timestamp": cap_ts,
                "error": str(e),
                "raw_row": row.to_dict(),
            })

    return records, errors
