"""Atomic dataset exporter for audit CSV and training-ready CSV/XLSX."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..contracts import CANONICAL_COLUMNS, QCStatus


def write_dataset_atomic(
    rows: List[Dict[str, Any]],
    output_path: Path | str,
    columns: List[str] = CANONICAL_COLUMNS,
    only_pass: bool = False,
) -> int:
    """Write tabular dataset atomically to CSV or XLSX via a temporary sibling file.

    Args:
        rows: List of row dictionaries.
        output_path: Target output path (.csv or .xlsx).
        columns: Canonical column order.
        only_pass: If True, filters only QC_Status == PASS rows.

    Returns:
        Number of rows exported.
    """
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if only_pass:
        filtered_rows = [r for r in rows if r.get("QC_Status") == QCStatus.PASS]
    else:
        filtered_rows = list(rows)

    df = pd.DataFrame(filtered_rows)

    # Ensure all canonical columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # Reorder strictly according to canonical column specification
    df = df[columns]

    ext = target.suffix.lower()
    temp_dir = target.parent
    temp_prefix = f".{target.stem}_tmp_"

    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, prefix=temp_prefix, suffix=ext) as tf:
        temp_path = Path(tf.name)

    try:
        if ext == ".csv":
            df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        elif ext in [".xlsx", ".xls"]:
            with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Dataset")
        else:
            raise ValueError(f"Unsupported export format: '{ext}'. Expected .csv or .xlsx")

        # Atomic replace
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return len(df)


def export_audit_and_training_datasets(
    rows: List[Dict[str, Any]],
    audit_csv_path: Path | str,
    final_csv_path: Path | str,
    final_xlsx_path: Path | str,
) -> Dict[str, int]:
    """Export canonical audit dataset and training-ready datasets atomically."""
    audit_count = write_dataset_atomic(rows, audit_csv_path, only_pass=False)
    training_csv_count = write_dataset_atomic(rows, final_csv_path, only_pass=True)
    training_xlsx_count = write_dataset_atomic(rows, final_xlsx_path, only_pass=True)

    return {
        "audit_total_rows": audit_count,
        "training_csv_rows": training_csv_count,
        "training_xlsx_rows": training_xlsx_count,
    }
