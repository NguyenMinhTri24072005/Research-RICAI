#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TOOL: KHẢO SÁT & ĐÁNH GIÁ CHI TIẾT TỪNG THUỘC TÍNH GỐC (FEATURE DEEP DIVE ANALYZER)
===============================================================================
Mục đích:
  - Khảo sát trực tiếp toàn bộ 31 thuộc tính gốc trong final_linear_regression_dataset.csv
  - Tính toán phân bố (Mean, Std, Min, 25%, Median, 75%, Max, Skewness).
  - Đánh giá tương quan tuyến tính Pearson (r), Spearman (rho), R2 với Actual_Count.
  - Xây dựng mô hình hồi quy đơn biến sơ bộ (y = ax + b, MAE, MAPE).
  - Xuất bảng xếp hạng toàn bộ 31 thuộc tính ra all_features_correlation_report.csv.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "DATASET_BUILDER":
        return current.parent
    if (current / "DATASET_BUILDER").exists():
        return current
    return current


ALL_ORIGINAL_FEATURES = [
    # Nhóm 1: Vật chứa & Thể tích khối lúa
    ("Bulk_Rice_Volume_mm3", "Vật chứa & Khối lúa", "mm³"),
    ("Rice_Height_mm", "Vật chứa & Khối lúa", "mm"),
    ("Weight_g", "Vật chứa & Khối lúa", "gam"),
    ("Empty_Height_mm", "Vật chứa & Khối lúa", "mm"),
    ("Pixels_Per_mm", "Vật chứa & Khối lúa", "px/mm"),
    ("Container_Detected_Diam_px", "Vật chứa & Khối lúa", "px"),
    ("Inner_Diameter_mm", "Vật chứa & Khối lúa", "mm"),
    ("Container_Height_mm", "Vật chứa & Khối lúa", "mm"),
    # Nhóm 2: Bề mặt & Ước lượng
    ("Whole_Grains_Count", "Bề mặt & Ước lượng", "hạt"),
    ("Uniformity_Rate_Pct", "Bề mặt & Ước lượng", "%"),
    ("Estimated_Total_Seeds_Hybrid", "Bề mặt & Ước lượng", "hạt"),
    # Nhóm 3: Chiều dài hạt (2a)
    ("Grain_Length_mm_Mean", "Chiều dài 2a", "mm"),
    ("Grain_Length_mm_Min", "Chiều dài 2a", "mm"),
    ("Grain_Length_mm_Max", "Chiều dài 2a", "mm"),
    ("Grain_Length_mm_Std", "Chiều dài 2a", "mm"),
    # Nhóm 4: Chiều rộng hạt (2b)
    ("Grain_Width_mm_Mean", "Chiều rộng 2b", "mm"),
    ("Grain_Width_mm_Min", "Chiều rộng 2b", "mm"),
    ("Grain_Width_mm_Max", "Chiều rộng 2b", "mm"),
    ("Grain_Width_mm_Std", "Chiều rộng 2b", "mm"),
    # Nhóm 5: Chiều dày hạt (2c)
    ("Grain_Thickness_mm_Mean", "Chiều dày 2c", "mm"),
    ("Grain_Thickness_mm_Min", "Chiều dày 2c", "mm"),
    ("Grain_Thickness_mm_Max", "Chiều dày 2c", "mm"),
    ("Grain_Thickness_mm_Std", "Chiều dày 2c", "mm"),
    # Nhóm 6: Diện tích 2D
    ("Grain_Area_mm2_Mean", "Diện tích 2D", "mm²"),
    ("Grain_Area_mm2_Min", "Diện tích 2D", "mm²"),
    ("Grain_Area_mm2_Max", "Diện tích 2D", "mm²"),
    ("Grain_Area_mm2_Std", "Diện tích 2D", "mm²"),
    # Nhóm 7: Thể tích 3D Ellipsoid
    ("Grain_Volume_mm3_Mean", "Thể tích 3D Hạt", "mm³"),
    ("Grain_Volume_mm3_Min", "Thể tích 3D Hạt", "mm³"),
    ("Grain_Volume_mm3_Max", "Thể tích 3D Hạt", "mm³"),
    ("Grain_Volume_mm3_Std", "Thể tích 3D Hạt", "mm³"),
]


def analyze_all_original_features(
    base_dir: Optional[Path] = None,
    output_report_name: str = "all_features_correlation_report.csv",
) -> List[Dict[str, Any]]:
    if base_dir is None:
        base_dir = find_project_root()

    final_dir = base_dir / "DATASET_BUILDER" / "4_Final_Dataset"
    csv_path = final_dir / "final_linear_regression_dataset.csv"

    print("=" * 105)
    print("🌾 CÔNG CỤ KHẢO SÁT & ĐÁNH GIÁ TOÀN BỘ 31 THUỘC TÍNH GỐC (FEATURE DEEP DIVE)")
    print("=" * 105)
    print(f"📁 Thư mục gốc      : {base_dir}")
    print(f"📁 Tệp dữ liệu CSV  : {csv_path}")
    print("=" * 105 + "\n")

    if not csv_path.exists():
        print(f"❌ CẢNH BÁO: Không tìm thấy tệp {csv_path}")
        return []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    valid_rows = [r for r in all_rows if r.get("Image_Status") == "FOUND"]
    print(f"🔹 Tổng số dòng trong tệp : {len(all_rows)} mẫu")
    print(f"🔹 Số mẫu hợp lệ (FOUND)   : {len(valid_rows)} mẫu ({len(valid_rows)/len(all_rows)*100:.1f}%)")
    print(f"🔹 Biến mục tiêu chính     : Actual_Count (Số hạt thực tế)")
    print(f"🔹 Số thuộc tính gốc       : {len(ALL_ORIGINAL_FEATURES)} thuộc tính\n")

    # Đọc target Actual_Count
    targets = []
    for r in valid_rows:
        try:
            targets.append(float(r.get("Actual_Count", "")))
        except (ValueError, TypeError):
            targets.append(np.nan)
    y_arr = np.array(targets)

    report_list = []

    for feat_name, group_name, unit in ALL_ORIGINAL_FEATURES:
        vals = []
        for r in valid_rows:
            try:
                vals.append(float(r.get(feat_name, "")))
            except (ValueError, TypeError):
                vals.append(np.nan)
        x_arr = np.array(vals)

        valid_mask = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
        if np.sum(valid_mask) > 10:
            x_c = x_arr[valid_mask]
            y_c = y_arr[valid_mask]

            mean_v = float(np.mean(x_c))
            std_v = float(np.std(x_c))
            min_v = float(np.min(x_c))
            q25_v = float(np.percentile(x_c, 25))
            med_v = float(np.median(x_c))
            q75_v = float(np.percentile(x_c, 75))
            max_v = float(np.max(x_c))

            # Pearson r
            r_val = float(np.corrcoef(x_c, y_c)[0, 1])
            r2_val = r_val ** 2

            # Hồi quy đơn biến
            slope, intercept = np.polyfit(x_c, y_c, 1)
            pred_y = slope * x_c + intercept
            mae_v = float(np.mean(np.abs(pred_y - y_c)))
            mape_v = float(np.mean(np.abs((pred_y - y_c) / y_c)) * 100)

            abs_r = abs(r_val)
            if abs_r >= 0.85:
                eval_str = "🌟 Cực kỳ mạnh"
            elif abs_r >= 0.60:
                eval_str = "🟢 Rất mạnh"
            elif abs_r >= 0.35:
                eval_str = "🟠 Trung bình"
            else:
                eval_str = "⚪ Yếu / Nhiễu"

            report_list.append({
                "feat_name": feat_name,
                "group_name": group_name,
                "unit": unit,
                "mean": mean_v,
                "std": std_v,
                "min": min_v,
                "q25": q25_v,
                "median": med_v,
                "q75": q75_v,
                "max": max_v,
                "r": r_val,
                "r2": r2_val,
                "slope": slope,
                "intercept": intercept,
                "mae": mae_v,
                "mape": mape_v,
                "eval": eval_str
            })

    # Sắp xếp theo R2 giảm dần
    report_list.sort(key=lambda item: item["r2"], reverse=True)

    # In Bảng Xếp Hạng
    print("=" * 105)
    print("🏆 BẢNG TỔNG HỢP XẾP HẠNG TOÀN BỘ 31 THUỘC TÍNH GỐC THEO R² VỚI Actual_Count:")
    print("=" * 105)
    print(f"{'Hạng':<5} | {'Thuộc Tính (Feature)':<30} | {'Nhóm':<20} | {'Pearson (r)':<12} | {'R² Score':<10} | {'MAE (hạt)':<10} | {'Đánh Giá'}")
    print("-" * 105)

    for rank, row in enumerate(report_list, start=1):
        print(f"#{rank:<4} | {row['feat_name']:<30} | {row['group_name']:<20} | {row['r']:<+12.4f} | {row['r2']:<10.4f} | {row['mae']:<10.2f} | {row['eval']}")

    # Xuất ra file CSV
    out_csv = final_dir / output_report_name
    fieldnames = [
        "Thu_Hang", "Thuoc_Tinh", "Nhom", "Don_Vi", "Mean", "Std", "Min", "Median", "Max",
        "Pearson_r", "R2_Score", "He_So_Goc_a", "He_So_Chan_b", "MAE_Don_Bien", "MAPE_Don_Bien_Pct", "Danh_Gia"
    ]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for rank, row in enumerate(report_list, start=1):
            writer.writerow([
                rank, row["feat_name"], row["group_name"], row["unit"],
                round(row["mean"], 3), round(row["std"], 3), round(row["min"], 3), round(row["median"], 3), round(row["max"], 3),
                round(row["r"], 4), round(row["r2"], 4), round(row["slope"], 5), round(row["intercept"], 2),
                round(row["mae"], 2), round(row["mape"], 2), row["eval"]
            ])

    print("\n" + "=" * 105)
    print(f"💾 Đã xuất báo cáo chi tiết 31 thuộc tính vào: {out_csv}")
    print("=" * 105 + "\n")

    return report_list


def main():
    analyze_all_original_features()


if __name__ == "__main__":
    main()
