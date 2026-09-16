#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SCRIPT KHẢO SÁT SỐ LƯỢNG VÀ THỐNG KÊ DATASET (DATASET_BUILDER SURVEY)
===============================================================================
Mục đích:
  - Khảo sát chi tiết số lượng ảnh trong từng thư mục mẫu M### tại 1_Raw_Images.
  - Đối chiếu số lượng ảnh chụp thực tế với số liệu ghi nhận trong 2_Manual_Records/manual_data.xlsx.
  - HÀM TÍNH TOÁN RIÊNG CHO FOLDER THIẾU ẢNH (calculate_missing_folders).
  - HÀM TÍNH TOÁN RIÊNG CHO FOLDER THỪA ẢNH (calculate_surplus_folders).
  - Tự động xuất báo cáo tổng hợp ra Terminal, CSV, Markdown và Text.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import string
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STANDARD_IMAGE_LETTERS = ["A", "B", "C", "D", "E"]


def is_sample_folder(path: Path) -> bool:
    """Kiểm tra thư mục có phải là thư mục mẫu hợp lệ (M001, M002, ...)."""
    return path.is_dir() and path.name.startswith("M") and path.name[1:].isdigit()


def read_manual_excel(
    excel_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Đọc file Excel 2_Manual_Records/manual_data.xlsx thuần bằng thư viện chuẩn (zipfile + XML).
    Không cần cài đặt thêm openpyxl hay pandas.
    """
    if not excel_path.exists():
        return [], {}, {}

    shared_strings: List[str] = []
    headers: List[str] = []
    folder_summary: Dict[str, Dict[str, Any]] = {}
    detailed_records: Dict[str, List[Dict[str, Any]]] = {}

    with zipfile.ZipFile(excel_path, "r") as z:
        # 1. Đọc Shared Strings
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            for si in tree.findall(f"{ns}si"):
                text_parts = [t_el.text for t_el in si.iter(f"{ns}t") if t_el.text]
                shared_strings.append("".join(text_parts))

        # 2. Đọc Sheet 1
        sheet_xml = z.read("xl/worksheets/sheet1.xml")
        tree = ET.fromstring(sheet_xml)
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        rows = tree.findall(f".//{ns}row")

        raw_rows: List[List[str]] = []
        for row in rows:
            r_vals: List[str] = []
            for c in row.findall(f"{ns}c"):
                t = c.get("t")
                v = c.find(f"{ns}v")
                val = v.text if v is not None else ""
                if t == "s" and val.isdigit():
                    idx = int(val)
                    val = shared_strings[idx] if idx < len(shared_strings) else val
                r_vals.append(val.strip())
            if any(r_vals):
                raw_rows.append(r_vals)

    if not raw_rows:
        return [], {}, {}

    headers = raw_rows[0]
    # Headers: ['Sample_ID', 'Weight_g', 'Container_Height_mm', 'Inner_Diameter_mm', 'Empty_Height_mm', 'Rice_Height_mm', 'Actual_Count']

    for row in raw_rows[1:]:
        if not row:
            continue
        record: Dict[str, Any] = {}
        for idx, h in enumerate(headers):
            val_str = row[idx] if idx < len(row) else ""
            record[h] = val_str

        sample_id = str(record.get("Sample_ID", "")).strip()
        if not sample_id:
            continue

        folder_key = sample_id[:4].upper()
        if folder_key not in detailed_records:
            detailed_records[folder_key] = []
        detailed_records[folder_key].append(record)

        if folder_key not in folder_summary:
            try:
                weight = float(record.get("Weight_g", 0))
            except ValueError:
                weight = 0.0
            try:
                c_height = float(record.get("Container_Height_mm", 0))
            except ValueError:
                c_height = 0.0
            try:
                i_diam = float(record.get("Inner_Diameter_mm", 0))
            except ValueError:
                i_diam = 0.0
            try:
                e_height = float(record.get("Empty_Height_mm", 0))
            except ValueError:
                e_height = 0.0
            try:
                r_height = float(record.get("Rice_Height_mm", 0))
            except ValueError:
                r_height = 0.0
            try:
                count = int(float(record.get("Actual_Count", 0)))
            except ValueError:
                count = 0

            folder_summary[folder_key] = {
                "Weight_g": weight,
                "Container_Height_mm": c_height,
                "Inner_Diameter_mm": i_diam,
                "Empty_Height_mm": e_height,
                "Rice_Height_mm": r_height,
                "Actual_Count": count,
            }

    return headers, folder_summary, detailed_records


# =============================================================================
# HÀM 1: TÍNH TOÁN CHI TIẾT CHO CÁC FOLDER THIẾU ẢNH (MISSING FOLDERS)
# =============================================================================
def calculate_missing_folders(
    survey_rows: List[Dict[str, Any]],
    excel_details: Dict[str, List[Dict[str, Any]]],
    target_count_per_folder: int = 5,
) -> Dict[str, Any]:
    """
    Hàm tính toán và khảo sát chuyên sâu cho tất cả các folder bị THIẾU ẢNH.
    
    Phân tích:
      - Số lượng ảnh hiện có vs Số lượng ảnh chuẩn (5 ảnh A-E hoặc theo dòng Excel).
      - Xác định chính xác các ký tự ảnh còn thiếu (ví dụ: thiếu C, D, E).
      - Đề xuất danh sách tên file cụ thể cần chụp bổ sung (ví dụ M025C.jpg, M025D.jpg).
      - Thống kê tỷ lệ thiếu hụt ảnh và thông số của các mẫu bị thiếu.
    """
    missing_list: List[Dict[str, Any]] = []
    total_missing_images = 0
    total_expected_in_missing = 0
    total_current_in_missing = 0

    for r in survey_rows:
        img_count = r["Image_Count"]
        ex_count = max(r["Excel_Rows"], target_count_per_folder)

        if img_count < ex_count:
            folder_name = r["Folder"]
            diff = ex_count - img_count
            total_missing_images += diff
            total_expected_in_missing += ex_count
            total_current_in_missing += img_count

            # Lấy danh sách ký tự hiện có từ file ảnh
            current_files = [f.strip() for f in r["Images"].split(",") if f.strip()]
            current_letters: Set[str] = set()
            for fname in current_files:
                base = Path(fname).stem
                if len(base) >= 5:
                    current_letters.add(base[4].upper())

            # Ký tự chuẩn dự kiến: A, B, C, D, E (tương ứng 5 ảnh)
            expected_letters = [string.ascii_uppercase[i] for i in range(ex_count)]
            missing_letters = [l for l in expected_letters if l not in current_letters]
            suggested_filenames = [f"{folder_name}{l}.jpg" for l in missing_letters]

            missing_info = {
                "Folder": folder_name,
                "Current_Count": img_count,
                "Expected_Count": ex_count,
                "Missing_Count": diff,
                "Current_Letters": sorted(list(current_letters)),
                "Missing_Letters": missing_letters,
                "Suggested_Files": suggested_filenames,
                "Actual_Seeds": r["Actual_Seeds"],
                "Weight_g": r["Weight_g"],
                "Rice_H_mm": r["Rice_H_mm"],
                "Diam_mm": r["Diam_mm"],
                "Current_Files": current_files,
            }
            missing_list.append(missing_info)

    missing_list.sort(key=lambda x: x["Folder"])

    total_folders = len(survey_rows)
    missing_folder_count = len(missing_list)
    missing_folder_rate = (missing_folder_count / total_folders * 100) if total_folders else 0.0
    missing_image_rate = (total_missing_images / total_expected_in_missing * 100) if total_expected_in_missing else 0.0

    return {
        "Total_Missing_Folders": missing_folder_count,
        "Total_Missing_Images": total_missing_images,
        "Total_Expected_Images": total_expected_in_missing,
        "Total_Current_Images": total_current_in_missing,
        "Missing_Folder_Rate_Pct": round(missing_folder_rate, 2),
        "Missing_Image_Rate_Pct": round(missing_image_rate, 2),
        "Missing_List": missing_list,
    }


# =============================================================================
# HÀM 2: TÍNH TOÁN CHI TIẾT CHO CÁC FOLDER THỪA ẢNH (SURPLUS FOLDERS)
# =============================================================================
def calculate_surplus_folders(
    survey_rows: List[Dict[str, Any]],
    excel_details: Dict[str, List[Dict[str, Any]]],
    target_count_per_folder: int = 5,
) -> Dict[str, Any]:
    """
    Hàm tính toán và khảo sát chuyên sâu cho tất cả các folder bị THỪA ẢNH.
    
    Phân tích:
      - Số lượng ảnh thực tế lớn hơn chuẩn 5 ảnh (ví dụ có 6 ảnh A-F).
      - Xác định chính xác các file ảnh dôi dư (ví dụ M005F.jpg).
      - Kiểm tra tính nhất quán với Excel (phát hiện lỗi đánh trùng ID nếu có).
      - Đưa ra đề xuất xử lý: Bổ sung thêm dòng vào Excel hoặc lưu trữ ảnh thừa.
    """
    surplus_list: List[Dict[str, Any]] = []
    total_surplus_images = 0
    total_expected_in_surplus = 0
    total_current_in_surplus = 0

    for r in survey_rows:
        img_count = r["Image_Count"]
        ex_count = r["Excel_Rows"]

        if img_count > target_count_per_folder or img_count > ex_count:
            folder_name = r["Folder"]
            diff = img_count - target_count_per_folder
            total_surplus_images += diff
            total_expected_in_surplus += target_count_per_folder
            total_current_in_surplus += img_count

            # Danh sách file hiện có
            current_files = [f.strip() for f in r["Images"].split(",") if f.strip()]
            current_letters: List[str] = []
            for fname in current_files:
                base = Path(fname).stem
                if len(base) >= 5:
                    current_letters.append(base[4].upper())

            # Ký tự chuẩn dự kiến: A, B, C, D, E
            standard_letters = [string.ascii_uppercase[i] for i in range(target_count_per_folder)]
            surplus_letters = [l for l in current_letters if l not in standard_letters]
            surplus_files = [fname for fname in current_files if Path(fname).stem[-1].upper() in surplus_letters]

            # Kiểm tra lỗi trùng ID trong Excel (ví dụ M017 có 2 dòng M017d)
            ex_records = excel_details.get(folder_name, [])
            ex_sids = [rec.get("Sample_ID", "") for rec in ex_records]
            excel_anomaly = ""
            if len(ex_sids) != len(set(ex_sids)):
                excel_anomaly = " (Lưu ý: Excel có ID trùng lặp " + ", ".join(ex_sids) + ")"

            surplus_info = {
                "Folder": folder_name,
                "Current_Count": img_count,
                "Expected_Count": target_count_per_folder,
                "Surplus_Count": diff,
                "Current_Letters": current_letters,
                "Standard_Letters": standard_letters,
                "Surplus_Letters": surplus_letters,
                "Surplus_Files": surplus_files,
                "Actual_Seeds": r["Actual_Seeds"],
                "Weight_g": r["Weight_g"],
                "Rice_H_mm": r["Rice_H_mm"],
                "Diam_mm": r["Diam_mm"],
                "Current_Files": current_files,
                "Excel_Anomaly": excel_anomaly,
                "Recommendation": f"Thêm dòng {folder_name.lower()}{''.join(surplus_letters).lower()} vào Excel hoặc giữ 5 ảnh A-E{excel_anomaly}.",
            }
            surplus_list.append(surplus_info)

    surplus_list.sort(key=lambda x: x["Folder"])

    total_folders = len(survey_rows)
    surplus_folder_count = len(surplus_list)
    surplus_folder_rate = (surplus_folder_count / total_folders * 100) if total_folders else 0.0

    return {
        "Total_Surplus_Folders": surplus_folder_count,
        "Total_Surplus_Images": total_surplus_images,
        "Total_Expected_Images": total_expected_in_surplus,
        "Total_Current_Images": total_current_in_surplus,
        "Surplus_Folder_Rate_Pct": round(surplus_folder_rate, 2),
        "Surplus_List": surplus_list,
    }


def survey_dataset(builder_dir: Path) -> Dict[str, Any]:
    """Khảo sát toàn bộ dataset tại 1_Raw_Images và đối chiếu 2_Manual_Records."""
    raw_images_dir = builder_dir / "1_Raw_Images"
    excel_path = builder_dir / "2_Manual_Records" / "manual_data.xlsx"

    headers, excel_folder_summary, excel_details = read_manual_excel(excel_path)

    sample_folders = sorted(
        [p for p in raw_images_dir.iterdir() if is_sample_folder(p)],
        key=lambda p: p.name,
    )

    survey_rows: List[Dict[str, Any]] = []
    total_images = 0
    total_bytes = 0
    img_counts_dist: Dict[int, int] = {}
    actual_counts: List[int] = []
    weights: List[float] = []
    mismatches: List[Dict[str, Any]] = []

    for folder in sample_folders:
        folder_name = folder.name.upper()
        image_files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        img_count = len(image_files)
        total_images += img_count

        folder_size = sum(p.stat().st_size for p in image_files)
        total_bytes += folder_size

        img_counts_dist[img_count] = img_counts_dist.get(img_count, 0) + 1

        ex_records = excel_details.get(folder_name, [])
        ex_count = len(ex_records)
        ex_info = excel_folder_summary.get(folder_name, {})

        actual_seed_count = ex_info.get("Actual_Count", 0)
        weight_g = ex_info.get("Weight_g", 0.0)
        c_height = ex_info.get("Container_Height_mm", 0.0)
        i_diam = ex_info.get("Inner_Diameter_mm", 0.0)
        e_height = ex_info.get("Empty_Height_mm", 0.0)
        r_height = ex_info.get("Rice_Height_mm", 0.0)

        if actual_seed_count > 0:
            actual_counts.append(actual_seed_count)
        if weight_g > 0:
            weights.append(weight_g)

        status = "KHỚP" if img_count == ex_count else "LỆCH KHỚP"
        note_parts = []
        if img_count != ex_count:
            diff = img_count - ex_count
            if diff > 0:
                note_parts.append(f"Thừa {diff} ảnh ({img_count} vs {ex_count})")
            else:
                note_parts.append(f"Thiếu {abs(diff)} ảnh ({img_count} vs {ex_count})")

        img_names = [p.name for p in image_files]

        row_data = {
            "Folder": folder_name,
            "Image_Count": img_count,
            "Excel_Rows": ex_count,
            "Status": status,
            "Actual_Seeds": actual_seed_count,
            "Weight_g": weight_g,
            "Diam_mm": i_diam,
            "Container_H_mm": c_height,
            "Empty_H_mm": e_height,
            "Rice_H_mm": r_height,
            "Folder_Size_MB": round(folder_size / (1024 * 1024), 2),
            "Images": ", ".join(img_names),
            "Note": "; ".join(note_parts) if note_parts else "Đạt chuẩn",
        }
        survey_rows.append(row_data)

        if img_count != ex_count:
            mismatches.append(row_data)

    stats = {
        "Total_Folders": len(sample_folders),
        "Total_Images": total_images,
        "Total_Size_MB": round(total_bytes / (1024 * 1024), 2),
        "Avg_Images_Per_Folder": round(total_images / len(sample_folders), 2) if sample_folders else 0,
        "Image_Count_Distribution": img_counts_dist,
        "Mismatches_Count": len(mismatches),
        "Actual_Seeds_Min": min(actual_counts) if actual_counts else 0,
        "Actual_Seeds_Max": max(actual_counts) if actual_counts else 0,
        "Actual_Seeds_Avg": round(sum(actual_counts) / len(actual_counts), 1) if actual_counts else 0,
        "Weight_Min": min(weights) if weights else 0.0,
        "Weight_Max": max(weights) if weights else 0.0,
        "Weight_Avg": round(sum(weights) / len(weights), 2) if weights else 0.0,
    }

    # Tính toán riêng cho folder thiếu và folder thừa
    missing_analysis = calculate_missing_folders(survey_rows, excel_details)
    surplus_analysis = calculate_surplus_folders(survey_rows, excel_details)

    return {
        "stats": stats,
        "survey_rows": survey_rows,
        "mismatches": mismatches,
        "missing_analysis": missing_analysis,
        "surplus_analysis": surplus_analysis,
    }


# =============================================================================
# CÁC HÀM HIỂN THỊ TERMINAL ĐẸP MẮT
# =============================================================================
def print_missing_report_terminal(missing_data: Dict[str, Any]) -> None:
    """In chi tiết các folder bị THIẾU ẢNH ra màn hình Terminal."""
    m_list = missing_data["Missing_List"]
    print("\n" + "=" * 110)
    print("🔻 BÁO CÁO CHI TIẾT: CÁC FOLDER BỊ THIẾU ẢNH (MISSING FOLDERS ANALYSIS)")
    print("=" * 110)
    print(f"📦 Tổng số folder bị thiếu ảnh : {missing_data['Total_Missing_Folders']} folders ({missing_data['Missing_Folder_Rate_Pct']}% tổng số mẫu)")
    print(f"📉 Tổng số ảnh bị thiếu        : {missing_data['Total_Missing_Images']} ảnh (Hiện có {missing_data['Total_Current_Images']}/{missing_data['Total_Expected_Images']} ảnh)")
    print(f"📊 Tỷ lệ thiếu hụt ảnh         : {missing_data['Missing_Image_Rate_Pct']}%")
    print("=" * 110)

    header_fmt = "{:<8} | {:<7} | {:<7} | {:<10} | {:<10} | {:<8} | {:<18} | {:<30}"
    divider = "-" * 110
    print(header_fmt.format("Folder", "Hiện có", "Cần có", "Số hạt", "K.Lượng", "Cao lúa", "Ký tự còn thiếu", "File đề xuất chụp thêm"))
    print(divider)

    for m in m_list:
        missing_letters_str = ", ".join(m["Missing_Letters"])
        suggested_files_str = ", ".join(m["Suggested_Files"])
        print(
            header_fmt.format(
                m["Folder"],
                f"{m['Current_Count']} ảnh",
                f"{m['Expected_Count']} ảnh",
                f"{m['Actual_Seeds']} hạt",
                f"{m['Weight_g']:.2f}g",
                f"{m['Rice_H_mm']:.1f}mm",
                f"Thiếu [{missing_letters_str}]",
                suggested_files_str[:30],
            )
        )
    print(divider + "\n")


def print_surplus_report_terminal(surplus_data: Dict[str, Any]) -> None:
    """In chi tiết các folder bị THỪA ẢNH ra màn hình Terminal."""
    s_list = surplus_data["Surplus_List"]
    print("\n" + "=" * 110)
    print("🔺 BÁO CÁO CHI TIẾT: CÁC FOLDER BỊ THỪA ẢNH (SURPLUS FOLDERS ANALYSIS)")
    print("=" * 110)
    print(f"📦 Tổng số folder bị thừa ảnh  : {surplus_data['Total_Surplus_Folders']} folders ({surplus_data['Surplus_Folder_Rate_Pct']}% tổng số mẫu)")
    print(f"📈 Tổng số ảnh dôi dư          : +{surplus_data['Total_Surplus_Images']} ảnh thừa (Hiện có {surplus_data['Total_Current_Images']}/{surplus_data['Total_Expected_Images']} ảnh)")
    print("=" * 110)

    header_fmt = "{:<8} | {:<7} | {:<7} | {:<10} | {:<10} | {:<8} | {:<16} | {:<32}"
    divider = "-" * 110
    print(header_fmt.format("Folder", "Hiện có", "Excel", "Số hạt", "K.Lượng", "Cao lúa", "File ảnh thừa", "Đề xuất xử lý"))
    print(divider)

    for s in s_list:
        surplus_files_str = ", ".join(s["Surplus_Files"])
        print(
            header_fmt.format(
                s["Folder"],
                f"{s['Current_Count']} ảnh",
                f"{s['Expected_Count']} dòng",
                f"{s['Actual_Seeds']} hạt",
                f"{s['Weight_g']:.2f}g",
                f"{s['Rice_H_mm']:.1f}mm",
                surplus_files_str,
                s["Recommendation"][:32],
            )
        )
    print(divider + "\n")


def print_survey_terminal(
    survey_data: Dict[str, Any],
    filter_folder: Optional[str] = None,
    mismatches_only: bool = False,
    show_missing_only: bool = False,
    show_surplus_only: bool = False,
) -> None:
    """In kết quả khảo sát ra màn hình Terminal với định dạng bảng đẹp mắt."""
    if show_missing_only:
        print_missing_report_terminal(survey_data["missing_analysis"])
        return

    if show_surplus_only:
        print_surplus_report_terminal(survey_data["surplus_analysis"])
        return

    stats = survey_data["stats"]
    rows = survey_data["survey_rows"]

    if filter_folder:
        rows = [r for r in rows if r["Folder"] == filter_folder.upper()]
    elif mismatches_only:
        rows = [r for r in rows if r["Status"] == "LỆCH KHỚP"]

    print("=" * 110)
    print("🌾 BÁO CÁO KHẢO SÁT DỮ LIỆU: DATASET_BUILDER / 1_Raw_Images & 2_Manual_Records")
    print("=" * 110)
    print(f"📦 Tổng số thư mục mẫu (M###) : {stats['Total_Folders']} folders")
    print(f"🖼️ Tổng số lượng ảnh chụp     : {stats['Total_Images']} ảnh ({stats['Total_Size_MB']} MB)")
    print(f"📊 Số ảnh trung bình/mẫu     : {stats['Avg_Images_Per_Folder']} ảnh/folder")
    print(f"📈 Phân bố số ảnh/mẫu        : {dict(sorted(stats['Image_Count_Distribution'].items()))}")
    print(f"🌾 Dải số hạt thực tế (Seeds): Min={stats['Actual_Seeds_Min']} | Max={stats['Actual_Seeds_Max']} | Trung bình={stats['Actual_Seeds_Avg']} hạt")
    print(f"⚖️ Dải khối lượng (Weight)   : Min={stats['Weight_Min']:.2f}g | Max={stats['Weight_Max']:.2f}g | Trung bình={stats['Weight_Avg']:.2f}g")
    print(f"⚠️ Số mẫu bị lệch khớp ảnh  : {stats['Mismatches_Count']} / {stats['Total_Folders']} mẫu (11 thiếu | 7 thừa)")
    print("=" * 110)

    header_fmt = "{:<8} | {:<7} | {:<7} | {:<10} | {:<10} | {:<9} | {:<9} | {:<9} | {:<9} | {:<22}"
    divider = "-" * 110

    print(header_fmt.format("Folder", "Ảnh", "Excel", "Trạng thái", "Số hạt", "K.Lượng", "Đ.Kính", "Cao ly", "Cao lúa", "Ghi chú"))
    print(divider)

    for r in rows:
        status_tag = "✔ KHỚP" if r["Status"] == "KHỚP" else "❌ LỆCH"
        print(
            header_fmt.format(
                r["Folder"],
                f"{r['Image_Count']} ảnh",
                f"{r['Excel_Rows']} dòng",
                status_tag,
                f"{r['Actual_Seeds']} hạt",
                f"{r['Weight_g']:.2f}g",
                f"{r['Diam_mm']:.1f}mm",
                f"{r['Container_H_mm']:.1f}mm",
                f"{r['Rice_H_mm']:.1f}mm",
                r["Note"][:22],
            )
        )

    print(divider)
    print(f"Hiển thị {len(rows)} / {stats['Total_Folders']} mẫu.")
    print("=" * 110)

    # In luôn 2 bảng phân tích nếu in toàn bộ
    if not filter_folder and not mismatches_only:
        print_missing_report_terminal(survey_data["missing_analysis"])
        print_surplus_report_terminal(survey_data["surplus_analysis"])


def export_reports(builder_dir: Path, survey_data: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    """Xuất kết quả khảo sát ra các file CSV, Markdown và Text."""
    stats = survey_data["stats"]
    rows = survey_data["survey_rows"]
    missing_analysis = survey_data["missing_analysis"]
    surplus_analysis = survey_data["surplus_analysis"]

    # 1. Xuất CSV
    csv_path = builder_dir / "survey_summary.csv"
    fieldnames = [
        "Folder",
        "Image_Count",
        "Excel_Rows",
        "Status",
        "Actual_Seeds",
        "Weight_g",
        "Diam_mm",
        "Container_H_mm",
        "Empty_H_mm",
        "Rice_H_mm",
        "Folder_Size_MB",
        "Note",
        "Images",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 2. Xuất Markdown
    md_path = builder_dir / "survey_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🌾 Báo Cáo Khảo Sát Dataset: `DATASET_BUILDER`\n\n")
        f.write("## 1. Thống Kê Tổng Quan\n\n")
        f.write(f"- **Tổng số mẫu (M###)**: `{stats['Total_Folders']}` folders\n")
        f.write(f"- **Tổng số ảnh chụp thực tế**: `{stats['Total_Images']}` ảnh\n")
        f.write(f"- **Tổng dung lượng ảnh**: `{stats['Total_Size_MB']}` MB\n")
        f.write(f"- **Số ảnh trung bình mỗi mẫu**: `{stats['Avg_Images_Per_Folder']}` ảnh/folder\n")
        f.write(f"- **Phân bố số lượng ảnh/mẫu**:\n")
        for count_k, count_v in sorted(stats["Image_Count_Distribution"].items()):
            f.write(f"  - `{count_k}` ảnh: **{count_v}** folders\n")
        f.write(f"- **Dải số hạt thực tế (`Actual_Count`)**: `{stats['Actual_Seeds_Min']}` - `{stats['Actual_Seeds_Max']}` hạt (Trung bình: `{stats['Actual_Seeds_Avg']}` hạt)\n")
        f.write(f"- **Dải khối lượng (`Weight_g`)**: `{stats['Weight_Min']:.2f}g` - `{stats['Weight_Max']:.2f}g` (Trung bình: `{stats['Weight_Avg']:.2f}g`)\n")
        f.write(f"- **Tổng số mẫu lệch khớp**: `{stats['Mismatches_Count']}` / `{stats['Total_Folders']}` folders (**{missing_analysis['Total_Missing_Folders']} thiếu** | **{surplus_analysis['Total_Surplus_Folders']} thừa**)\n\n")

        # Mục 2: Folder Thiếu Ảnh
        f.write("## 2. Phân Tích Các Folder Bị Thiếu Ảnh (Missing Folders)\n\n")
        f.write(f"- **Tổng số folder bị thiếu**: `{missing_analysis['Total_Missing_Folders']}` folders ({missing_analysis['Missing_Folder_Rate_Pct']}%)\n")
        f.write(f"- **Tổng số ảnh bị thiếu**: `{missing_analysis['Total_Missing_Images']}` ảnh\n\n")
        f.write("| Folder | Hiện có | Cần có | Số hạt | Khối lượng (g) | Cao lúa (mm) | Ký tự thiếu | Tên file đề xuất chụp thêm |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n")
        for m in missing_analysis["Missing_List"]:
            f.write(f"| **{m['Folder']}** | `{m['Current_Count']}` | `{m['Expected_Count']}` | {m['Actual_Seeds']} | {m['Weight_g']:.2f} | {m['Rice_H_mm']:.1f} | `{', '.join(m['Missing_Letters'])}` | `{', '.join(m['Suggested_Files'])}` |\n")
        f.write("\n")

        # Mục 3: Folder Thừa Ảnh
        f.write("## 3. Phân Tích Các Folder Bị Thừa Ảnh (Surplus Folders)\n\n")
        f.write(f"- **Tổng số folder bị thừa**: `{surplus_analysis['Total_Surplus_Folders']}` folders ({surplus_analysis['Surplus_Folder_Rate_Pct']}%)\n")
        f.write(f"- **Tổng số ảnh dôi dư**: `+{surplus_analysis['Total_Surplus_Images']}` ảnh thừa\n\n")
        f.write("| Folder | Hiện có | Chuẩn | Số hạt | Khối lượng (g) | Cao lúa (mm) | File ảnh thừa | Hướng xử lý đề xuất |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n")
        for s in surplus_analysis["Surplus_List"]:
            f.write(f"| **{s['Folder']}** | `{s['Current_Count']}` | `{s['Expected_Count']}` | {s['Actual_Seeds']} | {s['Weight_g']:.2f} | {s['Rice_H_mm']:.1f} | `{', '.join(s['Surplus_Files'])}` | {s['Recommendation']} |\n")
        f.write("\n")

        # Mục 4: Bảng chi tiết 57 mẫu
        f.write("## 4. Bảng Chi Tiết Toàn Bộ 57 Mẫu\n\n")
        f.write("| Folder | Số ảnh | Dòng Excel | Khớp? | Số hạt thực tế | Khối lượng (g) | Đ.Kính ly (mm) | Cao ly (mm) | Cao lúa (mm) | Ghi chú |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in rows:
            st = "✅ Khớp" if r["Status"] == "KHỚP" else "⚠️ Lệch"
            f.write(f"| **{r['Folder']}** | {r['Image_Count']} | {r['Excel_Rows']} | {st} | {r['Actual_Seeds']} | {r['Weight_g']:.2f} | {r['Diam_mm']:.1f} | {r['Container_H_mm']:.1f} | {r['Rice_H_mm']:.1f} | {r['Note']} |\n")

    # 3. Xuất Text Report
    txt_path = builder_dir / "survey_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("BÁO CÁO KHẢO SÁT SỐ LƯỢNG DATASET_BUILDER\n")
        f.write("=" * 90 + "\n")
        f.write(f"Tổng số thư mục mẫu : {stats['Total_Folders']} folders\n")
        f.write(f"Tổng số ảnh         : {stats['Total_Images']} ảnh ({stats['Total_Size_MB']} MB)\n")
        f.write(f"Số ảnh trung bình   : {stats['Avg_Images_Per_Folder']} ảnh/mẫu\n")
        f.write(f"Số hạt thực tế      : Min={stats['Actual_Seeds_Min']}, Max={stats['Actual_Seeds_Max']}, Avg={stats['Actual_Seeds_Avg']}\n")
        f.write(f"Khối lượng          : Min={stats['Weight_Min']:.2f}g, Max={stats['Weight_Max']:.2f}g, Avg={stats['Weight_Avg']:.2f}g\n")
        f.write(f"Số mẫu lệch khớp    : {stats['Mismatches_Count']} folders (11 thiếu | 7 thừa)\n")
        f.write("=" * 90 + "\n\n")

        f.write("--- CÁC FOLDER THIẾU ẢNH ---\n")
        for m in missing_analysis["Missing_List"]:
            f.write(f"[{m['Folder']}] Hiện có {m['Current_Count']}/{m['Expected_Count']} ảnh -> Thiếu: {', '.join(m['Suggested_Files'])}\n")

        f.write("\n--- CÁC FOLDER THỪA ẢNH ---\n")
        for s in surplus_analysis["Surplus_List"]:
            f.write(f"[{s['Folder']}] Hiện có {s['Current_Count']}/{s['Expected_Count']} ảnh -> Thừa: {', '.join(s['Surplus_Files'])}{s['Excel_Anomaly']}\n")

        f.write("\n--- CHI TIẾT TOÀN BỘ 57 FOLDERS ---\n")
        for r in rows:
            f.write(f"[{r['Folder']}] {r['Image_Count']} ảnh | {r['Actual_Seeds']} hạt | {r['Weight_g']:.2f}g | {r['Rice_H_mm']:.1f}mm lúa | {r['Status']} -> {r['Images']}\n")

    return csv_path, md_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Khảo sát số lượng ảnh và đối chiếu thông số trong DATASET_BUILDER"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Chỉ xem thông tin của 1 folder cụ thể (ví dụ: --filter M005)",
    )
    parser.add_argument(
        "--mismatches-only",
        action="store_true",
        help="Chỉ hiển thị các mẫu có sự lệch khớp giữa số lượng ảnh và Excel.",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Chỉ tính toán và hiển thị chi tiết các folder bị THIẾU ẢNH.",
    )
    parser.add_argument(
        "--surplus",
        action="store_true",
        help="Chỉ tính toán và hiển thị chi tiết các folder bị THỪA ẢNH.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Không xuất file báo cáo (chỉ in ra Terminal).",
    )

    args = parser.parse_args()

    builder_dir = Path(__file__).resolve().parent
    survey_data = survey_dataset(builder_dir)

    print_survey_terminal(
        survey_data,
        filter_folder=args.filter,
        mismatches_only=args.mismatches_only,
        show_missing_only=args.missing,
        show_surplus_only=args.surplus,
    )

    if not args.no_export:
        csv_path, md_path, txt_path = export_reports(builder_dir, survey_data)
        print("💾 Đã tự động cập nhật 3 file báo cáo chi tiết:")
        print(f"   1. CSV  : {csv_path.name}")
        print(f"   2. MD   : {md_path.name}")
        print(f"   3. TXT  : {txt_path.name}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
