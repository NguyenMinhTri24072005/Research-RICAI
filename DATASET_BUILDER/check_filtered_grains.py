#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TOOL: KIỂM TRA & THỐNG KÊ SỐ LƯỢNG HẠT NGUYÊN SAU KHI LỌC (GRAIN AUDIT TOOL)
===============================================================================
Mục đích:
  - Duyệt qua toàn bộ thư mục hạt nguyên đã bóc tách tại:
    DATASET_BUILDER/3_AI_Extracted/CROPPED_GRAINS/
  - Đếm chính xác số lượng ảnh hạt nguyên (clean whole grains) trong từng mã ảnh.
    Hỗ trợ cả dữ liệu cũ dạng M001/M001A.jpg và dữ liệu mới dạng phẳng M0001.jpg.
  - Cảnh báo các mã có 0 hạt hoặc quá ít hạt (< 3 hạt) để người dùng kiểm tra lại.
  - Xuất bảng báo cáo tổng hợp ra CSV và hiển thị bảng điều khiển trực quan.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Cấu hình encoding stdout để in tiếng Việt mượt mà trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def derive_sample_group(sample_id: str) -> str:
    """Trả về mã nhóm của ảnh legacy, nhưng giữ nguyên mã ảnh phẳng mới."""
    normalized = str(sample_id).strip().upper()
    if not normalized:
        return ""

    for separator in ("_", "-"):
        if separator in normalized:
            prefix, _separator, _suffix = normalized.partition(separator)
            return prefix or normalized

    # M001A hoặc M0001B thuộc nhóm M001 / M0001; M001 và M0001 là mã độc lập.
    legacy_match = re.fullmatch(r"([A-Z][A-Z0-9]*?\d+)([A-Z]+)", normalized)
    if legacy_match:
        return legacy_match.group(1)
    return normalized


def find_project_root() -> Path:
    """Tự động tìm thư mục gốc MAIN_SOURCES từ vị trí file script."""
    current = Path(__file__).resolve().parent
    if current.name == "DATASET_BUILDER":
        return current.parent
    if (current / "DATASET_BUILDER").exists():
        return current
    return current


def read_manual_sample_ids(manual_excel_path: Path) -> List[Dict[str, Any]]:
    """Đọc danh sách mã Sample_ID từ file Excel manual_data.xlsx không cần pandas."""
    if not manual_excel_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        shared_strings: List[str] = []
        raw_rows: List[List[str]] = []

        with zipfile.ZipFile(manual_excel_path, "r") as z:
            if "xl/sharedStrings.xml" in z.namelist():
                tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
                for si in tree.findall(f"{ns}si"):
                    parts = [t.text for t in si.iter(f"{ns}t") if t.text]
                    shared_strings.append("".join(parts))

            sheet_xml = z.read("xl/worksheets/sheet1.xml")
            tree = ET.fromstring(sheet_xml)
            ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            rows = tree.findall(f".//{ns}row")

            for row in rows:
                values_by_column: Dict[int, str] = {}
                last_column = -1
                for c in row.findall(f"{ns}c"):
                    cell_ref = c.get("r", "")
                    column_match = re.match(r"([A-Z]+)", cell_ref)
                    if not column_match:
                        continue

                    column_index = 0
                    for letter in column_match.group(1):
                        column_index = (column_index * 26) + (ord(letter) - ord("A") + 1)
                    column_index -= 1

                    cell_type = c.get("t")
                    value_node = c.find(f"{ns}v")
                    val = value_node.text if value_node is not None and value_node.text else ""
                    if cell_type == "s" and val.isdigit():
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    elif cell_type == "inlineStr":
                        val = "".join(t.text for t in c.iter(f"{ns}t") if t.text)

                    values_by_column[column_index] = val.strip()
                    last_column = max(last_column, column_index)

                r_vals = [values_by_column.get(i, "") for i in range(last_column + 1)]
                if any(r_vals):
                    raw_rows.append(r_vals)

        if raw_rows:
            headers = raw_rows[0]
            for row in raw_rows[1:]:
                rec: Dict[str, Any] = {}
                for i, h in enumerate(headers):
                    rec[h] = row[i] if i < len(row) else ""
                if rec.get("Sample_ID"):
                    records.append(rec)
    except Exception:
        pass

    return records


def audit_filtered_grains(
    base_dir: Optional[Path] = None,
    min_warning_threshold: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Thực hiện kiểm tra số lượng ảnh hạt nguyên sau khi lọc trong 3_AI_Extracted.
    """
    if base_dir is None:
        base_dir = find_project_root()

    builder_dir = base_dir / "DATASET_BUILDER"
    raw_images_dir = builder_dir / "1_Raw_Images"
    manual_excel_path = builder_dir / "2_Manual_Records" / "manual_data.xlsx"
    extracted_dir = builder_dir / "3_AI_Extracted"
    crops_dir = extracted_dir / "CROPPED_GRAINS"

    print("=" * 85)
    print("🌾 CÔNG CỤ KIỂM TRA SỐ LƯỢNG HẠT NGUYÊN SAU LỌC (GRAIN AUDIT TOOL)")
    print("=" * 85)
    print(f"📁 Thư mục gốc       : {base_dir}")
    print(f"📁 Thư mục hạt crops : {crops_dir}")
    print("=" * 85 + "\n")

    if not crops_dir.exists():
        print(f"❌ CẢNH BÁO: Không tìm thấy thư mục: {crops_dir}")
        print("   Vui lòng chạy Giai đoạn 1 của Pipeline để tạo các thư mục hạt nguyên trước.")
        return {}

    # 1. Thu thập danh sách mã mẫu từ thư mục hạt crops_dir
    sample_grain_counts: Dict[str, Dict[str, Any]] = {}

    for item in crops_dir.iterdir():
        if item.is_dir():
            sub_dirs = [d for d in item.iterdir() if d.is_dir()]
            if sub_dirs:
                # Dạng phân cấp: item = M001, sub_dirs = [M001A, M001B, ...]
                for sub in sorted(sub_dirs, key=lambda x: x.name):
                    img_files = sorted([f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
                    sample_grain_counts[sub.name.upper()] = {
                        "group": item.name.upper(),
                        "sample_id": sub.name.upper(),
                        "folder_path": str(sub),
                        "count": len(img_files),
                        "files": [f.name for f in img_files],
                    }
            else:
                # Dạng phẳng: item = M001A
                img_files = sorted([f for f in item.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
                group_name = derive_sample_group(item.name)
                sample_grain_counts[item.name.upper()] = {
                    "group": group_name,
                    "sample_id": item.name.upper(),
                    "folder_path": str(item),
                    "count": len(img_files),
                    "files": [f.name for f in img_files],
                }

    # 2. Thu thập danh sách tất cả các Sample_ID từ Excel hoặc từ thư mục ảnh thô
    manual_records = read_manual_sample_ids(manual_excel_path)
    all_sample_ids_set: Set[str] = set()

    if manual_records:
        for r in manual_records:
            if r.get("Sample_ID"):
                all_sample_ids_set.add(r["Sample_ID"].strip().upper())

    # Bổ sung các thư mục đã có trong crops_dir
    for sid in sample_grain_counts.keys():
        all_sample_ids_set.add(sid)

    # Bổ sung các file trong 1_Raw_Images. Capture App mới lưu ảnh trực tiếp
    # tại thư mục này (M0001.jpg), còn dữ liệu cũ nằm trong M001/M001A.jpg.
    raw_image_paths: Dict[str, List[Path]] = {}
    if raw_images_dir.exists():
        for raw_item in raw_images_dir.iterdir():
            if raw_item.is_file() and raw_item.suffix.lower() in IMAGE_EXTENSIONS:
                raw_image_paths.setdefault(raw_item.stem.upper(), []).append(raw_item)
            elif raw_item.is_dir() and raw_item.name.upper().startswith("M"):
                for f in raw_item.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        raw_image_paths.setdefault(f.stem.upper(), []).append(f)
                        all_sample_ids_set.add(f.stem.upper())

    all_sample_ids_set.update(raw_image_paths)

    all_sample_ids = sorted(list(all_sample_ids_set))

    audit_rows: List[Dict[str, Any]] = []
    total_grains = 0
    valid_samples_count = 0
    zero_grains_count = 0
    low_grains_count = 0
    missing_folder_count = 0

    grouped_stats: Dict[str, Dict[str, Any]] = {}

    for sid in all_sample_ids:
        group_key = derive_sample_group(sid)
        if group_key not in grouped_stats:
            grouped_stats[group_key] = {"total_grains": 0, "samples": {}}

        if sid in sample_grain_counts:
            info = sample_grain_counts[sid]
            cnt = info["count"]
            folder_exists = True
        else:
            cnt = 0
            folder_exists = False

        # Kiểm tra xem ảnh thô gốc có tồn tại không
        raw_paths = raw_image_paths.get(sid, [])
        raw_img_exists = bool(raw_paths)

        # Đánh giá trạng thái
        if not raw_img_exists and not folder_exists:
            status = "CHƯA CÓ ẢNH THÔ"
            status_symbol = "⚪"
        elif not folder_exists:
            status = "CHƯA BÓC TÁCH"
            status_symbol = "⏳"
            missing_folder_count += 1
        elif cnt == 0:
            status = "0 HẠT (ĐÃ XÓA HẾT / KHÔNG CÓ)"
            status_symbol = "🔴"
            zero_grains_count += 1
        elif cnt < min_warning_threshold:
            status = f"ÍT HẠT ({cnt} HẠT - CẦN LƯU Ý)"
            status_symbol = "🟠"
            low_grains_count += 1
            valid_samples_count += 1
            total_grains += cnt
        else:
            status = f"ĐẠT CHUẨN ({cnt} HẠT)"
            status_symbol = "🟢"
            valid_samples_count += 1
            total_grains += cnt

        row = {
            "Mã_Mẫu": sid,
            "Nhóm": group_key,
            "Số_Hạt_Nguyên": cnt if folder_exists else 0,
            "Trạng_Thái": status,
            "Biểu_Tượng": status_symbol,
            "Đã_Có_Ảnh_Gốc": "Có" if raw_img_exists else "Chưa",
            "Vị_Trí_Ảnh_Gốc": "; ".join(str(path.relative_to(raw_images_dir)) for path in raw_paths),
            "Đã_Tạo_Folder": "Có" if folder_exists else "Chưa",
        }
        audit_rows.append(row)
        grouped_stats[group_key]["total_grains"] += (cnt if folder_exists else 0)
        grouped_stats[group_key]["samples"][sid] = cnt if folder_exists else 0

    # 3. Xuất file báo cáo CSV
    report_csv_path = extracted_dir / "filtered_grains_audit_report.csv"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Mã_Mẫu", "Nhóm", "Số_Hạt_Nguyên", "Trạng_Thái", "Đã_Có_Ảnh_Gốc", "Vị_Trí_Ảnh_Gốc", "Đã_Tạo_Folder"]
    with open(report_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in audit_rows:
            row_dict = {k: r[k] for k in fieldnames}
            writer.writerow(row_dict)

    # 4. Hiển thị bảng tổng kết trên Console
    counts_list = [r["Số_Hạt_Nguyên"] for r in audit_rows if r["Số_Hạt_Nguyên"] > 0]
    avg_grains = (sum(counts_list) / len(counts_list)) if counts_list else 0.0
    min_grains = min(counts_list) if counts_list else 0
    max_grains = max(counts_list) if counts_list else 0

    print("📊 BẢNG TỔNG HỢP KIỂM KÊ SỐ LƯỢNG HẠT NGUYÊN (SAU LỌC):")
    print("=" * 85)
    print(f"  • Tổng số mã ảnh được kiểm tra : {len(audit_rows)} mã")
    print(f"  • Số mã ĐÃ CÓ HẠT NGUYÊN (✅)   : {valid_samples_count} mã")
    print(f"  • Tổng số hạt nguyên sạch      : {total_grains} hạt")
    print(f"  • Trung bình số hạt / 1 ảnh    : {avg_grains:.1f} hạt (Min: {min_grains}, Max: {max_grains})")
    print(f"  • Số mã có 0 HẠT (🔴)          : {zero_grains_count} mã")
    print(f"  • Số mã ÍT HẠT (<{min_warning_threshold}) (🟠)       : {low_grains_count} mã")
    print(f"  • Số mã chưa có thư mục (⏳)   : {missing_folder_count} mã")
    print(f"💾 Đã lưu báo cáo chi tiết vào  : {report_csv_path}")
    print("=" * 85 + "\n")

    # In chi tiết theo nhóm; không giả định một mẫu phải có 5 ảnh A-E.
    print("📋 CHI TIẾT SỐ LƯỢNG HẠT THEO NHÓM/MÃ ẢNH:")
    print("-" * 110)
    print(f"{'Nhóm':<12} | {'Số mã':<6} | {'Tổng hạt':<10} | {'Chi tiết mã: số hạt'}")
    print("-" * 110)

    for grp, g_info in sorted(grouped_stats.items()):
        samples = g_info["samples"]
        sample_details = ", ".join(f"{sid}:{count}" for sid, count in sorted(samples.items()))
        print(f"{grp:<12} | {len(samples):<6} | {g_info['total_grains']:<10} | {sample_details}")

    print("-" * 110 + "\n")

    # In danh sách các mã cần lưu ý (0 hạt hoặc ít hạt)
    flagged_samples = [r for r in audit_rows if r["Trạng_Thái"].startswith(("0 HẠT", "ÍT HẠT"))]
    if flagged_samples:
        print(f"⚠️ DANH SÁCH {len(flagged_samples)} MÃ CẦN LƯU Ý (0 hạt hoặc < {min_warning_threshold} hạt):")
        for r in flagged_samples:
            print(f"   {r['Biểu_Tượng']} {r['Mã_Mẫu']:<8} : {r['Số_Hạt_Nguyên']} hạt -> {r['Trạng_Thái']}")
        print()

    if verbose:
        print("📁 DANH SÁCH CHI TIẾT CÁC TẬP TIN HẠT NGUYÊN TRONG TỪNG MÃ:")
        for sid, sinfo in sorted(sample_grain_counts.items()):
            if sinfo["count"] > 0:
                print(f"  • {sid} ({sinfo['count']} hạt): {', '.join(sinfo['files'][:5])}{'...' if sinfo['count'] > 5 else ''}")
        print()

    return {
        "total_grains": total_grains,
        "valid_samples_count": valid_samples_count,
        "zero_grains_count": zero_grains_count,
        "low_grains_count": low_grains_count,
        "avg_grains": avg_grains,
        "report_csv_path": str(report_csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra số lượng hạt nguyên sau khi lọc trong DATASET_BUILDER")
    parser.add_argument("--min_warning", type=int, default=3, help="Ngưỡng cảnh báo số hạt ít (mặc định: 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="In chi tiết danh sách file trong từng mã")
    args = parser.parse_args()

    audit_filtered_grains(min_warning_threshold=args.min_warning, verbose=args.verbose)


if __name__ == "__main__":
    main()
