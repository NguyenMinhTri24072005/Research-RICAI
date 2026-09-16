#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 6: ĐIỀU PHỐI TRÍCH XUẤT ĐẶC TRƯNG DATASET (DATASET EXTRACTOR)
===============================================================================
Mục đích:
  - Duyệt qua từng dòng trong bảng Excel dữ liệu nhập tay manual_data.xlsx.
  - Ánh xạ tìm file ảnh tương ứng trong 1_Raw_Images/M###/M###X.jpg.
  - Xử lý điều kiện biên: Nếu mẫu CHƯA CÓ ẢNH thì để trống toàn bộ thuộc tính AI trích xuất (None / NaN).
  - GIAI ĐOẠN 1: Bóc tách vật chứa + SAHI YOLO-seg + Khử nhiễu 2 lần Cleaner + CNN v3_step2 -> Lưu ảnh hạt nguyên vào 3_AI_Extracted/CROPPED_GRAINS/M###/M###X/.
  - ĐIỂM DỪNG: Người dùng vào 3_AI_Extracted kiểm duyệt thủ công (xóa ảnh lỗi/ngoại lai nếu có).
  - GIAI ĐOẠN 2: Dò lại các hạt nguyên đã kiểm duyệt, tính toán toàn bộ kích thước 2D/3D (Dài, Rộng, Dày, Diện tích, Thể tích: Mean, Min, Max, Std).
  - Ghép với dữ liệu thô nhập tay và xuất bộ dữ liệu hoàn chỉnh sang 4_Final_Dataset/final_linear_regression_dataset.xlsx.
===============================================================================
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import string
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

from .container_detector import detect_container_and_scale
from .grain_segmenter import segment_grains_sahi
from .grain_crop_cleaner import clean_single_grain_crop
from .grain_classifier import GrainClassifier
from .ellipsoid_geometry import compute_folder_grains_summary, compute_single_grain_metrics
from .uniformity_evaluator import evaluate_batch_uniformity

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def derive_sample_group(sample_id: str) -> str:
    """Suy ra thư mục ảnh từ mã cũ hoặc mã có độ rộng tùy chỉnh."""
    sid = str(sample_id).strip().upper()
    if not sid:
        return ""
    for separator in ("_", "-"):
        if separator in sid:
            return sid.split(separator, 1)[0]
    match = re.fullmatch(r"([A-Z][A-Z0-9]*?\d+)([A-Z]+)", sid)
    if match:
        return match.group(1)
    return sid[:4] if len(sid) >= 4 else sid


class DatasetExtractor:
    """Class điều phối toàn bộ quy trình trích xuất đặc trưng cho Dataset."""

    def __init__(
        self,
        base_dir: Union[str, Path],
        yolo_model_path: Optional[Union[str, Path]] = None,
        cnn_model_path: Optional[Union[str, Path]] = None,
        packing_fraction: float = 0.62,
    ):
        self.base_dir = Path(base_dir)
        self.raw_images_dir = self.base_dir / "DATASET_BUILDER" / "1_Raw_Images"
        self.manual_excel_path = self.base_dir / "DATASET_BUILDER" / "2_Manual_Records" / "manual_data.xlsx"
        self.extracted_dir = self.base_dir / "DATASET_BUILDER" / "3_AI_Extracted"
        self.crops_dir = self.extracted_dir / "CROPPED_GRAINS"
        self.final_dataset_dir = self.base_dir / "DATASET_BUILDER" / "4_Final_Dataset"

        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.final_dataset_dir.mkdir(parents=True, exist_ok=True)

        if yolo_model_path is None:
            p1 = self.base_dir / "RESULTS" / "all-new-data-v1.yolov8_yolov8s-seg_trained" / "weights" / "best.pt"
            p2 = self.base_dir / "RESULTS" / "35_special_images_segmentation.v1i.yolov8_v1_trained" / "weights" / "best.pt"
            yolo_model_path = p1 if p1.exists() else p2

        if cnn_model_path is None:
            cnn_model_path = self.base_dir / "RESULTS" / "CNN_DenseNet121_Trained" / "best_v3_step2.keras"

        self.yolo_model_path = Path(yolo_model_path)
        self.cnn_model_path = Path(cnn_model_path)
        self.packing_fraction = packing_fraction

        self.yolo_detection_model = None
        self.cnn_classifier = None

    def init_models(self, conf: float = 0.5, device: Optional[str] = None) -> None:
        """Khởi tạo và nạp trọng số mô hình YOLO và CNN vào GPU/CPU."""
        import torch
        from sahi import AutoDetectionModel

        if device is None:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in device_str else "CPU"

        print("=" * 75)
        print("🚀 ĐANG KHỞI TẠO CÁC MÔ HÌNH THỊ GIÁC MÁY TÍNH...")
        print(f"   • YOLO Model : {self.yolo_model_path}")
        print(f"   • CNN Model  : {self.cnn_model_path}")
        print(f"   • Thiết bị   : {device_str} ({device_name})")
        print("=" * 75)

        # 1. Nạp YOLO với SAHI
        self.yolo_detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(self.yolo_model_path),
            confidence_threshold=conf,
            device=device_str,
        )

        # 2. Nạp DenseNet121 CNN
        self.cnn_classifier = GrainClassifier(
            model_path=self.cnn_model_path,
            class_names=["hat_khuyet_tat", "hat_nguyen"],
            target_size=(224, 224),
        )
        print("✨ Tất cả mô hình AI đã sẵn sàng hoạt động!\n")

    def read_manual_records(self) -> List[Dict[str, Any]]:
        """Đọc toàn bộ các dòng dữ liệu từ file Excel manual_data.xlsx."""
        if not self.manual_excel_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Excel tại: {self.manual_excel_path}")

        shared_strings: List[str] = []
        raw_rows: List[List[str]] = []

        with zipfile.ZipFile(self.manual_excel_path, "r") as z:
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
                fallback_column = 0
                for c in row.findall(f"{ns}c"):
                    cell_ref = c.get("r", "")
                    column_match = re.match(r"([A-Z]+)", cell_ref.upper())
                    if column_match:
                        column_number = 0
                        for char in column_match.group(1):
                            column_number = column_number * 26 + (ord(char) - 64)
                    else:
                        column_number = fallback_column + 1
                    fallback_column = column_number

                    t = c.get("t")
                    v = c.find(f"{ns}v")
                    val = v.text if v is not None and v.text is not None else ""
                    if t == "s" and val.isdigit():
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    elif t == "inlineStr":
                        inline = c.find(f"{ns}is")
                        parts = [node.text for node in inline.iter(f"{ns}t") if node.text] if inline is not None else []
                        val = "".join(parts)
                    values_by_column[column_number] = str(val).strip()

                max_column = max(values_by_column, default=0)
                r_vals = [values_by_column.get(column, "") for column in range(1, max_column + 1)]
                if any(r_vals):
                    raw_rows.append(r_vals)

        if not raw_rows:
            return []

        headers = raw_rows[0]
        records: List[Dict[str, Any]] = []

        for row in raw_rows[1:]:
            rec: Dict[str, Any] = {}
            for i, h in enumerate(headers):
                rec[h] = row[i] if i < len(row) else ""
            if not rec.get("Rice_Height_mm"):
                try:
                    container_height = float(rec.get("Container_Height_mm", ""))
                    empty_height = float(rec.get("Empty_Height_mm", ""))
                    rec["Rice_Height_mm"] = str(container_height - empty_height)
                except (TypeError, ValueError):
                    pass
            if rec.get("Sample_ID"):
                records.append(rec)

        return records

    def find_matching_image(self, sample_id: str) -> Optional[Path]:
        """
        Tìm ảnh theo cấu trúc phẳng mới hoặc cấu trúc thư mục cũ.

        Ví dụ mới: 1_Raw_Images/M001.jpg.
        Ví dụ cũ: 1_Raw_Images/M001/M001A.jpg.
        """
        sid = str(sample_id).strip()
        if not sid:
            return None

        target_stem = sid.upper()
        for ext in IMAGE_EXTENSIONS:
            candidate = self.raw_images_dir / f"{target_stem}{ext}"
            if candidate.is_file():
                return candidate
            candidate_lower = self.raw_images_dir / f"{sid.lower()}{ext}"
            if candidate_lower.is_file():
                return candidate_lower

        folder_name = derive_sample_group(sid)
        folder_path = self.raw_images_dir / folder_name

        if not folder_path.is_dir():
            return None

        for ext in IMAGE_EXTENSIONS:
            candidate = folder_path / f"{target_stem}{ext}"
            if candidate.is_file():
                return candidate
            candidate_lower = folder_path / f"{sid.lower()}{ext}"
            if candidate_lower.is_file():
                return candidate_lower

        for f in folder_path.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if f.stem.upper() == target_stem:
                    return f

        return None

    # =========================================================================
    # GIAI ĐOẠN 1: BÓC TÁCH VẬT CHỨA, 2 LẦN CLEAN & LỌC HẠT NGUYÊN
    # =========================================================================
    def run_stage1_crop_and_classify(
        self,
        conf_yolo: float = 0.50,
        slice_size: int = 640,
        overlap: float = 0.25,
        conf_cnn: float = 0.50,
        skip_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Chạy Giai đoạn 1:
        1. Bóc tách từng hạt lúa bằng SAHI YOLO-seg.
        2. Chạy 2 lần Cleaner (Step 1 + Step 2) bẻ eo dính và lọc mảnh vụn.
        3. Phân loại CNN v3_step2 để lấy các hạt nguyên.
        4. Lưu riêng các hạt nguyên vào 3_AI_Extracted/CROPPED_GRAINS/M###/M###X/.
        """
        if self.yolo_detection_model is None or self.cnn_classifier is None:
            self.init_models(conf=conf_yolo)

        records = self.read_manual_records()
        print("=" * 80)
        print("🌾 BẮT ĐẦU GIAI ĐOẠN 1: BÓC TÁCH SAHI, 2 LẦN CLEAN & SÀNG LỌC HẠT NGUYÊN")
        print(f"📋 Tổng số mẫu cần duyệt từ Excel : {len(records)} dòng")
        print(f"📁 Thư mục lưu ảnh hạt nguyên      : {self.crops_dir}")
        print("=" * 80 + "\n")

        processed_count = 0
        missing_count = 0
        total_whole_grains = 0

        for idx, rec in enumerate(records, start=1):
            sample_id = rec["Sample_ID"]
            folder_key = derive_sample_group(sample_id)
            sample_key = sample_id.upper()

            sample_crops_dir = self.crops_dir / folder_key / sample_key

            img_path = self.find_matching_image(sample_id)
            if img_path is None:
                missing_count += 1
                print(f"[{idx:03d}/{len(records)}] ⚪ {sample_id:8s} -> KHÔNG CÓ ẢNH (Bỏ qua, gán giá trị trống).")
                continue

            if skip_existing and sample_crops_dir.exists() and any(sample_crops_dir.iterdir()):
                print(f"[{idx:03d}/{len(records)}] ⏩ {sample_id:8s} -> Đã tồn tại thư mục cắt, bỏ qua.")
                processed_count += 1
                continue

            print(f"[{idx:03d}/{len(records)}] 🚀 {sample_id:8s} -> Đang xử lý: {img_path.name}...")

            # 1. Bóc tách hạt lúa thô bằng SAHI YOLO-seg
            temp_crop_dir = self.extracted_dir / "_temp_raw_crops" / sample_key
            raw_grains = segment_grains_sahi(
                detection_model=self.yolo_detection_model,
                image_path=img_path,
                output_crop_dir=temp_crop_dir,
                conf_threshold=conf_yolo,
                slice_size=slice_size,
                overlap_ratio=overlap,
            )

            # 2. Thực hiện 2 lần Clean chuyên sâu (Step 1 + Step 2)
            cleaned_grains = []
            for g_idx, g in enumerate(raw_grains):
                raw_crop = g["crop_rgba"]
                # Lần 1: Step 1
                c1 = clean_single_grain_crop(
                    raw_crop,
                    open_ksize=5,
                    min_neck_ratio=0.20,
                    min_area=35,
                    centrality_weight=2.5,
                    fill_holes=False,
                    sever_bridges=True,
                )
                # Lần 2: Step 2
                c2 = clean_single_grain_crop(
                    c1,
                    open_ksize=3,
                    min_neck_ratio=0.15,
                    min_area=25,
                    centrality_weight=2.2,
                    fill_holes=True,
                    sever_bridges=True,
                )
                alpha = c2[:, :, 3] if len(c2.shape) == 3 and c2.shape[2] == 4 else None
                if alpha is not None and cv2.countNonZero(alpha) >= 20:
                    g_clean = dict(g)
                    g_clean["crop_rgba"] = c2
                    bgr = c2[:, :, :3].copy()
                    bgr[c2[:, :, 3] == 0] = [0, 0, 0]
                    g_clean["crop_bgr"] = bgr
                    cleaned_grains.append(g_clean)

            # 3. Phân loại CNN v3_step2 để SÀNG LỌC RIÊNG HẠT NGUYÊN
            whole_grains, defective_grains = self.cnn_classifier.filter_grains(
                cleaned_grains, target_label="hat_nguyen", min_conf=conf_cnn
            )

            # 4. Lưu riêng các hạt nguyên vào 3_AI_Extracted/CROPPED_GRAINS/M###/M###X/
            if sample_crops_dir.exists():
                shutil.rmtree(sample_crops_dir)
            sample_crops_dir.mkdir(parents=True, exist_ok=True)

            for g_idx, g in enumerate(whole_grains):
                conf_val = g.get("confidence", 1.0)
                save_fpath = sample_crops_dir / f"{sample_key}_whole_{g_idx:03d}_conf{conf_val:.2f}.png"
                cv2.imwrite(str(save_fpath), g["crop_rgba"])

            # Xóa thư mục tạm
            if temp_crop_dir.exists():
                shutil.rmtree(temp_crop_dir)

            processed_count += 1
            total_whole_grains += len(whole_grains)
            print(f"   ✅ Bắt thô: {len(raw_grains)} | Sạch sau 2 lần clean: {len(cleaned_grains)} | 🌟 HẠT NGUYÊN: {len(whole_grains)} | 🗑️ Khuyết/Bỏ: {len(defective_grains)}")

        print("\n" + "=" * 80)
        print("🎉 HOÀN TẤT GIAI ĐOẠN 1:")
        print(f"   • Đã xử lý thành công : {processed_count} mẫu")
        print(f"   • Mẫu chưa có ảnh     : {missing_count} mẫu")
        print(f"   • Tổng số hạt nguyên  : {total_whole_grains} hạt")
        print(f"📁 Bạn có thể mở '{self.crops_dir}' để kiểm duyệt thủ công (Human-in-the-loop QA) trước khi chạy Giai đoạn 2.")
        print("=" * 80 + "\n")

        return {
            "processed_count": processed_count,
            "missing_count": missing_count,
            "total_whole_grains": total_whole_grains,
        }

    # =========================================================================
    # GIAI ĐOẠN 2: TÍNH TOÁN 2D/3D & XUẤT DATASET CUỐI SANG 4_FINAL_DATASET
    # =========================================================================
    def run_stage2_calculate_and_export(
        self,
        output_filename: str = "final_linear_regression_dataset.xlsx",
        show_progress: bool = True,
    ) -> Path:
        """
        Chạy Giai đoạn 2:
        1. Đọc các ảnh hạt nguyên đã kiểm duyệt trong CROPPED_GRAINS.
        2. Tính toán toàn bộ các đặc trưng 2D/3D (Dài, Rộng, Dày, Diện tích, Thể tích: Mean, Min, Max, Std).
        3. Ghép nối với dữ liệu thủ công nhập tay từ manual_data.xlsx.
        4. Xuất file hoàn chỉnh sang 4_Final_Dataset/.
        """
        import time
        start_t = time.time()
        records = self.read_manual_records()
        print("=" * 85)
        print("📊 BẮT ĐẦU GIAI ĐOẠN 2: TÍNH TOÁN HÌNH THÁI HỌC 2D & THỂ TÍCH 3D ELLIPSOID")
        print(f"📋 Tổng số mẫu xử lý : {len(records)} dòng")
        print("=" * 85 + "\n")

        enriched_rows: List[Dict[str, Any]] = []
        valid_count = 0
        missing_count = 0

        for idx, rec in enumerate(records, start=1):
            sample_id = rec["Sample_ID"]
            folder_key = derive_sample_group(sample_id)
            sample_key = sample_id.upper()

            sample_crops_dir = self.crops_dir / folder_key / sample_key
            img_path = self.find_matching_image(sample_id)

            row_out: Dict[str, Any] = dict(rec)

            if img_path is None or not sample_crops_dir.exists() or not any(sample_crops_dir.iterdir()):
                # MẪU CHƯA CÓ ẢNH HOẶC CHƯA CÓ HẠT -> ĐỂ TRỐNG TOÀN BỘ CÁC CỘT FEATURE AI
                row_out["Image_Status"] = "MISSING" if img_path is None else "NO_GRAINS"
                row_out["Image_Filename"] = img_path.name if img_path else ""
                row_out["Pixels_Per_mm"] = ""
                row_out["Container_Detected_Diam_px"] = ""
                row_out["Bulk_Rice_Volume_mm3"] = ""
                row_out["Whole_Grains_Count"] = 0 if sample_crops_dir.exists() else ""
                
                # Chiều dài 2a
                row_out["Grain_Length_mm_Mean"] = ""
                row_out["Grain_Length_mm_Min"] = ""
                row_out["Grain_Length_mm_Max"] = ""
                row_out["Grain_Length_mm_Std"] = ""
                
                # Chiều rộng 2b
                row_out["Grain_Width_mm_Mean"] = ""
                row_out["Grain_Width_mm_Min"] = ""
                row_out["Grain_Width_mm_Max"] = ""
                row_out["Grain_Width_mm_Std"] = ""

                # Chiều dày 2c
                row_out["Grain_Thickness_mm_Mean"] = ""
                row_out["Grain_Thickness_mm_Min"] = ""
                row_out["Grain_Thickness_mm_Max"] = ""
                row_out["Grain_Thickness_mm_Std"] = ""

                # Diện tích 2D
                row_out["Grain_Area_mm2_Mean"] = ""
                row_out["Grain_Area_mm2_Min"] = ""
                row_out["Grain_Area_mm2_Max"] = ""
                row_out["Grain_Area_mm2_Std"] = ""

                # Thể tích 3D Ellipsoid
                row_out["Grain_Volume_mm3_Mean"] = ""
                row_out["Grain_Volume_mm3_Min"] = ""
                row_out["Grain_Volume_mm3_Max"] = ""
                row_out["Grain_Volume_mm3_Std"] = ""
                
                row_out["Estimated_Total_Seeds_Hybrid"] = ""
                row_out["Uniformity_Rate_Pct"] = ""
                enriched_rows.append(row_out)
                missing_count += 1

                if show_progress:
                    print(f"[{idx:03d}/{len(records)}] ⚪ {sample_id:<8} -> Bỏ qua (Chưa có ảnh/hạt nguyên)", flush=True)
                continue

            # MẪU ĐÃ CÓ ẢNH VÀ ĐÃ CÓ TẬP HẠT NGUYÊN ĐƯỢC DUYỆT
            row_out["Image_Status"] = "FOUND"
            row_out["Image_Filename"] = img_path.name

            # 1. Tính toán tỷ lệ scale và thể tích khối lúa từ vật chứa
            try:
                inner_diam = float(rec.get("Inner_Diameter_mm", 17.8))
                c_height = float(rec.get("Container_Height_mm", 33.9))
                e_height = float(rec.get("Empty_Height_mm", 0.0))
                c_info = detect_container_and_scale(
                    img_path,
                    inner_diam_mm=inner_diam,
                    container_height_mm=c_height,
                    empty_height_mm=e_height,
                )
                pixels_per_mm = c_info["pixels_per_mm"]
                bulk_volume = c_info["bulk_rice_volume_mm3"]
                c_diam_px = c_info["inner_w_px"]
            except Exception:
                pixels_per_mm = 10.0
                bulk_volume = 0.0
                c_diam_px = 0

            # 2. Tính toán toàn bộ kích thước 2D và thể tích 3D cho các hạt nguyên trong thư mục
            summary = compute_folder_grains_summary(sample_crops_dir, pixels_per_mm=pixels_per_mm, label="hat_nguyen")

            if summary and summary.get("count", 0) > 0:
                whole_count = summary["count"]
                v_mean = summary["volume_mean"]

                # Ước lượng số hạt theo thể tích khối lúa và hệ số chèn lấp
                if v_mean > 0 and bulk_volume > 0:
                    est_seeds = (bulk_volume * self.packing_fraction) / v_mean
                else:
                    est_seeds = 0.0

                # Đánh giá độ đồng đều thể tích hạt sau lọc IQR
                vol_list = []
                for p in sample_crops_dir.glob("*.png"):
                    try:
                        m = compute_single_grain_metrics(p, pixels_per_mm, label="hat_nguyen")
                        vol_list.append(m["volume_mm3"])
                    except Exception:
                        pass
                unif_res = evaluate_batch_uniformity(vol_list)

                row_out["Pixels_Per_mm"] = round(pixels_per_mm, 2)
                row_out["Container_Detected_Diam_px"] = c_diam_px
                row_out["Bulk_Rice_Volume_mm3"] = round(bulk_volume, 2)
                row_out["Whole_Grains_Count"] = whole_count

                # Chiều dài 2a
                row_out["Grain_Length_mm_Mean"] = round(summary["length_mean"], 3)
                row_out["Grain_Length_mm_Min"] = round(summary["length_min"], 3)
                row_out["Grain_Length_mm_Max"] = round(summary["length_max"], 3)
                row_out["Grain_Length_mm_Std"] = round(summary["length_std"], 3)

                # Chiều rộng 2b
                row_out["Grain_Width_mm_Mean"] = round(summary["width_mean"], 3)
                row_out["Grain_Width_mm_Min"] = round(summary["width_min"], 3)
                row_out["Grain_Width_mm_Max"] = round(summary["width_max"], 3)
                row_out["Grain_Width_mm_Std"] = round(summary["width_std"], 3)

                # Chiều dày 2c
                row_out["Grain_Thickness_mm_Mean"] = round(summary.get("thickness_mean", summary["length_mean"]), 3)
                row_out["Grain_Thickness_mm_Min"] = round(summary.get("thickness_min", summary["length_min"]), 3)
                row_out["Grain_Thickness_mm_Max"] = round(summary.get("thickness_max", summary["length_max"]), 3)
                row_out["Grain_Thickness_mm_Std"] = round(summary.get("thickness_std", summary["length_std"]), 3)

                # Diện tích 2D
                row_out["Grain_Area_mm2_Mean"] = round(summary["area_mean"], 3)
                row_out["Grain_Area_mm2_Min"] = round(summary["area_min"], 3)
                row_out["Grain_Area_mm2_Max"] = round(summary["area_max"], 3)
                row_out["Grain_Area_mm2_Std"] = round(summary["area_std"], 3)

                # Thể tích 3D Ellipsoid
                row_out["Grain_Volume_mm3_Mean"] = round(summary["volume_mean"], 3)
                row_out["Grain_Volume_mm3_Min"] = round(summary["volume_min"], 3)
                row_out["Grain_Volume_mm3_Max"] = round(summary["volume_max"], 3)
                row_out["Grain_Volume_mm3_Std"] = round(summary["volume_std"], 3)

                row_out["Estimated_Total_Seeds_Hybrid"] = int(round(est_seeds))
                row_out["Uniformity_Rate_Pct"] = unif_res["uniformity_rate_pct"]
                valid_count += 1

                if show_progress:
                    print(f"[{idx:03d}/{len(records)}] 🌾 {sample_id:<8} -> {whole_count:2d} hạt | Dài={summary['length_mean']:.2f}mm | Rộng={summary['width_mean']:.2f}mm | V_hạt={summary['volume_mean']:.2f}mm³ | V_khối={bulk_volume:,.0f}mm³", flush=True)
            else:
                row_out["Whole_Grains_Count"] = 0
                row_out["Grain_Volume_mm3_Mean"] = ""
                row_out["Estimated_Total_Seeds_Hybrid"] = ""
                missing_count += 1
                if show_progress:
                    print(f"[{idx:03d}/{len(records)}] 🔴 {sample_id:<8} -> 0 hạt (Thư mục rỗng)", flush=True)

            enriched_rows.append(row_out)

        # 3. Xuất file CSV và Excel vào 4_Final_Dataset/
        final_csv_path = self.final_dataset_dir / output_filename.replace(".xlsx", ".csv")
        final_excel_path = self.final_dataset_dir / output_filename

        # Thu thập đầy đủ danh sách tất cả các cột xuất hiện
        fieldnames: List[str] = []
        for r in enriched_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        # Xuất file CSV
        with open(final_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(enriched_rows)

        # Xuất file Excel
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Final_Dataset"
            ws.append(fieldnames)
            for r in enriched_rows:
                ws.append([r.get(k, "") for k in fieldnames])
            wb.save(final_excel_path)
            print(f"🎉 Đã xuất file Excel thành công tại: {final_excel_path}")
        except Exception:
            print(f"ℹ️ Đã xuất file CSV tổng hợp thành công tại: {final_csv_path}")

        # Đồng thời lưu 1 bản sao vào 3_AI_Extracted/
        extracted_csv = self.extracted_dir / "ai_extracted_dataset.csv"
        with open(extracted_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(enriched_rows)

        total_t = time.time() - start_t
        print("\n" + "=" * 85)
        print("🏆 HOÀN THÀNH XUẤT DATASET CUỐI CHO HỒI QUY TUYẾN TÍNH:")
        print(f"   • Tổng số mẫu đã xử lý         : {len(enriched_rows)} mẫu (Có hạt: {valid_count} | Bỏ qua/Thiếu: {missing_count})")
        print(f"   • Thời gian thực thi           : {total_t:.2f} giây")
        print(f"   📁 File Excel                  : {final_excel_path}")
        print(f"   📁 File CSV                    : {final_csv_path}")
        print("=" * 85 + "\n")

        return final_excel_path
