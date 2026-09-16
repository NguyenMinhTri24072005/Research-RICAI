#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
🌾 RICE VISION AI - GRAIN CROP CLEANER (LOCAL RUNNER)
===============================================================================
Mục đích:
  - Chạy làm sạch, khử nhiễu và bẻ cầu nối dính cho ảnh hạt lúa crop TRỰC TIẾP TRÊN MÁY LOCAL.
  - Tối ưu Đa luồng (Multi-threading) tận dụng 100% CPU i3-1005G1 (hoặc máy trạm).
  - Tốc độ siêu nhanh: ~100 ảnh/giây, 7.000 ảnh chỉ mất ~1 phút.
  - TẤT CẢ THAM SỐ ĐƯỢC GOM GỌN Ở KHU VỰC CẤU HÌNH NGAY DƯỚI ĐÂY ĐỂ DỄ DÀNG ĐIỀU CHỈNH.
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# =============================================================================
# 🛠️ [BẢNG CẤU HÌNH TOÀN BỘ THAM SỐ - CONFIGURATION PANEL]
# =============================================================================
# Bạn có thể điều chỉnh trực tiếp tất cả các tham số của thuật toán ngay tại đây:

# 📁 1. THƯ MỤC DỮ LIỆU
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục chứa dữ liệu cần làm sạch:
# • DATASET_PATH = BASE_DIR / "DETECTED_OBJECTS" / "DATA3"      (Chỉ chạy riêng DATA3)
# • DATASET_PATH = BASE_DIR / "DETECTED_OBJECTS"               (Quét toàn bộ DATA1, DATA2, DATA3)
DATASET_PATH = BASE_DIR / "DETECTED_OBJECTS" / "DATA2"

TARGET_SUBFOLDER = "SINGLE_IMAGE_CROPPED_step3"   # Tên thư mục nguồn chứa ảnh crop từ YOLO
OUTPUT_SUBFOLDER = "SINGLE_IMAGE_CROPPED_step4"   # Tên thư mục đích lưu ảnh sau làm sạch


# ⚡ 2. TÙY CHỌN TĂNG TỐC VẬN HÀNH
SKIP_EXISTING = True    # True: Bỏ qua ảnh đã làm sạch trước đó (chạy 10s) | False: Ghi đè lại tất cả
NUM_WORKERS   = 4       # Số luồng CPU chạy song song (Chip i3-1005G1 có 4 threads -> đặt 4)
SHOW_PROGRESS = True    # Hiển thị thanh tiến trình % và tốc độ thời gian thực


# 🔬 3. THÔNG SỐ BẺ CẦU NỐI DÍNH & KHỬ DỊ VẬT NGOẠI LAI
SEVER_BRIDGES     = True  # True: Bật thuật toán bẻ cầu nối pixel dính (Distance Transform + Watershed)

# step 1
# OPEN_KSIZE        = 5
# MIN_NECK_RATIO    = 0.20
# MIN_AREA          = 35
# CENTRALITY_WEIGHT = 2.5
# FILL_HOLES        = False

# step 2
# OPEN_KSIZE        = 3
# MIN_NECK_RATIO    = 0.15
# MIN_AREA          = 25
# CENTRALITY_WEIGHT = 2.2
# FILL_HOLES        = True

# step 3
# OPEN_KSIZE        = 3
# MIN_NECK_RATIO    = 0.12
# MIN_AREA          = 20
# CENTRALITY_WEIGHT = 2.0
# FILL_HOLES        = True

# step 4
OPEN_KSIZE        = 3
MIN_NECK_RATIO    = 0.10
MIN_AREA          = 15
CENTRALITY_WEIGHT = 1.5
FILL_HOLES        = True

# ✂️ 4. TÙY CHỌN CẮT GỌN KHUNG HÌNH (BOUNDING BOX)
RE_CROP = False   # False: Giữ nguyên kích thước ảnh gốc | True: Cắt gọn khung ảnh ôm sát hạt lúa
PADDING = 2       # Khoảng đệm viền (pixel) nếu bật RE_CROP = True

# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (AN TOÀN UNICODE TRÊN WINDOWS)
# ─────────────────────────────────────────────────────────────────────────────
def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Đọc file ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(str(path), flags)


def imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    """Lưu file ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() if p.suffix else ".png"
        success, nparr = cv2.imencode(ext, img)
        if success:
            nparr.tofile(str(p))
            return True
        return False
    except Exception:
        return cv2.imwrite(str(path), img)


def sever_bridges_and_isolate_main(
    raw_mask: np.ndarray,
    min_area: int = 20,
    open_ksize: int = 3,
    min_neck_ratio: float = 0.15,
    centrality_weight: float = 2.5,
) -> np.ndarray:
    """Bẻ gãy các cầu nối pixel dính hẹp và trích xuất hạt giống lõi trung tâm."""
    h, w = raw_mask.shape[:2]
    if cv2.countNonZero(raw_mask) == 0:
        return raw_mask

    # 1. Biến đổi khoảng cách Euclidean (Distance Transform)
    dist = cv2.distanceTransform(raw_mask, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 1.0:
        return raw_mask

    # 2. Phép toán Mở (Morphological Opening) bẻ gãy cầu nối pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # 3. Trích xuất hạt giống lõi (Core Seeds) theo tỷ lệ eo dính
    _, core_seeds = cv2.threshold(dist, max_dist * min_neck_ratio, 255, cv2.THRESH_BINARY)
    core_seeds = core_seeds.astype(np.uint8)
    refined_seeds = cv2.bitwise_and(opened, core_seeds)

    # 4. Phân tích thành phần liên thông
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_seeds, connectivity=8)
    if n_labels <= 1:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    if n_labels <= 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)

    # 5. Chấm điểm độ nổi bật tâm ảnh (Salient Centrality)
    img_cx, img_cy = w / 2.0, h / 2.0
    max_d = np.sqrt(img_cx**2 + img_cy**2) + 1e-6

    best_idx = 1
    best_score = -1.0

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        norm_d = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2) / max_d
        score = float(area) / (1.0 + centrality_weight * (norm_d ** 2))
        if score > best_score:
            best_score = score
            best_idx = i

    # 6. Tái tạo viền hạt nguyên bằng Watershed
    markers = np.zeros((h, w), dtype=np.int32)
    markers[raw_mask == 0] = 1
    markers[labels == best_idx] = 2

    marker_id = 3
    for i in range(1, n_labels):
        if i != best_idx:
            markers[labels == i] = marker_id
            marker_id += 1

    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_inv = 255 - dist_norm
    ws_input = cv2.cvtColor(dist_inv, cv2.COLOR_GRAY2BGR)

    cv2.watershed(ws_input, markers)
    clean_mask = np.where(markers == 2, 255, 0).astype(np.uint8)

    cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(clean_mask, cnts, -1, 255, -1)

    return clean_mask


def clean_single_grain_crop(
    image_input: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    min_area: int = MIN_AREA,
    centrality_weight: float = CENTRALITY_WEIGHT,
    fill_holes: bool = FILL_HOLES,
    sever_bridges: bool = SEVER_BRIDGES,
    open_ksize: int = OPEN_KSIZE,
    min_neck_ratio: float = MIN_NECK_RATIO,
    re_crop: bool = RE_CROP,
    padding: int = PADDING,
    **kwargs,
) -> np.ndarray:
    """Làm sạch 1 ảnh hạt lúa đơn lẻ theo đầy đủ các tham số cấu hình."""
    if isinstance(image_input, (str, Path)):
        img_raw = imread_unicode(image_input, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_raw = image_input.copy()
    else:
        raise ValueError("image_input phải là đường dẫn hoặc numpy array.")

    is_rgba = (len(img_raw.shape) == 3 and img_raw.shape[2] == 4)
    h, w = img_raw.shape[:2]
    if h == 0 or w == 0:
        return img_raw

    # Binary Mask
    if is_rgba:
        alpha = img_raw[:, :, 3]
        raw_mask = (alpha > 10).astype(np.uint8) * 255 if np.any(alpha == 0) else \
                   cv2.threshold(cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2GRAY), 15, 255, cv2.THRESH_BINARY)[1]
    else:
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw
        _, raw_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(raw_mask) == 0:
        res = cv2.cvtColor(img_raw[:, :, :3] if is_rgba else img_raw, cv2.COLOR_BGR2BGRA)
        res[:, :, :] = 0
        if output_path is not None:
            imwrite_unicode(output_path, res)
        return res

    if sever_bridges:
        clean_mask = sever_bridges_and_isolate_main(
            raw_mask=raw_mask,
            min_area=min_area,
            open_ksize=open_ksize,
            min_neck_ratio=min_neck_ratio,
            centrality_weight=centrality_weight,
        )
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

        if num_labels <= 1:
            clean_mask = raw_mask.copy()
        else:
            img_cx, img_cy = w / 2.0, h / 2.0
            max_dist = np.sqrt(img_cx**2 + img_cy**2) + 1e-6
            best_idx = 1
            best_score = -1.0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                cx, cy = centroids[i]
                norm_dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2) / max_dist
                score = float(area) / (1.0 + centrality_weight * (norm_dist ** 2))
                if score > best_score:
                    best_score = score
                    best_idx = i
            clean_mask = np.zeros((h, w), dtype=np.uint8)
            clean_mask[labels == best_idx] = 255

    if fill_holes:
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(clean_mask, cnts, -1, 255, -1)

    bgr = img_raw[:, :, :3].copy() if is_rgba else img_raw.copy()
    bgr[clean_mask == 0] = [0, 0, 0]

    cleaned_rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    cleaned_rgba[:, :, 3] = clean_mask

    if re_crop:
        ys, xs = np.where(clean_mask == 255)
        if len(ys) > 0:
            y1 = max(0, int(ys.min()) - padding)
            y2 = min(h, int(ys.max()) + 1 + padding)
            x1 = max(0, int(xs.min()) - padding)
            x2 = min(w, int(xs.max()) + 1 + padding)
            cleaned_rgba = cleaned_rgba[y1:y2, x1:x2].copy()

    if output_path is not None:
        imwrite_unicode(output_path, cleaned_rgba)

    return cleaned_rgba


# ─────────────────────────────────────────────────────────────────────────────
# HÀM KIỂM TRA VÀ ĐỐI SOÁT DỮ LIỆU ĐẦU VÀO
# ─────────────────────────────────────────────────────────────────────────────
def audit_dataset(root_path: Union[str, Path], target_subfolder: str = TARGET_SUBFOLDER, valid_exts=None):
    """Kiểm tra và in bảng thống kê số lượng ảnh crop và ảnh đã clean."""
    if valid_exts is None:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    root_p = Path(root_path)
    cropped_folders = []
    for p in root_p.rglob("*"):
        if p.is_dir() and p.name.lower() == target_subfolder.lower():
            cropped_folders.append(p)
    if not cropped_folders and root_p.name.lower() == target_subfolder.lower():
        cropped_folders.append(root_p)

    print("\n" + "=" * 75)
    print("📊 BÁO CÁO KIỂM ĐỊNH SỐ LƯỢNG DATASET")
    print("=" * 75)
    print(f"📂 Thư mục quét: {root_p}")
    print(f"🔎 Tìm thấy {len(cropped_folders)} thư mục '{target_subfolder}'\n")

    total_cr, total_cl = 0, 0

    for idx, c_dir in enumerate(cropped_folders, 1):
        clean_dir = c_dir.parent / OUTPUT_SUBFOLDER
        cr_files = [f for f in c_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts and OUTPUT_SUBFOLDER.lower() not in str(f).lower()]
        cl_files = [f for f in clean_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts] if clean_dir.exists() else []

        total_cr += len(cr_files)
        total_cl += len(cl_files)

        prefix_counts = defaultdict(lambda: {"cropped": 0, "cleaned": 0})
        for f in cr_files:
            prefix_counts[f.stem.split("_")[0]]["cropped"] += 1
        for f in cl_files:
            prefix_counts[f.stem.split("_")[0]]["cleaned"] += 1

        parent_name = c_dir.parent.name if c_dir.parent != root_p else c_dir.name
        print(f"[{idx}/{len(cropped_folders)}] 📁 Folder: {parent_name}/{c_dir.name}")
        print(f"   • Tổng số ảnh Crop : {len(cr_files):,} ảnh")
        print(f"   • Đã Clean & Lưu   : {len(cl_files):,} ảnh")

        diff = len(cr_files) - len(cl_files)
        if diff > 0:
            print(f"   ⚠️ Còn thiếu: {diff:,} ảnh chưa được clean!")
        elif diff == 0 and len(cr_files) > 0:
            print(f"   ✅ Đã đồng bộ 100% ({len(cr_files):,} ảnh)")

        print("-" * 65)
        print(f"   {'Mẫu ảnh':<15} | {'Số hạt Crop':<15} | {'Số hạt Clean':<15} | {'Trạng thái'}")
        print("-" * 65)
        for pfx in sorted(prefix_counts.keys()):
            cr = prefix_counts[pfx]["cropped"]
            cl = prefix_counts[pfx]["cleaned"]
            st = "✅ Đủ" if cr == cl else f"⚠️ Thiếu {cr - cl}"
            print(f"   {pfx:<15} | {cr:<15} | {cl:<15} | {st}")
        print("-" * 65 + "\n")

    print(f"🎯 TỔNG CỘNG: {total_cr:,} ảnh Crop đầu vào | {total_cl:,} ảnh Clean đã có\n" + "=" * 75)
    return cropped_folders


# ─────────────────────────────────────────────────────────────────────────────
# HÀM XỬ LÝ HÀNG LOẠT ĐA LUỒNG TỐI ƯU (MULTI-THREADED BATCH CLEANER)
# ─────────────────────────────────────────────────────────────────────────────
def _process_single_image_worker(args_tuple):
    img_path, dest_path, min_area, centrality_weight, fill_holes, sever_bridges, open_ksize, min_neck_ratio, re_crop, padding = args_tuple
    try:
        clean_single_grain_crop(
            image_input=img_path,
            output_path=dest_path,
            min_area=min_area,
            centrality_weight=centrality_weight,
            fill_holes=fill_holes,
            sever_bridges=sever_bridges,
            open_ksize=open_ksize,
            min_neck_ratio=min_neck_ratio,
            re_crop=re_crop,
            padding=padding,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def run_clean_batch_local(
    root_path: Union[str, Path] = DATASET_PATH,
    target_subfolder: str = TARGET_SUBFOLDER,
    output_subfolder: str = OUTPUT_SUBFOLDER,
    min_area: int = MIN_AREA,
    centrality_weight: float = CENTRALITY_WEIGHT,
    fill_holes: bool = FILL_HOLES,
    sever_bridges: bool = SEVER_BRIDGES,
    open_ksize: int = OPEN_KSIZE,
    min_neck_ratio: float = MIN_NECK_RATIO,
    re_crop: bool = RE_CROP,
    padding: int = PADDING,
    skip_existing: bool = SKIP_EXISTING,
    num_workers: int = NUM_WORKERS,
    show_progress: bool = SHOW_PROGRESS,
):
    """Thực thi làm sạch hàng loạt toàn bộ dataset với đa luồng CPU cục bộ."""
    root_p = Path(root_path)
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    try:
        from tqdm.auto import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    target_dirs = []
    for p in root_p.rglob("*"):
        if p.is_dir() and p.name.lower() == target_subfolder.lower():
            target_dirs.append(p)
    if not target_dirs and root_p.name.lower() == target_subfolder.lower():
        target_dirs.append(root_p)

    if not target_dirs:
        print(f"❌ Không tìm thấy thư mục nào có tên '{target_subfolder}' trong: {root_p}")
        return

    print("\n" + "=" * 75)
    print("🧹 BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH ẢNH CỤC BỘ (LOCAL BATCH PROCESSING)")
    print("=" * 75)
    print(f"📂 Thư mục nguồn      : {root_p}")
    print(f"⚡ Đa luồng CPU       : {num_workers} workers (threads)")
    print(f"⏭️ Bỏ qua ảnh đã có   : {'BẬT (Skip Existing)' if skip_existing else 'TẮT (Ghi đè tất cả)'}")
    print(f"✂️ Bẻ cầu nối dính    : {'BẬT (Kernel ' + str(open_ksize) + 'x' + str(open_ksize) + ', Neck ratio: ' + str(min_neck_ratio) + ')' if sever_bridges else 'TẮT'}")
    print(f"🔲 Cắt viền hạt       : {'BẬT (Padding ' + str(padding) + 'px)' if re_crop else 'TẮT (Giữ nguyên size gốc)'}")
    print("=" * 75)

    start_all = time.time()
    grand_cleaned = 0
    grand_skipped = 0
    grand_errors = 0

    for idx, in_dir in enumerate(target_dirs, 1):
        out_dir = in_dir.parent / output_subfolder if in_dir.name.lower() == target_subfolder.lower() else in_dir / output_subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts and output_subfolder.lower() not in p.parts]

        tasks = []
        skipped_count = 0

        for img_p in all_imgs:
            rel_path = img_p.relative_to(in_dir)
            dest_p = out_dir / rel_path.with_suffix(".png")

            if skip_existing and dest_p.exists():
                skipped_count += 1
                continue

            tasks.append((img_p, dest_p, min_area, centrality_weight, fill_holes, sever_bridges, open_ksize, min_neck_ratio, re_crop, padding))

        parent_tag = in_dir.parent.name if in_dir.parent != root_p else in_dir.name
        print(f"\n[{idx}/{len(target_dirs)}] 📁 Đang xử lý: {parent_tag}/{in_dir.name}")
        print(f"   ↳ Tổng số: {len(all_imgs):,} ảnh | Cần clean: {len(tasks):,} | Đã có sẵn: {skipped_count:,}")

        cleaned_count = 0
        error_count = 0

        if tasks:
            if show_progress and has_tqdm:
                pbar = tqdm(total=len(tasks), desc=f"   ⏳ [{parent_tag}]", unit="ảnh", ncols=95)
            else:
                pbar = None

            last_t = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_single_image_worker, t) for t in tasks]
                for done_idx, fut in enumerate(as_completed(futures), 1):
                    ok, err = fut.result()
                    if ok:
                        cleaned_count += 1
                    else:
                        error_count += 1

                    if pbar:
                        pbar.update(1)
                    elif not has_tqdm:
                        curr = time.time()
                        if done_idx % 100 == 0 or done_idx == len(tasks) or (curr - last_t >= 3.0):
                            pct = (done_idx / len(tasks)) * 100.0
                            rate = done_idx / max(curr - start_all, 0.001)
                            print(f"   ⏳ Tiến độ: {done_idx}/{len(tasks)} ({pct:.1f}%) | Tốc độ: {rate:.1f} ảnh/s")
                            last_t = curr

            if pbar:
                pbar.close()

        grand_cleaned += cleaned_count
        grand_skipped += skipped_count
        grand_errors += error_count

        print(f"   ✅ Xong folder: Clean mới {cleaned_count:,} ảnh | Bỏ qua {skipped_count:,} ảnh | Lỗi: {error_count}")
        print(f"   📁 Lưu tại: {out_dir}")

    total_time = time.time() - start_all
    total_imgs = grand_cleaned + grand_skipped
    avg_speed = total_imgs / max(total_time, 0.001)

    print("\n" + "=" * 75)
    print("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH LÀM SẠCH CỤC BỘ!")
    print("=" * 75)
    print(f"📊 Tổng số ảnh mới đã clean : {grand_cleaned:,} ảnh")
    print(f"⏭️ Tổng số ảnh bỏ qua       : {grand_skipped:,} ảnh (đã có sẵn)")
    print(f"⚠️ Tổng số ảnh bị lỗi       : {grand_errors:,} ảnh")
    print(f"⏱️ Tổng thời gian chạy      : {total_time:.2f} giây (~{total_time/60:.1f} phút)")
    print(f"⚡ Tốc độ xử lý trung bình  : {avg_speed:.1f} ảnh / giây")
    print("=" * 75 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ĐIỂM BẮT ĐẦU CHẠY CHÍNH (MAIN ENTRYPOINT)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chạy làm sạch ảnh hạt lúa crop cục bộ (Local Runner)")
    parser.add_argument("--input", "-i", type=str, default=str(DATASET_PATH), help="Đường dẫn thư mục nguồn")
    parser.add_argument("--audit_only", action="store_true", help="Chỉ kiểm tra số lượng và thoát")
    parser.add_argument("--skip_existing", action="store_true", default=SKIP_EXISTING, help="Bỏ qua ảnh đã làm sạch")
    parser.add_argument("--force_all", action="store_true", help="Ghi đè tất cả (tắt skip_existing)")
    parser.add_argument("--workers", "-w", type=int, default=NUM_WORKERS, help="Số luồng CPU")
    parser.add_argument("--open_ksize", "-k", type=int, default=OPEN_KSIZE, help="Kích thước kernel mở (3, 5, 7)")
    parser.add_argument("--neck_ratio", "-r", type=float, default=MIN_NECK_RATIO, help="Tỷ lệ bề dày eo dính (0.10 - 0.25)")
    parser.add_argument("--re_crop", action="store_true", default=RE_CROP, help="Cắt gọn viền ôm sát hạt")

    args = parser.parse_args()

    input_path = Path(args.input)
    skip = not args.force_all if args.force_all else args.skip_existing

    # 1. Kiểm tra thống kê đầu vào trước
    audit_dataset(input_path, target_subfolder=TARGET_SUBFOLDER)

    if args.audit_only:
        print("🔍 Đã hoàn thành chế độ Audit Only.")
        return

    # 2. Chạy làm sạch hàng loạt
    run_clean_batch_local(
        root_path=input_path,
        target_subfolder=TARGET_SUBFOLDER,
        output_subfolder=OUTPUT_SUBFOLDER,
        min_area=MIN_AREA,
        centrality_weight=CENTRALITY_WEIGHT,
        fill_holes=FILL_HOLES,
        sever_bridges=SEVER_BRIDGES,
        open_ksize=args.open_ksize,
        min_neck_ratio=args.neck_ratio,
        re_crop=args.re_crop,
        padding=PADDING,
        skip_existing=skip,
        num_workers=args.workers,
        show_progress=SHOW_PROGRESS,
    )


if __name__ == "__main__":
    main()
