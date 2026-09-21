#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE: KHỬ NHIỄU, BẺ EO DÍNH & CÔ LẬP HẠT LÚA CHÍNH (GRAIN CROP CLEANER)
===============================================================================
Mục đích:
  - Loại bỏ các mảnh vụn, góc hạt lúa thừa xung quanh hạt lúa chính trong ảnh crop.
  - XỬ LÝ ĐẶC TRỊ TRƯỜNG HỢP HẠT CHÍNH BỊ DÍNH VÀO MẢNH VỤN QUA CẦU PIXEL HẸP (1-5 pixel)
    bằng thuật toán Bẻ cầu nối (Morphological Opening) kết hợp Biến đổi khoảng cách (Distance Transform Watershed).
  - Giữ lại duy nhất 1 đối tượng hạt lúa chủ đạo (nằm ở trung tâm crop và có diện tích lớn nhất).
  - Xuất ra ảnh 4 kênh RGBA với nền trong suốt (Alpha = 0, RGB = [0, 0, 0] cho vùng ngoài).
  - Hỗ trợ xử lý 1 ảnh đơn lẻ hoặc tự động quét đệ quy toàn bộ thư mục dữ liệu OUTPUT_CROPPED_GRAINS.
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """
    Đọc file ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt có dấu.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(str(path), flags)


def imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    """
    Lưu file ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt có dấu.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() if p.suffix else ".png"
        if not ext:
            ext = ".png"
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
    """
    Thuật toán cắt đứt các cầu nối pixel hẹp (Neck/Bridge Severing) giữa hạt chính và mảnh vụn góc,
    sau đó sử dụng Watershed trên bản đồ khoảng cách (Distance Transform) để mở rộng hạt chính
    về đúng kích thước gốc mà không bị dính lại mảnh vụn.
    """
    h, w = raw_mask.shape[:2]
    if cv2.countNonZero(raw_mask) == 0:
        return raw_mask

    # 1. Biến đổi khoảng cách Euclidean (Distance Transform)
    dist = cv2.distanceTransform(raw_mask, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 1.0:
        return raw_mask

    # 2. Phép toán Mở (Morphological Opening) bẻ gãy các cầu nối hẹp 1-3 pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # 3. Trích xuất hạt giống lõi (Core Seeds) dựa trên Distance Transform
    # Phần thân hạt lúa dày có giá trị distance cao, phần cầu dính hẹp có distance rất thấp
    _, core_seeds = cv2.threshold(dist, max_dist * min_neck_ratio, 255, cv2.THRESH_BINARY)
    core_seeds = core_seeds.astype(np.uint8)

    # Giao giữa mask đã mở và lõi trung tâm
    refined_seeds = cv2.bitwise_and(opened, core_seeds)

    # 4. Phân tích thành phần liên thông trên các hạt giống đã tách rời
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_seeds, connectivity=8)

    if n_labels <= 1:
        # Fallback thử với opened
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    if n_labels <= 1:
        # Nếu chỉ có 1 khối hoặc không có, trả về mask đóng chuẩn
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)

    # 5. Chấm điểm độ nổi bật (Salient Centrality) để chọn hạt giống của HẠT LÚA CHÍNH
    img_cx = w / 2.0
    img_cy = h / 2.0
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

    # 6. Watershed trên Inverse Distance Transform để tái tạo viền mịn màng gốc
    # Đánh dấu Marker:
    # Marker = 1: Vùng nền (bên ngoài raw_mask)
    # Marker = 2: Hạt lúa chính
    # Marker = 3, 4, ...: Các mảnh vụn góc
    markers = np.zeros((h, w), dtype=np.int32)
    markers[raw_mask == 0] = 1
    markers[labels == best_idx] = 2

    marker_id = 3
    for i in range(1, n_labels):
        if i != best_idx:
            markers[labels == i] = marker_id
            marker_id += 1

    # Tạo ảnh gradient cho Watershed từ Distance Transform đảo ngược
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_inv = 255 - dist_norm
    ws_input = cv2.cvtColor(dist_inv, cv2.COLOR_GRAY2BGR)

    cv2.watershed(ws_input, markers)

    # Lấy lưu vực của hạt lúa chính (Marker == 2)
    clean_mask = np.where(markers == 2, 255, 0).astype(np.uint8)
    
    # Phục hồi viền bị mất do thuật toán watershed (đường biên luôn bị đánh dấu là -1)
    kernel_restore = np.ones((3, 3), np.uint8)
    clean_mask = cv2.bitwise_and(cv2.dilate(clean_mask, kernel_restore, iterations=1), raw_mask)

    # Lấp đầy các lỗ rỗng bên trong thân hạt lúa
    cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(clean_mask, cnts, -1, 255, -1)

    return clean_mask


def clean_single_grain_crop(
    image_input: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    min_area: int = 20,
    centrality_weight: float = 2.5,
    fill_holes: bool = True,
    sever_bridges: bool = True,
    open_ksize: int = 3,
    min_neck_ratio: float = 0.15,
    re_crop: bool = False,
    padding: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Khử bỏ toàn bộ các đối tượng nhiễu xung quanh trong một ảnh hạt lúa crop,
    kể cả trường hợp hạt chính bị dính vào mảnh vụn qua vài pixel hẹp,
    chỉ giữ lại duy nhất 1 đối tượng hạt lúa chính.

    Parameters
    ----------
    image_input : str, Path hoặc np.ndarray
        Đường dẫn file ảnh hoặc ma trận ảnh (RGBA 4 kênh hoặc BGR 3 kênh).
    output_path : str hoặc Path, optional
        Đường dẫn lưu file ảnh kết quả (.png RGBA). Nếu None thì chỉ trả về ma trận numpy.
    min_area : int, default 20
        Diện tích tối thiểu (pixel) để xét một vùng là đối tượng hợp lệ.
    centrality_weight : float, default 2.5
        Trọng số ưu tiên đối tượng nằm gần tâm ảnh crop (tránh lấy nhầm hạt lúa góc biên).
    fill_holes : bool, default True
        Tự động lấp đầy các lỗ rỗng/vết lõm nhỏ bên trong thân hạt lúa chính.
    sever_bridges : bool, default True
        Bật thuật toán bẻ gãy các cầu nối pixel hẹp (1-5px) dính với mảnh vụn lân cận.
    open_ksize : int, default 3
        Kích thước kernel mở để bẻ cầu nối pixel (3 hoặc 5).
    min_neck_ratio : float, default 0.15
        Tỷ lệ bề dày eo dính so với bề dày lớn nhất của thân hạt (0.10 - 0.25).
    re_crop : bool, default False
        Nếu True: Cắt gọn lại viền ảnh (tight bounding box) ôm sát hạt lúa chính sau khi làm sạch.
        Nếu False: Giữ nguyên kích thước chiều rộng/cao gốc của ảnh crop.
    padding : int, default 2
        Khoảng đệm (pixel) khi re_crop=True.

    Returns
    -------
    np.ndarray
        Ảnh RGBA 4 kênh đã làm sạch, nền trong suốt (Alpha = 0).
    """
    # 1. Đọc và chuẩn hóa ảnh đầu vào
    if isinstance(image_input, (str, Path)):
        img_raw = imread_unicode(image_input, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh tại: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_raw = image_input.copy()
    else:
        raise ValueError("image_input phải là đường dẫn file (str, Path) hoặc numpy ndarray.")

    is_rgba = (len(img_raw.shape) == 3 and img_raw.shape[2] == 4)
    h, w = img_raw.shape[:2]

    if h == 0 or w == 0:
        return img_raw

    # 2. Trích xuất Binary Mask ban đầu của tiền cảnh
    if is_rgba:
        alpha = img_raw[:, :, 3]
        if np.any(alpha == 0):
            raw_mask = (alpha > 10).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, raw_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw
        _, raw_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(raw_mask) == 0:
        res = cv2.cvtColor(img_raw[:, :, :3] if is_rgba else img_raw, cv2.COLOR_BGR2BGRA)
        res[:, :, :] = 0
        if output_path is not None:
            imwrite_unicode(output_path, res)
        return res

    # 3. Thực hiện bẻ gãy cầu dính & tách hạt chính
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
            img_cx = w / 2.0
            img_cy = h / 2.0
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

    # 4. Lấp đầy lỗ rỗng nếu được yêu cầu
    if fill_holes:
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(clean_mask, cnts, -1, 255, -1)

    # 5. Xây dựng ảnh RGBA trong suốt
    bgr = img_raw[:, :, :3].copy() if is_rgba else img_raw.copy()
    bgr[clean_mask == 0] = [0, 0, 0]

    cleaned_rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    cleaned_rgba[:, :, 3] = clean_mask

    # 6. Cắt gọn khung viền (Tùy chọn re_crop)
    if re_crop:
        ys, xs = np.where(clean_mask == 255)
        if len(ys) > 0:
            y1 = max(0, int(ys.min()) - padding)
            y2 = min(h, int(ys.max()) + 1 + padding)
            x1 = max(0, int(xs.min()) - padding)
            x2 = min(w, int(xs.max()) + 1 + padding)
            cleaned_rgba = cleaned_rgba[y1:y2, x1:x2].copy()

    # 7. Lưu file nếu có yêu cầu
    if output_path is not None:
        imwrite_unicode(output_path, cleaned_rgba)

    return cleaned_rgba


def clean_dataset_cropped_grains(
    root_path: Union[str, Path],
    target_subfolder: str = "OUTPUT_CROPPED_GRAINS",
    output_subfolder: str = "OUTPUT_CLEANED_GRAINS",
    min_area: int = 20,
    centrality_weight: float = 2.5,
    fill_holes: bool = True,
    sever_bridges: bool = True,
    open_ksize: int = 3,
    min_neck_ratio: float = 0.15,
    re_crop: bool = False,
    padding: int = 2,
    skip_existing: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quét đệ quy toàn bộ thư mục dữ liệu, tìm tất cả các thư mục con có tên `target_subfolder`
    (ví dụ: OUTPUT_CROPPED_GRAINS), làm sạch từng ảnh hạt lúa và lưu sang thư mục cùng cấp `output_subfolder`.
    
    Parameters
    ----------
    open_ksize : int, default 3
        Kích thước kernel mở (3, 5, 7) dùng để bẻ gãy cầu nối pixel dính giữa hạt chính và vật phụ.
    min_neck_ratio : float, default 0.15
        Tỷ lệ bề dày eo dính so với thân hạt (0.10 - 0.25).
    skip_existing : bool, default False
        Nếu True, tự động bỏ qua các ảnh đã có sẵn trong output_subfolder để tiết kiệm thời gian.
    show_progress : bool, default True
        Hiển thị thanh tiến trình (tqdm) hoặc thông báo % tiến độ theo thời gian thực.
    """
    root_p = Path(root_path)
    if not root_p.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục gốc tại: {root_p}")

    start_time = time.time()
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    # Kiểm tra hỗ trợ tqdm
    has_tqdm = False
    if show_progress:
        try:
            from tqdm.auto import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

    target_dirs: List[Path] = []
    target_lower = target_subfolder.lower()

    if root_p.name.lower() == target_lower:
        target_dirs.append(root_p)
    else:
        for p in root_p.rglob("*"):
            if p.is_dir() and p.name.lower() == target_lower:
                target_dirs.append(p)

    if not target_dirs:
        direct_imgs = [f for f in root_p.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        if direct_imgs:
            target_dirs.append(root_p)

    if verbose:
        print("=" * 75)
        print("🧹 MODULE KHỬ NHIỄU, BẺ EO DÍNH & LÀM SẠCH ẢNH HẠT LÚA CROP")
        print("=" * 75)
        print(f"📂 Thư mục gốc           : {root_p}")
        print(f"🎯 Thư mục mục tiêu      : {target_subfolder}")
        print(f"📁 Thư mục xuất cùng cấp : {output_subfolder}")
        print(f"🔎 Tìm thấy              : {len(target_dirs)} thư mục cần xử lý.")
        print(f"✂️ Cơ chế bẻ cầu nối eo : {'BẬT (Sever Bridges Active)' if sever_bridges else 'TẮT'}")
        print(f"⚡ Bỏ qua ảnh đã có      : {'BẬT (Skip Existing)' if skip_existing else 'TẮT (Ghi đè tất cả)'}")
        print("=" * 75)

    total_images_processed = 0
    total_images_skipped = 0
    total_images_error = 0
    processed_folders_info: List[Dict[str, Any]] = []

    for idx, in_dir in enumerate(target_dirs, 1):
        if in_dir.name.lower() == target_lower:
            out_dir = in_dir.parent / output_subfolder
        else:
            out_dir = in_dir / output_subfolder

        out_dir.mkdir(parents=True, exist_ok=True)

        img_files: List[Path] = []
        for p in in_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_extensions:
                if output_subfolder.lower() not in p.parts:
                    img_files.append(p)

        if verbose:
            parent_tag = in_dir.parent.name if in_dir.parent != root_p else in_dir.name
            print(f"\n[{idx}/{len(target_dirs)}] 📁 Đang xử lý: {parent_tag}/{in_dir.name}")
            print(f"   ↳ Tổng số ảnh tìm thấy: {len(img_files):,} ảnh")

        folder_cleaned_count = 0
        folder_skipped_count = 0
        folder_error_count = 0

        # Cấu hình thanh tiến trình
        if show_progress and has_tqdm:
            iter_files = tqdm(
                img_files,
                desc=f"   ⏳ Tiến trình [{idx}/{len(target_dirs)}]",
                unit="ảnh",
                ncols=95,
                leave=True
            )
        else:
            iter_files = img_files

        last_print_time = time.time()

        for i_file, img_path in enumerate(iter_files, 1):
            rel_subpath = img_path.relative_to(in_dir)
            dest_path = out_dir / rel_subpath.with_suffix(".png")

            # Bỏ qua nếu ảnh đã tồn tại
            if skip_existing and dest_path.exists():
                folder_skipped_count += 1
                if show_progress and has_tqdm:
                    iter_files.set_postfix({
                        "clean": folder_cleaned_count,
                        "bỏ qua": folder_skipped_count
                    })
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)

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
                folder_cleaned_count += 1
                if show_progress and has_tqdm:
                    iter_files.set_postfix({
                        "clean": folder_cleaned_count,
                        "bỏ qua": folder_skipped_count
                    })
            except Exception as e:
                folder_error_count += 1
                if verbose and not has_tqdm:
                    print(f"   ⚠️ Lỗi khi xử lý {img_path.name}: {e}")

            # Fallback log khi không có tqdm (in mỗi 50 ảnh hoặc 3 giây)
            if verbose and not has_tqdm:
                curr_t = time.time()
                if i_file % 50 == 0 or i_file == len(img_files) or (curr_t - last_print_time >= 3.0):
                    pct = (i_file / len(img_files)) * 100.0
                    rate = i_file / max(curr_t - start_time, 0.001)
                    remain_s = (len(img_files) - i_file) / max(rate, 0.001)
                    print(f"   ⏳ [{i_file}/{len(img_files)}] ({pct:.1f}%) | Clean: {folder_cleaned_count} | Bỏ qua: {folder_skipped_count} | Tốc độ: {rate:.1f} ảnh/s | Còn: {remain_s:.0f}s")
                    last_print_time = curr_t

        total_images_processed += folder_cleaned_count
        total_images_skipped += folder_skipped_count
        total_images_error += folder_error_count

        processed_folders_info.append({
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "cleaned_count": folder_cleaned_count,
            "skipped_count": folder_skipped_count,
            "error_count": folder_error_count,
            "total_in_folder": len(img_files),
        })

        if verbose:
            print(f"   ✅ Hoàn tất thư mục: Đã clean mới {folder_cleaned_count:,} ảnh | Bỏ qua {folder_skipped_count:,} ảnh đã có | Lỗi: {folder_error_count}")
            print(f"   📁 Thư mục lưu: {out_dir}")

    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 75)
        print("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH LÀM SẠCH DATASET!")
        print(f"📊 Tổng số ảnh mới đã làm sạch : {total_images_processed:,} ảnh")
        if total_images_skipped > 0:
            print(f"⏭️ Tổng số ảnh bỏ qua (đã có sẵn): {total_images_skipped:,} ảnh")
        if total_images_error > 0:
            print(f"⚠️ Tổng số ảnh bị lỗi          : {total_images_error:,} ảnh")
        print(f"⏱️ Tổng thời gian thực thi     : {elapsed:.2f} giây ({((total_images_processed + total_images_skipped) / max(elapsed, 0.001)):.1f} ảnh/giây)")
        print("=" * 75)

    return {
        "total_cleaned": total_images_processed,
        "total_skipped": total_images_skipped,
        "total_errors": total_images_error,
        "elapsed_seconds": elapsed,
        "processed_folders": processed_folders_info,
    }


class GrainCropCleaner:
    """
    Class giao diện hướng đối tượng (OOP) quản lý việc làm sạch và khử nhiễu ảnh hạt lúa.
    """

    def __init__(
        self,
        min_area: int = 20,
        centrality_weight: float = 2.5,
        fill_holes: bool = True,
        sever_bridges: bool = True,
        open_ksize: int = 3,
        min_neck_ratio: float = 0.15,
        re_crop: bool = False,
        padding: int = 2,
    ):
        self.min_area = min_area
        self.centrality_weight = centrality_weight
        self.fill_holes = fill_holes
        self.sever_bridges = sever_bridges
        self.open_ksize = open_ksize
        self.min_neck_ratio = min_neck_ratio
        self.re_crop = re_crop
        self.padding = padding

    def clean_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Làm sạch 1 ảnh hạt lúa đơn lẻ."""
        return clean_single_grain_crop(
            image_input=image_input,
            output_path=output_path,
            min_area=self.min_area,
            centrality_weight=self.centrality_weight,
            fill_holes=self.fill_holes,
            sever_bridges=self.sever_bridges,
            open_ksize=self.open_ksize,
            min_neck_ratio=self.min_neck_ratio,
            re_crop=self.re_crop,
            padding=self.padding,
        )

    def clean_dataset(
        self,
        root_path: Union[str, Path],
        target_subfolder: str = "OUTPUT_CROPPED_GRAINS",
        output_subfolder: str = "OUTPUT_CLEANED_GRAINS",
        skip_existing: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Làm sạch hàng loạt thư mục chứa dữ liệu crop."""
        return clean_dataset_cropped_grains(
            root_path=root_path,
            target_subfolder=target_subfolder,
            output_subfolder=output_subfolder,
            min_area=self.min_area,
            centrality_weight=self.centrality_weight,
            fill_holes=self.fill_holes,
            sever_bridges=self.sever_bridges,
            open_ksize=self.open_ksize,
            min_neck_ratio=self.min_neck_ratio,
            re_crop=self.re_crop,
            padding=self.padding,
            skip_existing=skip_existing,
            show_progress=show_progress,
            verbose=verbose,
            **kwargs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khử nhiễu, bẻ eo dính & cô lập hạt lúa chính trong ảnh crop.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Đường dẫn đến 1 ảnh hoặc thư mục gốc dataset")
    parser.add_argument("--output", "-o", type=str, default=None, help="Đường dẫn file/thư mục xuất kết quả")
    parser.add_argument("--target_subfolder", type=str, default="OUTPUT_CROPPED_GRAINS", help="Tên thư mục con cần quét")
    parser.add_argument("--output_subfolder", type=str, default="OUTPUT_CLEANED_GRAINS", help="Tên thư mục đích cùng cấp")
    parser.add_argument("--re_crop", action="store_true", help="Cắt gọn lại khung hình ôm sát hạt lúa")

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        out_f = args.output or str(input_path.with_name(f"{input_path.stem}_cleaned.png"))
        res = clean_single_grain_crop(input_path, output_path=out_f, re_crop=args.re_crop)
        print(f"✅ Đã làm sạch ảnh {input_path.name} -> {out_f}")
    elif input_path.is_dir():
        out_sub = args.output_subfolder if not args.output else args.output
        clean_dataset_cropped_grains(
            root_path=input_path,
            target_subfolder=args.target_subfolder,
            output_subfolder=out_sub,
            re_crop=args.re_crop,
            verbose=True,
        )
    else:
        print(f"❌ Đường dẫn không tồn tại: {input_path}")
