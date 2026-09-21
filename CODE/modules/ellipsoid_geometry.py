#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 4: HÌNH THÁI HỌC 2D & THỂ TÍCH 3D ELLIPSOID HẠT LÚA (ELLIPSOID GEOMETRY)
===============================================================================
Mục đích:
  - Phóng đại nội suy x4 để khử răng cưa viền hạt.
  - Khớp Ellipse (cv2.fitEllipse) để tìm bán trục dài (a) và bán trục ngắn (b).
  - Mô hình hóa thể tích 3D hình cầu dẹt/Ellipsoid: V = (4/3) * pi * a * b * c.
  - Quy đổi toàn bộ kích thước sang đơn vị thực tế (mm, mm², mm³).
  - Thống kê các giá trị Min, Max, Mean, Std cho từng mẫu.
===============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


THICKNESS_RATIO = {
    "hat_nguyen": 1,      # Đo thực tế bằng thước kẹp: dày ≈ 80% chiều rộng
    "hat_khuyet_tat": 0.90,
    "undefined": 0.50,
}


def compute_single_grain_metrics(
    image_input: Union[str, Path, np.ndarray],
    pixels_per_mm: float,
    label: str = "hat_nguyen",
    scale_factor: int = 4,
    thickness_k: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tính toán chi tiết các thông số hình học 2D và thể tích 3D của 1 hạt lúa.

    Parameters
    ----------
    image_input : str, Path hoặc np.ndarray
        Ảnh hạt lúa RGBA 4 kênh hoặc BGR 3 kênh.
    pixels_per_mm : float
        Tỷ lệ quy đổi pixel sang mm tính từ miệng ly.
    label : str, default 'hat_nguyen'
        Nhãn của hạt lúa.
    scale_factor : int, default 4
        Hệ số phóng to để khử răng cưa đường viền hạt lúa.
    thickness_k : float, optional
        Hệ số tỷ lệ chiều dày c/a (mặc định lấy từ THICKNESS_RATIO).

    Returns
    -------
    dict
        Chứa các chỉ số pixel và mm:
        - length_mm, width_mm, thickness_mm
        - area_mm2, volume_mm3
        - a_px, b_px, c_px, area_px2, vol_px3
        - ellipse_params
    """
    if isinstance(image_input, (str, Path)):
        p_str = str(image_input)
        try:
            data = np.fromfile(p_str, dtype=np.uint8)
            img_raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        except Exception:
            img_raw = None
        if img_raw is None:
            img_raw = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"Không thể nạp ảnh: {image_input}")
    else:
        img_raw = image_input.copy()

    # 1. Tách mask hạt lúa từ Alpha channel hoặc threshold
    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        crop_bgr = img_raw[:, :, :3].copy()
        crop_bgr[alpha == 0] = [0, 0, 0]
    else:
        crop_bgr = img_raw.copy()
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(mask) == 0:
        raise ValueError("Ảnh rỗng, không tìm thấy hạt lúa trong mặt nạ!")

    # 2. Phóng to nội suy để bo tròn viền
    scaled_mask = cv2.resize(
        mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
    )

    # 3. Tìm viền contour lớn nhất
    contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Không trích xuất được đường viền hạt lúa!")

    largest_cnt = max(contours, key=cv2.contourArea)

    # Tính diện tích 2D (chia lại cho scale_factor^2)
    area_px2 = cv2.contourArea(largest_cnt) / (scale_factor ** 2)

    # 4. Khớp Ellipse để tìm bán trục a và b
    if largest_cnt.shape[0] >= 5:
        scaled_ellipse = cv2.fitEllipse(largest_cnt)
        (scx, scy), (sMA, sma), angle = scaled_ellipse
        MA = sMA / scale_factor
        ma = sma / scale_factor
        cx = scx / scale_factor
        cy = scy / scale_factor
        ellipse_params = ((cx, cy), (MA, ma), angle)
        a_px = max(MA, ma) / 2.0  # Bán trục dài
        b_px = min(MA, ma) / 2.0  # Bán trục ngắn
        fit_method = "fitEllipse"
    else:
        h, w = mask.shape[:2]
        a_px = max(w, h) / 2.0
        b_px = min(w, h) / 2.0
        cx, cy = w / 2.0, h / 2.0
        ellipse_params = ((cx, cy), (a_px * 2, b_px * 2), 0.0)
        fit_method = "BoundingFallback"

    # 5. Tính bán trục c (chiều dày) và Thể tích 3D Ellipsoid
    #    Chiều dày hạt lúa tỷ lệ với chiều RỘNG (b_px), không phải chiều DÀI (a_px).
    #    Hạt lúa: a > b > c (dài > rộng > dày). k=0.80 đo thực tế bằng thước kẹp.
    k = thickness_k if thickness_k is not None else THICKNESS_RATIO.get(label, 1.0)
    c_px = b_px * k
    vol_px3 = (4.0 / 3.0) * math.pi * a_px * b_px * c_px

    # 6. Quy đổi sang kích thước thực tế (mm, mm², mm³)
    if pixels_per_mm <= 0:
        pixels_per_mm = 1.0

    length_mm = (2.0 * a_px) / pixels_per_mm
    width_mm = (2.0 * b_px) / pixels_per_mm
    thickness_mm = (2.0 * c_px) / pixels_per_mm
    area_mm2 = area_px2 / (pixels_per_mm ** 2)
    volume_mm3 = vol_px3 / (pixels_per_mm ** 3)

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "thickness_mm": float(thickness_mm),
        "area_mm2": float(area_mm2),
        "volume_mm3": float(volume_mm3),
        "a_px": float(a_px),
        "b_px": float(b_px),
        "c_px": float(c_px),
        "area_px2": float(area_px2),
        "vol_px3": float(vol_px3),
        "k_factor": float(k),
        "fit_method": fit_method,
        "ellipse_params": ellipse_params,
    }


def compute_folder_grains_summary(
    grains_folder: Union[str, Path],
    pixels_per_mm: float,
    label: str = "hat_nguyen",
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số thống kê (Min, Max, Mean, Std) cho toàn bộ ảnh hạt lúa trong một thư mục.
    """
    grains_dir = Path(grains_folder)
    if not grains_dir.exists():
        return {}

    image_files = sorted([
        p for p in grains_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if not image_files:
        return {}

    lengths: List[float] = []
    widths: List[float] = []
    thicknesses: List[float] = []
    areas: List[float] = []
    volumes: List[float] = []

    for img_path in image_files:
        try:
            m = compute_single_grain_metrics(img_path, pixels_per_mm, label=label)
            lengths.append(m["length_mm"])
            widths.append(m["width_mm"])
            thicknesses.append(m["thickness_mm"])
            areas.append(m["area_mm2"])
            volumes.append(m["volume_mm3"])
        except Exception:
            continue

    if not volumes:
        return {}

    return {
        "count": len(volumes),
        "length_mean": float(np.mean(lengths)),
        "length_min": float(np.min(lengths)),
        "length_max": float(np.max(lengths)),
        "length_std": float(np.std(lengths)),
        "width_mean": float(np.mean(widths)),
        "width_min": float(np.min(widths)),
        "width_max": float(np.max(widths)),
        "width_std": float(np.std(widths)),
        "thickness_mean": float(np.mean(thicknesses)),
        "thickness_min": float(np.min(thicknesses)),
        "thickness_max": float(np.max(thicknesses)),
        "thickness_std": float(np.std(thicknesses)),
        "area_mean": float(np.mean(areas)),
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
        "area_std": float(np.std(areas)),
        "volume_mean": float(np.mean(volumes)),
        "volume_min": float(np.min(volumes)),
        "volume_max": float(np.max(volumes)),
        "volume_std": float(np.std(volumes)),
    }


def draw_grain_ellipse_overlay(
    crop_input: Union[str, Path, np.ndarray],
    metrics: Dict[str, Any],
    draw_axes: bool = True,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Vẽ viền mô hình 3D Ellipsoid, trục dài (2a), trục ngắn (2b) và tâm lên ảnh hạt lúa để hiển thị trực quan.
    """
    if isinstance(crop_input, (str, Path)):
        img_raw = cv2.imread(str(crop_input), cv2.IMREAD_UNCHANGED)
    else:
        img_raw = crop_input.copy()

    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        vis = cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        vis = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    ep = metrics.get("ellipse_params")
    if ep is not None:
        (cx, cy), (MA, ma), angle = ep
        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(MA / 2.0)), int(round(ma / 2.0)))

        # 1. Vẽ viền elip màu vàng neon (mô hình 3D)
        cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 240, 255), 2, lineType=cv2.LINE_AA)

        # 2. Vẽ 2 trục chính (Trục dài 2a màu xanh lá, Trục ngắn 2b màu đỏ)
        if draw_axes:
            theta_rad = math.radians(angle)
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

            # Trục dài 2a
            x_maj1 = int(round(cx + (MA / 2.0) * sin_t))
            y_maj1 = int(round(cy - (MA / 2.0) * cos_t))
            x_maj2 = int(round(cx - (MA / 2.0) * sin_t))
            y_maj2 = int(round(cy + (MA / 2.0) * cos_t))
            cv2.line(vis, (x_maj1, y_maj1), (x_maj2, y_maj2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # Trục ngắn 2b
            x_min1 = int(round(cx + (ma / 2.0) * cos_t))
            y_min1 = int(round(cy + (ma / 2.0) * sin_t))
            x_min2 = int(round(cx - (ma / 2.0) * cos_t))
            y_min2 = int(round(cy - (ma / 2.0) * sin_t))
            cv2.line(vis, (x_min1, y_min1), (x_min2, y_min2), (255, 60, 60), 2, lineType=cv2.LINE_AA)

            # Vẽ tâm hạt lúa
            cv2.circle(vis, center, 3, (255, 255, 255), -1)

    return vis
