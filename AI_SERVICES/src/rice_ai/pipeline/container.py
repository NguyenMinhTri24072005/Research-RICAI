"""Container detection and physical geometry stage."""
from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np

from rice_ai.contracts import ContainerResult, PipelineError
from rice_ai.vision.container_detector import detect_container_and_scale


def analyze_container(
    image: np.ndarray,
    diam_cm: float,
    height_cm: float,
    empty_cm: float,
    wall_thickness_cm: float = 0.1,
) -> ContainerResult:
    """Xác thực thông số vật lý và phân tích hình học vật chứa."""
    # 1. Kiểm tra tính hợp lệ của tham số vật lý
    if not math.isfinite(diam_cm) or diam_cm <= 0:
        raise PipelineError(422, "INVALID_INPUT", f"Đường kính ly (diam={diam_cm}cm) phải là số dương hữu hạn.")
    if not math.isfinite(height_cm) or height_cm <= 0:
        raise PipelineError(422, "INVALID_INPUT", f"Chiều cao ly (height={height_cm}cm) phải là số dương hữu hạn.")
    if not math.isfinite(empty_cm) or empty_cm < 0:
        raise PipelineError(422, "INVALID_INPUT", f"Khoảng trống miệng ly (empty={empty_cm}cm) phải >= 0.")
    if empty_cm > height_cm:
        raise PipelineError(422, "INVALID_INPUT", f"Khoảng trống (empty={empty_cm}cm) không được lớn hơn chiều cao ly ({height_cm}cm).")
    if not math.isfinite(wall_thickness_cm) or wall_thickness_cm < 0:
        raise PipelineError(422, "INVALID_INPUT", f"Độ dày thành ly (wall_thickness={wall_thickness_cm}cm) phải >= 0.")

    # 2. Quy đổi sang đơn vị chuẩn mm
    diam_mm = float(diam_cm) * 10.0
    height_mm = float(height_cm) * 10.0
    empty_mm = float(empty_cm) * 10.0
    wall_mm = float(wall_thickness_cm) * 10.0

    # 3. Gọi module thị giác phát hiện miệng ly và tỷ lệ scale
    raw_res = detect_container_and_scale(
        image_input=image,
        inner_diam_mm=diam_mm,
        container_height_mm=height_mm,
        empty_height_mm=empty_mm,
        wall_thickness_mm=wall_mm,
        detect_mode="inner",
    )

    pixels_per_mm = float(raw_res.get("pixels_per_mm", 0.0))
    if pixels_per_mm <= 0 or not math.isfinite(pixels_per_mm):
        raise PipelineError(
            status_code=422,
            error_code="DETECTION_FAILED",
            message="Không thể phát hiện miệng ly hoặc tỷ lệ pixels/mm không hợp lệ.",
        )

    bulk_vol_mm3 = float(raw_res.get("bulk_rice_volume_mm3", raw_res.get("bulk_volume_mm3", 0.0)))
    rice_h_mm = float(raw_res.get("rice_height_mm", max(0.0, height_mm - empty_mm)))
    if bulk_vol_mm3 <= 0 and rice_h_mm > 0:
        bulk_vol_mm3 = math.pi * ((diam_mm / 2.0) ** 2) * rice_h_mm

    return ContainerResult(
        inner_diam_mm=diam_mm,
        container_height_mm=height_mm,
        empty_height_mm=empty_mm,
        rice_height_mm=rice_h_mm,
        bulk_volume_mm3=bulk_vol_mm3,
        pixels_per_mm=pixels_per_mm,
        raw_dict=raw_res,
        visual_overlay=raw_res.get("visual_overlay"),
    )
