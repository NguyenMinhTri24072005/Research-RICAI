#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 2: BÓC TÁCH HẠT LÚA VỚI SAHI + YOLO-SEGMENTATION (GRAIN SEGMENTER)
===============================================================================
Mục đích:
  - Sử dụng SAHI (Slicing Aided Hyper Inference) kết hợp YOLO-seg.
  - Cắt polygon viền từng hạt lúa và xuất ảnh RGBA trong suốt (nền đen/alpha=0).
===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def segment_grains_sahi(
    detection_model: Any,
    image_path: Union[str, Path, np.ndarray],
    output_crop_dir: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.5,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    min_area_px: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chạy SAHI + YOLO-seg để bóc tách từng hạt lúa từ ảnh chụp độ phân giải cao.
    """
    from sahi.predict import get_sliced_prediction

    if isinstance(image_path, (str, Path)):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}")
        sahi_input = str(image_path)
        base_name = image_path.stem
    elif isinstance(image_path, np.ndarray):
        img_bgr = image_path.copy()
        sahi_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        base_name = "cropped_image"
    else:
        raise ValueError("image_path must be a string, Path, or numpy ndarray")

    if output_crop_dir is not None:
        out_dir = Path(output_crop_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Thực hiện SAHI sliced prediction
    result = get_sliced_prediction(
        sahi_input,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

    predictions = result.object_prediction_list

    extracted_grains: List[Dict[str, Any]] = []
    saved_index = 0

    for idx, obj in enumerate(predictions):
        if obj.score.value < conf_threshold:
            continue
        if obj.mask is None or not obj.mask.segmentation:
            continue

        local_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for polygon in obj.mask.segmentation:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(local_mask, [pts], 255, lineType=cv2.LINE_AA)

        ys, xs = np.where(local_mask == 255)
        if len(ys) == 0 or len(ys) < min_area_px:
            continue

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        crop_bgr = img_bgr[y1:y2, x1:x2].copy()
        crop_mask = local_mask[y1:y2, x1:x2].copy()

        crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_rgba[:, :, 3] = crop_mask

        crop_bgr_clean = crop_bgr.copy()
        crop_bgr_clean[crop_mask == 0] = [0, 0, 0]

        saved_path = None
        if out_dir is not None:
            filename = f"{base_name}_grain_{saved_index:03d}.png"
            file_dest = out_dir / filename
            cv2.imwrite(str(file_dest), crop_rgba)
            saved_path = str(file_dest)

        extracted_grains.append({
            "index": saved_index,
            "grain_id": saved_index,
            "crop_rgba": crop_rgba,
            "crop_bgr": crop_bgr_clean,
            "mask_uint8": crop_mask,
            "bbox": [x1, y1, x2, y2],
            "score": float(obj.score.value),
            "saved_path": saved_path,
        })
        saved_index += 1

    return extracted_grains
