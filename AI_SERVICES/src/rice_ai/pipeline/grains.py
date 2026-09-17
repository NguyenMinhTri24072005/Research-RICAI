"""Grain segmentation, cleaning, classification, and 3D geometry stage."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rice_ai.contracts import GrainAnalysis
from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.settings import Settings
from rice_ai.vision.ellipsoid_geometry import compute_single_grain_metrics
from rice_ai.vision.grain_crop_cleaner import clean_single_grain_crop
from rice_ai.vision.grain_segmenter import segment_grains_sahi
from rice_ai.vision.uniformity_evaluator import evaluate_batch_uniformity

logger = logging.getLogger("rice_ai.pipeline.grains")


def process_grains(
    image_input: Union[np.ndarray, Path],
    pixels_per_mm: float,
    vision_provider: VisionModelProvider,
    settings: Settings,
    crop_output_dir: Optional[Path] = None,
) -> GrainAnalysis:
    """Điều phối toàn bộ quy trình: SAHI -> Cleaner -> CNN -> 3D Geometry -> Uniformity."""
    # 1. Bóc tách hạt lúa qua YOLO SAHI
    t_sahi = time.perf_counter()
    detection_model = vision_provider.get_yolo_model(confidence=settings.sahi_conf_threshold)
    raw_crops = segment_grains_sahi(
        detection_model=detection_model,
        image_path=image_input,
        output_crop_dir=crop_output_dir,
        conf_threshold=settings.sahi_conf_threshold,
        slice_size=settings.sahi_slice_height,
        overlap_ratio=settings.sahi_overlap_height_ratio,
    )
    sahi_ms = (time.perf_counter() - t_sahi) * 1000

    total_detected = len(raw_crops)

    # 2. Làm sạch, phân loại CNN và đo 3D
    classifier = vision_provider.get_cnn_model()

    whole_grains: List[Dict[str, Any]] = []
    broken_grains: List[Dict[str, Any]] = []
    chalky_grains: List[Dict[str, Any]] = []
    foreign_objects: List[Dict[str, Any]] = []
    volumes_px3: List[float] = []

    classified_counts: Dict[str, int] = {
        "hat_nguyen": 0,
        "hat_khuyet_tat": 0,
    }
    skipped_count: int = 0
    cleaning_ms: float = 0.0
    classification_ms: float = 0.0
    measurement_ms: float = 0.0

    for item in raw_crops:
        rgba = item.get("crop_rgba")
        if rgba is None:
            continue

        t_clean = time.perf_counter()
        cleaned_rgba = clean_single_grain_crop(
            rgba,
            min_neck_ratio=settings.cleaner_neck_ratio,
            sever_bridges=True,
        )
        cleaning_ms += (time.perf_counter() - t_clean) * 1000

        t_cnn = time.perf_counter()
        cnn_result = classifier.predict(cleaned_rgba)
        classification_ms += (time.perf_counter() - t_cnn) * 1000
        label = cnn_result.get("label", "unknown")
        classified_counts[label] = classified_counts.get(label, 0) + 1

        if label == "hat_nguyen":
            t_meas = time.perf_counter()
            try:
                metrics = compute_single_grain_metrics(
                    image_input=cleaned_rgba,
                    pixels_per_mm=pixels_per_mm,
                    label="hat_nguyen",
                )
                whole_grains.append(metrics)
                if "vol_px3" in metrics and metrics["vol_px3"] is not None:
                    volumes_px3.append(float(metrics["vol_px3"]))
            except Exception as ex:
                skipped_count += 1
                logger.warning(f"Bỏ qua hạt nguyên đo lỗi: {ex}")
            finally:
                measurement_ms += (time.perf_counter() - t_meas) * 1000
        else:
            broken_grains.append({"crop_rgba": cleaned_rgba, "label": label})

    # 3. Đánh giá độ đồng đều thể tích các hạt nguyên
    t_unif = time.perf_counter()
    grain_volumes_mm3 = [g["volume_mm3"] for g in whole_grains if g.get("volume_mm3") is not None]
    uniformity_res = evaluate_batch_uniformity(grain_volumes_mm3)
    uniformity_ms = (time.perf_counter() - t_unif) * 1000

    timings = {
        "sahi_ms": round(sahi_ms, 2),
        "cleaning_ms": round(cleaning_ms, 2),
        "classification_ms": round(classification_ms, 2),
        "measurement_ms": round(measurement_ms, 2),
        "uniformity_ms": round(uniformity_ms, 2),
    }

    return GrainAnalysis(
        whole_grains=whole_grains,
        broken_grains=broken_grains,
        chalky_grains=chalky_grains,
        foreign_objects=foreign_objects,
        total_detected=total_detected,
        classified_counts=classified_counts,
        volumes_px3=volumes_px3,
        skipped_measurement_count=skipped_count,
        uniformity_metrics=uniformity_res,
        timings_ms=timings,
    )
