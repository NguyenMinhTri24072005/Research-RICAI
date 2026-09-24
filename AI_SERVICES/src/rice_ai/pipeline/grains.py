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
from rice_ai.vision.grain_size_filter import filter_grains_by_size
from rice_ai.vision.uniformity_evaluator import evaluate_batch_uniformity

logger = logging.getLogger("rice_ai.pipeline.grains")


def process_grains(
    image_input: Union[np.ndarray, Path],
    pixels_per_mm: float,
    vision_provider: VisionModelProvider,
    settings: Settings,
    crop_output_dir: Optional[Path] = None,
) -> GrainAnalysis:
    """Run the CODE-compatible segment -> clean -> batch CNN -> geometry flow."""
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

    cleaned_crops: List[Dict[str, Any]] = []
    cleaning_ms = 0.0
    for item in raw_crops:
        rgba = item.get("crop_rgba")
        if rgba is None:
            continue
        t_clean = time.perf_counter()
        cleaned_rgba = clean_single_grain_crop(
            clean_single_grain_crop(
                rgba,
                open_ksize=settings.cleaner_step1_open_ksize,
                min_neck_ratio=settings.cleaner_neck_ratio,
                min_area=settings.cleaner_step1_min_area,
                centrality_weight=settings.cleaner_step1_centrality_weight,
                fill_holes=False,
                sever_bridges=False,
            ),
            open_ksize=settings.cleaner_step2_open_ksize,
            min_neck_ratio=settings.cleaner_neck_ratio,
            min_area=settings.cleaner_step2_min_area,
            centrality_weight=settings.cleaner_step2_centrality_weight,
            fill_holes=True,
            sever_bridges=False,
        )
        cleaning_ms += (time.perf_counter() - t_clean) * 1000
        cleaned_crops.append({**item, "crop_rgba": cleaned_rgba})

    t_cnn = time.perf_counter()
    if cleaned_crops:
        classifier = vision_provider.get_cnn_model()
        accepted, rejected = classifier.filter_grains(
            cleaned_crops,
            target_label="hat_nguyen",
            min_conf=0.50,
        )
    else:
        accepted, rejected = [], []
    classification_ms = (time.perf_counter() - t_cnn) * 1000
    classified_counts: Dict[str, int] = {"hat_nguyen": 0, "hat_khuyet_tat": 0}
    for item in accepted + rejected:
        label = str(item.get("predicted_label", "unknown"))
        classified_counts[label] = classified_counts.get(label, 0) + 1

    whole_grains: List[Dict[str, Any]] = []
    broken_grains: List[Dict[str, Any]] = []
    volumes_px3: List[float] = []
    skipped_count = 0
    measurement_ms = 0.0
    for item in accepted:
        if float(item.get("confidence", 0.0)) < settings.cnn_whole_confidence:
            broken_grains.append(item)
            continue
        t_meas = time.perf_counter()
        try:
            metrics = compute_single_grain_metrics(
                image_input=item["crop_rgba"],
                pixels_per_mm=pixels_per_mm,
                label="hat_nguyen",
            )
            whole_grains.append({
                **metrics,
                "grain_id": item.get("grain_id", item.get("index")),
                "confidence": float(item.get("confidence", 0.0)),
                "probabilities": item.get("probabilities", {}),
                "bbox": item.get("bbox"),
            })
        except Exception as ex:
            skipped_count += 1
            logger.warning("Bỏ qua hạt nguyên đo lỗi: %s", ex)
        finally:
            measurement_ms += (time.perf_counter() - t_meas) * 1000
    broken_grains.extend(rejected)

    filter_input = [
        {"candidate_index": index, "grain_id": grain.get("grain_id"),
         "area_mm2": grain.get("area_mm2"), "volume_mm3": grain.get("volume_mm3")}
        for index, grain in enumerate(whole_grains)
    ]
    size_filter = filter_grains_by_size(
        filter_input,
        k=settings.size_filter_k,
        min_samples=settings.size_filter_min_samples,
        enabled=settings.enable_size_filter,
    )
    kept_indices = {record["candidate_index"] for record in size_filter["kept"]}
    physical_grains = [grain for index, grain in enumerate(whole_grains) if index in kept_indices]
    volumes_px3 = [float(grain["vol_px3"]) for grain in physical_grains if grain.get("vol_px3") is not None]

    t_unif = time.perf_counter()
    grain_volumes_mm3 = [g["volume_mm3"] for g in physical_grains if g.get("volume_mm3") is not None]
    uniformity_res = evaluate_batch_uniformity(grain_volumes_mm3)
    uniformity_ms = (time.perf_counter() - t_unif) * 1000

    # Add labels to the cleaned records used by the artifact writer.
    cleaned_by_id = {item.get("grain_id", item.get("index")): item for item in accepted + rejected}
    for item in cleaned_crops:
        grain_id = item.get("grain_id", item.get("index"))
        decision = cleaned_by_id.get(grain_id)
        if decision:
            item["predicted_label"] = decision.get("predicted_label", "unknown")
            item["confidence"] = decision.get("confidence")
            item["probabilities"] = decision.get("probabilities", {})

    return GrainAnalysis(
        whole_grains=whole_grains,
        broken_grains=broken_grains,
        chalky_grains=[],
        foreign_objects=[],
        total_detected=len(raw_crops),
        classified_counts=classified_counts,
        volumes_px3=volumes_px3,
        skipped_measurement_count=skipped_count,
        uniformity_metrics=uniformity_res,
        timings_ms={
            "sahi_ms": round(sahi_ms, 2),
            "cleaning_ms": round(cleaning_ms, 2),
            "classification_ms": round(classification_ms, 2),
            "measurement_ms": round(measurement_ms, 2),
            "uniformity_ms": round(uniformity_ms, 2),
        },
        physical_grains=physical_grains,
        size_filter_stats=size_filter["stats"],
        size_filter_rejected=size_filter["rejected"],
        raw_crops=raw_crops,
        cleaned_crops=cleaned_crops,
    )