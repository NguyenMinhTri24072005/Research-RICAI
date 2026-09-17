"""Pipeline Runner - Orchestrates all pipeline stages."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from rice_ai.contracts import (
    EstimateSet,
    PipelineError,
    PipelineResult,
    PredictionInput,
)
from rice_ai.estimation.feature_schema import validate_feature_vector
from rice_ai.estimation.fusion import select_final_estimate
from rice_ai.estimation.geometry import compute_geometry_estimate
from rice_ai.estimation.regression import predict_regression
from rice_ai.estimation.weight import compute_weight_estimate
from rice_ai.models.regression_loader import LoadedRegressionProvider
from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.pipeline.container import analyze_container
from rice_ai.pipeline.features import assemble_31_features
from rice_ai.pipeline.grains import process_grains
from rice_ai.pipeline.image_io import RequestWorkspace, decode_image
from rice_ai.settings import Settings

logger = logging.getLogger("rice_ai.pipeline.runner")


class RicePipeline:
    """Bộ điều phối luồng xử lý suy luận Rice Vision AI."""

    def __init__(
        self,
        settings: Settings,
        vision_provider: VisionModelProvider,
        regression_provider: LoadedRegressionProvider,
    ):
        self.settings = settings
        self.vision_provider = vision_provider
        self.regression_provider = regression_provider

    def run(
        self,
        input_params: PredictionInput,
        image_bytes: bytes,
        request_id: str,
    ) -> PipelineResult:
        """Thực thi pipeline tuần tự và trả về kết quả chuẩn hóa."""
        t_total_start = time.perf_counter()
        timings: Dict[str, float] = {}
        warnings_list = []

        # ── Bước 1: Giải mã ảnh ──────────────────────────────────────────────
        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        timings["decode_ms"] = (time.perf_counter() - t0) * 1000

        # ── Bước 2: Phân tích vật lý và miệng ly ──────────────────────────────
        t0 = time.perf_counter()
        container_res = analyze_container(
            image=img,
            diam_cm=input_params.diam,
            height_cm=input_params.height,
            empty_cm=input_params.empty,
            wall_thickness_cm=input_params.wall_thickness,
        )
        timings["container_ms"] = (time.perf_counter() - t0) * 1000

        # ── Bước 3: SAHI -> Cleaner -> CNN -> 3D Geometry ────────────────────
        t0 = time.perf_counter()
        with RequestWorkspace(debug=input_params.debug) as ws:
            grain_analysis = process_grains(
                image_input=img,
                pixels_per_mm=container_res.pixels_per_mm,
                vision_provider=self.vision_provider,
                settings=self.settings,
                crop_output_dir=ws.dir_path,
            )
        timings["segmentation_ms"] = grain_analysis.timings_ms.get("sahi_ms", (time.perf_counter() - t0) * 1000)
        timings["cleaning_ms"] = grain_analysis.timings_ms.get("cleaning_ms", 0.0)
        timings["classification_ms"] = grain_analysis.timings_ms.get("classification_ms", 0.0)
        timings["measurement_ms"] = grain_analysis.timings_ms.get("measurement_ms", 0.0)
        timings["uniformity_ms"] = grain_analysis.timings_ms.get("uniformity_ms", 0.0)

        if grain_analysis.skipped_measurement_count > 0:
            warnings_list.append(
                f"SKIPPED_GRAINS: Đã bỏ qua {grain_analysis.skipped_measurement_count} "
                "hạt do trích xuất số đo 3D thất bại."
            )

        # ── Bước 4: Ước lượng hình học và cân mẫu ────────────────────────────
        t0 = time.perf_counter()
        geometry_est = compute_geometry_estimate(
            bulk_volume_mm3=container_res.bulk_volume_mm3,
            pixels_per_mm=container_res.pixels_per_mm,
            volumes_px3_list=grain_analysis.volumes_px3,
            packing_fraction=self.settings.packing_fraction_geometry,
        )
        weight_est = compute_weight_estimate(
            weight_total=input_params.weight_total,
            sample_weight=input_params.sample_weight,
            sample_count=input_params.sample_count,
        )
        timings["geometry_ms"] = (time.perf_counter() - t0) * 1000

        # ── Bước 5: Tập hợp vector 31 đặc trưng & Validate ───────────────────
        form_inputs = {
            "weight_g": input_params.weight_total,
            "empty_height_mm": container_res.empty_height_mm,
            "inner_diameter_mm": container_res.inner_diam_mm,
            "container_height_mm": container_res.container_height_mm,
        }
        features_dict = assemble_31_features(
            container=container_res,
            grain_analysis=grain_analysis,
            form_inputs=form_inputs,
        )

        feat_val = validate_feature_vector(features_dict, require_grains=True)
        if feat_val.warnings:
            warnings_list.extend(feat_val.warnings)

        # ── Bước 6: Dự đoán hồi quy ──────────────────────────────────────────
        t0 = time.perf_counter()
        regression_est: Optional[float] = None
        regression_model_name: Optional[str] = None

        mode = input_params.estimator_mode.strip().lower() if input_params.estimator_mode else "auto"

        if mode in ("auto", "regression"):
            if not feat_val.valid:
                if mode == "regression":
                    raise PipelineError(
                        status_code=422,
                        error_code=feat_val.error_code or "INVALID_INPUT",
                        message=feat_val.error_message or "Vector đặc trưng không hợp lệ cho hồi quy.",
                    )
                warnings_list.append(
                    f"REGRESSION_SKIPPED: {feat_val.error_message or 'Vector đặc trưng không đạt chuẩn'}"
                )
            else:
                try:
                    try:
                        loaded_reg = self.regression_provider.get_regression()
                    except Exception as load_ex:
                        if mode == "regression":
                            raise PipelineError(
                                status_code=503,
                                error_code="MODEL_UNAVAILABLE",
                                message=f"Mô hình hồi quy không khả dụng: {load_ex}",
                            )
                        raise

                    try:
                        pred_val, m_name = predict_regression(features_dict, loaded_reg)
                        regression_est = pred_val
                        regression_model_name = m_name
                        if loaded_reg.warnings:
                            for w in loaded_reg.warnings:
                                if w not in warnings_list:
                                    warnings_list.append(w)
                    except Exception as inf_ex:
                        if mode == "regression":
                            raise PipelineError(
                                status_code=500,
                                error_code="INFERENCE_FAILED",
                                message=f"Dự đoán hồi quy thất bại: {inf_ex}",
                            )
                        raise
                except PipelineError:
                    raise
                except Exception as ex:
                    logger.warning(f"[{request_id}] Dự đoán hồi quy thất bại: {ex}")
                    warnings_list.append(f"INFERENCE_FAILED: {ex}")

        timings["regression_ms"] = (time.perf_counter() - t0) * 1000

        # ── Bước 7: Quyết định kết quả cuối cùng qua fusion policy ────────────
        estimates = select_final_estimate(
            mode=mode,
            geometry_est=geometry_est,
            weight_est=weight_est,
            regression_est=regression_est,
            regression_model_name=regression_model_name,
        )

        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000

        return PipelineResult(
            request_id=request_id,
            estimates=estimates,
            container=container_res,
            grain_analysis=grain_analysis,
            features_31=features_dict,
            timings_ms=timings,
            warnings=warnings_list,
        )
