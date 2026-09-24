"""Pipeline Runner - Orchestrates the deployable CODE-compatible stages."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from rice_ai.contracts import PipelineError, PipelineResult, PredictionInput
from rice_ai.estimation.feature_schema import validate_feature_vector
from rice_ai.estimation.fusion import select_final_estimate
from rice_ai.estimation.geometry import compute_geometry_estimate
from rice_ai.estimation.regression import predict_regression
from rice_ai.estimation.weight import compute_weight_estimate
from rice_ai.models.regression_loader import LoadedRegressionProvider, LoadedRegression
from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.pipeline.artifacts import InferenceArtifactWriter
from rice_ai.pipeline.container import analyze_container
from rice_ai.pipeline.debug_visuals import build_debug_visuals
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
        self.artifact_writer = InferenceArtifactWriter(
            settings.get_results_root(), enabled=settings.save_inference_artifacts
        )

    def run(
        self,
        input_params: PredictionInput,
        image_bytes: bytes,
        request_id: str,
    ) -> PipelineResult:
        """Execute the pipeline and persist a request-scoped result when enabled."""
        t_total_start = time.perf_counter()
        timings: Dict[str, float] = {}
        warnings_list = []

        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        timings["decode_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        container_res = analyze_container(
            image=img,
            diam_mm=input_params.diam,
            height_mm=input_params.height,
            empty_mm=input_params.empty,
            wall_thickness_mm=input_params.wall_thickness,
        )
        timings["container_ms"] = (time.perf_counter() - t0) * 1000

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
        for key in ("cleaning_ms", "classification_ms", "measurement_ms", "uniformity_ms"):
            timings[key] = grain_analysis.timings_ms.get(key, 0.0)

        if grain_analysis.skipped_measurement_count > 0:
            warnings_list.append(
                f"SKIPPED_GRAINS: Đã bỏ qua {grain_analysis.skipped_measurement_count} hạt do trích xuất số đo 3D thất bại."
            )

        t0 = time.perf_counter()
        physical_mean_volume_mm3 = grain_analysis.uniformity_metrics.get("mean_clean")
        if physical_mean_volume_mm3 is not None and float(physical_mean_volume_mm3) > 0:
            geometry_est = int(round(
                container_res.bulk_volume_mm3 * self.settings.packing_fraction_geometry
                / float(physical_mean_volume_mm3)
            ))
        else:
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

        mode = input_params.estimator_mode.strip().lower() if input_params.estimator_mode else "auto"
        loaded_reg_for_features: Optional[LoadedRegression] = None
        if mode in ("auto", "regression"):
            try:
                loaded_reg_for_features = self.regression_provider.get_regression()
                for warning in loaded_reg_for_features.warnings:
                    if warning not in warnings_list:
                        warnings_list.append(warning)
            except Exception as load_ex:
                if mode == "regression":
                    raise PipelineError(503, "MODEL_UNAVAILABLE", f"Mô hình hồi quy không khả dụng: {load_ex}")
                warnings_list.append(f"REGRESSION_SKIPPED: {load_ex}")

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
            hybrid_packing_fraction=(
                loaded_reg_for_features.hybrid_packing_fraction
                if loaded_reg_for_features is not None
                else self.settings.packing_fraction_feature_hybrid
            ),
        )
        feat_val = validate_feature_vector(features_dict, require_grains=True)
        warnings_list.extend(w for w in feat_val.warnings if w not in warnings_list)

        t0 = time.perf_counter()
        regression_est: Optional[float] = None
        regression_model_name: Optional[str] = None
        if mode in ("auto", "regression"):
            if not feat_val.valid:
                if mode == "regression":
                    raise PipelineError(
                        422,
                        feat_val.error_code or "INVALID_INPUT",
                        feat_val.error_message or "Vector đặc trưng không hợp lệ cho hồi quy.",
                    )
                warnings_list.append(f"REGRESSION_SKIPPED: {feat_val.error_message or 'Vector đặc trưng không đạt chuẩn'}")
            elif loaded_reg_for_features is not None:
                try:
                    regression_est, regression_model_name = predict_regression(features_dict, loaded_reg_for_features)
                except Exception as inf_ex:
                    if mode == "regression":
                        raise PipelineError(500, "INFERENCE_FAILED", f"Dự đoán hồi quy thất bại: {inf_ex}")
                    warnings_list.append(f"INFERENCE_FAILED: {inf_ex}")
        timings["regression_ms"] = (time.perf_counter() - t0) * 1000

        estimates = select_final_estimate(
            mode=mode,
            geometry_est=geometry_est,
            weight_est=weight_est,
            regression_est=regression_est,
            regression_model_name=regression_model_name,
        )
        debug_visuals = build_debug_visuals(img, container_res, grain_analysis) if input_params.debug else {}
        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000

        result = PipelineResult(
            request_id=request_id,
            estimates=estimates,
            container=container_res,
            grain_analysis=grain_analysis,
            features_31=features_dict,
            timings_ms=timings,
            warnings=warnings_list,
            debug_visuals=debug_visuals,
        )
        try:
            result.artifact_manifest = self.artifact_writer.write(
                result,
                source_image=img,
                request_payload=input_params.__dict__,
            )
        except Exception as artifact_ex:
            # Scientific prediction remains inspectable, but persistence failure
            # must be explicit and must not be represented as a completed bundle.
            logger.exception("[%s] Không thể lưu artifact", request_id)
            result.warnings.append(f"ARTIFACT_WRITE_FAILED: {artifact_ex}")
            result.artifact_manifest = self.artifact_writer.write_failure(
                request_id, artifact_ex, input_params.__dict__
            )
        return result