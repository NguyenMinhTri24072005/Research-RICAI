"""FastAPI Route Definitions for Rice Vision AI."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from rice_ai.api.schemas import (
    ComponentInfo,
    ComponentStatus,
    ErrorCode,
    ErrorDetail,
    EstimationResult,
    PredictResponse,
    ServiceReadiness,
    StatusResponse,
    TimingInfo,
)
from rice_ai.contracts import PipelineError, PredictionInput
from rice_ai.pipeline.runner import RicePipeline

logger = logging.getLogger("rice_ai.api.routes")
router = APIRouter()


def get_pipeline(request: Request) -> RicePipeline:
    return request.app.state.pipeline


def get_settings(request: Request):
    return request.app.state.settings


def get_vision_provider(request: Request):
    return request.app.state.vision_provider


def get_regression_provider(request: Request):
    return request.app.state.regression_provider


@router.get("/health")
async def health_check():
    """Liveness check nhẹ — phản hồi ngay lập tức, không chặn."""
    return {"status": "ok"}


@router.get("/api/status")
async def get_system_status(
    request: Request,
    settings=Depends(get_settings),
    vision_provider=Depends(get_vision_provider),
    regression_provider=Depends(get_regression_provider),
):
    """Báo cáo trạng thái sẵn sàng thực tế của hệ thống và các thành phần mô hình."""
    components: Dict[str, ComponentInfo] = {}
    v_status = vision_provider.get_status()

    # 1. Trạng thái YOLO
    yolo_info = v_status.get("yolo", {})
    try:
        yolo_path = settings.get_yolo_path()
        yolo_name = yolo_path.name
    except Exception:
        yolo_name = "unknown"

    if yolo_info.get("status") == "error":
        components["yolo"] = ComponentInfo(status=ComponentStatus.ERROR, path=yolo_info.get("error", "load_failed"))
    elif yolo_info.get("status") == "loaded":
        components["yolo"] = ComponentInfo(status=ComponentStatus.LOADED, path=yolo_name)
    elif yolo_info.get("configured"):
        components["yolo"] = ComponentInfo(status=ComponentStatus.CONFIGURED, path=yolo_name)
    else:
        components["yolo"] = ComponentInfo(status=ComponentStatus.NOT_FOUND)

    # 2. Trạng thái CNN
    cnn_info = v_status.get("cnn", {})
    try:
        cnn_path = settings.get_cnn_path()
        cnn_name = cnn_path.name
    except Exception:
        cnn_name = "unknown"

    if cnn_info.get("status") == "error":
        components["cnn"] = ComponentInfo(status=ComponentStatus.ERROR, path=cnn_info.get("error", "load_failed"))
    elif cnn_info.get("status") == "loaded":
        components["cnn"] = ComponentInfo(status=ComponentStatus.LOADED, path=cnn_name)
    elif cnn_info.get("configured"):
        components["cnn"] = ComponentInfo(status=ComponentStatus.CONFIGURED, path=cnn_name)
    else:
        components["cnn"] = ComponentInfo(status=ComponentStatus.NOT_FOUND)

    # 3. Trạng thái Regression Bundle
    bundle_id = None
    if regression_provider.is_loaded():
        reg = regression_provider.get_regression()
        bundle_id = reg.model_name
        components["regression"] = ComponentInfo(
            status=ComponentStatus.LOADED,
            path=reg.model_dir.name,
        )
    elif regression_provider.get_error():
        raw_err = regression_provider.get_error()
        clean_err = raw_err.split(":")[0] if ":" in raw_err else raw_err
        components["regression"] = ComponentInfo(
            status=ComponentStatus.ERROR,
            path=clean_err,
        )
    else:
        try:
            reg_dir = settings.get_regression_dir()
            components["regression"] = ComponentInfo(
                status=ComponentStatus.CONFIGURED,
                path=reg_dir.name,
            )
        except Exception:
            components["regression"] = ComponentInfo(status=ComponentStatus.NOT_FOUND)

    # Đánh giá mức độ sẵn sàng chung
    yolo_ok = components.get("yolo", ComponentInfo()).status in (ComponentStatus.LOADED, ComponentStatus.CONFIGURED)
    cnn_ok = components.get("cnn", ComponentInfo()).status in (ComponentStatus.LOADED, ComponentStatus.CONFIGURED)
    reg_loaded = components.get("regression", ComponentInfo()).status == ComponentStatus.LOADED

    if yolo_ok and cnn_ok and reg_loaded:
        readiness = ServiceReadiness.READY
    elif yolo_ok and cnn_ok:
        readiness = ServiceReadiness.DEGRADED
    else:
        readiness = ServiceReadiness.NOT_READY

    response = StatusResponse(
        service="Rice Vision AI Inference API",
        readiness=readiness,
        version="2.2.0",
        schema_version="31v1",
        bundle_id=bundle_id,
        components=components,
        capabilities=[
            "container_detection",
            "grain_segmentation_sahi",
            "densenet_classification",
            "3d_ellipsoid_geometry",
            "seed_count_regression",
            "auto_fallback_fusion",
        ],
    )
    return response.to_dict()


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    diam: float = Form(...),
    height: float = Form(...),
    empty: float = Form(...),
    wall_thickness: float = Form(0.1),
    weight_total: Optional[float] = Form(None),
    sample_weight: Optional[float] = Form(None),
    sample_count: Optional[int] = Form(None),
    estimator_mode: str = Form("auto"),
    debug: bool = Form(False),
    pipeline: RicePipeline = Depends(get_pipeline),
    regression_provider: LoadedRegressionProvider = Depends(get_regression_provider),
):
    """Dự đoán số lượng hạt và phân tích chất lượng mẫu lúa."""
    request_id = uuid.uuid4().hex[:12]

    # Kiểm tra cổng nhập (Admission gate) để bảo vệ inference đồng thời
    semaphore: asyncio.Semaphore = getattr(request.app.state, "admission_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
        request.app.state.admission_semaphore = semaphore
    try:
        # Thử acquire không chờ đợi để tránh request treo vô hạn
        await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        error_resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=ErrorCode.SERVER_BUSY,
                message="Hệ thống đang bận xử lý tác vụ khác. Vui lòng thử lại sau giây lát.",
                stage="admission_gate",
            ),
        )
        return JSONResponse(status_code=503, content=error_resp.to_dict())

    submitted = False
    try:
        image_bytes = await file.read()

        parsed_weight = float(weight_total) if weight_total is not None else None
        parsed_s_weight = float(sample_weight) if sample_weight is not None else None
        parsed_s_count = int(sample_count) if sample_count is not None else None

        input_params = PredictionInput(
            diam=float(diam),
            height=float(height),
            empty=float(empty),
            wall_thickness=float(wall_thickness) if wall_thickness is not None else 0.1,
            weight_total=parsed_weight,
            sample_weight=parsed_s_weight,
            sample_count=parsed_s_count,
            estimator_mode=str(estimator_mode or "auto"),
            debug=bool(debug),
        )
        input_params.validate()

        # Chạy pipeline đồng bộ trong worker thread độc lập
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            None,
            pipeline.run,
            input_params,
            image_bytes,
            request_id,
        )

        pending_workers: set = getattr(request.app.state, "pending_workers", None)
        if pending_workers is None:
            pending_workers = set()
            request.app.state.pending_workers = pending_workers

        task_wrapper = asyncio.ensure_future(fut)
        pending_workers.add(task_wrapper)
        submitted = True

        def _on_worker_done(_t):
            semaphore.release()
            pending_workers.discard(task_wrapper)

        task_wrapper.add_done_callback(_on_worker_done)

        result = await asyncio.shield(task_wrapper)

        # Định danh model bundle
        bundle_id = None
        if regression_provider.is_loaded():
            bundle_id = regression_provider.get_regression().model_name
        elif result.estimates.regression_est is not None:
            bundle_id = "regression"

        # Chuẩn hóa cấu trúc phản hồi
        estimation = EstimationResult(
            final=result.estimates.final,
            regression_est=result.estimates.regression_est,
            geometry_est=result.estimates.geometry_est,
            weight_est=result.estimates.weight_est,
            method_used=result.estimates.method_used,
            feature_schema_version="31v1",
            model_bundle=bundle_id or "default",
            ai_est=result.estimates.geometry_est,
        )

        timings = TimingInfo(
            total_ms=result.timings_ms.get("total_ms", 0.0),
            decode_ms=result.timings_ms.get("decode_ms", 0.0),
            container_ms=result.timings_ms.get("container_ms", 0.0),
            segmentation_ms=result.timings_ms.get("segmentation_ms", 0.0),
            cleaning_ms=result.timings_ms.get("cleaning_ms", 0.0),
            classification_ms=result.timings_ms.get("classification_ms", 0.0),
            geometry_ms=result.timings_ms.get("geometry_ms", 0.0),
            regression_ms=result.timings_ms.get("regression_ms", 0.0),
        )

        whole_grains = result.grain_analysis.whole_grains
        lengths = [g["length_mm"] for g in whole_grains if g.get("length_mm") is not None]
        widths = [g["width_mm"] for g in whole_grains if g.get("width_mm") is not None]
        thicknesses = [g["thickness_mm"] for g in whole_grains if g.get("thickness_mm") is not None]

        avg_length_mm = round(float(np.mean(lengths)), 2) if lengths else None
        avg_width_mm = round(float(np.mean(widths)), 2) if widths else None
        avg_thickness_mm = round(float(np.mean(thicknesses)), 2) if thicknesses else None

        metrics_summary = {
            "total_detected_grains": result.grain_analysis.total_detected,
            "total_grains_detected": result.grain_analysis.total_detected,
            "whole_grains_count": len(whole_grains),
            "whole_grains_surface": len(whole_grains),
            "classified_counts": result.grain_analysis.classified_counts,
            "uniformity_rate_pct": result.grain_analysis.uniformity_metrics.get("uniformity_rate_pct"),
            "bulk_volume_mm3": round(result.container.bulk_volume_mm3, 2),
            "pixels_per_mm": round(result.container.pixels_per_mm, 2),
            "avg_length_mm": avg_length_mm,
            "avg_width_mm": avg_width_mm,
            "avg_thickness_mm": avg_thickness_mm,
            "regression_model": result.estimates.method_used,
            "estimator_mode": input_params.estimator_mode,
        }

        debug_data = None
        if input_params.debug:
            debug_data = {
                "features_vector": {
                    k: (round(v, 4) if v is not None else None)
                    for k, v in result.features_31.items()
                } if result.features_31 else {},
                "grain_details": {
                    "whole_grains": len(result.grain_analysis.whole_grains),
                    "broken_grains": len(result.grain_analysis.broken_grains),
                    "skipped_grains": result.grain_analysis.skipped_measurement_count,
                },
            }

        resp = PredictResponse(
            status="success",
            request_id=request_id,
            estimation=estimation,
            metrics_summary=metrics_summary,
            features_used=result.features_31 if input_params.debug else None,
            warnings=result.warnings,
            timings_ms=timings,
            debug_info=debug_data,
        )
        return JSONResponse(status_code=200, content=resp.to_dict())

    except PipelineError as pe:
        logger.warning(f"[{request_id}] PipelineError: {pe.message} (code={pe.error_code})")
        err_resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=pe.error_code,
                message=pe.message,
                stage="pipeline",
            ),
        )
        return JSONResponse(status_code=pe.status_code, content=err_resp.to_dict())

    except Exception as ex:
        logger.error(f"[{request_id}] Lỗi không mong đợi trong /predict: {ex}", exc_info=True)
        err_resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=ErrorCode.INFERENCE_FAILED,
                message=f"Lỗi hệ thống nội bộ: {str(ex)}",
                stage="internal",
            ),
        )
        return JSONResponse(status_code=500, content=err_resp.to_dict())

    finally:
        if not submitted:
            semaphore.release()
