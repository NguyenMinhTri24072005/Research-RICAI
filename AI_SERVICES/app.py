#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
🌾 RICE VISION AI — BACKEND INFERENCE SERVICE (FastAPI)
===============================================================================
Mục đích:
  - Cung cấp API suy luận trực tuyến cho ứng dụng RICE_ESTIMATION_APPLICATION.
  - Tự động nhận diện vật chứa, phân đoạn hạt lúa bằng SAHI + YOLO-seg.
  - Làm sạch và phân loại phẩm cấp hạt bằng DenseNet121.
  - Mô hình hóa thể tích 3D Ellipsoid và ước lượng số hạt.
  - Hồi quy 31 biến bằng Extra Trees Regressor.
  - Package độc lập, tự chứa (Self-contained) với modules nội bộ.
===============================================================================
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ KIỂM TRA TƯƠNG THÍCH PHIÊN BẢN PYTHON AN TOÀN
# ─────────────────────────────────────────────────────────────────────────────
if sys.version_info >= (3, 13) or sys.version_info < (3, 10):
    import warnings
    warnings.warn(
        f"\n[CANH BAO MOI TRUONG] Python {sys.version_info.major}.{sys.version_info.minor} "
        f"co the khong tuong thich hoan hao voi TensorFlow/PyTorch.\n"
        f"Khuyen nghi su dung Python 3.10, 3.11 hoac 3.12 (64-bit) de dam bao on dinh tuyet doi!\n"
    )

# Đảm bảo import được các module nội bộ của AI_SERVICES
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import từ thư mục modules nội bộ của AI_SERVICES
from modules.container_detector import detect_container_and_scale
from modules.grain_segmenter import segment_grains_sahi
from modules.grain_classifier import GrainClassifier
from modules.ellipsoid_geometry import compute_single_grain_metrics
from modules.uniformity_evaluator import evaluate_batch_uniformity
from modules.grain_crop_cleaner import clean_single_grain_crop

# Import feature schema, model registry, schemas
from feature_schema import (
    ALL_31_FEATURES,
    FEATURE_SCHEMA_VERSION,
    validate_feature_vector,
)
from model_registry import ModelRegistry
from schemas import (
    ErrorCode,
    ErrorDetail,
    EstimationResult,
    TimingInfo,
    PredictResponse,
    StatusResponse,
    ComponentInfo,
    ComponentStatus,
    ServiceReadiness,
)

# Import regression engine
from regression_engine import (
    assemble_31_features,
    predict_from_tree,
    predict_from_equation,
    predict_regression,
)

app = FastAPI(
    title="Rice Vision AI Inference API",
    description="Dịch vụ AI ước lượng số lượng và đánh giá chất lượng hạt giống lúa",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH (TỰ ĐỘNG PHÁT HIỆN LOCAL & COLAB)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PROJECT_DIR = CURRENT_DIR.parent


def resolve_model_paths() -> Dict[str, Optional[Path]]:
    """Tự động tìm kiếm đường dẫn các file trọng số mô hình hợp lệ."""
    yolo_candidates = [
        BASE_PROJECT_DIR / "RESULTS" / "all-new-data-v1.yolov8_yolov8s-seg_trained" / "weights" / "best.pt",
        BASE_PROJECT_DIR / "RESULTS" / "35_special_images_segmentation.v1i.yolov8_v1_trained" / "weights" / "best.pt",
        BASE_PROJECT_DIR / "MODELS" / "yolo26s-seg.pt",
        Path("/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt"),
    ]
    cnn_candidates = [
        BASE_PROJECT_DIR / "RESULTS" / "CNN_DenseNet121_Trained" / "best_v3_step2.keras",
        BASE_PROJECT_DIR / "RESULTS" / "CNN_DenseNet121_Trained" / "best_rice_densenet121.keras",
        BASE_PROJECT_DIR / "RESULTS" / "35_special_images_segmentation.v1i.yolov8_v1_trained" / "rice_grain_classifier_cnn.h5",
        Path("/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/DETECTED_OBJECTS/35_special_images_segmentation.v1i.yolov8_v1_trained/rice_grain_classifier_cnn.h5"),
    ]

    yolo_path = next((p for p in yolo_candidates if p.exists()), None)
    cnn_path = next((p for p in cnn_candidates if p.exists()), None)

    return {
        "yolo": yolo_path,
        "cnn": cnn_path,
    }


MODEL_PATHS = resolve_model_paths()
PACKING_FRACTION = 0.82  # Xem manifest.json cho ghi chú về discrepancy

# Bộ nhớ đệm giữ mô hình nạp sẵn (Lazy load)
_MODELS_CACHE: Dict[str, Any] = {
    "yolo_detection_model": None,
    "cnn_classifier": None,
}

# Model Registry cho regression bundle
_registry = ModelRegistry(project_root=BASE_PROJECT_DIR)


def get_yolo_model(confidence: float = 0.5):
    """Nạp YOLO detection model qua SAHI."""
    if _MODELS_CACHE["yolo_detection_model"] is None:
        yolo_path = MODEL_PATHS["yolo"]
        if not yolo_path:
            raise FileNotFoundError("Không tìm thấy trọng số YOLO model trong hệ thống!")

        import torch
        from sahi import AutoDetectionModel

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"📦 [AI SERVER] Nạp YOLO SAHI model từ: {yolo_path} ({device_str})...")
        _MODELS_CACHE["yolo_detection_model"] = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(yolo_path),
            confidence_threshold=confidence,
            device=device_str,
        )
        print("✅ [AI SERVER] Đã nạp YOLO thành công!")
    return _MODELS_CACHE["yolo_detection_model"]


def get_cnn_classifier():
    """Nạp CNN classifier (DenseNet121)."""
    if _MODELS_CACHE["cnn_classifier"] is None:
        cnn_path = MODEL_PATHS["cnn"]
        if not cnn_path:
            raise FileNotFoundError("Không tìm thấy trọng số CNN model trong hệ thống!")

        print(f"📦 [AI SERVER] Nạp CNN Classifier từ: {cnn_path}...")
        _MODELS_CACHE["cnn_classifier"] = GrainClassifier(
            model_path=cnn_path,
            class_names=["hat_khuyet_tat", "hat_nguyen"],
            target_size=(224, 224),
        )
        print("✅ [AI SERVER] Đã nạp CNN thành công!")
    return _MODELS_CACHE["cnn_classifier"]


# ─────────────────────────────────────────────────────────────────────────────
# 🔬 CÁC HÀM XỬ LÝ NGHIỆP VỤ BÓC TÁCH & ĐO LƯỜNG
# ─────────────────────────────────────────────────────────────────────────────

def execute_container_analysis(
    image_path: Path,
    diam_cm: float,
    height_cm: float,
    empty_cm: float,
    wall_thickness_cm: float = 0.1,
) -> Dict[str, Any]:
    """Nhận diện miệng ly, tính tỷ lệ pixels/mm và thể tích khối lúa."""
    diam_mm = float(diam_cm) * 10.0
    height_mm = float(height_cm) * 10.0
    empty_mm = float(empty_cm) * 10.0
    wall_mm = float(wall_thickness_cm) * 10.0

    return detect_container_and_scale(
        image_input=image_path,
        inner_diam_mm=diam_mm,
        container_height_mm=height_mm,
        empty_height_mm=empty_mm,
        wall_thickness_mm=wall_mm,
        detect_mode="inner",
    )


def execute_sahi_crops(
    image_path: Path,
    output_crop_dir: Path,
    confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """Cắt lát phân giải cao bóc tách từng hạt lúa."""
    detection_model = get_yolo_model(confidence=confidence)
    return segment_grains_sahi(
        detection_model=detection_model,
        image_path=image_path,
        output_crop_dir=output_crop_dir,
        conf_threshold=confidence,
        slice_size=640,
        overlap_ratio=0.20,
    )


def execute_grain_classification_and_metrics(
    raw_grains: List[Dict[str, Any]],
    pixels_per_mm: float,
) -> Tuple[List[Dict[str, Any]], List[float], int]:
    """Làm sạch, phân loại hạt nguyên/khuyết tật và đo thể tích 3D."""
    classifier = get_cnn_classifier()
    whole_grains: List[Dict[str, Any]] = []
    volumes_px3: List[float] = []
    total_detected = len(raw_grains)

    for item in raw_grains:
        rgba = item.get("crop_rgba")
        if rgba is None:
            continue

        cleaned_rgba = clean_single_grain_crop(rgba, min_neck_ratio=0.15, sever_bridges=True)
        cnn_result = classifier.predict(cleaned_rgba)
        label = cnn_result["label"]

        if label == "hat_nguyen":
            try:
                metrics = compute_single_grain_metrics(
                    image_input=cleaned_rgba,
                    pixels_per_mm=pixels_per_mm,
                    label="hat_nguyen",
                )
                whole_grains.append(metrics)
                volumes_px3.append(metrics["vol_px3"])
            except Exception:
                pass

    return whole_grains, volumes_px3, total_detected


def compute_final_estimates(
    bulk_volume_mm3: float,
    pixels_per_mm: float,
    volumes_px3_list: List[float],
    weight_total: float = 0.0,
    sample_count: int = 0,
    sample_weight: float = 0.0,
) -> Dict[str, Any]:
    """Tính toán số lượng hạt theo Geometry (Pixel Volume) và Weight."""
    pixels_per_mm3 = pixels_per_mm ** 3
    bulk_volume_px3 = bulk_volume_mm3 * pixels_per_mm3
    effective_bulk_px3 = bulk_volume_px3 * PACKING_FRACTION

    # 1. Ước tính hình học (Geometry Volume Method)
    median_grain_vol_px3 = float(np.median(volumes_px3_list)) if volumes_px3_list else 0.0
    if median_grain_vol_px3 > 0:
        geometry_est = int(round(effective_bulk_px3 / median_grain_vol_px3))
    else:
        geometry_est = None

    # 2. Ước tính theo cân mẫu
    weight_est = None
    if weight_total > 0 and sample_weight > 0 and sample_count > 0:
        weight_est = int(round((weight_total / sample_weight) * sample_count))

    # 3. Hybrid
    if weight_est is not None and geometry_est is not None:
        final_est = int(round((geometry_est + weight_est) / 2))
        used_hybrid = True
    elif geometry_est is not None:
        final_est = geometry_est
        used_hybrid = False
    elif weight_est is not None:
        final_est = weight_est
        used_hybrid = False
    else:
        final_est = None
        used_hybrid = False

    return {
        "final": final_est,
        "geometry_est": geometry_est,
        "ai_est": geometry_est,  # Alias cho tương thích client cũ
        "weight_est": weight_est,
        "hybrid": used_hybrid,
        "median_grain_vol_px3": round(median_grain_vol_px3, 2) if median_grain_vol_px3 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🌐 ENDPOINTS API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness check nhẹ — không kiểm tra model."""
    return {"status": "ok"}


@app.get("/api/status")
def check_status():
    """Kiểm tra tình trạng sẵn sàng thực tế của hệ thống và các model."""
    # Check registry
    bundle_status = _registry.get_status()

    components = {}

    # YOLO
    yolo_path = MODEL_PATHS.get("yolo")
    if yolo_path:
        if _MODELS_CACHE["yolo_detection_model"] is not None:
            components["yolo"] = ComponentInfo(status="loaded", path=yolo_path.name)
        else:
            components["yolo"] = ComponentInfo(status="configured", path=yolo_path.name)
    else:
        components["yolo"] = ComponentInfo(status="not_found")

    # CNN
    cnn_path = MODEL_PATHS.get("cnn")
    if cnn_path:
        if _MODELS_CACHE["cnn_classifier"] is not None:
            components["cnn"] = ComponentInfo(status="loaded", path=cnn_path.name)
        else:
            components["cnn"] = ComponentInfo(status="configured", path=cnn_path.name)
    else:
        components["cnn"] = ComponentInfo(status="not_found")

    # Regression bundle
    if bundle_status.verified:
        components["regression"] = ComponentInfo(status="verified")
    elif bundle_status.loaded:
        components["regression"] = ComponentInfo(status="loaded")
    elif bundle_status.error:
        components["regression"] = ComponentInfo(status="error")
    else:
        components["regression"] = ComponentInfo(status="not_found")

    # Determine readiness
    all_configured = (yolo_path is not None and cnn_path is not None)
    if all_configured and bundle_status.loaded:
        readiness = "ready"
    elif all_configured or bundle_status.loaded:
        readiness = "degraded"
    else:
        readiness = "not_ready"

    capabilities = []
    if bundle_status.verified:
        capabilities.append("regression_31v1")
    if yolo_path:
        capabilities.append("yolo_segmentation")
    if cnn_path:
        capabilities.append("cnn_classification")

    response = StatusResponse(
        readiness=readiness,
        schema_version=FEATURE_SCHEMA_VERSION,
        bundle_id=bundle_status.bundle_id,
        components=components,
        capabilities=capabilities,
    )
    return response.to_dict()


@app.post("/predict")
async def predict(
    file:           UploadFile = File(...),
    diam:           float      = Form(...),
    height:         float      = Form(...),
    empty:          float      = Form(...),
    wall_thickness: float      = Form(0.1),
    weight_total:   float      = Form(0.0),
    sample_count:   int        = Form(0),
    sample_weight:  float      = Form(0.0),
    estimator_mode: str        = Form("auto"),
    debug:          bool       = Form(False),
):
    """
    Endpoint chính: Nhận ảnh cốc lúa + thông số vật lý → trả về kết quả ước lượng số hạt.
    Hỗ trợ:
      - estimator_mode: 'auto' (ưu tiên regression, fallback geometry/weight),
                        'regression' (bắt buộc hồi quy, lỗi nếu thiếu feature/model),
                        'geometry' (chỉ dùng đo thể tích pixel),
                        'weight' (chỉ dùng cân mẫu).
      - debug: True (trả thêm chi tiết timings và feature vector).
    """
    request_id = uuid.uuid4().hex[:12]
    timings = TimingInfo()
    t_total_start = time.perf_counter()
    temp_work_dir = None
    response_warnings: List[str] = []

    # ── Input Validation ────────────────────────────────────────────────────
    errors: List[str] = []
    if diam <= 0:
        errors.append("diam phải > 0")
    if height <= 0:
        errors.append("height phải > 0")
    if empty < 0:
        errors.append("empty không được âm")
    if empty > height:
        errors.append("empty phải <= height")
    if wall_thickness < 0:
        errors.append("wall_thickness không được âm")
    if weight_total < 0:
        errors.append("weight_total không được âm")
    if sample_weight < 0:
        errors.append("sample_weight không được âm")
    if sample_count < 0:
        errors.append("sample_count không được âm")
    if estimator_mode not in ("auto", "regression", "geometry", "weight"):
        errors.append(f"estimator_mode '{estimator_mode}' không hợp lệ (auto, regression, geometry, weight)")

    if errors:
        resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=ErrorCode.INVALID_INPUT,
                message="; ".join(errors),
                stage="input_validation",
            ),
        )
        return JSONResponse(status_code=422, content=resp.to_dict())

    try:
        # ── Bước 0: Đọc ảnh ────────────────────────────────────────────────
        t0 = time.perf_counter()
        temp_work_dir = Path(tempfile.mkdtemp(prefix="rice_predict_"))
        temp_img_path = temp_work_dir / f"input_{request_id}.jpg"
        crop_dir = temp_work_dir / "crops"

        image_bytes = await file.read()
        if len(image_bytes) == 0:
            resp = PredictResponse(
                status="error",
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.INVALID_IMAGE,
                    message="File ảnh rỗng",
                    stage="decode",
                ),
            )
            return JSONResponse(status_code=422, content=resp.to_dict())

        with open(temp_img_path, "wb") as buffer:
            buffer.write(image_bytes)

        # Validate image can be decoded
        test_img = cv2.imread(str(temp_img_path))
        if test_img is None:
            resp = PredictResponse(
                status="error",
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.INVALID_IMAGE,
                    message="Không thể giải mã file ảnh",
                    stage="decode",
                ),
            )
            return JSONResponse(status_code=422, content=resp.to_dict())

        timings.decode_ms = (time.perf_counter() - t0) * 1000

        # ── Bước 1: Container Detection ─────────────────────────────────────
        t1 = time.perf_counter()
        container_res = execute_container_analysis(
            image_path=temp_img_path,
            diam_cm=diam,
            height_cm=height,
            empty_cm=empty,
            wall_thickness_cm=wall_thickness,
        )
        pixels_per_mm = container_res["pixels_per_mm"]
        bulk_volume_mm3 = container_res["bulk_rice_volume_mm3"]
        timings.container_ms = (time.perf_counter() - t1) * 1000

        # ── Bước 2: SAHI + YOLO-seg ─────────────────────────────────────────
        t2 = time.perf_counter()
        raw_crops = execute_sahi_crops(
            image_path=temp_img_path,
            output_crop_dir=crop_dir,
            confidence=0.5,
        )
        timings.segmentation_ms = (time.perf_counter() - t2) * 1000

        # ── Bước 3: Clean + CNN + Geometry ───────────────────────────────────
        t3 = time.perf_counter()
        whole_grains, volumes_px3, total_detected = execute_grain_classification_and_metrics(
            raw_grains=raw_crops,
            pixels_per_mm=pixels_per_mm,
        )
        timings.classification_ms = (time.perf_counter() - t3) * 1000

        # ── Bước 4: Uniformity ───────────────────────────────────────────────
        grain_volumes_mm3 = [g["volume_mm3"] for g in whole_grains]
        uniformity_res = evaluate_batch_uniformity(grain_volumes_mm3)

        # ── Bước 5: Geometry/Weight estimates ────────────────────────────────
        t5 = time.perf_counter()
        estimates = compute_final_estimates(
            bulk_volume_mm3=bulk_volume_mm3,
            pixels_per_mm=pixels_per_mm,
            volumes_px3_list=volumes_px3,
            weight_total=weight_total,
            sample_count=sample_count,
            sample_weight=sample_weight,
        )
        timings.geometry_ms = (time.perf_counter() - t5) * 1000

        # ── Bước 6: Assemble 31 features ─────────────────────────────────────
        diam_mm = float(diam) * 10.0
        height_mm = float(height) * 10.0
        empty_mm = float(empty) * 10.0

        form_inputs = {
            "weight_g": float(weight_total),
            "empty_height_mm": empty_mm,
            "inner_diameter_mm": diam_mm,
            "container_height_mm": height_mm,
        }

        # TODO: Estimated_Total_Seeds_Hybrid discrepancy —
        # Training extractor: Round(bulk_vol * packing / vol_mean_mm3)
        # Runtime: estimates["final"] (hybrid of geometry + weight)
        # This mismatch is documented in manifest.json pipeline_config
        hybrid_for_regression = float(estimates.get("final") or 0)

        features_dict = assemble_31_features(
            container_res=container_res,
            whole_grains=whole_grains,
            uniformity_res=uniformity_res,
            form_inputs=form_inputs,
            hybrid_estimate=hybrid_for_regression,
        )

        # ── Bước 7: Validate features trước regression ───────────────────────
        feature_validation = validate_feature_vector(features_dict, require_grains=True)
        if feature_validation.warnings:
            response_warnings.extend(feature_validation.warnings)

        # ── Bước 8: Regression (theo estimator_mode) ────────────────────────
        t8 = time.perf_counter()
        regression_est: Optional[float] = None
        regression_method: Optional[str] = None

        if estimator_mode in ("auto", "regression"):
            if not feature_validation.valid and feature_validation.grain_count == 0:
                if estimator_mode == "regression":
                    resp = PredictResponse(
                        status="error",
                        request_id=request_id,
                        error=ErrorDetail(
                            code=ErrorCode.NO_VALID_GRAINS,
                            message="Không có hạt nguyên nào được phát hiện để chạy hồi quy.",
                            stage="regression_validation",
                        ),
                    )
                    return JSONResponse(status_code=422, content=resp.to_dict())
                response_warnings.append(
                    "NO_VALID_GRAINS: Regression bỏ qua vì không có hạt nguyên nào."
                )
            else:
                # Load bundle nếu chưa
                bundle_status = _registry.load_bundle()

                if bundle_status.verified:
                    model, scaler = _registry.get_model_and_scaler()
                    try:
                        # Thay None bằng 0.0 cho regression (model cần vector đầy đủ)
                        features_for_model = {
                            k: (v if v is not None else 0.0) for k, v in features_dict.items()
                        }
                        pred, method = predict_regression(
                            features_dict=features_for_model,
                            model=model,
                            scaler=scaler,
                        )
                        regression_est = pred
                        regression_method = method
                    except Exception as e:
                        print(f"[{request_id}] Regression failed: {e}")
                        if estimator_mode == "regression":
                            resp = PredictResponse(
                                status="error",
                                request_id=request_id,
                                error=ErrorDetail(
                                    code=ErrorCode.INFERENCE_FAILED,
                                    message=f"Hồi quy thất bại: {e}",
                                    stage="regression",
                                ),
                            )
                            return JSONResponse(status_code=500, content=resp.to_dict())
                        response_warnings.append(f"INFERENCE_FAILED: {e}")
                else:
                    msg = bundle_status.error or "Bundle chưa verified"
                    if estimator_mode == "regression":
                        resp = PredictResponse(
                            status="error",
                            request_id=request_id,
                            error=ErrorDetail(
                                code=ErrorCode.MODEL_UNAVAILABLE,
                                message=f"Mô hình hồi quy không khả dụng: {msg}",
                                stage="model_registry",
                            ),
                        )
                        return JSONResponse(status_code=503, content=resp.to_dict())
                    response_warnings.append(f"MODEL_UNAVAILABLE: {msg}")

        timings.regression_ms = (time.perf_counter() - t8) * 1000

        # ── Bước 9: Quyết định final theo estimator_mode ───────────────────────
        geometry_est = estimates.get("geometry_est")
        weight_est = estimates.get("weight_est")

        if estimator_mode == "regression":
            final_est = int(round(regression_est)) if regression_est is not None else None
            method_used = f"regression_{regression_method}" if regression_est is not None else "none"
        elif estimator_mode == "geometry":
            final_est = geometry_est
            method_used = "geometry"
        elif estimator_mode == "weight":
            final_est = weight_est
            method_used = "weight"
        else:  # auto
            if regression_est is not None and regression_est >= 0:
                final_est = int(round(regression_est))
                method_used = f"regression_{regression_method}"
            elif estimates.get("final") is not None:
                final_est = estimates["final"]
                method_used = "hybrid" if estimates.get("hybrid") else (
                    "geometry" if geometry_est is not None else "weight"
                )
            else:
                final_est = None
                method_used = "none"

        # ── Build response ────────────────────────────────────────────────────
        timings.total_ms = (time.perf_counter() - t_total_start) * 1000

        estimation = EstimationResult(
            final=final_est,
            regression_est=regression_est,
            geometry_est=geometry_est,
            weight_est=weight_est,
            method_used=method_used,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_bundle=_registry.get_status().bundle_id,
            ai_est=geometry_est,  # backward compat
        )

        metrics_summary = {
            "total_grains_detected": total_detected,
            "whole_grains_surface": len(whole_grains),
            "uniformity_rate_pct": uniformity_res.get("uniformity_rate_pct", 0.0),
            "pixels_per_mm": round(pixels_per_mm, 2),
            "bulk_rice_volume_mm3": round(bulk_volume_mm3, 2),
            "avg_length_mm": round(float(np.mean([g["length_mm"] for g in whole_grains])), 2) if whole_grains else None,
            "avg_width_mm": round(float(np.mean([g["width_mm"] for g in whole_grains])), 2) if whole_grains else None,
            "avg_thickness_mm": round(float(np.mean([g["thickness_mm"] for g in whole_grains])), 2) if whole_grains else None,
            "regression_model": regression_method,
            "estimator_mode": estimator_mode,
        }

        resp_content = PredictResponse(
            status="success",
            request_id=request_id,
            estimation=estimation,
            metrics_summary=metrics_summary,
            features_used=features_dict if debug else None,
            warnings=response_warnings,
            timings_ms=timings,
        ).to_dict()

        if debug:
            resp_content["debug_info"] = {
                "features_vector": {k: (v if v is not None else 0.0) for k, v in features_dict.items()},
                "feature_validation": {
                    "valid": feature_validation.valid,
                    "warnings": feature_validation.warnings,
                    "grain_count": feature_validation.grain_count,
                },
                "total_grains_detected": total_detected,
            }

        print(f"[{request_id}] ✅ Predict complete: final={final_est}, method={method_used}, time={timings.total_ms:.0f}ms")
        return resp_content

    except FileNotFoundError as e:
        print(f"[{request_id}] Model not found: {e}")
        traceback.print_exc()
        resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=ErrorCode.MODEL_UNAVAILABLE,
                message=str(e),
                stage="model_loading",
            ),
        )
        return JSONResponse(status_code=503, content=resp.to_dict())

    except Exception as e:
        print(f"[{request_id}] ❌ Predict error: {e}")
        traceback.print_exc()
        resp = PredictResponse(
            status="error",
            request_id=request_id,
            error=ErrorDetail(
                code=ErrorCode.INFERENCE_FAILED,
                message=str(e),
                stage="pipeline",
            ),
        )
        return JSONResponse(status_code=500, content=resp.to_dict())

    finally:
        # Dọn dẹp an toàn thư mục tạm
        if temp_work_dir and temp_work_dir.exists():
            try:
                shutil.rmtree(str(temp_work_dir))
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Khoi dong Rice Vision AI Service tai http://localhost:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)