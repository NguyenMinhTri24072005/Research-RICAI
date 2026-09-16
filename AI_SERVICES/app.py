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
  - Mô hình hóa thể tích 3D Ellipsoid và ước lượng số hạt (Pure AI, Regression, Hybrid).
  - Package độc lập, tự chứa (Self-contained) với modules nội bộ trong AI_SERVICES/modules.
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
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

# Import module hoi quy 31 bien
from regression_engine import (
    assemble_31_features,
    load_tree_model,
    predict_regression,
    ALL_31_FEATURES,
)

app = FastAPI(
    title="Rice Vision AI Inference API",
    description="Dịch vụ AI ước lượng số lượng và đánh giá chất lượng hạt giống lúa",
    version="2.0.0",
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
    regression_model_path = BASE_PROJECT_DIR / "LINEAR_REGRESSION_MODEL" / "models" / "best_linear_regression_model.joblib"
    scaler_path = BASE_PROJECT_DIR / "LINEAR_REGRESSION_MODEL" / "models" / "scaler.joblib"
    tree_ensemble_path = BASE_PROJECT_DIR / "LINEAR_REGRESSION_MODEL" / "models" / "best_tree_ensemble_model.joblib"

    yolo_path = next((p for p in yolo_candidates if p.exists()), None)
    cnn_path = next((p for p in cnn_candidates if p.exists()), None)

    return {
        "yolo": yolo_path,
        "cnn": cnn_path,
        "regression": regression_model_path if regression_model_path.exists() else None,
        "scaler": scaler_path if scaler_path.exists() else None,
        "tree_ensemble": tree_ensemble_path if tree_ensemble_path.exists() else None,
    }

MODEL_PATHS = resolve_model_paths()
PACKING_FRACTION = 0.82

# Bộ nhớ đệm giữ mô hình nạp sẵn (Lazy load)
_MODELS_CACHE: Dict[str, Any] = {
    "yolo_detection_model": None,
    "cnn_classifier": None,
    "regression_model": None,
    "scaler": None,
}


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


def get_regression_artifacts():
    """Nap Extra Trees model va scaler qua regression_engine."""
    tree_model, tree_scaler = load_tree_model(
        model_path=MODEL_PATHS.get("tree_ensemble"),
        scaler_path=MODEL_PATHS.get("scaler"),
    )
    return tree_model, tree_scaler


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

        # 1. Bẻ cầu nối dính và cô lập hạt chủ đạo
        cleaned_rgba = clean_single_grain_crop(rgba, min_neck_ratio=0.15, sever_bridges=True)

        # 2. Phan loai pham cap bang CNN DenseNet121
        cnn_result = classifier.predict(cleaned_rgba)
        label = cnn_result["label"]
        conf = cnn_result["confidence"]

        # 3. Tinh toan thong so 3D cho hat nguyen
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
    """Tính toán số lượng hạt theo Pure AI (Pixel Volume), Regression và Hybrid."""
    # Quy đổi thể tích khối lúa sang px³
    pixels_per_mm3 = pixels_per_mm ** 3
    bulk_volume_px3 = bulk_volume_mm3 * pixels_per_mm3
    effective_bulk_px3 = bulk_volume_px3 * PACKING_FRACTION

    # 1. Ước tính thuần AI (Pure AI Volume Method)
    median_grain_vol_px3 = float(np.median(volumes_px3_list)) if volumes_px3_list else 0.0
    if median_grain_vol_px3 > 0:
        ai_est = int(round(effective_bulk_px3 / median_grain_vol_px3))
    else:
        ai_est = 0

    # 2. Ước tính theo tỷ trọng cân mẫu thực nghiệm
    weight_est = 0
    if weight_total > 0 and sample_weight > 0 and sample_count > 0:
        weight_est = int(round((weight_total / sample_weight) * sample_count))

    # 3. Ước tính Hybrid dung hòa
    if weight_est > 0 and ai_est > 0:
        final_est = int(round((ai_est + weight_est) / 2))
        used_hybrid = True
    elif ai_est > 0:
        final_est = ai_est
        used_hybrid = False
    else:
        final_est = weight_est
        used_hybrid = False

    return {
        "final": final_est,
        "ai_est": ai_est,
        "weight_est": weight_est,
        "hybrid": used_hybrid,
        "median_grain_vol_px3": round(median_grain_vol_px3, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🌐 ENDPOINTS API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
@app.get("/health")
def check_status():
    """Kiểm tra tình trạng sẵn sàng của hệ thống và các model."""
    return {
        "service": "Rice Vision AI Inference API",
        "status": "ready",
        "models_configured": {
            "yolo_detected": MODEL_PATHS["yolo"] is not None,
            "cnn_detected": MODEL_PATHS["cnn"] is not None,
            "regression_detected": MODEL_PATHS["regression"] is not None,
            "tree_ensemble_detected": MODEL_PATHS.get("tree_ensemble") is not None,
        },
        "model_paths": {k: str(v) if v else None for k, v in MODEL_PATHS.items()},
    }


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
):
    """
    Endpoint chính: Nhận ảnh cốc lúa + thông số vật lý $\to$ trả về kết quả ước lượng số hạt.
    """
    temp_work_dir = None
    try:
        # Tạo thư mục làm việc tạm thời an toàn
        temp_work_dir = Path(tempfile.mkdtemp(prefix="rice_predict_"))
        temp_img_path = temp_work_dir / f"input_{uuid.uuid4().hex[:8]}.jpg"
        crop_dir = temp_work_dir / "crops"

        with open(temp_img_path, "wb") as buffer:
            buffer.write(await file.read())

        # Bước 1: Nhận diện vật chứa & tính tỷ lệ quy đổi
        container_res = execute_container_analysis(
            image_path=temp_img_path,
            diam_cm=diam,
            height_cm=height,
            empty_cm=empty,
            wall_thickness_cm=wall_thickness,
        )
        pixels_per_mm = container_res["pixels_per_mm"]
        bulk_volume_mm3 = container_res["bulk_rice_volume_mm3"]

        # Bước 2: Cắt lát SAHI + YOLO-seg
        raw_crops = execute_sahi_crops(
            image_path=temp_img_path,
            output_crop_dir=crop_dir,
            confidence=0.5,
        )

        # Bước 3: Làm sạch + Phân loại CNN + Đo kích thước Ellipsoid 3D
        whole_grains, volumes_px3, total_detected = execute_grain_classification_and_metrics(
            raw_grains=raw_crops,
            pixels_per_mm=pixels_per_mm,
        )

        # Buoc 4: Danh gia do dong deu me lua
        grain_volumes_mm3 = [g["volume_mm3"] for g in whole_grains]
        uniformity_res = evaluate_batch_uniformity(grain_volumes_mm3)

        # Buoc 5: Tinh toan cac con so uoc luong (Pure AI + Weight)
        estimates = compute_final_estimates(
            bulk_volume_mm3=bulk_volume_mm3,
            pixels_per_mm=pixels_per_mm,
            volumes_px3_list=volumes_px3,
            weight_total=weight_total,
            sample_count=sample_count,
            sample_weight=sample_weight,
        )

        # ★ Buoc 6 [MOI]: Dung vector 31 dac trung cho Regression
        diam_mm = float(diam) * 10.0
        height_mm = float(height) * 10.0
        empty_mm = float(empty) * 10.0

        form_inputs = {
            "weight_g": float(weight_total),
            "empty_height_mm": empty_mm,
            "inner_diameter_mm": diam_mm,
            "container_height_mm": height_mm,
        }

        features_dict = assemble_31_features(
            container_res=container_res,
            whole_grains=whole_grains,
            uniformity_res=uniformity_res,
            form_inputs=form_inputs,
            hybrid_estimate=float(estimates.get("final", 0)),
        )

        # ★ Buoc 7 [MOI]: Suy luan hoi quy (Extra Trees primary, OLS fallback)
        tree_model, tree_scaler = get_regression_artifacts()
        regression_est, regression_method = predict_regression(
            features_dict=features_dict,
            model=tree_model,
            scaler=tree_scaler,
        )

        # Quyet dinh gia tri cuoi cung: uu tien Regression > Hybrid > Pure AI
        if regression_est > 0:
            final_est = int(round(regression_est))
            method_used = f"regression_{regression_method}"
        elif estimates.get("final", 0) > 0:
            final_est = estimates["final"]
            method_used = "hybrid" if estimates.get("hybrid") else "pure_ai"
        else:
            final_est = 0
            method_used = "none"

        # Tong hop thong so hinh thai hoc trung binh cua me lua
        metrics_summary = {
            "total_grains_detected": total_detected,
            "whole_grains_surface": len(whole_grains),
            "uniformity_rate_pct": uniformity_res.get("uniformity_rate_pct", 0.0),
            "pixels_per_mm": round(pixels_per_mm, 2),
            "bulk_rice_volume_mm3": round(bulk_volume_mm3, 2),
            "avg_length_mm": round(float(np.mean([g["length_mm"] for g in whole_grains])), 2) if whole_grains else 0.0,
            "avg_width_mm": round(float(np.mean([g["width_mm"] for g in whole_grains])), 2) if whole_grains else 0.0,
            "avg_thickness_mm": round(float(np.mean([g["thickness_mm"] for g in whole_grains])), 2) if whole_grains else 0.0,
            "regression_model": regression_method,
        }

        return {
            "status": "success",
            "estimation": {
                "final": final_est,
                "regression_est": regression_est,
                "ai_est": estimates.get("ai_est", 0),
                "weight_est": estimates.get("weight_est", 0),
                "method_used": method_used,
                "feature_schema_version": "31v_ExtraTrees_2026",
            },
            "metrics_summary": metrics_summary,
            "features_used": {k: round(v, 4) for k, v in features_dict.items()},
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
        }

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