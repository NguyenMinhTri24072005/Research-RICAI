#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
REGRESSION ENGINE — Module Suy Luan Hoi Quy 31 Bien
===============================================================================
Muc dich:
  - Dung vector 31 dac trung tu ket qua pipeline (container, grains, uniformity).
  - Suy luan bang Extra Trees Regressor (primary) hoac OLS equation (fallback).
  - Duoc su dung boi ca app.py (local) va API_Server.ipynb (Colab).
===============================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Import from feature_schema
try:
    from feature_schema import (
        ALL_31_FEATURES,
        validate_feature_vector,
        feature_vector_to_ordered_list,
        compute_trained_hybrid_feature,
    )
except ImportError:
    from AI_SERVICES.feature_schema import (
        ALL_31_FEATURES,
        validate_feature_vector,
        feature_vector_to_ordered_list,
        compute_trained_hybrid_feature,
    )

# Export ALL_31_FEATURES for backward compatibility
__all__ = [
    "ALL_31_FEATURES",
    "assemble_31_features",
    "predict_from_tree",
    "predict_from_equation",
    "predict_regression",
]

# ---------------------------------------------------------------------------
# OLS Equation Fallback (nhung truc tiep tu regression_equation.json)
# ---------------------------------------------------------------------------
_OLS_INTERCEPT = -120.0382943054218
_OLS_COEFFICIENTS = {
    "Bulk_Rice_Volume_mm3": 0.0009464250517212867,
    "Rice_Height_mm": -0.38582191648388264,
    "Weight_g": 39.6233414875361,
    "Empty_Height_mm": 0.5583980698754893,
    "Pixels_Per_mm": 1.154820515651178,
    "Container_Detected_Diam_px": -0.04919910600746634,
    "Inner_Diameter_mm": 1.2606910685238584,
    "Container_Height_mm": 1.9698297925652095,
    "Whole_Grains_Count": 0.08405378563407165,
    "Uniformity_Rate_Pct": -0.025703714334778867,
    "Estimated_Total_Seeds_Hybrid": 0.00025299307465455356,
    "Grain_Length_mm_Mean": -114.5325161777479,
    "Grain_Length_mm_Min": 1171.311753642038,
    "Grain_Length_mm_Max": 469.4544519621923,
    "Grain_Length_mm_Std": 189.20844150177598,
    "Grain_Width_mm_Mean": -2.3244221812461037,
    "Grain_Width_mm_Min": 2.5568315863498023,
    "Grain_Width_mm_Max": 0.09289271571035551,
    "Grain_Width_mm_Std": 3.0412187448544157,
    "Grain_Thickness_mm_Mean": 135.50886687278884,
    "Grain_Thickness_mm_Min": -1377.3771112898262,
    "Grain_Thickness_mm_Max": -553.4549519373832,
    "Grain_Thickness_mm_Std": -219.14112355879118,
    "Grain_Area_mm2_Mean": 0.16405810156564238,
    "Grain_Area_mm2_Min": -0.21826057179973116,
    "Grain_Area_mm2_Max": 0.04054252345237438,
    "Grain_Area_mm2_Std": -0.33632687072716155,
    "Grain_Volume_mm3_Mean": -0.00493315054514439,
    "Grain_Volume_mm3_Min": 0.0053977642119641105,
    "Grain_Volume_mm3_Max": -0.0003464764847389859,
    "Grain_Volume_mm3_Std": 0.006517007819043831,
}


def _safe_stat(values: List[float], func) -> Optional[float]:
    """Tinh toan thong ke an toan, tra ve None neu list rong."""
    if not values:
        return None
    return float(func(values))


# ═══════════════════════════════════════════════════════════════════════════
# 1. DUNG VECTOR 31 DAC TRUNG
# ═══════════════════════════════════════════════════════════════════════════


def assemble_31_features(
    container_res: Dict[str, Any],
    whole_grains: List[Dict[str, Any]],
    uniformity_res: Dict[str, Any],
    form_inputs: Dict[str, float],
    hybrid_estimate: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Dung vector 31 bien tu ket qua pipeline.

    Parameters
    ----------
    container_res : dict
        Ket qua tu detect_container_and_scale().
    whole_grains : list[dict]
        Danh sach metrics cua tung hat nguyen (tu compute_single_grain_metrics).
    uniformity_res : dict
        Ket qua tu evaluate_batch_uniformity().
    form_inputs : dict
        Thong so tu form: weight_g, empty_height_mm, inner_diameter_mm,
        container_height_mm.
    hybrid_estimate : float
        Uoc tinh Hybrid (ai + weight) da tinh truoc.

    Returns
    -------
    dict[str, Optional[float]]
        Vector 31 bien. Co the chua None neu thieu gia tri.
    """
    # --- Trich xuat thong so hat ---
    lengths = [float(g["length_mm"]) for g in whole_grains if "length_mm" in g]
    widths = [float(g["width_mm"]) for g in whole_grains if "width_mm" in g]
    thicknesses = [float(g["thickness_mm"]) for g in whole_grains if "thickness_mm" in g]
    areas = [float(g["area_mm2"]) for g in whole_grains if "area_mm2" in g]
    volumes = [float(g["volume_mm3"]) for g in whole_grains if "volume_mm3" in g]

    # --- Tinh Rice_Height_mm ---
    container_height_mm = form_inputs.get("container_height_mm")
    empty_height_mm = form_inputs.get("empty_height_mm")
    
    rice_height_mm = None
    if container_height_mm is not None and empty_height_mm is not None:
        rice_height_mm = container_height_mm - empty_height_mm
        if rice_height_mm < 0:
            rice_height_mm = container_res.get("rice_height_mm")
    else:
        rice_height_mm = container_res.get("rice_height_mm")

    container_diam_px = container_res.get("inner_w_px")
    if container_diam_px is not None:
        container_diam_px = float(container_diam_px)

    # --- Tinh Estimated_Total_Seeds_Hybrid chuan theo training semantics ---
    bulk_vol = container_res.get("bulk_rice_volume_mm3")
    trained_hybrid = compute_trained_hybrid_feature(bulk_vol, volumes)
    if trained_hybrid is not None:
        hybrid_feature = float(trained_hybrid)
    elif hybrid_estimate is not None and hybrid_estimate > 0:
        hybrid_feature = float(hybrid_estimate)
    else:
        hybrid_feature = None

    features = {
        # Group 1: Container & Bulk
        "Bulk_Rice_Volume_mm3": container_res.get("bulk_rice_volume_mm3"),
        "Rice_Height_mm": rice_height_mm,
        "Weight_g": form_inputs.get("weight_g"),
        "Empty_Height_mm": empty_height_mm,
        "Pixels_Per_mm": container_res.get("pixels_per_mm"),
        "Container_Detected_Diam_px": container_diam_px,
        "Inner_Diameter_mm": form_inputs.get("inner_diameter_mm"),
        "Container_Height_mm": container_height_mm,
        # Group 2: Surface & Estimation
        "Whole_Grains_Count": float(len(whole_grains)),
        "Uniformity_Rate_Pct": uniformity_res.get("uniformity_rate_pct"),
        "Estimated_Total_Seeds_Hybrid": hybrid_feature,
        # Group 3: Length 2a
        "Grain_Length_mm_Mean": _safe_stat(lengths, np.mean),
        "Grain_Length_mm_Min": _safe_stat(lengths, np.min),
        "Grain_Length_mm_Max": _safe_stat(lengths, np.max),
        "Grain_Length_mm_Std": _safe_stat(lengths, np.std),
        # Group 4: Width 2b
        "Grain_Width_mm_Mean": _safe_stat(widths, np.mean),
        "Grain_Width_mm_Min": _safe_stat(widths, np.min),
        "Grain_Width_mm_Max": _safe_stat(widths, np.max),
        "Grain_Width_mm_Std": _safe_stat(widths, np.std),
        # Group 5: Thickness 2c
        "Grain_Thickness_mm_Mean": _safe_stat(thicknesses, np.mean),
        "Grain_Thickness_mm_Min": _safe_stat(thicknesses, np.min),
        "Grain_Thickness_mm_Max": _safe_stat(thicknesses, np.max),
        "Grain_Thickness_mm_Std": _safe_stat(thicknesses, np.std),
        # Group 6: 2D Area
        "Grain_Area_mm2_Mean": _safe_stat(areas, np.mean),
        "Grain_Area_mm2_Min": _safe_stat(areas, np.min),
        "Grain_Area_mm2_Max": _safe_stat(areas, np.max),
        "Grain_Area_mm2_Std": _safe_stat(areas, np.std),
        # Group 7: 3D Volume
        "Grain_Volume_mm3_Mean": _safe_stat(volumes, np.mean),
        "Grain_Volume_mm3_Min": _safe_stat(volumes, np.min),
        "Grain_Volume_mm3_Max": _safe_stat(volumes, np.max),
        "Grain_Volume_mm3_Std": _safe_stat(volumes, np.std),
    }

    # Validate — log warnings nhưng không chặn (app.py quyết định policy)
    validation = validate_feature_vector(features, require_grains=False)
    if validation.warnings:
        for w in validation.warnings:
            print(f"[REGRESSION] Warning: {w}")
    if validation.missing_features:
        print(f"[REGRESSION] Missing features: {validation.missing_features}")
    if validation.non_finite_features:
        print(f"[REGRESSION] Non-finite features: {validation.non_finite_features}")

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 2. SUY LUAN
# ═══════════════════════════════════════════════════════════════════════════


def predict_from_tree(
    model: Any,
    scaler: Any,
    features_dict: Dict[str, Optional[float]],
) -> float:
    """
    Suy luan bang Extra Trees Regressor (primary).
    Thuc hien scaler.transform() roi model.predict().
    Tu choi vector thieu du lieu bat buoc hoac chua None (khong tu ep 0.0).
    """
    ordered_list = feature_vector_to_ordered_list(features_dict, allow_missing=False)
    feature_vector = np.array([ordered_list])

    if scaler is not None:
        feature_vector = scaler.transform(feature_vector)

    prediction = model.predict(feature_vector)
    return float(prediction[0])


def predict_from_equation(features_dict: Dict[str, float]) -> float:
    """
    Suy luan bang phuong trinh OLS tuyen tinh (fallback).
    y = intercept + sum(w_i * x_i)
    """
    result = _OLS_INTERCEPT
    for feature_name, coeff in _OLS_COEFFICIENTS.items():
        result += coeff * features_dict.get(feature_name, 0.0)
    return result


def predict_regression(
    features_dict: Dict[str, float],
    model: Any = None,
    scaler: Any = None,
) -> Tuple[float, str]:
    """
    Suy luan mo hinh tree, hoac throw exception neu that bai.
    Khong fallback ngam sang OLS, de ung dung chu dong goi OLS neu can.
    """
    if model is None:
        raise ValueError("Model is None. Cannot predict using tree.")
        
    try:
        pred = predict_from_tree(model, scaler, features_dict)
        pred = max(0.0, pred)
        return round(pred, 1), "ExtraTrees"
    except Exception as e:
        raise RuntimeError(f"Tree model prediction failed: {e}") from e
