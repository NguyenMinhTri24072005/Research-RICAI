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

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warnings

# Bo qua InconsistentVersionWarning tu scikit-learn khi unpickle tren phien ban moi hon
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 31 Features theo dung thu tu da dung khi train model
# ---------------------------------------------------------------------------
ALL_31_FEATURES: List[str] = [
    # Group 1: Container & Bulk (8 bien)
    "Bulk_Rice_Volume_mm3",
    "Rice_Height_mm",
    "Weight_g",
    "Empty_Height_mm",
    "Pixels_Per_mm",
    "Container_Detected_Diam_px",
    "Inner_Diameter_mm",
    "Container_Height_mm",
    # Group 2: Surface & Estimation (3 bien)
    "Whole_Grains_Count",
    "Uniformity_Rate_Pct",
    "Estimated_Total_Seeds_Hybrid",
    # Group 3: Length 2a (4 bien)
    "Grain_Length_mm_Mean",
    "Grain_Length_mm_Min",
    "Grain_Length_mm_Max",
    "Grain_Length_mm_Std",
    # Group 4: Width 2b (4 bien)
    "Grain_Width_mm_Mean",
    "Grain_Width_mm_Min",
    "Grain_Width_mm_Max",
    "Grain_Width_mm_Std",
    # Group 5: Thickness 2c (4 bien)
    "Grain_Thickness_mm_Mean",
    "Grain_Thickness_mm_Min",
    "Grain_Thickness_mm_Max",
    "Grain_Thickness_mm_Std",
    # Group 6: 2D Area (4 bien)
    "Grain_Area_mm2_Mean",
    "Grain_Area_mm2_Min",
    "Grain_Area_mm2_Max",
    "Grain_Area_mm2_Std",
    # Group 7: 3D Volume (4 bien)
    "Grain_Volume_mm3_Mean",
    "Grain_Volume_mm3_Min",
    "Grain_Volume_mm3_Max",
    "Grain_Volume_mm3_Std",
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

# ---------------------------------------------------------------------------
# Cache cho tree model (lazy load)
# ---------------------------------------------------------------------------
_TREE_CACHE: Dict[str, Any] = {"model": None, "scaler": None, "loaded": False}


def _safe_stat(values: List[float], func, default: float = 0.0) -> float:
    """Tinh toan thong ke an toan, tra ve default neu list rong."""
    if not values:
        return default
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
) -> Dict[str, float]:
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
    dict[str, float]
        Vector 31 bien theo dung thu tu ALL_31_FEATURES.
    """
    # --- Trich xuat thong so hat ---
    lengths = [g["length_mm"] for g in whole_grains if "length_mm" in g]
    widths = [g["width_mm"] for g in whole_grains if "width_mm" in g]
    thicknesses = [g["thickness_mm"] for g in whole_grains if "thickness_mm" in g]
    areas = [g["area_mm2"] for g in whole_grains if "area_mm2" in g]
    volumes = [g["volume_mm3"] for g in whole_grains if "volume_mm3" in g]

    # --- Tinh Rice_Height_mm ---
    container_height_mm = form_inputs.get("container_height_mm", 0.0)
    empty_height_mm = form_inputs.get("empty_height_mm", 0.0)
    rice_height_mm = container_height_mm - empty_height_mm
    if rice_height_mm < 0:
        rice_height_mm = container_res.get("rice_height_mm", 0.0)

    features = {
        # Group 1: Container & Bulk
        "Bulk_Rice_Volume_mm3": container_res.get("bulk_rice_volume_mm3", 0.0),
        "Rice_Height_mm": rice_height_mm,
        "Weight_g": form_inputs.get("weight_g", 0.0),
        "Empty_Height_mm": empty_height_mm,
        "Pixels_Per_mm": container_res.get("pixels_per_mm", 0.0),
        "Container_Detected_Diam_px": float(container_res.get("inner_w_px", 0)),
        "Inner_Diameter_mm": form_inputs.get("inner_diameter_mm", 0.0),
        "Container_Height_mm": container_height_mm,
        # Group 2: Surface & Estimation
        "Whole_Grains_Count": float(len(whole_grains)),
        "Uniformity_Rate_Pct": uniformity_res.get("uniformity_rate_pct", 0.0),
        "Estimated_Total_Seeds_Hybrid": hybrid_estimate,
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

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 2. NAP MO HINH EXTRA TREES
# ═══════════════════════════════════════════════════════════════════════════


def load_tree_model(
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """
    Nap Extra Trees Regressor va StandardScaler tu file joblib.
    Su dung lazy cache: chi nap 1 lan.

    Returns
    -------
    (model, scaler) hoac (None, None) neu khong tim thay file.
    """
    if _TREE_CACHE["loaded"]:
        return _TREE_CACHE["model"], _TREE_CACHE["scaler"]

    if model_path is None or not Path(model_path).exists():
        _TREE_CACHE["loaded"] = True
        return None, None

    try:
        import joblib

        model = joblib.load(model_path)
        scaler = None
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)

        _TREE_CACHE["model"] = model
        _TREE_CACHE["scaler"] = scaler
        _TREE_CACHE["loaded"] = True

        print(f"[REGRESSION] Da nap Extra Trees tu: {model_path}")
        if scaler:
            print(f"[REGRESSION] Da nap Scaler tu: {scaler_path}")
        return model, scaler

    except Exception as e:
        print(f"[REGRESSION] Khong the nap tree model: {e}")
        _TREE_CACHE["loaded"] = True
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# 3. SUY LUAN
# ═══════════════════════════════════════════════════════════════════════════


def predict_from_tree(
    model: Any,
    scaler: Any,
    features_dict: Dict[str, float],
) -> float:
    """
    Suy luan bang Extra Trees Regressor (primary).
    Thuc hien scaler.transform() roi model.predict().
    """
    # Xay dung vector theo dung thu tu
    feature_vector = np.array(
        [[features_dict.get(f, 0.0) for f in ALL_31_FEATURES]]
    )

    if scaler is not None:
        feature_vector = scaler.transform(feature_vector)

    prediction = model.predict(feature_vector)
    return float(prediction[0])


def predict_from_equation(features_dict: Dict[str, float]) -> float:
    """
    Suy luan bang phuong trinh OLS tuyen tinh (fallback).
    y = intercept + sum(w_i * x_i)
    Khong can scikit-learn hay joblib.
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
    Ham tong hop: tu dong chon Extra Trees (primary) hoac OLS (fallback).

    Returns
    -------
    (predicted_count, method_name)
        - predicted_count: so hat du doan (lam tron)
        - method_name: "ExtraTrees" hoac "OLS_Equation"
    """
    # Uu tien 1: Extra Trees
    if model is not None:
        try:
            pred = predict_from_tree(model, scaler, features_dict)
            # Dam bao gia tri khong am
            pred = max(0.0, pred)
            return round(pred, 1), "ExtraTrees"
        except Exception as e:
            print(f"[REGRESSION] Extra Trees loi, fallback sang OLS: {e}")

    # Uu tien 2: OLS Equation
    try:
        pred = predict_from_equation(features_dict)
        pred = max(0.0, pred)
        return round(pred, 1), "OLS_Equation"
    except Exception as e:
        print(f"[REGRESSION] OLS cung that bai: {e}")
        return 0.0, "failed"
