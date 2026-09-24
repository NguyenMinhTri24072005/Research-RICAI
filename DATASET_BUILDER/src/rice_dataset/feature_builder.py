"""Feature construction for 31-feature regression contract and physical diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import ORDERED_FEATURES
from .vision.uniformity_evaluator import evaluate_batch_uniformity


def build_regression_features(
    raw_valid_metrics: List[Dict[str, Any]],
    bulk_volume_mm3: float,
    rice_height_mm: float,
    weight_g: float,
    empty_height_mm: float,
    pixels_per_mm: float,
    container_detected_diam_px: float,
    inner_diameter_mm: float,
    container_height_mm: float,
    trained_hybrid_packing_fraction: float = 0.62,
) -> Dict[str, float]:
    """Construct the exact 31 regression features from raw-valid CNN-whole grain metrics.

    Scientific Contract:
    - regression population is raw_valid_cnn_whole only.
    - Actual_Count is strictly excluded.
    - ddof=0 for population standard deviations.
    - Estimated_Total_Seeds_Hybrid uses trained_hybrid_packing_fraction (0.62) and raw mean volume.
    """
    if len(raw_valid_metrics) == 0:
        raise ValueError("Cannot assemble features: raw_valid_metrics is empty.")

    if weight_g is None or not math.isfinite(weight_g) or weight_g <= 0:
        raise ValueError(f"Invalid Weight_g: {weight_g}")

    lens_arr = np.array([float(m["length_mm"]) for m in raw_valid_metrics], dtype=np.float64)
    wids_arr = np.array([float(m["width_mm"]) for m in raw_valid_metrics], dtype=np.float64)
    thks_arr = np.array([float(m["thickness_mm"]) for m in raw_valid_metrics], dtype=np.float64)
    area_arr = np.array([float(m["area_mm2"]) for m in raw_valid_metrics], dtype=np.float64)
    vols_arr = np.array([float(m["volume_mm3"]) for m in raw_valid_metrics], dtype=np.float64)

    if (
        np.any(~np.isfinite(lens_arr))
        or np.any(~np.isfinite(wids_arr))
        or np.any(~np.isfinite(thks_arr))
        or np.any(~np.isfinite(area_arr))
        or np.any(~np.isfinite(vols_arr))
        or np.any(lens_arr <= 0)
        or np.any(wids_arr <= 0)
        or np.any(thks_arr <= 0)
        or np.any(area_arr <= 0)
        or np.any(vols_arr <= 0)
    ):
        raise ValueError("Grain measurements contain non-finite or non-positive dimensions/area/volume.")

    def calc_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "Mean": round(float(np.mean(arr)), 3),
            "Min": round(float(np.min(arr)), 3),
            "Max": round(float(np.max(arr)), 3),
            "Std": round(float(np.std(arr, ddof=0)), 3),
        }

    lens_st = calc_stats(lens_arr)
    wids_st = calc_stats(wids_arr)
    thks_st = calc_stats(thks_arr)
    area_st = calc_stats(area_arr)
    vols_st = calc_stats(vols_arr)

    # Batch uniformity calculation
    batch_uniformity = evaluate_batch_uniformity(vols_arr.tolist())
    unif_rate = round(float(batch_uniformity.get("uniformity_rate_pct", 0.0)), 2)

    raw_bulk_volume = float(bulk_volume_mm3)
    mean_volume = float(np.mean(vols_arr))
    if mean_volume <= 0:
        raise ValueError("Mean grain volume must be positive.")

    hybrid_estimate = int(round((raw_bulk_volume * trained_hybrid_packing_fraction) / mean_volume))

    feat_dict: Dict[str, float] = {
        "Bulk_Rice_Volume_mm3": round(raw_bulk_volume, 2),
        "Rice_Height_mm": round(float(rice_height_mm), 2),
        "Weight_g": float(weight_g),
        "Empty_Height_mm": float(empty_height_mm),
        "Pixels_Per_mm": round(float(pixels_per_mm), 2),
        "Container_Detected_Diam_px": round(float(container_detected_diam_px), 2),
        "Inner_Diameter_mm": float(inner_diameter_mm),
        "Container_Height_mm": float(container_height_mm),
        "Whole_Grains_Count": float(len(raw_valid_metrics)),
        "Uniformity_Rate_Pct": unif_rate,
        "Estimated_Total_Seeds_Hybrid": float(hybrid_estimate),
        "Grain_Length_mm_Mean": lens_st["Mean"],
        "Grain_Length_mm_Min": lens_st["Min"],
        "Grain_Length_mm_Max": lens_st["Max"],
        "Grain_Length_mm_Std": lens_st["Std"],
        "Grain_Width_mm_Mean": wids_st["Mean"],
        "Grain_Width_mm_Min": wids_st["Min"],
        "Grain_Width_mm_Max": wids_st["Max"],
        "Grain_Width_mm_Std": wids_st["Std"],
        "Grain_Thickness_mm_Mean": thks_st["Mean"],
        "Grain_Thickness_mm_Min": thks_st["Min"],
        "Grain_Thickness_mm_Max": thks_st["Max"],
        "Grain_Thickness_mm_Std": thks_st["Std"],
        "Grain_Area_mm2_Mean": area_st["Mean"],
        "Grain_Area_mm2_Min": area_st["Min"],
        "Grain_Area_mm2_Max": area_st["Max"],
        "Grain_Area_mm2_Std": area_st["Std"],
        "Grain_Volume_mm3_Mean": vols_st["Mean"],
        "Grain_Volume_mm3_Min": vols_st["Min"],
        "Grain_Volume_mm3_Max": vols_st["Max"],
        "Grain_Volume_mm3_Std": vols_st["Std"],
    }

    # Strict validation of feature count, order, and finite numbers
    if len(feat_dict) != 31:
        raise ValueError(f"Expected 31 features, got {len(feat_dict)}")

    for k in ORDERED_FEATURES:
        if k not in feat_dict:
            raise ValueError(f"Missing required feature: {k}")
        v = feat_dict[k]
        if not math.isfinite(v):
            raise ValueError(f"Feature '{k}' is non-finite: {v}")

    return {k: feat_dict[k] for k in ORDERED_FEATURES}


def build_physical_diagnostics(
    filtered_volumes: List[float],
    raw_volumes: List[float],
    bulk_volume_mm3: float,
    pixels_per_mm: float,
    container_detected_diam_px: float,
    physical_packing_fraction: float = 0.55,
) -> Dict[str, Any]:
    """Calculate physical diagnostic values (isolated from regression feature contract)."""
    raw_mean_vol = float(np.mean(raw_volumes)) if raw_volumes else None

    if filtered_volumes and len(filtered_volumes) > 0:
        batch_unif = evaluate_batch_uniformity(filtered_volumes)
        mean_clean = float(batch_unif.get("mean_clean", 0.0))
        if mean_clean > 0:
            phys_count = int(round((bulk_volume_mm3 * physical_packing_fraction) / mean_clean))
        else:
            phys_count = None
    else:
        mean_clean = None
        phys_count = None

    return {
        "Physical_Estimated_Seeds": phys_count,
        "Whole_Grains_Filtered_Count": len(filtered_volumes),
        "Whole_Grains_Raw_Count": len(raw_volumes),
        "Mean_Clean_Grain_Volume_mm3": round(mean_clean, 3) if mean_clean is not None else None,
        "Mean_Raw_Grain_Volume_mm3": round(raw_mean_vol, 3) if raw_mean_vol is not None else None,
        "Scale_Pixels_Per_mm": round(float(pixels_per_mm), 2),
    }
