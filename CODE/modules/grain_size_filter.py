import numpy as np
from typing import List, Dict, Any, Tuple

def filter_grains_by_size(
    measurements: List[Dict[str, Any]], 
    k: float = 1.0, 
    min_samples: int = 8, 
    enabled: bool = True
) -> Dict[str, Any]:
    """
    Filter grain measurements based on IQR (LOWER bound only)
    for BOTH Area (area_mm2) and Volume (volume_3d_mm3 or volume_mm3).
    Returns a dict with kept measurements, rejected ones (with reason), and statistics.
    """
    if not enabled:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "disabled",
                "count_before": len(measurements),
                "count_after": len(measurements),
                "q1": None, "q3": None, "iqr": None, "lower_bound": None
            }
        }
    
    if not measurements:
        return {
            "kept": [],
            "rejected": [],
            "stats": {
                "status": "empty_input",
                "count_before": 0, "count_after": 0,
            }
        }
        
    n = len(measurements)
    if n < min_samples:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "insufficient_samples",
                "count_before": n,
                "count_after": n,
            }
        }
        
    areas = []
    volumes = []
    
    for m in measurements:
        area = m.get("area_mm2")
        vol = m.get("volume_3d_mm3") if "volume_3d_mm3" in m else m.get("volume_mm3")
        if area is not None and not np.isnan(area):
            areas.append(float(area))
        if vol is not None and not np.isnan(vol):
            volumes.append(float(vol))
            
    if len(areas) == 0 and len(volumes) == 0:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "invalid_areas_or_volumes",
                "count_before": n, "count_after": n,
            }
        }

    # Tính toán IQR cho Area
    lower_bound_a = 4.0 # Fallback
    if len(areas) > 0:
        q1_a = float(np.percentile(areas, 25))
        q3_a = float(np.percentile(areas, 75))
        iqr_a = q3_a - q1_a
        if iqr_a > 0:
            lb = q1_a - k * iqr_a
            if lb > 0:
                lower_bound_a = lb
    
    # Tính toán IQR cho Volume
    lower_bound_v = 4.0 # Fallback
    if len(volumes) > 0:
        q1_v = float(np.percentile(volumes, 25))
        q3_v = float(np.percentile(volumes, 75))
        iqr_v = q3_v - q1_v
        if iqr_v > 0:
            lb = q1_v - k * iqr_v
            if lb > 0:
                lower_bound_v = lb

    kept = []
    rejected = []
    
    for m in measurements:
        area = m.get("area_mm2")
        vol = m.get("volume_3d_mm3") if "volume_3d_mm3" in m else m.get("volume_mm3")
        
        is_rejected = False
        reason = ""
        
        if area is not None and not np.isnan(area):
            a_val = float(area)
            if a_val < lower_bound_a:
                is_rejected = True
                reason += f"Area ({a_val:.2f}) < mức tối thiểu ({lower_bound_a:.2f}). "
                
        if vol is not None and not np.isnan(vol):
            v_val = float(vol)
            if v_val < lower_bound_v:
                is_rejected = True
                reason += f"Volume ({v_val:.2f}) < mức tối thiểu ({lower_bound_v:.2f})."
                
        if is_rejected:
            rej = m.copy()
            rej["reject_reason"] = reason.strip()
            rejected.append(rej)
        else:
            kept.append(m)
            
    return {
        "kept": kept,
        "rejected": rejected,
        "stats": {
            "status": "filtered",
            "count_before": n,
            "count_after": len(kept),
            "lower_bound": lower_bound_a,  # giữ format cũ cho tương thích với print notebook
            "lower_bound_v": lower_bound_v
        }
    }
