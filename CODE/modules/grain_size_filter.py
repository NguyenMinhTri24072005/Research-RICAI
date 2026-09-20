import numpy as np
from typing import List, Dict, Any, Tuple

def filter_grains_by_size(
    measurements: List[Dict[str, Any]], 
    k: float = 1.5, 
    min_samples: int = 8, 
    enabled: bool = True
) -> Dict[str, Any]:
    """
    Filter grain measurements based on area IQR (lower bound only).
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
                "q1": None, "q3": None, "iqr": None, "lower_bound": None
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
                "q1": None, "q3": None, "iqr": None, "lower_bound": None
            }
        }
        
    areas = []
    for m in measurements:
        area = m.get("area_mm2")
        if area is not None and not np.isnan(area):
            areas.append(float(area))
            
    if len(areas) == 0:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "invalid_areas",
                "count_before": n, "count_after": n,
                "q1": None, "q3": None, "iqr": None, "lower_bound": None
            }
        }

    q1 = float(np.percentile(areas, 25))
    q3 = float(np.percentile(areas, 75))
    iqr = q3 - q1
    
    if iqr <= 1e-9:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "degenerate_iqr",
                "count_before": n, "count_after": n,
                "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": None
            }
        }
        
    lower_bound = q1 - k * iqr
    # Sửa thành cắt cứng mốc 40% các hạt nhỏ nhất
    # lower_bound = float(np.percentile(areas, 40))
    
    if lower_bound <= 0:
        return {
            "kept": measurements.copy(),
            "rejected": [],
            "stats": {
                "status": "non_positive_lower_bound",
                "count_before": n, "count_after": n,
                "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower_bound
            }
        }
        
    kept = []
    rejected = []
    
    for m in measurements:
        area = m.get("area_mm2")
        if area is not None and not np.isnan(area):
            if float(area) < lower_bound:
                rej = m.copy()
                rej["reject_reason"] = f"area ({area:.2f}) < lower_bound ({lower_bound:.2f})"
                rejected.append(rej)
            else:
                kept.append(m)
        else:
            # Keep invalid areas or handle differently? The prompt says "Dùng IQR một phía trên diện tích: chỉ loại outlier nhỏ". So if invalid, maybe keep it? Or it's a measurement error.
            # Plan says: keep it if it cannot be evaluated against lower_bound, or maybe it wouldn't have area_mm2.
            # Actually we'll keep it.
            kept.append(m)
            
    return {
        "kept": kept,
        "rejected": rejected,
        "stats": {
            "status": "filtered",
            "count_before": n,
            "count_after": len(kept),
            "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower_bound
        }
    }
