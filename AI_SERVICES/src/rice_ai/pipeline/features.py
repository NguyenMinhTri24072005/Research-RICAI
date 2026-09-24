"""31-Feature Vector Assembly Stage."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from rice_ai.contracts import ContainerResult, GrainAnalysis
from rice_ai.estimation.feature_schema import (
    ALL_31_FEATURES,
    compute_trained_hybrid_feature,
    validate_feature_vector,
)

logger = logging.getLogger("rice_ai.pipeline.features")


def _safe_stat(values: List[float], func) -> Optional[float]:
    """Tính toán hàm thống kê an toàn (mean, min, max, std ddof=0). Trả về None nếu list rỗng."""
    if not values:
        return None
    return float(func(values))


def assemble_31_features(
    container: Union[ContainerResult, Dict[str, Any]],
    grain_analysis: Union[GrainAnalysis, List[Dict[str, Any]]],
    uniformity_res: Optional[Dict[str, Any]] = None,
    form_inputs: Optional[Dict[str, Any]] = None,
    hybrid_estimate: float = 0.0,
    hybrid_packing_fraction: float = 0.62,
) -> Dict[str, Optional[float]]:
    """Tập hợp chính xác vector 31 đặc trưng theo đúng hợp đồng 31v1.
    
    Quy tắc:
    - Feature 11 (Estimated_Total_Seeds_Hybrid) được tính theo hệ số 0.62 / mean grain volume (mm³).
    - Độ lệch chuẩn std sử dụng ddof=0 (population standard deviation).
    - Không gán 0.0 cho các giá trị thống kê khi không có hạt hợp lệ (giữ None).
    """
    # Adapter nếu truyền dict hoặc ContainerResult
    if isinstance(container, ContainerResult):
        bulk_volume_mm3 = container.bulk_volume_mm3
        rice_height_mm = container.rice_height_mm
        empty_height_mm = container.empty_height_mm
        pixels_per_mm = container.pixels_per_mm
        inner_diam_mm = container.inner_diam_mm
        container_height_mm = container.container_height_mm
        container_diam_px = float(container.raw_dict.get("inner_w_px", 0.0)) or None
    else:
        bulk_volume_mm3 = container.get("bulk_rice_volume_mm3", container.get("bulk_volume_mm3"))
        pixels_per_mm = container.get("pixels_per_mm")
        container_diam_px = container.get("inner_w_px")
        if container_diam_px is not None:
            container_diam_px = float(container_diam_px)
        
        inputs = form_inputs or {}
        container_height_mm = inputs.get("container_height_mm", container.get("container_height_mm"))
        empty_height_mm = inputs.get("empty_height_mm", container.get("empty_height_mm"))
        inner_diam_mm = inputs.get("inner_diameter_mm", container.get("inner_diam_mm"))

        rice_height_mm = None
        if container_height_mm is not None and empty_height_mm is not None:
            rice_height_mm = float(container_height_mm) - float(empty_height_mm)
            if rice_height_mm < 0:
                rice_height_mm = container.get("rice_height_mm")
        else:
            rice_height_mm = container.get("rice_height_mm")

    # Adapter nếu truyền GrainAnalysis hoặc list
    if isinstance(grain_analysis, GrainAnalysis):
        whole_grains = grain_analysis.whole_grains
        if uniformity_res is None:
            uniformity_res = grain_analysis.uniformity_metrics
    else:
        whole_grains = grain_analysis
        if uniformity_res is None:
            uniformity_res = {}

    inputs = form_inputs or {}
    weight_g = inputs.get("weight_g")
    if weight_g is not None:
        weight_g = float(weight_g)

    # Trích xuất số đo các hạt nguyên
    lengths = [float(g["length_mm"]) for g in whole_grains if g.get("length_mm") is not None]
    widths = [float(g["width_mm"]) for g in whole_grains if g.get("width_mm") is not None]
    thicknesses = [float(g["thickness_mm"]) for g in whole_grains if g.get("thickness_mm") is not None]
    areas = [float(g["area_mm2"]) for g in whole_grains if g.get("area_mm2") is not None]
    volumes = [float(g["volume_mm3"]) for g in whole_grains if g.get("volume_mm3") is not None]

    # Tính Feature 11: Estimated_Total_Seeds_Hybrid (0.62 * bulk / mean_grain_vol)
    trained_hybrid = compute_trained_hybrid_feature(
        bulk_volume_mm3, volumes, packing_fraction=hybrid_packing_fraction
    )
    if trained_hybrid is not None:
        hybrid_feature = float(trained_hybrid)
    elif hybrid_estimate is not None and hybrid_estimate > 0:
        hybrid_feature = float(hybrid_estimate)
    else:
        hybrid_feature = None

    features: Dict[str, Optional[float]] = {
        # Group 1: Container & Bulk (8)
        "Bulk_Rice_Volume_mm3": float(bulk_volume_mm3) if bulk_volume_mm3 is not None else None,
        "Rice_Height_mm": float(rice_height_mm) if rice_height_mm is not None else None,
        "Weight_g": weight_g,
        "Empty_Height_mm": float(empty_height_mm) if empty_height_mm is not None else None,
        "Pixels_Per_mm": float(pixels_per_mm) if pixels_per_mm is not None else None,
        "Container_Detected_Diam_px": container_diam_px,
        "Inner_Diameter_mm": float(inner_diam_mm) if inner_diam_mm is not None else None,
        "Container_Height_mm": float(container_height_mm) if container_height_mm is not None else None,
        # Group 2: Surface & Estimation (3)
        "Whole_Grains_Count": float(len(whole_grains)),
        "Uniformity_Rate_Pct": float(uniformity_res.get("uniformity_rate_pct")) if uniformity_res.get("uniformity_rate_pct") is not None else None,
        "Estimated_Total_Seeds_Hybrid": hybrid_feature,
        # Group 3: Length (4)
        "Grain_Length_mm_Mean": _safe_stat(lengths, np.mean),
        "Grain_Length_mm_Min": _safe_stat(lengths, np.min),
        "Grain_Length_mm_Max": _safe_stat(lengths, np.max),
        "Grain_Length_mm_Std": _safe_stat(lengths, np.std),
        # Group 4: Width (4)
        "Grain_Width_mm_Mean": _safe_stat(widths, np.mean),
        "Grain_Width_mm_Min": _safe_stat(widths, np.min),
        "Grain_Width_mm_Max": _safe_stat(widths, np.max),
        "Grain_Width_mm_Std": _safe_stat(widths, np.std),
        # Group 5: Thickness (4)
        "Grain_Thickness_mm_Mean": _safe_stat(thicknesses, np.mean),
        "Grain_Thickness_mm_Min": _safe_stat(thicknesses, np.min),
        "Grain_Thickness_mm_Max": _safe_stat(thicknesses, np.max),
        "Grain_Thickness_mm_Std": _safe_stat(thicknesses, np.std),
        # Group 6: Area (4)
        "Grain_Area_mm2_Mean": _safe_stat(areas, np.mean),
        "Grain_Area_mm2_Min": _safe_stat(areas, np.min),
        "Grain_Area_mm2_Max": _safe_stat(areas, np.max),
        "Grain_Area_mm2_Std": _safe_stat(areas, np.std),
        # Group 7: Volume (4)
        "Grain_Volume_mm3_Mean": _safe_stat(volumes, np.mean),
        "Grain_Volume_mm3_Min": _safe_stat(volumes, np.min),
        "Grain_Volume_mm3_Max": _safe_stat(volumes, np.max),
        "Grain_Volume_mm3_Std": _safe_stat(volumes, np.std),
    }

    return features
