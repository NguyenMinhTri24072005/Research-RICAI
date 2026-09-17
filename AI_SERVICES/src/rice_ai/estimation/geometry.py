"""Geometry-based Seed Count Estimation."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def compute_geometry_estimate(
    bulk_volume_mm3: float,
    pixels_per_mm: float,
    volumes_px3_list: Sequence[float],
    packing_fraction: float = 0.82,
) -> Optional[int]:
    """Tính toán số lượng hạt theo phương pháp hình học không gian (Pixel Volume).
    
    Công thức:
        bulk_volume_px3 = bulk_volume_mm3 * (pixels_per_mm ** 3)
        effective_bulk_px3 = bulk_volume_px3 * packing_fraction
        ai_est = round(effective_bulk_px3 / median(volumes_px3_list))
    
    Lưu ý:
        Hệ số packing_fraction của phương pháp hình học thực tế là 0.82
        (độc lập với hệ số 0.62 của Feature 11 trong vector hồi quy 31 biến).
    """
    if not volumes_px3_list or pixels_per_mm <= 0 or bulk_volume_mm3 <= 0:
        return None

    valid_vols = [float(v) for v in volumes_px3_list if v is not None and v > 0 and math.isfinite(v)]
    if not valid_vols:
        return None

    pixels_per_mm3 = pixels_per_mm ** 3
    bulk_volume_px3 = bulk_volume_mm3 * pixels_per_mm3
    effective_bulk_px3 = bulk_volume_px3 * packing_fraction

    median_grain_vol_px3 = float(np.median(valid_vols))
    if median_grain_vol_px3 <= 0 or not math.isfinite(median_grain_vol_px3):
        return None

    geometry_est = int(round(effective_bulk_px3 / median_grain_vol_px3))
    return max(0, geometry_est)
