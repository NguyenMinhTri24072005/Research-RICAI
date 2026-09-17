"""Weight-based Seed Count Estimation."""
from __future__ import annotations

import math
from typing import Optional


def compute_weight_estimate(
    weight_total: Optional[float],
    sample_weight: Optional[float],
    sample_count: Optional[int],
) -> Optional[int]:
    """Tính toán số lượng hạt theo phương pháp cân mẫu khối lượng.
    
    Công thức:
        weight_est = round((weight_total / sample_weight) * sample_count)
    """
    if (
        weight_total is not None
        and sample_weight is not None
        and sample_count is not None
        and weight_total > 0
        and sample_weight > 0
        and sample_count > 0
        and math.isfinite(weight_total)
        and math.isfinite(sample_weight)
    ):
        return int(round((weight_total / sample_weight) * sample_count))
    return None
