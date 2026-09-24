"""Optional lower-tail IQR filter for physical grain-volume estimation."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def filter_grains_by_size(
    measurements: List[Dict[str, Any]], k: float = 1.0, min_samples: int = 8, enabled: bool = True,
) -> Dict[str, Any]:
    """Filter only lower-tail area/volume outliers; numerical zero is never missing."""
    count = len(measurements)
    base_stats = {"count_before": count, "count_after": count}
    if not enabled:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "disabled", **base_stats}}
    if not measurements:
        return {"kept": [], "rejected": [], "stats": {"status": "empty_input", **base_stats}}
    if count < min_samples:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "insufficient_samples", **base_stats}}

    def valid_values(key: str) -> List[float]:
        return [float(item[key]) for item in measurements if item.get(key) is not None and np.isfinite(item[key])]

    areas, volumes = valid_values("area_mm2"), valid_values("volume_mm3")
    if not areas and not volumes:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "invalid_measurements", **base_stats}}

    def lower_bound(values: List[float]) -> float | None:
        if not values:
            return None
        q1, q3 = np.percentile(values, [25, 75])
        candidate = float(q1 - float(k) * (q3 - q1))
        return candidate if candidate > 0 else None

    area_lower, volume_lower = lower_bound(areas), lower_bound(volumes)
    kept, rejected = [], []
    for item in measurements:
        reasons = []
        area, volume = item.get("area_mm2"), item.get("volume_mm3")
        if area_lower is not None and area is not None and np.isfinite(area) and float(area) < area_lower:
            reasons.append(f"area_mm2={float(area):.3f} < {area_lower:.3f}")
        if volume_lower is not None and volume is not None and np.isfinite(volume) and float(volume) < volume_lower:
            reasons.append(f"volume_mm3={float(volume):.3f} < {volume_lower:.3f}")
        if reasons:
            rejected.append({**item, "reject_reason": "; ".join(reasons)})
        else:
            kept.append(item)

    return {"kept": kept, "rejected": rejected, "stats": {
        "status": "filtered", "count_before": count, "count_after": len(kept),
        "area_lower_bound": area_lower, "volume_lower_bound": volume_lower,
    }}
