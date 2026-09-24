"""Adaptive lower-tail IQR filter shared by the research and API pipelines."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def filter_grains_by_size(
    measurements: List[Dict[str, Any]],
    k: float = 0.1,
    min_samples: int = 8,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Reject only lower-tail outliers; never apply a fixed physical cutoff.

    A missing measurement is not a zero.  The volume key accepts both names
    used by historical notebook cells.
    """
    count = len(measurements)
    base = {"count_before": count, "count_after": count}
    if not enabled:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "disabled", **base}}
    if not measurements:
        return {"kept": [], "rejected": [], "stats": {"status": "empty_input", **base}}
    if count < min_samples:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "insufficient_samples", **base}}

    def value(item: Dict[str, Any], key: str) -> Any:
        if key == "volume_mm3":
            return item.get("volume_mm3", item.get("volume_3d_mm3"))
        return item.get(key)

    def finite_values(key: str) -> List[float]:
        values = []
        for item in measurements:
            raw = value(item, key)
            if raw is not None and np.isfinite(raw):
                values.append(float(raw))
        return values

    def lower_bound(values: List[float]) -> float | None:
        if not values:
            return None
        q1, q3 = np.percentile(values, [25, 75])
        candidate = float(q1 - float(k) * (q3 - q1))
        return candidate if candidate > 0 else None

    areas = finite_values("area_mm2")
    volumes = finite_values("volume_mm3")
    area_lower = lower_bound(areas)
    volume_lower = lower_bound(volumes)
    if area_lower is None and volume_lower is None:
        return {"kept": measurements.copy(), "rejected": [], "stats": {"status": "invalid_measurements", **base}}

    kept, rejected = [], []
    for item in measurements:
        reasons = []
        area, volume = value(item, "area_mm2"), value(item, "volume_mm3")
        if area_lower is not None and area is not None and np.isfinite(area) and float(area) < area_lower:
            reasons.append(f"area_mm2={float(area):.3f} < {area_lower:.3f}")
        if volume_lower is not None and volume is not None and np.isfinite(volume) and float(volume) < volume_lower:
            reasons.append(f"volume_mm3={float(volume):.3f} < {volume_lower:.3f}")
        if reasons:
            rejected.append({**item, "reject_reason": "; ".join(reasons)})
        else:
            kept.append(item)

    return {
        "kept": kept,
        "rejected": rejected,
        "stats": {
            "status": "filtered",
            "count_before": count,
            "count_after": len(kept),
            "area_lower_bound": area_lower,
            "volume_lower_bound": volume_lower,
        },
    }