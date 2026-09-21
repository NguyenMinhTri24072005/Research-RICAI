"""Counterfactual geometry audit for the supplied 450-grain raw sample.

This script does not change production code.  It compares the notebook's
cylindrical bulk-volume assumption with a linearly tapered conical-frustum
interior implied by the sample's own measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent

# Exact values configured in notebook Cell 3 and reproduced by the raw audit.
TOP_INNER_DIAMETER_MM = 32.7
RICE_SURFACE_DIAMETER_MM = 22.0
CONTAINER_HEIGHT_MM = 48.7
EMPTY_HEIGHT_MM = 23.3
PACKING_FRACTION = 0.55
MEAN_GRAIN_VOLUME_MM3 = 6.618097055225826
ACTUAL_COUNT = 450


def cylinder_volume(diameter_mm: float, height_mm: float) -> float:
    return math.pi * (diameter_mm / 2.0) ** 2 * height_mm


def frustum_volume(diameter_bottom_mm: float, diameter_top_mm: float, height_mm: float) -> float:
    return math.pi * height_mm * (
        diameter_bottom_mm ** 2
        + diameter_bottom_mm * diameter_top_mm
        + diameter_top_mm ** 2
    ) / 12.0


def main() -> None:
    rice_height = CONTAINER_HEIGHT_MM - EMPTY_HEIGHT_MM

    # Linear interior taper: diameter is 32.7 mm at rim (depth 0), 22.0 mm
    # at rice surface (depth 25.4), and is extrapolated to the physical bottom
    # at depth 48.7. This is a testable assumption, not a claimed direct caliper
    # measurement of the unseen bottom.
    bottom_diameter = (
        RICE_SURFACE_DIAMETER_MM * CONTAINER_HEIGHT_MM
        - TOP_INNER_DIAMETER_MM * rice_height
    ) / EMPTY_HEIGHT_MM
    cylinder = cylinder_volume(RICE_SURFACE_DIAMETER_MM, rice_height)
    frustum = frustum_volume(bottom_diameter, RICE_SURFACE_DIAMETER_MM, rice_height)
    cylinder_count = cylinder * PACKING_FRACTION / MEAN_GRAIN_VOLUME_MM3
    frustum_count = frustum * PACKING_FRACTION / MEAN_GRAIN_VOLUME_MM3
    excess = cylinder_count - ACTUAL_COUNT

    result = {
        "evidence_kind": "Counterfactual from notebook Cell 3 sample inputs and independently reproduced selected-grain mean. Bottom diameter is inferred under a linear-taper assumption.",
        "inputs": {
            "top_inner_diameter_mm": TOP_INNER_DIAMETER_MM,
            "rice_surface_diameter_mm": RICE_SURFACE_DIAMETER_MM,
            "container_height_mm": CONTAINER_HEIGHT_MM,
            "empty_height_mm": EMPTY_HEIGHT_MM,
            "rice_height_mm": rice_height,
            "packing_fraction": PACKING_FRACTION,
            "mean_grain_volume_mm3": MEAN_GRAIN_VOLUME_MM3,
            "actual_count": ACTUAL_COUNT,
        },
        "inferred_bottom_diameter_mm_if_linear_taper": bottom_diameter,
        "cylinder": {
            "bulk_volume_mm3": cylinder,
            "physical_count_raw": cylinder_count,
            "physical_count_rounded": round(cylinder_count),
        },
        "conical_frustum": {
            "bulk_volume_mm3": frustum,
            "volume_ratio_to_cylinder": frustum / cylinder,
            "physical_count_raw": frustum_count,
            "physical_count_rounded": round(frustum_count),
            "error_to_actual_count": frustum_count - ACTUAL_COUNT,
            "percent_of_original_excess_removed": (cylinder_count - frustum_count) / excess * 100,
        },
        "interpretation": "This demonstrates that a cylinder is sufficient to produce the observed 802 only if the 22-mm rice surface diameter is incorrectly treated as constant down to the bottom. It does not prove the internal wall is exactly linear; caliper or depth-profile measurement is required before deploying frustum geometry.",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / "actual_geometry_counterfactual.json").write_text(text, encoding="utf-8")
    (OUT / "actual_geometry_counterfactual.log").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
