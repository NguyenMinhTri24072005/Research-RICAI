#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for geometry, weight, and fusion estimation modules.
"""

import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.contracts import PipelineError
from rice_ai.estimation.fusion import select_final_estimate
from rice_ai.estimation.geometry import compute_geometry_estimate
from rice_ai.estimation.weight import compute_weight_estimate


class TestEstimators(unittest.TestCase):
    def test_geometry_estimate_calculation(self):
        bulk_vol_mm3 = 1000.0
        pixels_per_mm = 10.0  # 1 mm3 = 1000 px3
        volumes_px3 = [5000.0, 6000.0, 7000.0]  # median = 6000 px3
        # bulk_px3 = 1000 * 1000 = 1,000,000 px3
        # effective = 1,000,000 * 0.82 = 820,000 px3
        # ai_est = round(820,000 / 6000) = 137 hạt
        est = compute_geometry_estimate(bulk_vol_mm3, pixels_per_mm, volumes_px3, packing_fraction=0.82)
        self.assertEqual(est, 137)

    def test_geometry_estimate_empty_volumes_returns_none(self):
        est = compute_geometry_estimate(1000.0, 10.0, [])
        self.assertIsNone(est)

    def test_geometry_estimate_invalid_volume_returns_none(self):
        est = compute_geometry_estimate(0.0, 10.0, [100.0])
        self.assertIsNone(est)

    def test_weight_estimate_calculation(self):
        # 10.0g tổng / 2.0g mẫu * 50 hạt mẫu = 250 hạt
        est = compute_weight_estimate(weight_total=10.0, sample_weight=2.0, sample_count=50)
        self.assertEqual(est, 250)

    def test_weight_estimate_missing_returns_none(self):
        self.assertIsNone(compute_weight_estimate(None, 2.0, 50))
        self.assertIsNone(compute_weight_estimate(10.0, 0.0, 50))
        self.assertIsNone(compute_weight_estimate(10.0, 2.0, 0))

    def test_fusion_regression_mode(self):
        # Success
        estimates = select_final_estimate(
            mode="regression",
            geometry_est=100,
            weight_est=110,
            regression_est=105.4,
            regression_model_name="ARDRegression",
        )
        self.assertEqual(estimates.final, 105)
        self.assertEqual(estimates.method_used, "regression_ARDRegression")

        # Failure when regression is None
        with self.assertRaises(PipelineError) as ctx:
            select_final_estimate("regression", 100, 110, None)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_fusion_geometry_mode(self):
        estimates = select_final_estimate(
            mode="geometry",
            geometry_est=150,
            weight_est=140,
            regression_est=130.0,
        )
        self.assertEqual(estimates.final, 150)
        self.assertEqual(estimates.method_used, "geometry")

        with self.assertRaises(PipelineError) as ctx:
            select_final_estimate("geometry", None, 140, 130.0)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_fusion_weight_mode(self):
        estimates = select_final_estimate(
            mode="weight",
            geometry_est=150,
            weight_est=140,
            regression_est=130.0,
        )
        self.assertEqual(estimates.final, 140)
        self.assertEqual(estimates.method_used, "weight")

        with self.assertRaises(PipelineError) as ctx:
            select_final_estimate("weight", 150, None, 130.0)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_fusion_auto_priority_policy(self):
        # 1. Regression priority
        e1 = select_final_estimate("auto", 100, 120, 115.2, "ExtraTrees")
        self.assertEqual(e1.final, 115)
        self.assertEqual(e1.method_used, "regression_ExtraTrees")

        # 2. Fallback to hybrid geometry+weight when regression is None
        e2 = select_final_estimate("auto", 100, 120, None)
        self.assertEqual(e2.final, 110)
        self.assertEqual(e2.method_used, "hybrid")

        # 3. Fallback to geometry only
        e3 = select_final_estimate("auto", 100, None, None)
        self.assertEqual(e3.final, 100)
        self.assertEqual(e3.method_used, "geometry")

        # 4. Fallback to weight only
        e4 = select_final_estimate("auto", None, 120, None)
        self.assertEqual(e4.final, 120)
        self.assertEqual(e4.method_used, "weight")


if __name__ == "__main__":
    unittest.main()
