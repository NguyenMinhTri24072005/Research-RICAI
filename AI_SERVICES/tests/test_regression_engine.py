#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for regression_engine.py
"""

import sys
import unittest
from pathlib import Path

# Thêm AI_SERVICES vào sys.path
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from feature_schema import ALL_31_FEATURES
from model_registry import ModelRegistry
from regression_engine import (
    assemble_31_features,
    predict_from_tree,
    predict_from_equation,
    predict_regression,
)


class TestRegressionEngine(unittest.TestCase):
    def setUp(self):
        self.manifest_path = AI_SERVICES_DIR / "artifacts" / "manifest.json"
        self.registry = ModelRegistry(project_root=PROJECT_ROOT, manifest_path=self.manifest_path)
        self.registry.load_bundle()

    def test_assemble_31_features_keys(self):
        container_res = {
            "pixels_per_mm": 67.0,
            "bulk_rice_volume_mm3": 8200.0,
            "rice_height_mm": 23.0,
            "inner_w_px": 1200.0,
        }
        whole_grains = [
            {"length_mm": 7.5, "width_mm": 2.5, "thickness_mm": 6.5, "area_mm2": 30.0, "volume_mm3": 750.0},
            {"length_mm": 7.7, "width_mm": 2.4, "thickness_mm": 6.4, "area_mm2": 29.5, "volume_mm3": 740.0},
        ]
        uniformity_res = {"uniformity_rate_pct": 95.0}
        form_inputs = {
            "weight_g": 4.2,
            "empty_height_mm": 11.0,
            "inner_diameter_mm": 20.0,
            "container_height_mm": 34.0,
        }

        vec = assemble_31_features(
            container_res=container_res,
            whole_grains=whole_grains,
            uniformity_res=uniformity_res,
            form_inputs=form_inputs,
            hybrid_estimate=150.0,
        )

        for key in ALL_31_FEATURES:
            self.assertIn(key, vec)
        self.assertEqual(vec["Whole_Grains_Count"], 2.0)
        self.assertEqual(vec["Weight_g"], 4.2)
        self.assertAlmostEqual(vec["Rice_Height_mm"], 23.0)

    def test_predict_from_tree_inference(self):
        model, scaler = self.registry.get_model_and_scaler()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)

        # Vector mẫu
        vec = {k: 5.0 for k in ALL_31_FEATURES}
        vec["Bulk_Rice_Volume_mm3"] = 8000.0
        vec["Rice_Height_mm"] = 23.0
        vec["Weight_g"] = 4.0

        pred = predict_from_tree(model, scaler, vec)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

    def test_predict_from_equation(self):
        vec = {k: 5.0 for k in ALL_31_FEATURES}
        pred = predict_from_equation(vec)
        self.assertIsInstance(pred, float)

    def test_predict_regression_raises_on_none_model(self):
        with self.assertRaises(ValueError):
            predict_regression(features_dict={}, model=None, scaler=None)


if __name__ == "__main__":
    unittest.main()
