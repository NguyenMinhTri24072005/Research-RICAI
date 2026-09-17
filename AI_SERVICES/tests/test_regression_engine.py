#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for regression prediction and 31-feature assembly in rice_ai.
Migrated from legacy regression_engine.py to modular rice_ai package.
"""

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TESTS_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.contracts import ContainerResult, GrainAnalysis
from rice_ai.estimation.feature_schema import ALL_31_FEATURES, compute_trained_hybrid_feature
from rice_ai.estimation.regression import predict_regression
from rice_ai.models.regression_loader import LoadedRegressionProvider, load_regression_folder
from rice_ai.pipeline.features import assemble_31_features
from rice_ai.settings import Settings


class TestRegressionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.provider = LoadedRegressionProvider(cls.settings)
        cls.loaded_reg = cls.provider.get_regression()

    def test_assemble_31_features_keys(self):
        container = ContainerResult(
            inner_diam_mm=20.0,
            container_height_mm=34.0,
            empty_height_mm=11.0,
            rice_height_mm=23.0,
            bulk_volume_mm3=8200.0,
            pixels_per_mm=67.0,
        )
        whole_grains = [
            {"length_mm": 7.5, "width_mm": 2.5, "thickness_mm": 6.5, "area_mm2": 30.0, "volume_mm3": 750.0},
            {"length_mm": 7.7, "width_mm": 2.4, "thickness_mm": 6.4, "area_mm2": 29.5, "volume_mm3": 740.0},
        ]
        grain_analysis = GrainAnalysis(
            whole_grains=whole_grains,
            broken_grains=[],
            chalky_grains=[],
            foreign_objects=[],
            total_detected=2,
            classified_counts={"hat_nguyen": 2},
            volumes_px3=[],
            uniformity_metrics={"uniformity_rate_pct": 95.0},
        )
        form_inputs = {"weight_g": 4.2}

        vec = assemble_31_features(
            container=container,
            grain_analysis=grain_analysis,
            form_inputs=form_inputs,
        )

        for key in ALL_31_FEATURES:
            self.assertIn(key, vec)
        self.assertEqual(vec["Whole_Grains_Count"], 2.0)
        self.assertEqual(vec["Weight_g"], 4.2)
        self.assertAlmostEqual(vec["Rice_Height_mm"], 23.0)

    def test_predict_regression_inference(self):
        vec = {k: 5.0 for k in ALL_31_FEATURES}
        vec["Bulk_Rice_Volume_mm3"] = 8000.0
        vec["Rice_Height_mm"] = 23.0
        vec["Weight_g"] = 4.0

        pred, model_name = predict_regression(vec, self.loaded_reg)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)
        self.assertTrue(len(model_name) > 0)

    def test_assemble_hybrid_feature_invariance_to_weight(self):
        container = ContainerResult(
            inner_diam_mm=20.0,
            container_height_mm=34.0,
            empty_height_mm=11.0,
            rice_height_mm=23.0,
            bulk_volume_mm3=8000.0,
            pixels_per_mm=67.0,
        )
        whole_grains = [
            {"length_mm": 7.0, "width_mm": 2.0, "thickness_mm": 6.0, "area_mm2": 20.0, "volume_mm3": 20.0},
        ]
        grain_analysis = GrainAnalysis(
            whole_grains=whole_grains,
            broken_grains=[],
            chalky_grains=[],
            foreign_objects=[],
            total_detected=1,
            classified_counts={"hat_nguyen": 1},
            volumes_px3=[],
            uniformity_metrics={"uniformity_rate_pct": 95.0},
        )

        vec1 = assemble_31_features(
            container=container,
            grain_analysis=grain_analysis,
            form_inputs={"weight_g": 2.0},
        )
        vec2 = assemble_31_features(
            container=container,
            grain_analysis=grain_analysis,
            form_inputs={"weight_g": 200.0},
        )

        expected_hybrid = compute_trained_hybrid_feature(8000.0, [20.0])
        self.assertEqual(vec1["Estimated_Total_Seeds_Hybrid"], expected_hybrid)
        self.assertEqual(vec2["Estimated_Total_Seeds_Hybrid"], expected_hybrid)
        self.assertEqual(vec1["Estimated_Total_Seeds_Hybrid"], 248)

    def test_predict_rejects_missing_required_feature(self):
        vec = {k: 5.0 for k in ALL_31_FEATURES}
        del vec["Grain_Volume_mm3_Mean"]
        with self.assertRaises(ValueError) as ctx:
            predict_regression(vec, self.loaded_reg)
        self.assertIn("Grain_Volume_mm3_Mean", str(ctx.exception))

    def test_predict_rejects_nan_feature(self):
        vec = {k: 5.0 for k in ALL_31_FEATURES}
        vec["Bulk_Rice_Volume_mm3"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            predict_regression(vec, self.loaded_reg)
        self.assertIn("Bulk_Rice_Volume_mm3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
