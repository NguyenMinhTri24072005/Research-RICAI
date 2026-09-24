#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for RicePipeline runner and stage coordination.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.contracts import (
    ContainerResult,
    GrainAnalysis,
    PipelineError,
    PredictionInput,
)
from rice_ai.models.regression_loader import LoadedRegressionProvider
from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.pipeline.runner import RicePipeline
from rice_ai.pipeline.grains import process_grains
from rice_ai.settings import Settings


class TestRicePipeline(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.vision_provider = MagicMock(spec=VisionModelProvider)
        self.regression_provider = LoadedRegressionProvider(self.settings)
        self.pipeline = RicePipeline(self.settings, self.vision_provider, self.regression_provider)

        # Synthetic test image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".jpg", img)
        self.image_bytes = buf.tobytes()

    @patch("rice_ai.pipeline.runner.analyze_container")
    @patch("rice_ai.pipeline.runner.process_grains")
    def test_pipeline_run_success(self, mock_process_grains, mock_analyze_container):
        mock_analyze_container.return_value = ContainerResult(
            inner_diam_mm=17.8,
            container_height_mm=33.9,
            empty_height_mm=10.9,
            rice_height_mm=23.0,
            bulk_volume_mm3=5724.8,
            pixels_per_mm=67.0,
            raw_dict={"inner_w_px": 1192.6},
        )

        mock_process_grains.return_value = GrainAnalysis(
            whole_grains=[
                {"length_mm": 7.5, "width_mm": 2.5, "thickness_mm": 1.8, "area_mm2": 15.0, "volume_mm3": 17.5, "vol_px3": 5200.0},
                {"length_mm": 7.6, "width_mm": 2.4, "thickness_mm": 1.9, "area_mm2": 15.2, "volume_mm3": 17.8, "vol_px3": 5300.0},
            ],
            broken_grains=[],
            chalky_grains=[],
            foreign_objects=[],
            total_detected=2,
            classified_counts={"hat_nguyen": 2, "hat_khuyet_tat": 0},
            volumes_px3=[5200.0, 5300.0],
            skipped_measurement_count=0,
            uniformity_metrics={"uniformity_rate_pct": 94.5},
        )

        inputs = PredictionInput(
            diam=1.78,
            height=3.39,
            empty=1.09,
            weight_total=2.69,
            estimator_mode="auto",
        )

        result = self.pipeline.run(inputs, self.image_bytes, request_id="test_req_01")

        self.assertEqual(result.request_id, "test_req_01")
        self.assertIsNotNone(result.estimates.final)
        self.assertGreater(result.estimates.final, 0)
        self.assertIn("regression", result.estimates.method_used)
        self.assertIsNotNone(result.estimates.regression_est)
        self.assertIsNotNone(result.estimates.geometry_est)
        self.assertIn("decode_ms", result.timings_ms)
        self.assertIn("total_ms", result.timings_ms)
        self.assertEqual(len(result.features_31), 31)

    def test_pipeline_invalid_image_raises_error(self):
        inputs = PredictionInput(diam=1.78, height=3.39, empty=1.09)
        with self.assertRaises(PipelineError) as ctx:
            self.pipeline.run(inputs, b"NOT_AN_IMAGE", request_id="err_req")
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.error_code, "INVALID_IMAGE")

    @patch("rice_ai.pipeline.grains.segment_grains_sahi", return_value=[])
    @patch("rice_ai.pipeline.grains.evaluate_batch_uniformity", return_value={})
    def test_grain_stage_returns_filter_contract(self, _uniformity, _segment):
        result = process_grains(
            image_input=np.zeros((20, 20, 3), dtype=np.uint8),
            pixels_per_mm=10.0,
            vision_provider=self.vision_provider,
            settings=self.settings,
        )
        self.assertEqual(result.size_filter_rejected, [])
        self.assertEqual(result.size_filter_stats["status"], "empty_input")


if __name__ == "__main__":
    unittest.main()
