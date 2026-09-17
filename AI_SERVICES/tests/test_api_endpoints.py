#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API contract tests for FastAPI app.py
"""

import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Thêm AI_SERVICES vào sys.path
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from fastapi.testclient import TestClient
from app import app


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("readiness", data)
        self.assertIn(data["readiness"], ["ready", "degraded", "not_ready"])
        self.assertEqual(data.get("schema_version"), "31v1")
        self.assertIn("components", data)
        self.assertIn("regression", data["components"])
        # Kiểm tra không lộ đường dẫn filesystem tuyệt đối
        for comp_name, comp_info in data["components"].items():
            file_val = comp_info.get("file", "")
            self.assertNotIn(":", file_val)
            self.assertNotIn("/", file_val)
            self.assertNotIn("\\", file_val)

    def test_predict_empty_file_rejected(self):
        data = {
            "diam": 1.78,
            "height": 3.39,
            "empty": 1.09,
        }
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        res = self.client.post("/predict", data=data, files=files)
        self.assertEqual(res.status_code, 422)
        res_json = res.json()
        self.assertEqual(res_json.get("status"), "error")
        self.assertEqual(res_json.get("error", {}).get("code"), "INVALID_IMAGE")

    def test_predict_corrupted_image_rejected(self):
        data = {
            "diam": 1.78,
            "height": 3.39,
            "empty": 1.09,
        }
        files = {"file": ("bad.jpg", b"NOT_AN_IMAGE_CONTENT_HERE", "image/jpeg")}
        res = self.client.post("/predict", data=data, files=files)
        self.assertEqual(res.status_code, 422)
        res_json = res.json()
        self.assertEqual(res_json.get("status"), "error")
        self.assertEqual(res_json.get("error", {}).get("code"), "INVALID_IMAGE")

    def test_predict_invalid_dimensions(self):
        _, img_buf = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        files = {"file": ("dummy.jpg", img_buf.tobytes(), "image/jpeg")}
        
        # diam <= 0
        res = self.client.post("/predict", data={"diam": 0, "height": 3.0, "empty": 1.0}, files=files)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json().get("error", {}).get("code"), "INVALID_INPUT")

        # height <= 0
        res = self.client.post("/predict", data={"diam": 2.0, "height": -1.0, "empty": 1.0}, files=files)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json().get("error", {}).get("code"), "INVALID_INPUT")

        # empty > height
        res = self.client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": 4.0}, files=files)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json().get("error", {}).get("code"), "INVALID_INPUT")

        # invalid estimator_mode
        res = self.client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": 1.0, "estimator_mode": "invalid_mode"}, files=files)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json().get("error", {}).get("code"), "INVALID_INPUT")

    def test_predict_regression_mode_fails_validation_when_grains_empty(self):
        from unittest.mock import patch
        from rice_ai.contracts import ContainerResult, GrainAnalysis
        with patch("rice_ai.pipeline.runner.analyze_container") as mock_cnt, \
             patch("rice_ai.pipeline.runner.process_grains") as mock_grains:

            mock_cnt.return_value = ContainerResult(
                inner_diam_mm=20.0,
                container_height_mm=30.0,
                empty_height_mm=10.0,
                rice_height_mm=20.0,
                bulk_volume_mm3=8000.0,
                pixels_per_mm=67.0,
                raw_dict={"inner_w_px": 1200.0},
            )
            mock_grains.return_value = GrainAnalysis(
                whole_grains=[],
                broken_grains=[],
                chalky_grains=[],
                foreign_objects=[],
                total_detected=0,
                classified_counts={"hat_nguyen": 0, "hat_khuyet_tat": 0},
                volumes_px3=[],
                skipped_measurement_count=0,
                uniformity_metrics={},
            )

            import cv2
            import numpy as np
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, buf = cv2.imencode(".jpg", img)

            files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
            data = {
                "diam": 2.0,
                "height": 3.0,
                "empty": 1.0,
                "estimator_mode": "regression",
            }
            res = self.client.post("/predict", data=data, files=files)
            self.assertEqual(res.status_code, 422)
            res_json = res.json()
            self.assertEqual(res_json.get("status"), "error")
            self.assertEqual(res_json.get("error", {}).get("code"), "MISSING_FEATURE")


if __name__ == "__main__":
    unittest.main()
