#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for API contracts, lifecycle, admission semaphore cancellation, and diagnostics.
"""

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.api.application import create_app
from rice_ai.contracts import PipelineResult, EstimateSet, ContainerResult, GrainAnalysis
from rice_ai.settings import Settings


def make_dummy_jpeg() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class TestAPILifecycle(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.mock_pipeline = MagicMock()
        self.app = create_app(self.settings, pipeline=self.mock_pipeline)

    def test_negative_physical_inputs_rejected_422(self):
        """Các giá trị âm cho diam, height, empty, weights phải bị từ chối 422 trước khi vào pipeline."""
        with TestClient(self.app) as client:
            dummy_file = ("test.jpg", make_dummy_jpeg(), "image/jpeg")

            # diam âm
            res = client.post("/predict", data={"diam": -1.0, "height": 3.0, "empty": 1.0}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

            # height âm
            res = client.post("/predict", data={"diam": 2.0, "height": -3.0, "empty": 1.0}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

            # empty âm
            res = client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": -1.0}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

            # weight_total âm
            res = client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": 1.0, "weight_total": -5.0}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

            # sample_weight âm
            res = client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": 1.0, "sample_weight": -0.5}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

            # sample_count âm
            res = client.post("/predict", data={"diam": 2.0, "height": 3.0, "empty": 1.0, "sample_count": -10}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 422)

    def test_empty_equal_height_boundary_preserved(self):
        """empty == height là hợp lệ tại boundary (cốc rỗng)."""
        from rice_ai.contracts import PredictionInput
        p_input = PredictionInput(diam=2.0, height=3.0, empty=3.0)
        # Không được ném lỗi ở input boundary validation
        p_input.validate()

    def test_legacy_response_keys_present_in_predict_success(self):
        """Phản hồi /predict thành công phải khôi phục đầy đủ các key cũ cho React & Phone:
        whole_grains_surface, total_grains_detected, avg_length_mm, avg_width_mm, avg_thickness_mm, regression_model, estimator_mode.
        """
        # Mock pipeline.run để trả về kết quả giả định nhanh
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PipelineResult(
            request_id="req123",
            estimates=EstimateSet(final=85, regression_est=85.2, geometry_est=280, weight_est=None, method_used="regression_ExtraTrees"),
            container=ContainerResult(inner_diam_mm=17.8, container_height_mm=33.9, empty_height_mm=10.9, rice_height_mm=23.0, bulk_volume_mm3=5723.4, pixels_per_mm=39.0),
            grain_analysis=GrainAnalysis(
                whole_grains=[{"length_mm": 5.0, "width_mm": 2.0, "thickness_mm": 1.8, "volume_mm3": 20.0}],
                broken_grains=[],
                chalky_grains=[],
                foreign_objects=[],
                total_detected=40,
                classified_counts={"hat_nguyen": 35, "hat_khuyet_tat": 5},
                volumes_px3=[1500.0],
                uniformity_metrics={"uniformity_rate_pct": 92.5},
            ),
            features_31={"Bulk_Rice_Volume_mm3": 5723.4},
            timings_ms={"total_ms": 500.0, "segmentation_ms": 200.0, "cleaning_ms": 50.0, "classification_ms": 100.0},
            warnings=[],
        )

        app = create_app(self.settings, pipeline=mock_pipeline)
        with TestClient(app) as client:
            dummy_file = ("test.jpg", make_dummy_jpeg(), "image/jpeg")
            res = client.post("/predict", data={"diam": 1.78, "height": 3.39, "empty": 1.09, "debug": "false"}, files={"file": dummy_file})
            self.assertEqual(res.status_code, 200)
            data = res.json()

            summary = data.get("metrics_summary", {})
            self.assertIn("whole_grains_surface", summary)
            self.assertIn("total_grains_detected", summary)
            self.assertIn("avg_length_mm", summary)
            self.assertIn("avg_width_mm", summary)
            self.assertIn("avg_thickness_mm", summary)
            self.assertIn("regression_model", summary)
            self.assertIn("estimator_mode", summary)

            # features_used chỉ trả khi debug=True
            self.assertNotIn("features_used", data) or self.assertIsNone(data.get("features_used"))

    def test_cancellation_preserves_admission_gate_while_worker_active(self):
        """Khi coroutine HTTP caller bị cancel hoặc timeout, admission gate KHÔNG được thả sớm nếu worker thread vẫn đang chạy."""
        # Tạo pipeline worker giả lập chạy lâu (0.4s)
        import threading
        worker_started = threading.Event()

        def slow_run(*args, **kwargs):
            worker_started.set()
            time.sleep(0.4)
            return PipelineResult(
                request_id="slow",
                estimates=EstimateSet(final=50, regression_est=50.0, geometry_est=50, weight_est=None, method_used="mock"),
                container=ContainerResult(inner_diam_mm=10, container_height_mm=20, empty_height_mm=5, rice_height_mm=15, bulk_volume_mm3=1000, pixels_per_mm=10),
                grain_analysis=GrainAnalysis([], [], [], [], 0, {}, [], 0, {}),
                features_31={},
                timings_ms={},
            )

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = slow_run
        app = create_app(self.settings, pipeline=mock_pipeline)

        with TestClient(app) as client:
            dummy_file = ("test.jpg", make_dummy_jpeg(), "image/jpeg")
            
            # Start background async task that gets canceled
            # We test through asyncio
            async def run_test():
                import httpx
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as aclient:
                    # Request 1: gửi request
                    task1 = asyncio.create_task(
                        aclient.post(
                            "/predict",
                            data={"diam": 2.0, "height": 3.0, "empty": 1.0},
                            files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
                        )
                    )
                    # Chờ worker thread thực sự bắt đầu chạy trong threadpool
                    for _ in range(100):
                        if worker_started.is_set():
                            break
                        await asyncio.sleep(0.01)
                    self.assertTrue(worker_started.is_set(), "Worker thread chưa kịp bắt đầu!")

                    # Hủy request HTTP trong khi worker thread vẫn đang chạy slow_run
                    task1.cancel()
                    try:
                        await task1
                    except asyncio.CancelledError:
                        pass

                    # Tại thời điểm này worker vẫn đang chạy!
                    # Request 2 gửi tới phải nhận 503 (Server Busy) vì gate vẫn đang bị worker 1 chiếm giữ
                    res2 = await aclient.post(
                        "/predict",
                        data={"diam": 2.0, "height": 3.0, "empty": 1.0},
                        files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
                    )
                    self.assertEqual(res2.status_code, 503, "Request 2 phải nhận 503 khi worker 1 vẫn đang chạy!")

                    # Chờ worker 1 kết thúc
                    await asyncio.sleep(0.45)

                    # Sau khi worker 1 kết thúc, Request 3 gửi tới phải thành công
                    res3 = await aclient.post(
                        "/predict",
                        data={"diam": 2.0, "height": 3.0, "empty": 1.0},
                        files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
                    )
                    self.assertEqual(res3.status_code, 200, "Request 3 phải thành công sau khi worker 1 kết thúc!")

            asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
