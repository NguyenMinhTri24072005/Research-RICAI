#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end integration test with real image fixture (Sample M001a)
Ground Truth: Actual_Count = 85
"""

import sys
import unittest
from pathlib import Path

AI_SERVICES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from fastapi.testclient import TestClient
from app import app, MODEL_PATHS


class TestRealPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.fixture_path = PROJECT_ROOT / "DATASET_BUILDER" / "1_Raw_Images" / "M001" / "M001A.jpg"
        cls.ground_truth_count = 85

    def test_real_image_inference(self):
        if not self.fixture_path.exists():
            self.skipTest(f"Fixture image không tồn tại tại {self.fixture_path}")

        # Kiểm tra xem YOLO và CNN weights có tồn tại không
        yolo_path = MODEL_PATHS.get("yolo")
        cnn_path = MODEL_PATHS.get("cnn")
        if not yolo_path or not yolo_path.exists():
            self.skipTest("YOLO weights không tồn tại trong môi trường test này.")
        if not cnn_path or not cnn_path.exists():
            self.skipTest("CNN weights không tồn tại trong môi trường test này.")

        data = {
            "diam": 1.78,
            "height": 3.39,
            "empty": 1.09,
            "weight_total": 2.69,
            "estimator_mode": "auto",
            "debug": "true",
        }

        with open(self.fixture_path, "rb") as img_file:
            files = {"file": ("M001A.jpg", img_file, "image/jpeg")}
            res = self.client.post("/predict", data=data, files=files)

        self.assertEqual(res.status_code, 200, f"Inference failed with body: {res.text}")
        payload = res.json()

        self.assertEqual(payload.get("status"), "success")
        self.assertIn("request_id", payload)
        self.assertIn("estimation", payload)

        estimation = payload["estimation"]
        final_est = estimation.get("final")
        self.assertIsNotNone(final_est, "Ước lượng final_est không được là None")
        self.assertGreater(final_est, 0, "final_est phải > 0")

        # So sánh với Ground Truth
        abs_err = abs(final_est - self.ground_truth_count)
        pct_err = (abs_err / self.ground_truth_count) * 100.0

        hybrid_f11 = payload.get("debug_info", {}).get("features_vector", {}).get("Estimated_Total_Seeds_Hybrid")

        print("\n" + "=" * 70)
        print("🎯 KẾT QUẢ KIỂM THỬ TÍCH HỢP TRÊN FIXTURE M001a:")
        print(f"   • Ground Truth (Actual_Count): {self.ground_truth_count} hạt")
        print(f"   • Ước lượng của hệ thống      : {final_est} hạt")
        print(f"   • Phương pháp sử dụng         : {estimation.get('method_used')}")
        print(f"   • Hồi quy Extra Trees         : {estimation.get('regression_est')}")
        print(f"   • Hình học (Geometry/AI)      : {estimation.get('geometry_est')}")
        print(f"   • Cân mẫu (Weight)            : {estimation.get('weight_est')}")
        print(f"   • Feature 11 (Trained Hybrid) : {hybrid_f11}")
        print(f"   • Sai số tuyệt đối            : {abs_err} hạt")
        print(f"   • Tỷ lệ sai số (MAPE)         : {pct_err:.2f}%")
        print("=" * 70)

        # Kiểm tra timings
        self.assertIn("timings_ms", payload)
        timings = payload["timings_ms"]
        self.assertGreater(timings.get("total_ms", 0), 0)
        self.assertGreater(timings.get("container_ms", 0), 0)

        # Kiểm tra debug_info
        self.assertIn("debug_info", payload)
        self.assertIn("features_vector", payload["debug_info"])
        self.assertEqual(len(payload["debug_info"]["features_vector"]), 31)


if __name__ == "__main__":
    unittest.main()
