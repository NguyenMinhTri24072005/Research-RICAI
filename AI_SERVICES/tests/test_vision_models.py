#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for vision models device resolution and provider state.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.settings import Settings


class TestVisionModelsDevice(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_yolo_device_auto_cpu_when_no_cuda(self):
        """Khi YOLO_DEVICE=auto và không có CUDA, thiết bị giải quyết phải là 'cpu'."""
        os.environ.pop("YOLO_DEVICE", None)
        with patch("torch.cuda.is_available", return_value=False):
            settings = Settings()
            self.assertEqual(settings.get_yolo_device(), "cpu")

    def test_yolo_device_auto_cuda_when_available(self):
        """Khi YOLO_DEVICE=auto và có CUDA, thiết bị giải quyết phải là 'cuda:0'."""
        os.environ.pop("YOLO_DEVICE", None)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            settings = Settings()
            self.assertEqual(settings.get_yolo_device(), "cuda:0")

    def test_yolo_device_explicit_cuda_fails_when_unavailable(self):
        """Khi cấu hình tường minh YOLO_DEVICE=cuda mà hệ thống không có GPU -> phải báo lỗi rõ ràng, không ngầm fallback CPU."""
        os.environ["YOLO_DEVICE"] = "cuda"
        with patch("torch.cuda.is_available", return_value=False):
            settings = Settings()
            with self.assertRaises(RuntimeError) as ctx:
                settings.get_yolo_device()
            self.assertIn("CUDA", str(ctx.exception))

    def test_vision_provider_uses_resolved_device(self):
        """VisionModelProvider khi nạp AutoDetectionModel phải truyền đúng device từ Settings."""
        os.environ["YOLO_DEVICE"] = "cpu"
        settings = Settings()
        provider = VisionModelProvider(settings)

        mock_auto_det = MagicMock()
        with patch("sahi.AutoDetectionModel.from_pretrained", return_value=mock_auto_det) as mock_load:
            provider.get_yolo_model()
            self.assertTrue(mock_load.called)
            kwargs = mock_load.call_args[1]
            self.assertEqual(kwargs.get("device"), "cpu")

    def test_vision_provider_tracks_load_errors_without_crashing_health(self):
        """Nếu weights bị hỏng hoặc load thất bại, VisionModelProvider phải ghi nhận error state chứ không crash."""
        settings = Settings()
        provider = VisionModelProvider(settings)
        with patch("sahi.AutoDetectionModel.from_pretrained", side_effect=RuntimeError("Corrupt weights")):
            with self.assertRaises(RuntimeError):
                provider.get_yolo_model()
            # Trạng thái provider phải phản ánh lỗi
            status = provider.get_status()
            self.assertEqual(status.get("yolo", {}).get("status"), "error")
            self.assertIn("Corrupt weights", status.get("yolo", {}).get("error", ""))


if __name__ == "__main__":
    unittest.main()
