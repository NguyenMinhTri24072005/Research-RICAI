#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Settings and path resolution in rice_ai.settings
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Thêm AI_SERVICES/src vào sys.path
TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.settings import Settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_paths_exist(self):
        settings = Settings()
        # Regression default must point to LINEAR_REGRESSION_MODEL/models
        reg_dir = settings.get_regression_dir()
        self.assertTrue(reg_dir.exists(), f"Default regression dir {reg_dir} must exist")
        self.assertTrue(reg_dir.is_dir())

        # YOLO default
        yolo_p = settings.get_yolo_path()
        self.assertTrue(yolo_p.exists(), f"Default YOLO path {yolo_p} must exist")

        # CNN default
        cnn_p = settings.get_cnn_path()
        self.assertTrue(cnn_p.exists(), f"Default CNN path {cnn_p} must exist")

    def test_relative_path_resolved_from_ai_services_root(self):
        settings = Settings()
        # Regardless of cwd, a relative path must resolve from AI_SERVICES
        rel_path = "artifacts/regression"
        resolved = settings.resolve_path(rel_path)
        expected = (settings.ai_services_root / rel_path).resolve()
        self.assertEqual(resolved, expected)

    def test_explicit_yolo_path_missing_raises_error_no_fallback(self):
        os.environ["YOLO_MODEL_PATH"] = "artifacts/yolo/non_existent_model/best.pt"
        settings = Settings()
        with self.assertRaises(FileNotFoundError) as ctx:
            settings.get_yolo_path()
        self.assertIn("non_existent_model", str(ctx.exception))

    def test_explicit_cnn_path_missing_raises_error_no_fallback(self):
        os.environ["CNN_MODEL_PATH"] = "artifacts/cnn/non_existent_cnn/best.keras"
        settings = Settings()
        with self.assertRaises(FileNotFoundError) as ctx:
            settings.get_cnn_path()
        self.assertIn("non_existent_cnn", str(ctx.exception))

    def test_explicit_regression_dir_missing_raises_error_no_fallback(self):
        os.environ["REGRESSION_MODEL_DIR"] = "artifacts/regression/missing_folder"
        settings = Settings()
        with self.assertRaises(FileNotFoundError) as ctx:
            settings.get_regression_dir()
        self.assertIn("missing_folder", str(ctx.exception))

    def test_precedence_process_env_over_env_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_env = Path(tmp_dir) / ".env"
            tmp_env.write_text("YOLO_MODEL_PATH=from_file.pt\nPORT_AI=9000\n", encoding="utf-8")

            # Remove PORT_AI from os.environ so it is taken from tmp_env
            os.environ.pop("PORT_AI", None)
            # Process env overrides file
            os.environ["YOLO_MODEL_PATH"] = "from_process.pt"
            settings = Settings(env_file=tmp_env)
            self.assertEqual(settings.yolo_model_path_str, "from_process.pt")
            self.assertEqual(settings.port_ai, 9000)

    def test_unicode_and_spaces_in_path(self):
        settings = Settings()
        # Main repo path has Vietnamese with spaces & accents ("NGHIÊN CỨU KHOA HỌC")
        self.assertTrue(settings.project_root.exists())
        resolved = settings.resolve_path("artifacts/README.md")
        self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
