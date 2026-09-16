#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for model_registry.py
"""

import sys
import unittest
from pathlib import Path

# Thêm AI_SERVICES vào sys.path
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.manifest_path = AI_SERVICES_DIR / "artifacts" / "manifest.json"
        self.registry = ModelRegistry(project_root=PROJECT_ROOT, manifest_path=self.manifest_path)

    def test_load_and_verify_bundle(self):
        status = self.registry.load_bundle(force=True)
        self.assertTrue(status.loaded, f"Bundle load failed: {status.error}")
        self.assertTrue(status.verified, f"Bundle verify failed: {status.error}")
        self.assertEqual(status.bundle_id, "rice_vision_extratrees_31v1_20260824")
        self.assertEqual(status.n_features, 31)
        self.assertTrue(status.has_scaler)
        self.assertTrue(self.registry.is_ready())

    def test_get_model_and_scaler(self):
        model, scaler = self.registry.get_model_and_scaler()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(getattr(model, "n_features_in_", None), 31)
        self.assertEqual(getattr(scaler, "n_features_in_", None), 31)

    def test_missing_manifest_handling(self):
        bad_registry = ModelRegistry(
            project_root=PROJECT_ROOT,
            manifest_path=AI_SERVICES_DIR / "artifacts" / "non_existent_manifest.json",
        )
        status = bad_registry.load_bundle(force=True)
        self.assertFalse(status.loaded)
        self.assertFalse(status.verified)
        self.assertIsNotNone(status.error)
        self.assertIn("không tồn tại", status.error.lower())


if __name__ == "__main__":
    unittest.main()
