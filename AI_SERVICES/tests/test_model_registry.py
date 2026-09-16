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

    def test_model_sha256_mismatch_fails_verification(self):
        import json, tempfile
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["model"]["sha256"] = "0" * 64
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
            json.dump(manifest_data, tmp)
            tmp_path = Path(tmp.name)
        try:
            reg = ModelRegistry(project_root=PROJECT_ROOT, manifest_path=tmp_path)
            status = reg.load_bundle(force=True)
            self.assertFalse(status.verified)
            self.assertIn("Model SHA-256 mismatch", str(status.error))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_schema_hash_mismatch_fails_verification(self):
        import json, tempfile
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["schema_hash"] = "f" * 64
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
            json.dump(manifest_data, tmp)
            tmp_path = Path(tmp.name)
        try:
            reg = ModelRegistry(project_root=PROJECT_ROOT, manifest_path=tmp_path)
            status = reg.load_bundle(force=True)
            self.assertFalse(status.verified)
            self.assertIn("Schema hash mismatch", str(status.error))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_model_class_mismatch_fails_verification(self):
        import json, tempfile
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["model"]["expected_class"] = "sklearn.linear_model.LinearRegression"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
            json.dump(manifest_data, tmp)
            tmp_path = Path(tmp.name)
        try:
            reg = ModelRegistry(project_root=PROJECT_ROOT, manifest_path=tmp_path)
            status = reg.load_bundle(force=True)
            self.assertFalse(status.verified)
            self.assertIn("Model class mismatch", str(status.error))
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
