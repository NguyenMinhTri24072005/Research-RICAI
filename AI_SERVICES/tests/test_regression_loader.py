#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for regression_loader.py
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

TEST_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TEST_DIR.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.estimation.feature_schema import ALL_31_FEATURES
from rice_ai.models.regression_loader import (
    LoadedRegression,
    load_regression_folder,
)


class TestRegressionLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_dir = PROJECT_ROOT / "LINEAR_REGRESSION_MODEL" / "models"
        cls.ard_dir = PROJECT_ROOT / "LINEAR_REGRESSION_MODEL" / "models" / "ard"

    def test_load_legacy_adapter(self):
        """Kiểm tra nạp thư mục legacy (best_tree_ensemble_model.joblib + scaler_params.json)."""
        loaded = load_regression_folder(self.legacy_dir)
        self.assertTrue(loaded.is_legacy)
        self.assertEqual(loaded.model_name, "ExtraTrees")
        self.assertEqual(len(loaded.feature_names), 31)
        self.assertIsNotNone(loaded.scaler)
        # Test predict
        pred = loaded.predict([10.0] * 31)
        self.assertIsInstance(pred, float)
        self.assertGreaterEqual(pred, 0.0)

    def test_load_canonical_ard_bundle(self):
        """Kiểm tra nạp thư mục canonical bundle ARD mới hợp nhất."""
        if not self.ard_dir.exists():
            self.skipTest(f"Thư mục ARD {self.ard_dir} không tồn tại")

        loaded = load_regression_folder(self.ard_dir)
        self.assertFalse(loaded.is_legacy)
        self.assertEqual(loaded.model_name, "ARDRegression")
        self.assertEqual(len(loaded.feature_names), 31)
        pred = loaded.predict([10.0] * 31)
        self.assertIsInstance(pred, float)

    def test_synthetic_ridge_and_standard_scaler(self):
        """Kiểm tra bundle hợp lệ với Ridge và StandardScaler."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            X = np.random.randn(20, 31)
            y = np.random.uniform(50, 150, 20)

            scaler = StandardScaler().fit(X)
            model = Ridge().fit(scaler.transform(X), y)

            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            loaded = load_regression_folder(bundle_p)
            self.assertEqual(loaded.model_name, "Ridge")
            pred = loaded.predict([1.0] * 31)
            self.assertIsInstance(pred, float)

    def test_synthetic_decision_tree_and_minmax_scaler(self):
        """Kiểm tra khả năng tương thích với họ scaler khác (MinMaxScaler)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            X = np.random.uniform(1, 100, (20, 31))
            y = np.random.uniform(50, 150, 20)

            scaler = MinMaxScaler().fit(X)
            model = DecisionTreeRegressor(random_state=42).fit(scaler.transform(X), y)

            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            loaded = load_regression_folder(bundle_p)
            self.assertEqual(loaded.model_name, "DecisionTreeRegressor")
            pred = loaded.predict([10.0] * 31)
            self.assertIsInstance(pred, float)

    def test_explicit_no_scaler_bundle(self):
        """Kiểm tra mô hình có khai báo tường minh 'preprocessing': 'none'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            X = np.random.randn(20, 31)
            y = np.random.uniform(50, 150, 20)
            model = Ridge().fit(X, y)

            joblib.dump(model, bundle_p / "model.joblib")
            with open(bundle_p / "config.json", "w", encoding="utf-8") as f:
                json.dump({"preprocessing": "none"}, f)
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            loaded = load_regression_folder(bundle_p)
            self.assertIsNone(loaded.scaler)
            pred = loaded.predict([1.0] * 31)
            self.assertIsInstance(pred, float)

    def test_missing_model_file_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            # Only scaler and schema, no model.joblib
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)
            with self.assertRaises(FileNotFoundError):
                load_regression_folder(bundle_p)

    def test_missing_scaler_without_explicit_none_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            X = np.random.randn(10, 31)
            y = np.random.uniform(10, 50, 10)
            joblib.dump(Ridge().fit(X, y), bundle_p / "model.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            # Thiếu scaler.joblib và config không có 'none' -> raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_regression_folder(bundle_p)

    def test_pipeline_in_model_file_rejected(self):
        """Bảo đảm từ chối nếu model.joblib là một sklearn Pipeline (chống double preprocessing)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            X = np.random.randn(10, 31)
            y = np.random.uniform(10, 50, 10)
            pipe.fit(X, y)

            joblib.dump(pipe, bundle_p / "model.joblib")
            joblib.dump(StandardScaler(), bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertIn("Pipeline", str(ctx.exception))

    def test_schema_feature_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            joblib.dump(Ridge(), bundle_p / "model.joblib")
            joblib.dump(StandardScaler(), bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ["feat1", "feat2"]}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertIn("31", str(ctx.exception))

    def test_schema_reordered_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            joblib.dump(Ridge(), bundle_p / "model.joblib")
            joblib.dump(StandardScaler(), bundle_p / "scaler.joblib")
            reordered = list(ALL_31_FEATURES)
            reordered[0], reordered[1] = reordered[1], reordered[0]  # Swap 2 features
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": reordered}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertIn("ALL_31_FEATURES", str(ctx.exception))

    def test_folder_relocation_parity(self):
        """Sao chép bundle sang thư mục mới và xác nhận kết quả dự đoán đồng nhất tuyệt đối."""
        if not self.ard_dir.exists():
            self.skipTest(f"Thư mục ARD {self.ard_dir} không tồn tại")

        with tempfile.TemporaryDirectory() as tmp_dir:
            relocated_dir = Path(tmp_dir) / "relocated_ard"
            shutil.copytree(self.ard_dir, relocated_dir)

            loaded_orig = load_regression_folder(self.ard_dir)
            loaded_reloc = load_regression_folder(relocated_dir)

            test_vec = [15.0] * 31
            pred_orig = loaded_orig.predict(test_vec)
            pred_reloc = loaded_reloc.predict(test_vec)

            self.assertAlmostEqual(pred_orig, pred_reloc, places=7)

    def test_wrong_schema_version_rejected(self):
        """Bundle có schema_version khác '31v1' phải bị từ chối."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            model = Ridge().fit(np.zeros((5, 31)), np.zeros(5))
            scaler = StandardScaler().fit(np.zeros((5, 31)))
            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "30v1", "features": ALL_31_FEATURES}, f)

            with self.assertRaises((ValueError, KeyError)) as ctx:
                load_regression_folder(bundle_p)
            self.assertIn("version", str(ctx.exception).lower())

    def test_missing_schema_version_rejected(self):
        """Bundle thiếu schema_version phải bị từ chối (không ngầm gán default)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            model = Ridge().fit(np.zeros((5, 31)), np.zeros(5))
            scaler = StandardScaler().fit(np.zeros((5, 31)))
            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"features": ALL_31_FEATURES}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertIn("version", str(ctx.exception).lower())

    def test_reversed_object_feature_names_rejected(self):
        """Mô hình hoặc scaler có feature_names_in_ đảo thứ tự hoặc không khớp ALL_31_FEATURES phải bị từ chối."""
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            rev_cols = list(reversed(ALL_31_FEATURES))
            df = pd.DataFrame(np.zeros((5, 31)), columns=rev_cols)
            model = Ridge().fit(df, np.zeros(5))
            scaler = StandardScaler().fit(df)
            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertTrue(
                "feature_names_in_" in str(ctx.exception) or "thứ tự" in str(ctx.exception).lower() or "khớp" in str(ctx.exception).lower()
            )

    def test_multi_output_model_rejected(self):
        """Mô hình có nhiều đầu ra (multi-output) phải bị từ chối, không được lấy [0][0] ngầm."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            # fit với 2 targets -> multi-output shape (n_samples, 2)
            y_multi = np.zeros((5, 2))
            model = Ridge().fit(np.zeros((5, 31)), y_multi)
            scaler = StandardScaler().fit(np.zeros((5, 31)))
            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertTrue(
                "multi-output" in str(ctx.exception).lower() or "scalar" in str(ctx.exception).lower() or "shape" in str(ctx.exception).lower()
            )

    def test_preprocessing_none_conflicts_with_scaler_file(self):
        """Nếu config khai báo 'preprocessing': 'none' nhưng scaler.joblib vẫn tồn tại -> từ chối ambiguous bundle."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            model = Ridge().fit(np.zeros((5, 31)), np.zeros(5))
            scaler = StandardScaler().fit(np.zeros((5, 31)))
            joblib.dump(model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)
            with open(bundle_p / "config.json", "w", encoding="utf-8") as f:
                json.dump({"preprocessing": "none"}, f)

            with self.assertRaises(ValueError) as ctx:
                load_regression_folder(bundle_p)
            self.assertTrue(
                "ambiguous" in str(ctx.exception).lower() or "scaler" in str(ctx.exception).lower()
            )

    def test_unfitted_model_or_scaler_rejected(self):
        """Mô hình hoặc scaler chưa được fit phải bị từ chối."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_p = Path(tmp_dir)
            unfitted_model = Ridge()  # Not fitted!
            scaler = StandardScaler().fit(np.zeros((5, 31)))
            joblib.dump(unfitted_model, bundle_p / "model.joblib")
            joblib.dump(scaler, bundle_p / "scaler.joblib")
            with open(bundle_p / "feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "31v1", "features": ALL_31_FEATURES}, f)

            with self.assertRaises((ValueError, Exception)) as ctx:
                load_regression_folder(bundle_p)
            self.assertTrue(
                "fit" in str(ctx.exception).lower()
            )


if __name__ == "__main__":
    unittest.main()

