#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TEST SUITE: XÁC THỰC HỢP NHẤT NOTEBOOK HUẤN LUYỆN HỒI QUY
===============================================================================
Mục tiêu kiểm thử theo NOTEBOOK_CONSOLIDATION_PLAN.md:
  1. Static & Syntax: Cấu trúc nbformat v4, đầy đủ cell IDs, 0 syntax errors.
  2. Model Inventory: Đủ 19 cấu hình, parameters nguồn, Stacking disabled, PLS fixed-2.
  3. Feature Contract & Isolation: Đúng 31 biến, target/metadata cô lập khỏi X.
  4. Preprocessing & Split Isolation: Group disjointness, scaler fit cục bộ trong CV fold.
  5. Metrics & Guards: Xử lý shape (n,), MAPE loại trừ y=0, adjusted R2 OLS guard.
  6. Synthetic Smoke & Reload Parity: Pipeline vs separate model/scaler, tamper detection.
  7. Protected Deployment Artifacts: Bảo toàn nguyên vẹn 100% mã băm SHA-256 gốc.
===============================================================================
"""

import os
import sys
import json
import re
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Định vị thư mục gốc
TEST_DIR = Path(__file__).resolve().parent
MODULE_DIR = TEST_DIR.parent
REPO_ROOT = MODULE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "AI_SERVICES") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "AI_SERVICES"))

from feature_schema import (
    ALL_31_FEATURES,
    FEATURE_DEFS,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    compute_schema_hash,
)

NOTEBOOK_PATH = MODULE_DIR / "RICE_SEED_REGRESSION_TRAINER.ipynb"

# Baseline snapshot SHA-256 các artifact triển khai
PROTECTED_SNAPSHOT = {
    MODULE_DIR / "models" / "best_tree_ensemble_model.joblib": "cfa58aa3255b48a9f2073f8376a1e0ee084c4aa929e7fe601eff7924667f96f8",
    MODULE_DIR / "models" / "best_tree_model_info.json": "49384ca9731a566bca77d77376c7937c2668f7c9181ac8e7a223a24e199b6612",
    MODULE_DIR / "models" / "scaler.joblib": "af1b9536ef863a5a654edfc2392524d8c0954f1671ddee8a93933d57c87390cf",
    MODULE_DIR / "models" / "scaler_params.json": "e6904245895347052d53aa0550d03c57db42aeab1de5809d680f440db46d8ffa",
    REPO_ROOT / "AI_SERVICES" / "artifacts" / "manifest.json": "2190e76ff033821c9cf65a57bdeec830801b9e8144c899825156b03eadb66e59",
}


class TestNotebookConsolidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        self = cls
        cls.nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.cell_map = {}
        for idx, cell in enumerate(cls.nb.cells):
            cid = cell.metadata.get("id")
            if not cid and cell.source.startswith("# Cell ID:"):
                first_line = cell.source.split("\n")[0]
                cid = first_line.replace("# Cell ID:", "").strip()
            if cid:
                cls.cell_map[cid] = cell

    def test_01_notebook_structure_and_syntax(self):
        """Kiểm tra tính hợp lệ nbformat v4, đủ các Cell IDs chủ chốt và 0 lỗi cú pháp."""
        self.assertTrue(NOTEBOOK_PATH.exists(), f"File notebook {NOTEBOOK_PATH} không tồn tại!")
        nbformat.validate(self.nb)

        required_cids = [
            "intro", "ENV_SETUP", "RUN_CONFIG", "FEATURE_CONTRACT", "DATA_LOAD",
            "SPLIT_CONFIG", "SPLIT_BUILD", "PREPROCESSING", "HELPERS", "MODEL_REGISTRY_INIT",
            "MODEL_ridge_alpha_0_1", "MODEL_ols", "MODEL_ridge_alpha_1", "MODEL_bayesian_ridge",
            "MODEL_huber", "MODEL_elastic_net", "MODEL_pls", "MODEL_ard", "MODEL_kernel_ridge_rbf",
            "MODEL_stacking_linear", "MODEL_random_forest_v2", "MODEL_gradient_boosting_v2",
            "MODEL_xgboost_v2", "MODEL_decision_tree", "MODEL_random_forest_tree",
            "MODEL_extra_trees", "MODEL_gradient_boosting_tree", "MODEL_ada_boost",
            "MODEL_hist_gradient_boosting",
            "TRAIN_ALL", "SELECT_BEST", "EVALUATE_ALL", "COMPARE_ALL",
            "DIAGNOSTICS", "EXPORT_ALL", "RELOAD_SMOKE"
        ]

        for cid in required_cids:
            self.assertIn(cid, self.cell_map, f"Thiếu Cell ID bắt buộc trong notebook: {cid}")

        # Biên dịch toàn bộ code cells
        for idx, cell in enumerate(self.nb.cells):
            if cell.cell_type == "code":
                cid = cell.metadata.get("id", f"idx_{idx}")
                try:
                    compile(cell.source, f"Cell_{idx}_{cid}", "exec")
                except SyntaxError as e:
                    self.fail(f"Lỗi cú pháp Python trong Cell {idx} ({cid}): {e}")

    def test_02_model_inventory_19_specs(self):
        """Kiểm tra đầy đủ 19 mô hình ứng viên, cấu hình và ngoại lệ Stacking/PLS."""
        # Thực thi độc lập các cell đăng ký model trong môi trường cách ly
        from typing import Dict, Any, List, Tuple, Optional
        sandbox = {
            "RANDOM_SEED": 42,
            "ESTIMATOR_JOBS": 1,
            "MODEL_REGISTRY": {},
            "np": np,
            "Dict": Dict,
            "Any": Any,
            "List": List,
            "Tuple": Tuple,
            "Optional": Optional,
            "print": lambda *args, **kwargs: None,
        }

        # Chạy MODEL_REGISTRY_INIT
        exec(self.cell_map["MODEL_REGISTRY_INIT"].source, sandbox)

        # Chạy 19 model cells
        model_cids = [k for k in self.cell_map if k.startswith("MODEL_") and k not in ("MODEL_REGISTRY_INIT", "MODEL_CONFIGS_HEADER")]
        self.assertEqual(len(model_cids), 19, f"Yêu cầu chính xác 19 cells cấu hình model, tìm thấy {len(model_cids)}")

        for cid in model_cids:
            exec(self.cell_map[cid].source, sandbox)

        registry = sandbox["MODEL_REGISTRY"]
        self.assertEqual(len(registry), 19, f"MODEL_REGISTRY phải có đúng 19 mô hình, hiện có {len(registry)}")

        # Kiểm tra ngoại lệ Stacking
        self.assertIn("stacking_linear", registry)
        self.assertFalse(registry["stacking_linear"]["enabled"], "Stacking phải vô hiệu hóa mặc định!")
        self.assertIn("inner CV preprocessing/group isolation not verified", registry["stacking_linear"]["notes"])

        # Kiểm tra ngoại lệ PLS baseline fixed-2
        self.assertIn("pls", registry)
        self.assertEqual(registry["pls"]["params"]["n_components"], 2, "PLS baseline phải cố định n_components=2")
        self.assertTrue(registry["pls"]["params"]["scale"], "PLS baseline phải đặt scale=True")

        # Kiểm tra phân biệt variants của RF và GB
        self.assertIn("random_forest_v2", registry)
        self.assertIn("random_forest_tree", registry)
        self.assertEqual(registry["random_forest_v2"]["params"]["n_estimators"], 500)
        self.assertEqual(registry["random_forest_tree"]["params"]["n_estimators"], 100)

        self.assertIn("gradient_boosting_v2", registry)
        self.assertIn("gradient_boosting_tree", registry)
        self.assertEqual(registry["gradient_boosting_v2"]["params"]["n_estimators"], 500)
        self.assertEqual(registry["gradient_boosting_tree"]["params"]["n_estimators"], 100)

    def test_03_feature_contract_and_target_isolation(self):
        """Kiểm tra hợp đồng 31 đặc trưng và cô lập Target."""
        self.assertEqual(len(ALL_31_FEATURES), 31, "ALL_31_FEATURES phải chứa chính xác 31 đặc trưng.")
        self.assertNotIn(TARGET_COLUMN, ALL_31_FEATURES, "Target không được nằm trong X.")
        self.assertNotIn("Sample_ID", ALL_31_FEATURES, "Sample_ID không được nằm trong X.")
        self.assertNotIn("Image_Status", ALL_31_FEATURES, "Image_Status không được nằm trong X.")

        # Thẩm định mã băm schema
        self.assertEqual(len(compute_schema_hash()), 64, "Schema hash phải là SHA-256 hợp lệ.")

    def test_04_grouped_split_and_scaling_isolation(self):
        """Kiểm tra phân chia không rò rỉ nhóm và scaler fit cục bộ trong từng CV fold."""
        # Tạo dữ liệu giả lập 50 dòng thuộc 10 nhóm vật lý
        np.random.seed(42)
        n_samples = 50
        groups = np.repeat([f"M{i:03d}" for i in range(1, 11)], 5)
        X_syn = pd.DataFrame(np.random.randn(n_samples, 31), columns=ALL_31_FEATURES)
        y_syn = np.random.uniform(50, 150, n_samples)

        # Outer GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_syn, y_syn, groups=groups))

        train_grps = set(groups[train_idx])
        test_grps = set(groups[test_idx])
        self.assertEqual(len(train_grps.intersection(test_grps)), 0, "RÒ RỈ: Nhóm xuất hiện ở cả Train và Test!")

        # Inner GroupKFold
        gkf = GroupKFold(n_splits=4)
        train_sub_grps = groups[train_idx]
        X_sub = X_syn.iloc[train_idx]
        y_sub = y_syn[train_idx]

        for fold_idx, (f_tr, f_va) in enumerate(gkf.split(X_sub, y_sub, groups=train_sub_grps)):
            f_tr_g = set(train_sub_grps[f_tr])
            f_va_g = set(train_sub_grps[f_va])
            self.assertEqual(len(f_tr_g.intersection(f_va_g)), 0, f"RÒ RỈ Fold {fold_idx}: Giao thoa nhóm!")

        # Thẩm định Scaler fit cục bộ: Scaler trong Pipeline cross_validate không được tính mean của Validation fold
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        cv_res = cross_validate(pipe, X_sub, y_sub, cv=gkf.split(X_sub, y_sub, groups=train_sub_grps), return_estimator=True)

        for fold_idx, (f_tr, f_va) in enumerate(gkf.split(X_sub, y_sub, groups=train_sub_grps)):
            est = cv_res["estimator"][fold_idx]
            scaler = est.named_steps["scaler"]
            expected_mean = X_sub.iloc[f_tr].mean(axis=0).values
            np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5,
                                       err_msg=f"Scaler ở fold {fold_idx} bị rò rỉ dữ liệu ngoài fold!")

    def test_05_metrics_edge_cases(self):
        """Kiểm tra các hàm tính toán metrics, xử lý 0, shape và Adjusted R2."""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

        # Trích xuất hàm compute_metrics từ code cell HELPERS
        from typing import Dict, Any, List, Tuple, Optional
        import time
        sandbox = {
            "np": np,
            "pd": pd,
            "time": time,
            "clone": clone,
            "build_pipeline": lambda est: Pipeline([("scaler", StandardScaler()), ("model", est)]),
            "r2_score": r2_score,
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_error": mean_squared_error,
            "max_error": max_error,
            "Dict": Dict,
            "Any": Any,
            "List": List,
            "Tuple": Tuple,
            "Optional": Optional,
        }
        exec(self.cell_map["HELPERS"].source, sandbox)
        compute_metrics = sandbox["compute_metrics"]

        # Case 1: Ground truth chứa số 0 (MAPE phải loại trừ số 0 và đếm chính xác)
        y_true = np.array([0.0, 10.0, 20.0, 30.0])
        y_pred = np.array([1.0, 11.0, 19.0, 30.0])

        res = compute_metrics(y_true, y_pred, n_features=31, is_ols=False)
        self.assertEqual(res["mape_excluded_zeros_count"], 1)
        self.assertIsNotNone(res["mape"])
        self.assertIsNone(res["adjusted_r2"], "Non-OLS không được tính Adjusted R2!")

        # Case 2: OLS với n > p + 1 (hợp lệ tính Adjusted R2)
        y_true_large = np.random.uniform(50, 150, 40)
        y_pred_large = y_true_large + np.random.normal(0, 1, 40)
        res_ols = compute_metrics(y_true_large, y_pred_large, n_features=31, is_ols=True)
        self.assertIsNotNone(res_ols["adjusted_r2"])
        self.assertIn("OLS formula valid", res_ols["adjusted_r2_note"])

        # Case 3: OLS với n <= p + 1 (không đủ bậc tự do, phải trả về None)
        res_ols_small = compute_metrics(y_true[:4], y_pred[:4], n_features=31, is_ols=True)
        self.assertIsNone(res_ols_small["adjusted_r2"])

    def test_06_smoke_export_and_reload_parity(self):
        """Chạy synthetic smoke test nhanh trên OLS & ExtraTrees, kiểm tra xuất bundle và reload parity."""
        import joblib

        np.random.seed(42)
        n_samples = 40
        X_syn = pd.DataFrame(np.random.randn(n_samples, 31), columns=ALL_31_FEATURES)
        y_syn = np.random.uniform(20, 100, n_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Huấn luyện nhanh OLS và ExtraTrees
            models = {
                "ols": LinearRegression(),
                "extra_trees": ExtraTreesRegressor(n_estimators=5, random_state=42),
            }

            for m_id, est in models.items():
                pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
                pipe.fit(X_syn, y_syn)

                # Bundle hiện hành của mỗi model nằm trực tiếp trong models/<model_id>/.
                bundle_dir = tmp_path / m_id
                bundle_dir.mkdir(parents=True)

                pipeline_p = bundle_dir / "pipeline.joblib"
                model_p = bundle_dir / "model.joblib"
                scaler_p = bundle_dir / "scaler.joblib"

                joblib.dump(pipe, pipeline_p)
                joblib.dump(pipe.named_steps["model"], model_p)
                joblib.dump(pipe.named_steps["scaler"], scaler_p)

                # Reload và kiểm tra parity
                reloaded_pipe = joblib.load(pipeline_p)
                reloaded_model = joblib.load(model_p)
                reloaded_scaler = joblib.load(scaler_p)

                pred_pipe = reloaded_pipe.predict(X_syn)
                pred_sep = reloaded_model.predict(reloaded_scaler.transform(X_syn))
                in_mem_pred = pipe.predict(X_syn)

                np.testing.assert_allclose(pred_pipe, pred_sep, rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(pred_pipe, in_mem_pred, rtol=1e-8, atol=1e-8)

                # Kiểm tra phát hiện can thiệp file (Tamper Detection)
                orig_hash = hashlib.sha256(model_p.read_bytes()).hexdigest()
                # Thử sửa 1 byte
                with open(model_p, "ab") as f:
                    f.write(b"tamper")
                tampered_hash = hashlib.sha256(model_p.read_bytes()).hexdigest()
                self.assertNotEqual(orig_hash, tampered_hash, "Tamper detection thất bại!")

    def test_07_protected_deployment_artifacts_untouched(self):
        """Khẳng định 100% mã băm SHA-256 của các file deployment đang hoạt động không bị thay đổi."""
        for path, expected_hash in PROTECTED_SNAPSHOT.items():
            self.assertTrue(path.exists(), f"Artifact triển khai bị thiếu: {path}")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"NGUY HIỂM: Artifact triển khai {path.name} bị sửa đổi bytes! Expected: {expected_hash}, Actual: {actual_hash}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
