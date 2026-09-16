#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CLI: XÁC THỰC ARTIFACTS TRƯỚC KHI KHỞI CHẠY (PREFLIGHT ARTIFACT VERIFIER)
===============================================================================
Mục đích:
  - Đọc và thẩm định manifest.json của model bundle.
  - Sử dụng ModelRegistry để nạp thử model + scaler và kiểm tra n_features_in_ = 31.
  - Kiểm tra thứ tự và danh sách đặc trưng khớp với feature_schema.
  - Chạy smoke prediction trên vector 31 chiều để bảo đảm kết quả hữu hạn.
  - Trả về mã exit code 0 nếu đạt chuẩn (PASS), khác 0 nếu thất bại (FAIL).
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Đảm bảo import được các module từ AI_SERVICES
CURRENT_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent

if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from feature_schema import ALL_31_FEATURES, FEATURE_SCHEMA_VERSION
from model_registry import ModelRegistry


def verify_artifacts(manifest_path: Path, project_root: Path) -> bool:
    print("=" * 80)
    print("🔍 RICE VISION AI — PREFLIGHT ARTIFACT VERIFICATION")
    print("=" * 80)
    print(f"📁 Project Root    : {project_root}")
    print(f"📄 Manifest Path   : {manifest_path}")
    print("=" * 80)

    if not manifest_path.exists():
        print(f"❌ FAIL: Manifest không tồn tại tại {manifest_path}")
        return False

    registry = ModelRegistry(project_root=project_root, manifest_path=manifest_path)
    status = registry.load_bundle(force=True)

    print(f"\n📦 Bundle ID         : {status.bundle_id}")
    print(f"📐 Schema Version    : {status.schema_version}")
    print(f"🤖 Model Family      : {status.model_family}")
    print(f"⚙️ Preprocessing     : {status.preprocessing_type}")
    print(f"📊 Number of Features: {status.n_features}")
    print(f"⚖️ Scaler Available  : {status.has_scaler}")
    print(f"✅ Loaded Status     : {status.loaded}")
    print(f"🛡️ Verified Status   : {status.verified}")

    if not status.loaded or not status.verified:
        print(f"\n❌ FAIL: Nạp hoặc xác minh bundle thất bại: {status.error}")
        return False

    model, scaler = registry.get_model_and_scaler()
    if model is None:
        print("\n❌ FAIL: Model object is None sau khi load.")
        return False

    import numpy as np

    # 1. Kiểm tra class type
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    print(f"🌲 Model Class       : {model_class} (Family: {type(model).__name__})")
    if type(model).__name__ != "ExtraTreesRegressor":
        print(f"\n❌ FAIL: Model family là {type(model).__name__}, kỳ vọng ExtraTreesRegressor.")
        return False

    if scaler is not None:
        scaler_class = f"{type(scaler).__module__}.{type(scaler).__qualname__}"
        print(f"📏 Scaler Class      : {scaler_class}")
        if type(scaler).__name__ != "StandardScaler":
            print(f"\n❌ FAIL: Scaler class là {type(scaler).__name__}, kỳ vọng StandardScaler.")
            return False

    # 2. Kiểm tra n_features_in_
    n_in = getattr(model, "n_features_in_", None)
    if n_in != 31:
        print(f"\n❌ FAIL: Model có n_features_in_ = {n_in}, kỳ vọng 31.")
        return False
    print(f"🔢 Input Features    : {n_in} / 31 (OK)")

    # 3. Kiểm tra scaler nếu có
    if status.has_scaler:
        if scaler is None:
            print("\n❌ FAIL: Scaler flag bật nhưng scaler object is None.")
            return False
        scaler_n = getattr(scaler, "n_features_in_", None)
        if scaler_n != 31:
            print(f"\n❌ FAIL: Scaler có n_features_in_ = {scaler_n}, kỳ vọng 31.")
            return False
        print(f"⚖️ Scaler Features   : {scaler_n} / 31 (OK)")

    # 4. Kiểm tra feature names nếu model có
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if list(feature_names) != ALL_31_FEATURES:
            print("\n❌ FAIL: feature_names_in_ của model không khớp với ALL_31_FEATURES.")
            return False
        print("📋 Feature Names In  : Khớp chính xác 31 đặc trưng theo thứ tự (OK)")

    # 5. Kiểm tra scaler_params.json cross-validation
    manifest = registry.get_manifest()
    sp_rel = manifest.get("preprocessing", {}).get("scaler_params_relative_path")
    if sp_rel:
        import json
        sp_path = project_root / sp_rel
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                sp_data = json.load(f)
            mean_diff = float(np.max(np.abs(scaler.mean_ - np.array(sp_data["mean"]))))
            scale_diff = float(np.max(np.abs(scaler.scale_ - np.array(sp_data["scale"]))))
            if mean_diff > 1e-9 or scale_diff > 1e-9:
                print(f"\n❌ FAIL: Scaler joblib không khớp scaler_params.json (mean_diff={mean_diff}, scale_diff={scale_diff})")
                return False
            print(f"🔍 Scaler Metadata   : Khớp chính xác 100% với scaler_params.json (diff={max(mean_diff, scale_diff)})")

    # 6. Kiểm tra best_tree_model_info.json cross-validation
    mi_rel = manifest.get("model", {}).get("metadata_path")
    if mi_rel:
        import json
        mi_path = project_root / mi_rel
        if mi_path.exists():
            with open(mi_path, "r", encoding="utf-8") as f:
                mi_data = json.load(f)
            info_importances = {item["name"]: item["importance_score"] for item in mi_data.get("feature_importances", [])}
            actual_importances = dict(zip(ALL_31_FEATURES, model.feature_importances_))
            imp_diffs = [abs(actual_importances[name] - info_importances[name]) for name in ALL_31_FEATURES if name in info_importances]
            max_imp_diff = max(imp_diffs) if imp_diffs else 0.0
            if max_imp_diff > 1e-9:
                print(f"\n❌ FAIL: Model importances không khớp best_tree_model_info.json (diff={max_imp_diff})")
                return False
            print(f"🌳 Model Metadata    : Khớp chính xác 100% với best_tree_model_info.json (diff={max_imp_diff})")

    # 7. Smoke predict: kiểm tra suy luận với vector số 0 và vector chuẩn
    test_vec = np.zeros((1, 31), dtype=np.float64)
    scaled_vec = scaler.transform(test_vec) if scaler is not None else test_vec
    pred = model.predict(scaled_vec)
    if not np.isfinite(pred[0]):
        print(f"\n❌ FAIL: Smoke prediction trả về non-finite ({pred[0]}).")
        return False
    print(f"🚀 Smoke Prediction  : Thành công (pred={pred[0]:.4f}, hữu hạn)")

    print("\n" + "=" * 80)
    print("🎉 PASS: Toàn bộ artifacts và pipeline preprocessing đã được xác minh thành công!")
    print("=" * 80)
    return True


def main():
    default_manifest = AI_SERVICES_DIR / "artifacts" / "manifest.json"
    parser = argparse.ArgumentParser(description="Xác minh artifacts model bundle theo manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(default_manifest),
        help=f"Đường dẫn file manifest.json (mặc định: {default_manifest})",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(PROJECT_ROOT),
        help=f"Thư mục gốc của project (mặc định: {PROJECT_ROOT})",
    )
    args = parser.parse_args()

    manifest_p = Path(args.manifest).resolve()
    root_p = Path(args.project_root).resolve()

    success = verify_artifacts(manifest_path=manifest_p, project_root=root_p)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
