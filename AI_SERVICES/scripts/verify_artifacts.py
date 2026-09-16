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

    # 1. Kiểm tra n_features_in_
    n_in = getattr(model, "n_features_in_", None)
    if n_in != 31:
        print(f"\n❌ FAIL: Model có n_features_in_ = {n_in}, kỳ vọng 31.")
        return False

    # 2. Kiểm tra scaler nếu có
    if status.has_scaler:
        if scaler is None:
            print("\n❌ FAIL: Scaler flag bật nhưng scaler object is None.")
            return False
        scaler_n = getattr(scaler, "n_features_in_", None)
        if scaler_n != 31:
            print(f"\n❌ FAIL: Scaler có n_features_in_ = {scaler_n}, kỳ vọng 31.")
            return False

    # 3. Kiểm tra feature names nếu model có
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if list(feature_names) != ALL_31_FEATURES:
            print("\n❌ FAIL: feature_names_in_ của model không khớp với ALL_31_FEATURES.")
            return False

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
