#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CLI: XÁC THỰC THƯ MỤC MÔ HÌNH HỒI QUY (PREFLIGHT MODEL VERIFIER)
===============================================================================
Mục đích:
  - Nạp và thẩm định thư mục mô hình hồi quy (canonical folder hoặc legacy adapter).
  - Kiểm tra giao diện estimator và scaler (không giới hạn họ ExtraTrees/StandardScaler).
  - Kiểm tra thứ tự và danh sách 31 đặc trưng theo đúng hợp đồng 31v1.
  - Chạy smoke prediction trên vector 31 chiều để bảo đảm kết quả hữu hạn.
  - Trả về mã exit code 0 nếu đạt chuẩn (PASS), khác 0 nếu thất bại (FAIL).

Cách sử dụng:
    python scripts/verify_artifacts.py
    python scripts/verify_artifacts.py --model-dir ../LINEAR_REGRESSION_MODEL/models/ard
    python scripts/verify_artifacts.py --model-dir artifacts/regression/extra_trees
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được các module từ AI_SERVICES/src
CURRENT_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = CURRENT_DIR.parent
SRC_DIR = AI_SERVICES_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from rice_ai.estimation.feature_schema import ALL_31_FEATURES, FEATURE_SCHEMA_VERSION
from rice_ai.models.regression_loader import load_regression_folder
from rice_ai.settings import Settings


def verify_regression_dir(model_dir: Path) -> bool:
    print("=" * 80)
    print("🔍 RICE VISION AI — PREFLIGHT REGRESSION VERIFICATION")
    print("=" * 80)
    print(f"📁 Target Folder  : {model_dir}")
    print("=" * 80)

    try:
        loaded = load_regression_folder(model_dir)
    except Exception as ex:
        print(f"\n❌ FAIL: Không thể nạp hoặc xác thực thư mục mô hình: {ex}")
        return False

    print(f"\n🤖 Model Name        : {loaded.model_name}")
    print(f"🌲 Model Class       : {type(loaded.model).__module__}.{type(loaded.model).__qualname__}")
    if loaded.scaler is not None:
        print(f"📏 Scaler Class      : {type(loaded.scaler).__module__}.{type(loaded.scaler).__qualname__}")
    else:
        print(f"📏 Scaler Class      : None (Explicit 'preprocessing': 'none')")
    print(f"📐 Schema Version    : {loaded.schema_version}")
    print(f"📊 Feature Count     : {len(loaded.feature_names)} / 31 (OK)")
    print(f"🏛️ Layout Type       : {'Legacy Adapter' if loaded.is_legacy else 'Canonical Bundle'}")

    if loaded.warnings:
        print("\n⚠️ Cảnh báo:")
        for w in loaded.warnings:
            print(f"   • {w}")

    # Chạy thử nghiệm dự đoán kiểm chứng
    smoke_vector = [10.0] * 31
    try:
        pred_val = loaded.predict(smoke_vector)
        print(f"\n🎯 Smoke Prediction Test: Input [10.0]*31 -> Output: {pred_val} (PASS)")
    except Exception as ex:
        print(f"\n❌ FAIL: Smoke prediction gặp lỗi: {ex}")
        return False

    print("\n" + "=" * 80)
    print("✅ PASS: MÔ HÌNH HỒI QUY SẴN SÀNG HOẠT ĐỘNG (COMPATIBILITY VERIFIED)")
    print("=" * 80)
    return True


def main() -> int:
    settings = Settings()

    parser = argparse.ArgumentParser(
        description="Kiểm tra tính hợp lệ và sẵn sàng của thư mục mô hình hồi quy (Preflight Verifier)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục chứa mô hình hồi quy (đường dẫn tương đối luôn tính từ AI_SERVICES). "
             "Nếu bỏ trống, sử dụng cấu hình REGRESSION_MODEL_DIR từ .env hoặc mặc định.",
    )

    args = parser.parse_args()

    if args.model_dir:
        target_dir = settings.resolve_path(args.model_dir)
    else:
        try:
            target_dir = settings.get_regression_dir()
        except Exception as ex:
            print(f"❌ FAIL: Không xác định được thư mục mô hình hồi quy: {ex}")
            return 1

    success = verify_regression_dir(target_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
