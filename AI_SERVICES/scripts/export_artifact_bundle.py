#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CLI: XUẤT BẢN MODEL BUNDLE RA THƯ MỤC ĐỘC LẬP (ARTIFACT BUNDLE EXPORTER)
===============================================================================
Mục đích:
  - Đóng gói toàn bộ các file trọng số, scaler, manifest thành một bundle
    hoàn chỉnh trong thư mục chỉ định (ví dụ để deploy lên Google Drive / Colab).
  - Tự động tính toán mã băm SHA-256 cho từng file trong bundle và ghi vào manifest.
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent

if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from model_registry import ModelRegistry


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(
    manifest_path: Path,
    output_dir: Path,
    project_root: Path,
) -> bool:
    print("=" * 80)
    print("📦 RICE VISION AI — EXPORT MODEL ARTIFACT BUNDLE")
    print("=" * 80)
    print(f"📄 Nguồn Manifest : {manifest_path}")
    print(f"📁 Thư mục đích   : {output_dir}")
    print("=" * 80)

    if not manifest_path.exists():
        print(f"❌ FAIL: Manifest không tồn tại tại {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data: Dict[str, Any] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_out = output_dir / "models"
    models_out.mkdir(parents=True, exist_ok=True)

    # 1. Copy model
    model_rel = manifest_data.get("model", {}).get("relative_path")
    if not model_rel:
        print("❌ FAIL: Manifest thiếu model.relative_path")
        return False

    model_src = (project_root / model_rel).resolve()
    if not model_src.exists():
        print(f"❌ FAIL: File model nguồn không tồn tại: {model_src}")
        return False

    model_dst = models_out / model_src.name
    shutil.copy2(model_src, model_dst)
    model_hash = compute_sha256(model_dst)
    manifest_data["model"]["sha256"] = model_hash
    manifest_data["model"]["relative_path"] = f"models/{model_src.name}"
    print(f"✅ Đã copy model: {model_src.name} (SHA256: {model_hash[:12]}...)")

    # 2. Copy scaler nếu có
    if manifest_data.get("preprocessing", {}).get("type") == "scaler":
        scaler_rel = manifest_data.get("preprocessing", {}).get("scaler_relative_path")
        if scaler_rel:
            scaler_src = (project_root / scaler_rel).resolve()
            if scaler_src.exists():
                scaler_dst = models_out / scaler_src.name
                shutil.copy2(scaler_src, scaler_dst)
                scaler_hash = compute_sha256(scaler_dst)
                manifest_data["preprocessing"]["scaler_sha256"] = scaler_hash
                manifest_data["preprocessing"]["scaler_relative_path"] = f"models/{scaler_src.name}"
                print(f"✅ Đã copy scaler: {scaler_src.name} (SHA256: {scaler_hash[:12]}...)")

        scaler_params_rel = manifest_data.get("preprocessing", {}).get("scaler_params_relative_path")
        if scaler_params_rel:
            params_src = (project_root / scaler_params_rel).resolve()
            if params_src.exists():
                params_dst = models_out / params_src.name
                shutil.copy2(params_src, params_dst)
                manifest_data["preprocessing"]["scaler_params_relative_path"] = f"models/{params_src.name}"
                print(f"✅ Đã copy scaler params: {params_src.name}")

    # 3. Ghi manifest cập nhật vào output_dir
    manifest_out = output_dir / "manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Đã lưu bundle manifest tại: {manifest_out}")
    print("\n" + "=" * 80)
    print("🎉 EXPORT THÀNH CÔNG! Bundle đã sẵn sàng để triển khai độc lập.")
    print("=" * 80)
    return True


def main():
    default_manifest = AI_SERVICES_DIR / "artifacts" / "manifest.json"
    default_output = AI_SERVICES_DIR / "exported_bundle"

    parser = argparse.ArgumentParser(description="Export model bundle thành gói tự chứa độc lập.")
    parser.add_argument("--manifest", type=str, default=str(default_manifest), help="Đường dẫn manifest.json nguồn")
    parser.add_argument("--output", type=str, default=str(default_output), help="Thư mục xuất bundle")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT), help="Thư mục gốc project")
    args = parser.parse_args()

    success = export_bundle(
        manifest_path=Path(args.manifest).resolve(),
        output_dir=Path(args.output).resolve(),
        project_root=Path(args.project_root).resolve(),
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
