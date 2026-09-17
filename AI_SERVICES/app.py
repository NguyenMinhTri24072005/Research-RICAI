#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
🌾 RICE VISION AI — INFERENCE SERVICE APPLICATION LAUNCHER
===============================================================================
Đây là file bootstrap ứng dụng. Toàn bộ logic nghiệp vụ, thuật toán thị giác
và mô hình hồi quy đã được tách vào gói nguồn chuẩn `rice_ai` theo kiến trúc module.

Khởi chạy dịch vụ:
    python app.py
hoặc:
    uvicorn app:app --host 0.0.0.0 --port 8000
===============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Thêm AI_SERVICES/src vào sys.path tại thời điểm bootstrap
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import uvicorn
from rice_ai.api.application import create_app
from rice_ai.settings import Settings

# Khởi tạo Settings tập trung và FastAPI app
settings = Settings()
app = create_app(settings=settings)

# Khởi tạo thông tin MODEL_PATHS tương thích ngược cho launcher và client cũ
try:
    _yolo_p: Optional[Path] = settings.get_yolo_path()
except Exception:
    _yolo_p = None

try:
    _cnn_p: Optional[Path] = settings.get_cnn_path()
except Exception:
    _cnn_p = None

MODEL_PATHS: Dict[str, Optional[Path]] = {
    "yolo": _yolo_p,
    "cnn": _cnn_p,
}

PACKING_FRACTION = settings.packing_fraction_geometry


if __name__ == "__main__":
    port = settings.port
    print(f"🚀 [RICE AI] Khởi động máy chủ Inference tại cổng {port}...")
    uvicorn.run(app, host=settings.host, port=port)