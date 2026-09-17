"""Image Decoding and Workspace Management."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rice_ai.contracts import PipelineError


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Giải mã mảng byte thành ảnh OpenCV BGR (H, W, 3)."""
    if not image_bytes:
        raise PipelineError(
            status_code=422,
            error_code="INVALID_IMAGE",
            message="Dữ liệu ảnh rỗng.",
        )

    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as ex:
        raise PipelineError(
            status_code=422,
            error_code="INVALID_IMAGE",
            message=f"Lỗi khi giải mã ảnh: {ex}",
        )

    if img is None or img.size == 0 or len(img.shape) != 3:
        raise PipelineError(
            status_code=422,
            error_code="INVALID_IMAGE",
            message="Không thể giải mã tệp ảnh thành ma trận màu hợp lệ. Vui lòng kiểm tra định dạng ảnh (JPEG, PNG).",
        )

    return img


class RequestWorkspace:
    """Quản lý thư mục tạm cho từng request xử lý, đảm bảo tự giải phóng an toàn."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.dir_path: Optional[Path] = None

    def __enter__(self) -> RequestWorkspace:
        if self.debug:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="rice_ai_req_")
            self.dir_path = Path(self._temp_dir.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
