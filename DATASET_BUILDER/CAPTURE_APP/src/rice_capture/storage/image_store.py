from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path


MAX_IMAGE_BYTES = 30 * 1024 * 1024


def validate_jpeg_bytes(payload: bytes) -> tuple[int, int]:
    if not payload:
        raise ValueError("Ảnh tải lên đang trống.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 30 MB.")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise ValueError("Tệp tải lên không phải ảnh hợp lệ.") from exc
    if image_format not in {"JPEG", "JPG"}:
        raise ValueError("Điện thoại phải gửi ảnh ở định dạng JPEG.")
    if width < 320 or height < 240:
        raise ValueError("Độ phân giải ảnh quá nhỏ.")
    return width, height


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".jpg.part", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(payload)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

