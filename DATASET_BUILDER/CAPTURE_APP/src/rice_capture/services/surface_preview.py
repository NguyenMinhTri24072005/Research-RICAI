"""Read-only preview: never allocates sample IDs or writes raw images."""
import base64
import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps

from rice_capture.storage.image_store import validate_jpeg_bytes
from rice_capture.vision.container_detector import detect_container_and_scale

GEOMETRY_FIELDS = ('Inner_Diameter_mm', 'Container_Height_mm', 'Empty_Height_mm')


def preview_frame(frame, values):
    geometry = []
    for key in GEOMETRY_FIELDS:
        try:
            value = float(str(values.get(key, '')).strip().replace(',', '.'))
        except ValueError as exc:
            raise ValueError(f'Vui lòng nhập thông số hợp lệ: {key}') from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Thông số không hợp lệ: {key}')
        geometry.append(value)
    diam, height, empty = geometry
    if diam <= 0 or height <= 0 or empty > height:
        raise ValueError('Đường kính và chiều cao phải > 0; khoảng trống không được lớn hơn chiều cao ly.')
    result = detect_container_and_scale(
        frame, diam, height, empty, surface_method='adaptive', on_failure='dict')
    if result['status'] != 'ok':
        return {'status': 'failed', 'reason': result['reason']}

    def encode(image):
        h, w = image.shape[:2]
        if max(h, w) > 1000:
            image = cv2.resize(image, (round(w*1000/max(h,w)), round(h*1000/max(h,w))),
                               interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode('.jpg', image)
        if not ok:
            raise ValueError('Không thể tạo ảnh xem trước.')
        return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')

    return {'status': 'ok', 'detect_type': result['detect_type'],
            'pixels_per_mm': result['pixels_per_mm'],
            'inner_w_px': result['inner_w_px'],
            'diagnostics': result['surface_diagnostics'],
            'warnings': result['warnings'],
            'overlay': encode(result['overlay_bgr']),
            'crop': encode(result['cropped_bgr'])}


def preview_jpeg(payload, values):
    width, height = validate_jpeg_bytes(payload)
    if width * height > 60_000_000:
        raise ValueError('Ảnh vượt quá giới hạn xem trước 60 megapixel.')
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    return preview_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), values)
