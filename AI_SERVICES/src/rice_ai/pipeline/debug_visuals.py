"""Small, request-local visual summaries for the inference UI."""
from __future__ import annotations

import base64
from typing import Dict

import cv2
import numpy as np

from rice_ai.contracts import ContainerResult, GrainAnalysis


def _as_data_url(image_bgr: np.ndarray) -> str | None:
    if image_bgr is None or image_bgr.size == 0:
        return None
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest > 1400:
        ratio = 1400.0 / longest
        image_bgr = cv2.resize(image_bgr, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_debug_visuals(
    source_bgr: np.ndarray,
    container: ContainerResult,
    grains: GrainAnalysis,
) -> Dict[str, str]:
    """Return compact visual evidence without persisting request images to disk."""
    visuals: Dict[str, str] = {}
    container_overlay = container.raw_dict.get("overlay_bgr")
    if not isinstance(container_overlay, np.ndarray):
        container_overlay = container.raw_dict.get("visual_overlay")
    if isinstance(container_overlay, np.ndarray):
        data_url = _as_data_url(container_overlay)
        if data_url:
            visuals["container_detection"] = data_url

    detections = source_bgr.copy()
    for grain in grains.broken_grains:
        bbox = grain.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(detections, (x1, y1), (x2, y2), (0, 0, 255), 2)
    physical_ids = {grain.get("grain_id") for grain in grains.physical_grains}
    for grain in grains.whole_grains:
        bbox = grain.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 180, 0) if grain.get("grain_id") in physical_ids else (0, 165, 255)
            cv2.rectangle(detections, (x1, y1), (x2, y2), color, 2)
    data_url = _as_data_url(detections)
    if data_url:
        visuals["grain_decisions"] = data_url
    return visuals