"""Vision Model Providers for YOLO and CNN."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from rice_ai.settings import Settings

logger = logging.getLogger("rice_ai.models.vision_models")


class VisionModelProvider:
    """Quản lý nạp và lưu bộ nhớ đệm an toàn cho mô hình YOLO và CNN."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._yolo_model: Optional[Any] = None
        self._cnn_classifier: Optional[Any] = None
        self._lock = threading.Lock()
        self._yolo_state: dict[str, Any] = {
            "status": "configured",
            "device": self.settings.yolo_device_str,
            "file": None,
            "error": None,
        }
        self._cnn_state: dict[str, Any] = {
            "status": "configured",
            "device": "cpu",
            "file": None,
            "error": None,
        }

    def get_yolo_model(self, confidence: Optional[float] = None) -> Any:
        """Nạp hoặc lấy mô hình YOLO SAHI AutoDetectionModel từ cache."""
        conf = confidence if confidence is not None else self.settings.sahi_conf_threshold
        with self._lock:
            if self._yolo_model is None:
                device = self.settings.get_yolo_device()
                yolo_path = self.settings.get_yolo_path()
                logger.info(f"Đang nạp mô hình YOLO SAHI từ: {yolo_path} (confidence={conf}, device={device})...")
                try:
                    from sahi import AutoDetectionModel
                    self._yolo_model = AutoDetectionModel.from_pretrained(
                        model_type="yolov8",
                        model_path=str(yolo_path),
                        confidence_threshold=conf,
                        device=device,
                    )
                    self._yolo_state = {
                        "status": "loaded",
                        "device": device,
                        "file": yolo_path.name,
                        "error": None,
                    }
                    logger.info(f"Đã nạp YOLO SAHI thành công trên thiết bị {device}!")
                except Exception as ex:
                    self._yolo_state = {
                        "status": "error",
                        "device": device,
                        "file": yolo_path.name,
                        "error": str(ex),
                    }
                    raise
            return self._yolo_model

    def get_cnn_model(self) -> Any:
        """Nạp hoặc lấy mô hình CNN phân loại hạt (GrainClassifier) từ cache."""
        with self._lock:
            if self._cnn_classifier is None:
                cnn_path = self.settings.get_cnn_path()
                logger.info(f"Đang nạp mô hình CNN từ: {cnn_path}...")
                try:
                    from rice_ai.vision.grain_classifier import GrainClassifier
                    classifier = GrainClassifier(model_path=str(cnn_path))
                    self._cnn_classifier = classifier
                    self._cnn_state = {
                        "status": "loaded",
                        "device": getattr(classifier, "device", "cpu"),
                        "file": cnn_path.name,
                        "error": None,
                    }
                    logger.info("Đã nạp CNN thành công!")
                except Exception as ex:
                    self._cnn_state = {
                        "status": "error",
                        "device": "cpu",
                        "file": cnn_path.name,
                        "error": str(ex),
                    }
                    raise
            return self._cnn_classifier

    def is_yolo_loaded(self) -> bool:
        return self._yolo_model is not None

    def is_cnn_loaded(self) -> bool:
        return self._cnn_classifier is not None

    def get_status(self) -> dict[str, Any]:
        """Trả về trạng thái nạp và thông tin thiết bị (không lộ đường dẫn tuyệt đối)."""
        return {
            "yolo": dict(self._yolo_state),
            "cnn": dict(self._cnn_state),
        }
