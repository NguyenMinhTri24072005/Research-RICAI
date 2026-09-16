#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODEL REGISTRY — Quản lý nạp, cache và validate model bundle
===============================================================================
Mục đích:
  - Đọc manifest.json để biết artifact nào cần load.
  - Validate file tồn tại, n_features_in_, feature order trước khi đánh verified.
  - Cache keyed theo bundle_id; không cache lỗi vĩnh viễn.
  - Load model+scaler là một đơn vị — tránh model OK nhưng scaler thất bại.
  - Thread-safe: dùng Lock khi nạp model lần đầu.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from feature_schema import ALL_31_FEATURES, FEATURE_SCHEMA_VERSION

# Suppress sklearn version mismatch warning — but LOG it, don't silence completely
import warnings
import logging

logger = logging.getLogger("model_registry")


@dataclass
class BundleStatus:
    """Trạng thái nạp và xác minh bundle."""
    loaded: bool = False
    verified: bool = False
    error: Optional[str] = None
    bundle_id: Optional[str] = None
    schema_version: Optional[str] = None
    model_family: Optional[str] = None
    n_features: Optional[int] = None
    has_scaler: bool = False
    preprocessing_type: Optional[str] = None


@dataclass
class LoadedBundle:
    """Bundle đã nạp thành công."""
    model: Any = None
    scaler: Any = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    status: BundleStatus = field(default_factory=BundleStatus)


class ModelRegistry:
    """
    Quản lý vòng đời nạp/validate/cache cho model bundle.

    Sử dụng:
        registry = ModelRegistry(project_root)
        registry.load_bundle()  # Nạp model + scaler từ manifest
        model, scaler = registry.get_model_and_scaler()
        status = registry.get_status()
    """

    def __init__(
        self,
        project_root: Path,
        manifest_path: Optional[Path] = None,
    ):
        self.project_root = Path(project_root)
        if manifest_path is None:
            manifest_path = self.project_root / "AI_SERVICES" / "artifacts" / "manifest.json"
        self.manifest_path = Path(manifest_path)

        self._bundle: Optional[LoadedBundle] = None
        self._lock = threading.Lock()
        self._load_attempted = False

    def _resolve_path(self, relative_path: str) -> Path:
        """Giải quyết đường dẫn tương đối từ project root."""
        p = self.project_root / relative_path
        if p.exists():
            return p
        # Thử từ AI_SERVICES parent nếu project_root là AI_SERVICES
        alt = self.project_root.parent / relative_path
        if alt.exists():
            return alt
        return p

    def _read_manifest(self) -> Dict[str, Any]:
        """Đọc manifest.json."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest không tồn tại: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _file_sha256(self, filepath: Path, chunk_size: int = 65536) -> str:
        """Tính SHA-256 checksum."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def load_bundle(self, force: bool = False) -> BundleStatus:
        """
        Nạp model bundle theo manifest. Thread-safe, lazy load.

        Parameters
        ----------
        force : bool
            Nếu True, bỏ qua cache và nạp lại.

        Returns
        -------
        BundleStatus
        """
        with self._lock:
            if self._bundle is not None and self._bundle.status.loaded and not force:
                return self._bundle.status

            # Reset nếu force reload
            self._bundle = LoadedBundle()
            status = self._bundle.status

            try:
                manifest = self._read_manifest()
                self._bundle.manifest = manifest

                status.bundle_id = manifest.get("bundle_id")
                status.schema_version = manifest.get("schema_version")
                status.model_family = manifest.get("model", {}).get("family")
                status.preprocessing_type = manifest.get("preprocessing", {}).get("type")

                # 1. Resolve model path
                model_rel = manifest.get("model", {}).get("relative_path")
                if not model_rel:
                    status.error = "Manifest thiếu model.relative_path"
                    return status

                model_path = self._resolve_path(model_rel)
                if not model_path.exists():
                    status.error = f"Model file không tồn tại: {model_rel}"
                    return status

                # 2. Resolve scaler path (nếu preprocessing yêu cầu)
                scaler_path: Optional[Path] = None
                if status.preprocessing_type == "scaler":
                    scaler_rel = manifest.get("preprocessing", {}).get("scaler_relative_path")
                    if scaler_rel:
                        scaler_path = self._resolve_path(scaler_rel)
                        if not scaler_path.exists():
                            status.error = f"Scaler file không tồn tại: {scaler_rel}"
                            return status

                # 3. Load model + scaler as a unit
                import joblib

                # Capture sklearn version warnings
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = joblib.load(model_path)

                for w in caught:
                    logger.warning(
                        "sklearn version mismatch khi load model: %s", str(w.message)
                    )

                scaler = None
                if scaler_path:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        scaler = joblib.load(scaler_path)
                    for w in caught:
                        logger.warning(
                            "sklearn version mismatch khi load scaler: %s", str(w.message)
                        )

                self._bundle.model = model
                self._bundle.scaler = scaler
                status.loaded = True
                status.has_scaler = scaler is not None

                # 4. Validate n_features
                expected_n = manifest.get("model", {}).get("expected_n_features", 31)
                actual_n = getattr(model, "n_features_in_", None)
                if actual_n is not None and actual_n != expected_n:
                    status.error = (
                        f"n_features_in_={actual_n} không khớp manifest ({expected_n})"
                    )
                    status.loaded = False
                    return status
                status.n_features = actual_n or expected_n

                # 5. Validate feature_names_in_ (nếu model có)
                model_features = getattr(model, "feature_names_in_", None)
                if model_features is not None:
                    model_features_list = list(model_features)
                    if model_features_list != ALL_31_FEATURES:
                        status.error = "feature_names_in_ không khớp ALL_31_FEATURES"
                        status.loaded = False
                        return status

                # 6. Validate scaler dimensions
                if scaler is not None:
                    scaler_n = getattr(scaler, "n_features_in_", None)
                    if scaler_n is not None and scaler_n != expected_n:
                        status.error = (
                            f"Scaler n_features_in_={scaler_n} không khớp ({expected_n})"
                        )
                        status.loaded = False
                        return status

                # 7. Smoke test: predict on zero vector → must be finite
                try:
                    test_vec = np.zeros((1, expected_n), dtype=np.float64)
                    if scaler is not None:
                        test_vec_scaled = scaler.transform(test_vec)
                    else:
                        test_vec_scaled = test_vec
                    test_pred = model.predict(test_vec_scaled)
                    if not np.isfinite(test_pred[0]):
                        status.error = "Smoke test trả về giá trị không hữu hạn"
                        status.loaded = False
                        return status
                except Exception as e:
                    status.error = f"Smoke test thất bại: {e}"
                    status.loaded = False
                    return status

                # All checks passed
                status.verified = True
                logger.info(
                    "Bundle loaded & verified: %s (n_features=%d, scaler=%s)",
                    status.bundle_id, status.n_features or 0, status.has_scaler,
                )

            except FileNotFoundError as e:
                status.error = str(e)
            except json.JSONDecodeError as e:
                status.error = f"Manifest JSON không hợp lệ: {e}"
            except Exception as e:
                status.error = f"Lỗi nạp bundle: {e}"
                logger.error("Bundle load error: %s", traceback.format_exc())

            self._load_attempted = True
            return status

    def get_model_and_scaler(self) -> Tuple[Any, Any]:
        """
        Trả về (model, scaler) đã nạp. Tự động gọi load nếu chưa.

        Returns
        -------
        (model, scaler) hoặc (None, None) nếu load thất bại.
        """
        if self._bundle is None or not self._bundle.status.loaded:
            self.load_bundle()

        if self._bundle and self._bundle.status.loaded:
            return self._bundle.model, self._bundle.scaler
        return None, None

    def get_status(self) -> BundleStatus:
        """Trả về trạng thái hiện tại của bundle."""
        if self._bundle is None:
            return BundleStatus()
        return self._bundle.status

    def get_manifest(self) -> Dict[str, Any]:
        """Trả về manifest dict."""
        if self._bundle is None:
            return {}
        return self._bundle.manifest

    def is_ready(self) -> bool:
        """Bundle đã loaded và verified?"""
        if self._bundle is None:
            return False
        return self._bundle.status.loaded and self._bundle.status.verified
