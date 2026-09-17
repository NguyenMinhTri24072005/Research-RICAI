"""Folder-based Regression Model and Scaler Loader."""
from __future__ import annotations

import json
import logging
import math
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from rice_ai.estimation.feature_schema import (
    ALL_31_FEATURES,
    FEATURE_SCHEMA_VERSION,
)
from rice_ai.settings import Settings
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger("rice_ai.models.regression_loader")


def _predict_single_vector(
    model: Any,
    scaler: Optional[Any],
    feature_vector_ordered: Sequence[float],
) -> float:
    """Routine dùng chung cho cả smoke test và runtime inference.
    
    Quy tắc nghiêm ngặt:
    1. Kiểm tra độ dài đúng 31, mọi phần tử hữu hạn.
    2. Dựng DataFrame nếu scaler yêu cầu feature names, ngược lại ndarray.
    3. scaler.transform đúng 1 lần (nếu có scaler).
    4. Kiểm tra output scaler hữu hạn (1, 31).
    5. Dựng DataFrame nếu model yêu cầu feature names, ngược lại ndarray.
    6. model.predict thực thi.
    7. Kiểm tra output shape: chỉ chấp nhận (1,) hoặc (1, 1). Bất kỳ shape nào khác (ví dụ (1, 2) multi-output)
       phải bị từ chối, không được lấy [0][0] ngầm để che lỗi.
    8. Kiểm tra tính hữu hạn. Trả về raw scalar float.
    """
    if len(feature_vector_ordered) != 31:
        raise ValueError(f"Vector đặc trưng phải có đúng 31 phần tử, nhận được {len(feature_vector_ordered)}")

    for i, v in enumerate(feature_vector_ordered):
        if v is None or not math.isfinite(v):
            raise ValueError(f"Phần tử thứ {i} trong vector đặc trưng không hữu hạn: {v}")

    # 1. Chuẩn bị input cho Scaler
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        import pandas as pd
        x_scaler_in = pd.DataFrame([feature_vector_ordered], columns=ALL_31_FEATURES)
    else:
        x_scaler_in = np.asarray([feature_vector_ordered], dtype=np.float64)

    # 2. Transform đúng 1 lần
    if scaler is not None:
        x_scaled = scaler.transform(x_scaler_in)
        if not isinstance(x_scaled, np.ndarray):
            x_scaled = np.asarray(x_scaled, dtype=np.float64)
        if x_scaled.shape != (1, 31):
            raise ValueError(f"Scaler output shape không hợp lệ: {x_scaled.shape}, kỳ vọng (1, 31)")
        if not np.all(np.isfinite(x_scaled)):
            raise ValueError("Scaler output chứa giá trị NaN hoặc vô hạn!")
    else:
        x_scaled = np.asarray([feature_vector_ordered], dtype=np.float64)

    # 3. Chuẩn bị input cho Model
    if hasattr(model, "feature_names_in_"):
        import pandas as pd
        x_model_in = pd.DataFrame(x_scaled, columns=ALL_31_FEATURES)
    else:
        x_model_in = x_scaled

    # 4. Dự đoán qua Model
    pred_raw = model.predict(x_model_in)
    pred_arr = np.asarray(pred_raw)

    if pred_arr.size != 1 or (pred_arr.ndim > 1 and pred_arr.shape[-1] > 1):
        raise ValueError(
            f"Mô hình hồi quy trả về kết quả đa mục tiêu (shape {pred_arr.shape}). "
            "Hệ thống chỉ chấp nhận mô hình đơn mục tiêu (scalar prediction)."
        )

    val = float(pred_arr.ravel()[0])
    if not math.isfinite(val):
        raise ValueError(f"Mô hình hồi quy trả về giá trị không hữu hạn ({val})")

    return val


def _validate_fitted_and_metadata(model_or_scaler: Any, is_scaler: bool = False) -> None:
    """Xác thực trạng thái đã huấn luyện và metadata biến của estimator/scaler."""
    role = "Scaler" if is_scaler else "Mô hình"
    try:
        check_is_fitted(model_or_scaler)
    except Exception as ex:
        raise ValueError(f"{role} chưa được fit (huấn luyện): {ex}")

    if hasattr(model_or_scaler, "n_features_in_"):
        n_feat = getattr(model_or_scaler, "n_features_in_")
        if n_feat != 31:
            raise ValueError(f"{role} yêu cầu {n_feat} đặc trưng, nhưng hệ thống yêu cầu đúng 31 đặc trưng.")

    if hasattr(model_or_scaler, "feature_names_in_"):
        names = list(getattr(model_or_scaler, "feature_names_in_"))
        if names != ALL_31_FEATURES:
            raise ValueError(f"Thuộc tính feature_names_in_ trên {role} không khớp đúng thứ tự ALL_31_FEATURES.")


@dataclass
class LoadedRegression:
    """Gói mô hình hồi quy và tiền xử lý đã nạp và xác minh thành công."""
    model: Any
    scaler: Optional[Any]
    feature_names: List[str]
    schema_version: str
    model_name: str
    model_dir: Path
    config: Optional[Dict[str, Any]] = None
    is_legacy: bool = False
    warnings: List[str] = field(default_factory=list)

    def predict(self, feature_vector_ordered: Sequence[float]) -> float:
        """Dự đoán từ vector 31 đặc trưng đã xếp đúng thứ tự."""
        raw_val = _predict_single_vector(self.model, self.scaler, feature_vector_ordered)
        return max(0.0, round(raw_val, 1))


def load_regression_folder(folder_path: Union[str, Path]) -> LoadedRegression:
    """Nạp và kiểm chứng toàn vẹn bundle mô hình hồi quy từ thư mục.
    
    Hỗ trợ 2 quy chuẩn bố cục:
    1. Canonical Folder:
       - model.joblib (bắt buộc)
       - scaler.joblib (bắt buộc, trừ khi config.json chỉ định rõ preprocessing: 'none')
       - feature_schema.json (bắt buộc, khớp 31 features và version)
       - config.json (tùy chọn)
    2. Legacy Adapter Folder:
       - best_tree_ensemble_model.joblib
       - scaler.joblib
       - scaler_params.json
       (Chỉ kích hoạt khi không có model.joblib)
    """
    folder = Path(folder_path).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Thư mục mô hình hồi quy không tồn tại: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Đường dẫn không phải là thư mục: {folder}")

    canonical_model_file = folder / "model.joblib"
    legacy_model_file = folder / "best_tree_ensemble_model.joblib"

    collected_warnings: List[str] = []

    # =========================================================================
    # TRƯỜNG HỢP 1: CANONICAL BUNDLE (model.joblib)
    # =========================================================================
    if canonical_model_file.exists():
        schema_file = folder / "feature_schema.json"
        if not schema_file.exists():
            raise FileNotFoundError(
                f"Thư mục bundle canonical thiếu tệp hợp đồng feature_schema.json tại {folder}"
            )

        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except Exception as ex:
            raise ValueError(f"Không thể đọc file feature_schema.json tại {folder}: {ex}")

        schema_features = schema_data.get("features")
        if not isinstance(schema_features, list) or len(schema_features) != 31:
            raise ValueError(
                f"feature_schema.json phải chứa danh sách đúng 31 đặc trưng ('features'). Tìm thấy: {len(schema_features) if isinstance(schema_features, list) else type(schema_features)}"
            )

        if schema_features != ALL_31_FEATURES:
            raise ValueError(
                f"Danh sách đặc trưng trong {schema_file} không khớp chính xác với ALL_31_FEATURES của hệ thống!"
            )

        schema_ver = schema_data.get("schema_version")
        if not schema_ver or str(schema_ver).strip() != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"feature_schema.json phải có schema_version='{FEATURE_SCHEMA_VERSION}'. "
                f"Tìm thấy: '{schema_ver}'"
            )

        # Đọc config.json (nếu có)
        config_data: Optional[Dict[str, Any]] = None
        config_file = folder / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as ex:
                raise ValueError(f"Tệp cấu hình config.json bị lỗi cú pháp tại {folder}: {ex}")

        # Kiểm tra yêu cầu Scaler
        is_no_scaler_explicit = False
        if config_data and isinstance(config_data, dict):
            prep_setting = str(config_data.get("preprocessing", "")).strip().lower()
            if prep_setting == "none":
                is_no_scaler_explicit = True

        scaler_file = folder / "scaler.joblib"
        if is_no_scaler_explicit and scaler_file.exists():
            raise ValueError(
                f"Mâu thuẫn cấu hình bundle tại {folder}: config.json khai báo 'preprocessing': 'none' "
                "nhưng tệp scaler.joblib vẫn tồn tại trong thư mục (ambiguous bundle)."
            )

        scaler_obj = None
        if not is_no_scaler_explicit:
            if not scaler_file.exists():
                raise FileNotFoundError(
                    f"Thiếu tệp scaler.joblib tại {folder}. Nếu mô hình không cần scaler, "
                    "hãy xóa scaler.joblib và khai báo rõ ràng '\"preprocessing\": \"none\"' trong config.json."
                )

        # Nạp tệp nhị phân qua joblib
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model_obj = joblib.load(canonical_model_file)
            if scaler_file.exists() and not is_no_scaler_explicit:
                scaler_obj = joblib.load(scaler_file)

        for w in captured:
            collected_warnings.append(f"{w.category.__name__}: {w.message}")

        # Kiểm tra an toàn: không nhận đối tượng Pipeline trong model.joblib
        if isinstance(model_obj, Pipeline):
            raise ValueError(
                f"Tệp model.joblib tại {folder} là một sklearn Pipeline. "
                "Quy chuẩn kiến trúc yêu cầu tách rời model.joblib và scaler.joblib để tránh double-scaling."
            )

        # Kiểm tra giao diện fitted và metadata đặc trưng
        _validate_fitted_and_metadata(model_obj, is_scaler=False)
        if not hasattr(model_obj, "predict") or not callable(getattr(model_obj, "predict")):
            raise ValueError(f"Mô hình nạp từ {canonical_model_file} không có phương thức callable 'predict'.")

        if scaler_obj is not None:
            _validate_fitted_and_metadata(scaler_obj, is_scaler=True)
            if not hasattr(scaler_obj, "transform") or not callable(getattr(scaler_obj, "transform")):
                raise ValueError(f"Scaler nạp từ {scaler_file} không có phương thức callable 'transform'.")

        # Xác định tên hiển thị
        cls_name = model_obj.__class__.__name__
        model_name = "ExtraTrees" if cls_name == "ExtraTreesRegressor" else cls_name

        # Smoke prediction kiểm tra tính tương thích toán học
        _run_smoke_test(model_obj, scaler_obj)

        return LoadedRegression(
            model=model_obj,
            scaler=scaler_obj,
            feature_names=ALL_31_FEATURES,
            schema_version=schema_ver,
            model_name=model_name,
            model_dir=folder,
            config=config_data,
            is_legacy=False,
            warnings=collected_warnings,
        )

    # =========================================================================
    # TRƯỜNG HỢP 2: LEGACY ADAPTER (best_tree_ensemble_model.joblib)
    # =========================================================================
    elif legacy_model_file.exists():
        scaler_file = folder / "scaler.joblib"
        scaler_params_file = folder / "scaler_params.json"

        if not scaler_file.exists() or not scaler_params_file.exists():
            raise FileNotFoundError(
                f"Thư mục legacy {folder} có best_tree_ensemble_model.joblib nhưng thiếu "
                f"scaler.joblib hoặc scaler_params.json."
            )

        try:
            with open(scaler_params_file, "r", encoding="utf-8") as f:
                params_data = json.load(f)
        except Exception as ex:
            raise ValueError(f"Không thể đọc scaler_params.json tại {folder}: {ex}")

        legacy_features = params_data.get("features")
        if legacy_features != ALL_31_FEATURES:
            raise ValueError(
                f"Danh sách features trong legacy scaler_params.json không khớp với ALL_31_FEATURES!"
            )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model_obj = joblib.load(legacy_model_file)
            scaler_obj = joblib.load(scaler_file)

        for w in captured:
            collected_warnings.append(f"{w.category.__name__}: {w.message}")

        _validate_fitted_and_metadata(model_obj, is_scaler=False)
        _validate_fitted_and_metadata(scaler_obj, is_scaler=True)

        collected_warnings.append(
            "LEGACY_LAYOUT: Đang nạp mô hình qua adapter tương thích legacy (best_tree_ensemble_model.joblib)."
        )

        _run_smoke_test(model_obj, scaler_obj)

        return LoadedRegression(
            model=model_obj,
            scaler=scaler_obj,
            feature_names=ALL_31_FEATURES,
            schema_version="31v1",
            model_name="ExtraTrees",
            model_dir=folder,
            config=None,
            is_legacy=True,
            warnings=collected_warnings,
        )

    # =========================================================================
    # LỖI: KHÔNG TÌM THẤY TỆP MÔ HÌNH HỢP LỆ
    # =========================================================================
    else:
        raise FileNotFoundError(
            f"Thư mục '{folder}' không chứa tệp mô hình hợp lệ (cần 'model.joblib' cho canonical "
            f"hoặc 'best_tree_ensemble_model.joblib' cho legacy)."
        )


def _run_smoke_test(model: Any, scaler: Optional[Any]) -> None:
    """Chạy smoke test trên vector tổng hợp hợp lệ để kiểm chứng không lỗi runtime."""
    smoke_vec = [
        1000.0, 20.0, 2.5, 5.0, 10.0, 200.0, 20.0, 30.0,  # container (8)
        10.0, 85.0, 150.0,                                 # surface (3)
        6.5, 5.0, 8.0, 0.5,                                # length (4)
        2.5, 2.0, 3.0, 0.2,                                # width (4)
        1.8, 1.5, 2.1, 0.1,                                # thickness (4)
        12.0, 9.0, 15.0, 1.0,                              # area (4)
        15.0, 10.0, 20.0, 2.0,                             # volume (4)
    ]
    _predict_single_vector(model, scaler, smoke_vec)


class LoadedRegressionProvider:
    """Provider quản lý vòng đời và bộ nhớ đệm cho LoadedRegression."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._loaded: Optional[LoadedRegression] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    def get_regression(self) -> LoadedRegression:
        """Lấy LoadedRegression. Nếu chưa nạp, nạp tự động với lock an toàn luồng."""
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if self._load_error is not None:
                raise RuntimeError(f"Mô hình hồi quy nạp thất bại: {self._load_error}")

            try:
                target_dir = self.settings.get_regression_dir()
                self._loaded = load_regression_folder(target_dir)
                logger.info(
                    f"Đã nạp thành công mô hình hồi quy '{self._loaded.model_name}' "
                    f"từ {self._loaded.model_dir} (Legacy: {self._loaded.is_legacy})"
                )
                return self._loaded
            except Exception as ex:
                self._load_error = str(ex)
                logger.error(f"Lỗi khi nạp mô hình hồi quy: {ex}")
                raise

    def is_loaded(self) -> bool:
        return self._loaded is not None

    def get_error(self) -> Optional[str]:
        return self._load_error

    def get_model_name(self) -> Optional[str]:
        if self._loaded:
            return self._loaded.model_name
        return None
