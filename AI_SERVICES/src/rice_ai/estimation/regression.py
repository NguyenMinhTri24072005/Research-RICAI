"""Regression-based Seed Count Prediction."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from rice_ai.estimation.feature_schema import feature_vector_to_ordered_list
from rice_ai.models.regression_loader import LoadedRegression


def predict_regression(
    features_dict: Dict[str, Optional[float]],
    loaded_regression: LoadedRegression,
) -> Tuple[float, str]:
    """Thực hiện dự đoán hồi quy 31 đặc trưng qua mô hình đã nạp.
    
    Quy tắc:
    1. Chuyển đổi dict thành vector 31 phần tử theo đúng thứ tự ALL_31_FEATURES.
    2. Chuẩn hóa qua scaler.transform nếu có.
    3. Dự đoán qua model.predict.
    4. Trả về (giá trị dự đoán dạng số thực, tên mô hình thực tế).
    """
    ordered_features = feature_vector_to_ordered_list(features_dict, allow_missing=False)
    pred_val = loaded_regression.predict(ordered_features)
    return pred_val, loaded_regression.model_name
