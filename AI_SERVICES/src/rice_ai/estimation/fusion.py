"""Fusion and Final Estimate Selection Policy."""
from __future__ import annotations

from typing import Optional

from rice_ai.contracts import EstimateSet, PipelineError


def select_final_estimate(
    mode: str,
    geometry_est: Optional[int],
    weight_est: Optional[int],
    regression_est: Optional[float],
    regression_model_name: Optional[str] = None,
) -> EstimateSet:
    """Quyết định kết quả ước lượng cuối cùng theo chính sách phân cấp.
    
    Quy tắc:
    - regression: Bắt buộc có kết quả hồi quy; ném PipelineError 503 nếu không khả dụng.
    - geometry: Bắt buộc có kết quả hình học; ném PipelineError 422 nếu thiếu mẫu hạt.
    - weight: Bắt buộc có kết quả cân mẫu; ném PipelineError 422 nếu thiếu tham số cân.
    - auto: Ưu tiên hồi quy -> fallback trung bình geometry+weight -> fallback 1 nhánh còn lại.
            Tuyệt đối không tự ý trung bình cả 3 nhánh vì regression đã sử dụng đặc trưng hình học/cân nặng.
    """
    mode_normalized = mode.strip().lower() if mode else "auto"

    if mode_normalized == "regression":
        if regression_est is None:
            raise PipelineError(
                status_code=503,
                error_code="MODEL_UNAVAILABLE",
                message="Mô hình hồi quy không khả dụng hoặc dự đoán thất bại ở chế độ regression mode.",
            )
        final_val = int(round(regression_est))
        method = f"regression_{regression_model_name or 'model'}"

    elif mode_normalized == "geometry":
        if geometry_est is None:
            raise PipelineError(
                status_code=422,
                error_code="INVALID_INPUT",
                message="Không thể ước lượng theo phương pháp hình học do không có hạt hợp lệ được đo.",
            )
        final_val = geometry_est
        method = "geometry"

    elif mode_normalized == "weight":
        if weight_est is None:
            raise PipelineError(
                status_code=422,
                error_code="INVALID_INPUT",
                message="Không thể ước lượng theo cân nặng do thiếu hoặc sai lệch thông số cân mẫu.",
            )
        final_val = weight_est
        method = "weight"

    elif mode_normalized == "auto":
        if regression_est is not None and regression_est >= 0:
            final_val = int(round(regression_est))
            method = f"regression_{regression_model_name or 'model'}"
        elif geometry_est is not None and weight_est is not None:
            final_val = int(round((geometry_est + weight_est) / 2.0))
            method = "hybrid"
        elif geometry_est is not None:
            final_val = geometry_est
            method = "geometry"
        elif weight_est is not None:
            final_val = weight_est
            method = "weight"
        else:
            final_val = None
            method = "none"

    else:
        raise PipelineError(
            status_code=422,
            error_code="INVALID_INPUT",
            message=f"estimator_mode '{mode}' không hợp lệ. Các mode được hỗ trợ: auto, regression, geometry, weight.",
        )

    return EstimateSet(
        geometry_est=geometry_est,
        weight_est=weight_est,
        regression_est=regression_est,
        final=final_val,
        method_used=method,
    )
