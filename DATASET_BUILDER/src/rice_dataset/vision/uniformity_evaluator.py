#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 5: ĐÁNH GIÁ ĐỘ ĐỒNG ĐỀU & LỌC NGOẠI LAI IQR (UNIFORMITY EVALUATOR)
===============================================================================
Mục đích:
  - Lọc bỏ các hạt ngoại lai (quá to hoặc quá nhỏ bất thường) bằng chuẩn IQR.
  - Tính tỷ lệ đồng đều kích thước hạt theo tiêu chuẩn nông nghiệp (>= 80%).
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union
import numpy as np


def evaluate_batch_uniformity(
    input_sizes: Sequence[Union[int, float]],
    threshold_rate: float = 0.80,
) -> Dict[str, Any]:
    """
    Đánh giá độ đồng đều kích thước hạt lúa dựa trên phương pháp thống kê IQR.

    Parameters
    ----------
    input_sizes : Sequence of float
        Danh sách kích thước hoặc thể tích các hạt lúa.
    threshold_rate : float, default 0.80
        Ngưỡng tỷ lệ hạt đạt chuẩn yêu cầu (80%).

    Returns
    -------
    dict
        - mean_clean: float (Giá trị trung bình sau lọc ngoại lai)
        - var_clean: float (Phương sai sau lọc ngoại lai)
        - uniformity_rate_pct: float (Tỷ lệ hạt đồng đều %)
        - is_uniform: bool (True nếu >= threshold_rate)
        - total_count: int
        - clean_count: int
        - lower_limit: float
        - upper_limit: float
    """
    sizes_arr = np.array(input_sizes, dtype=float)
    total_objects = len(sizes_arr)

    if total_objects == 0:
        return {
            "mean_clean": 0.0,
            "var_clean": 0.0,
            "uniformity_rate_pct": 0.0,
            "is_uniform": False,
            "total_count": 0,
            "clean_count": 0,
            "lower_limit": 0.0,
            "upper_limit": 0.0,
        }

    q1, q3 = np.percentile(sizes_arr, [25, 75])
    iqr = q3 - q1
    lower_limit = float(q1 - 1.5 * iqr)
    upper_limit = float(q3 + 1.5 * iqr)

    standard_seeds = sizes_arr[(sizes_arr >= lower_limit) & (sizes_arr <= upper_limit)]
    clean_count = len(standard_seeds)
    standard_rate = clean_count / total_objects
    is_uniform = bool(standard_rate >= threshold_rate)

    mean_clean = float(np.mean(standard_seeds)) if clean_count > 0 else 0.0
    var_clean = float(np.var(standard_seeds)) if clean_count > 0 else 0.0

    return {
        "mean_clean": round(mean_clean, 3),
        "var_clean": round(var_clean, 3),
        "uniformity_rate_pct": round(standard_rate * 100.0, 2),
        "is_uniform": is_uniform,
        "total_count": total_objects,
        "clean_count": clean_count,
        "lower_limit": round(lower_limit, 3),
        "upper_limit": round(upper_limit, 3),
    }
