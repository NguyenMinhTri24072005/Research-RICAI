#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
FEATURE SCHEMA — Nguồn duy nhất cho 31 đặc trưng hồi quy (Single Source of Truth)
===============================================================================
Mục đích:
  - Khai báo tập trung danh sách, thứ tự, đơn vị và ràng buộc miền giá trị
    của 31 đặc trưng dùng trong mô hình hồi quy hệ thống.
  - Cung cấp hàm validate vector đặc trưng trước khi đưa vào model.
  - Tất cả module khác import từ đây, không tự khai báo danh sách features riêng.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEATURE_SCHEMA_VERSION = "31v1"
TARGET_COLUMN = "Actual_Count"
TRAINED_HYBRID_PACKING_FRACTION: float = 0.62  # Xác minh 100% (254/254 mẫu FOUND trên final_linear_regression_dataset.csv)


def compute_trained_hybrid_feature(
    bulk_volume_mm3: Optional[float],
    grain_volumes_mm3: Sequence[float],
    packing_fraction: float = TRAINED_HYBRID_PACKING_FRACTION,
) -> Optional[int]:
    """
    Tính đặc trưng Estimated_Total_Seeds_Hybrid (index 10) chuẩn hóa
    theo đúng công thức trích xuất dataset huấn luyện.

    Công thức:
        int(round(Bulk_Rice_Volume_mm3 * 0.62 / Grain_Volume_mm3_Mean))

    Ràng buộc:
      - Hoàn toàn độc lập với Actual_Count và cân mẫu (weight).
      - Trả về None nếu không đủ thông tin (không có thể tích hạt hợp lệ).
    """
    if bulk_volume_mm3 is None or bulk_volume_mm3 <= 0 or not math.isfinite(bulk_volume_mm3):
        return None
    if not grain_volumes_mm3:
        return None

    valid_vols = [float(v) for v in grain_volumes_mm3 if v is not None and v > 0 and math.isfinite(v)]
    if not valid_vols:
        return None

    mean_vol = sum(valid_vols) / len(valid_vols)
    if mean_vol <= 0 or not math.isfinite(mean_vol):
        return None

    if not math.isfinite(packing_fraction) or not 0.0 < packing_fraction <= 1.0:
        raise ValueError(f"packing_fraction không hợp lệ: {packing_fraction}")
    return int(round(bulk_volume_mm3 * packing_fraction / mean_vol))


def compute_schema_hash() -> str:
    """
    Tính SHA-256 canonical hash của schema version và danh sách ALL_31_FEATURES.
    Dùng để phát hiện bất kỳ sự thay đổi hoặc xáo trộn thứ tự feature nào.
    """
    import hashlib
    import json
    canonical = json.dumps({"version": FEATURE_SCHEMA_VERSION, "features": ALL_31_FEATURES}, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Định nghĩa metadata cho từng đặc trưng
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureDef:
    """Metadata mô tả một đặc trưng đầu vào."""
    name: str
    group: str
    unit: str
    min_val: Optional[float] = None    # None = không ràng buộc dưới
    max_val: Optional[float] = None    # None = không ràng buộc trên
    required: bool = True              # False = cho phép None/missing
    allow_zero: bool = True            # False = giá trị 0 không hợp lệ
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Danh sách 31 đặc trưng — ĐÚNG THỨ TỰ huấn luyện
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DEFS: Tuple[FeatureDef, ...] = (
    # Nhóm 1: Vật chứa & Thể tích khối lúa (8 biến)
    FeatureDef("Bulk_Rice_Volume_mm3",       "container", "mm³",  min_val=0.0, allow_zero=False, description="Thể tích khối lúa trong ly"),
    FeatureDef("Rice_Height_mm",             "container", "mm",   min_val=0.0, allow_zero=False, description="Chiều cao cột lúa"),
    FeatureDef("Weight_g",                   "container", "g",    min_val=0.0, required=False,   description="Khối lượng tổng (có thể bỏ trống)"),
    FeatureDef("Empty_Height_mm",            "container", "mm",   min_val=0.0,                   description="Khoảng trống từ miệng ly đến mặt lúa"),
    FeatureDef("Pixels_Per_mm",              "container", "px/mm", min_val=0.1, allow_zero=False, description="Tỷ lệ quy đổi pixel sang mm"),
    FeatureDef("Container_Detected_Diam_px", "container", "px",   min_val=1.0, allow_zero=False, description="Đường kính miệng ly phát hiện (px)"),
    FeatureDef("Inner_Diameter_mm",          "container", "mm",   min_val=1.0, allow_zero=False, description="Đường kính trong thực tế"),
    FeatureDef("Container_Height_mm",        "container", "mm",   min_val=1.0, allow_zero=False, description="Chiều cao toàn bộ ly"),
    # Nhóm 2: Bề mặt & Ước lượng (3 biến)
    FeatureDef("Whole_Grains_Count",         "surface",   "hạt",  min_val=0.0,                   description="Số hạt nguyên bề mặt"),
    FeatureDef("Uniformity_Rate_Pct",        "surface",   "%",    min_val=0.0, max_val=100.0,    description="Tỷ lệ đồng đều kích thước"),
    FeatureDef("Estimated_Total_Seeds_Hybrid","surface",  "hạt",  min_val=0.0,                   description="Ước lượng Hybrid (hình học+cân)"),
    # Nhóm 3: Chiều dài hạt 2a (4 biến thống kê)
    FeatureDef("Grain_Length_mm_Mean",        "length",   "mm",   min_val=0.0, description="Chiều dài trung bình"),
    FeatureDef("Grain_Length_mm_Min",         "length",   "mm",   min_val=0.0, description="Chiều dài nhỏ nhất"),
    FeatureDef("Grain_Length_mm_Max",         "length",   "mm",   min_val=0.0, description="Chiều dài lớn nhất"),
    FeatureDef("Grain_Length_mm_Std",         "length",   "mm",   min_val=0.0, description="Độ lệch chuẩn chiều dài"),
    # Nhóm 4: Chiều rộng hạt 2b (4 biến thống kê)
    FeatureDef("Grain_Width_mm_Mean",         "width",    "mm",   min_val=0.0, description="Chiều rộng trung bình"),
    FeatureDef("Grain_Width_mm_Min",          "width",    "mm",   min_val=0.0, description="Chiều rộng nhỏ nhất"),
    FeatureDef("Grain_Width_mm_Max",          "width",    "mm",   min_val=0.0, description="Chiều rộng lớn nhất"),
    FeatureDef("Grain_Width_mm_Std",          "width",    "mm",   min_val=0.0, description="Độ lệch chuẩn chiều rộng"),
    # Nhóm 5: Chiều dày hạt 2c (4 biến thống kê)
    FeatureDef("Grain_Thickness_mm_Mean",     "thickness","mm",   min_val=0.0, description="Chiều dày trung bình"),
    FeatureDef("Grain_Thickness_mm_Min",      "thickness","mm",   min_val=0.0, description="Chiều dày nhỏ nhất"),
    FeatureDef("Grain_Thickness_mm_Max",      "thickness","mm",   min_val=0.0, description="Chiều dày lớn nhất"),
    FeatureDef("Grain_Thickness_mm_Std",      "thickness","mm",   min_val=0.0, description="Độ lệch chuẩn chiều dày"),
    # Nhóm 6: Diện tích 2D (4 biến thống kê)
    FeatureDef("Grain_Area_mm2_Mean",         "area",     "mm²",  min_val=0.0, description="Diện tích 2D trung bình"),
    FeatureDef("Grain_Area_mm2_Min",          "area",     "mm²",  min_val=0.0, description="Diện tích 2D nhỏ nhất"),
    FeatureDef("Grain_Area_mm2_Max",          "area",     "mm²",  min_val=0.0, description="Diện tích 2D lớn nhất"),
    FeatureDef("Grain_Area_mm2_Std",          "area",     "mm²",  min_val=0.0, description="Độ lệch chuẩn diện tích"),
    # Nhóm 7: Thể tích 3D Ellipsoid (4 biến thống kê)
    FeatureDef("Grain_Volume_mm3_Mean",       "volume",   "mm³",  min_val=0.0, description="Thể tích 3D trung bình"),
    FeatureDef("Grain_Volume_mm3_Min",        "volume",   "mm³",  min_val=0.0, description="Thể tích 3D nhỏ nhất"),
    FeatureDef("Grain_Volume_mm3_Max",        "volume",   "mm³",  min_val=0.0, description="Thể tích 3D lớn nhất"),
    FeatureDef("Grain_Volume_mm3_Std",        "volume",   "mm³",  min_val=0.0, description="Độ lệch chuẩn thể tích"),
)

# Danh sách tên theo đúng thứ tự huấn luyện (dùng cho model.predict)
ALL_31_FEATURES: List[str] = [f.name for f in FEATURE_DEFS]

# Tra cứu nhanh theo tên
_FEATURE_MAP: Dict[str, FeatureDef] = {f.name: f for f in FEATURE_DEFS}

# Các nhóm thống kê grain (cần ít nhất 1 hạt hợp lệ)
GRAIN_STAT_GROUPS = ("length", "width", "thickness", "area", "volume")

# Các feature thuộc nhóm grain stats
GRAIN_STAT_FEATURES: List[str] = [
    f.name for f in FEATURE_DEFS if f.group in GRAIN_STAT_GROUPS
]

# Các feature container/form (luôn phải có từ input)
CONTAINER_FEATURES: List[str] = [
    f.name for f in FEATURE_DEFS if f.group == "container"
]


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureValidationResult:
    """Kết quả kiểm tra vector đặc trưng."""
    valid: bool = True
    missing_features: List[str] = field(default_factory=list)
    non_finite_features: List[str] = field(default_factory=list)
    out_of_domain: List[Tuple[str, float, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    grain_count: int = 0

    @property
    def error_code(self) -> Optional[str]:
        if self.missing_features:
            return "MISSING_FEATURE"
        if self.non_finite_features:
            return "NON_FINITE_FEATURE"
        if self.out_of_domain:
            return "INVALID_INPUT"
        return None

    @property
    def error_message(self) -> Optional[str]:
        if self.missing_features:
            return f"Thiếu đặc trưng bắt buộc: {', '.join(self.missing_features)}"
        if self.non_finite_features:
            return f"Giá trị không hữu hạn (NaN/Inf): {', '.join(self.non_finite_features)}"
        if self.out_of_domain:
            details = "; ".join(
                f"{name}={val} ({reason})" for name, val, reason in self.out_of_domain
            )
            return f"Giá trị ngoài miền: {details}"
        return None


def validate_feature_vector(
    features_dict: Dict[str, Optional[float]],
    require_grains: bool = True,
) -> FeatureValidationResult:
    """
    Kiểm tra vector 31 đặc trưng trước khi đưa vào model.

    Parameters
    ----------
    features_dict : dict
        Vector đặc trưng {tên: giá trị}.
    require_grains : bool
        Nếu True, yêu cầu các nhóm thống kê hạt phải có giá trị > 0
        (tức phải có ít nhất 1 hạt nguyên được đo).

    Returns
    -------
    FeatureValidationResult
    """
    result = FeatureValidationResult()

    for fdef in FEATURE_DEFS:
        val = features_dict.get(fdef.name)

        # Kiểm tra missing
        if val is None:
            if fdef.required:
                result.missing_features.append(fdef.name)
                result.valid = False
            continue

        # Kiểm tra finite
        if not math.isfinite(val):
            result.non_finite_features.append(fdef.name)
            result.valid = False
            continue

        # Kiểm tra miền giá trị
        if fdef.min_val is not None and val < fdef.min_val:
            result.out_of_domain.append((fdef.name, val, f"phải >= {fdef.min_val}"))
            result.valid = False
        if fdef.max_val is not None and val > fdef.max_val:
            result.out_of_domain.append((fdef.name, val, f"phải <= {fdef.max_val}"))
            result.valid = False
        if not fdef.allow_zero and val == 0.0:
            result.out_of_domain.append((fdef.name, val, "không được bằng 0"))
            result.valid = False

    # Kiểm tra logic nghiệp vụ
    grain_count = features_dict.get("Whole_Grains_Count", 0)
    result.grain_count = int(grain_count) if grain_count else 0

    if result.grain_count == 0 and require_grains:
        result.valid = False
        result.warnings.append("NO_VALID_GRAINS: Không có hạt nguyên nào được đo.")

    # Kiểm tra std = 0 khi chỉ có 1 hạt (hợp lệ, không phải lỗi)
    if result.grain_count == 1:
        for fdef in FEATURE_DEFS:
            if fdef.name.endswith("_Std"):
                val = features_dict.get(fdef.name, 0.0)
                if val == 0.0:
                    result.warnings.append(
                        f"SINGLE_GRAIN_STD: {fdef.name}=0 vì chỉ có 1 hạt (hợp lệ)."
                    )

    # Weight_g = 0 cảnh báo nhưng không lỗi (mode auto cho phép bỏ trống)
    weight = features_dict.get("Weight_g", 0.0)
    if weight is not None and weight == 0.0:
        result.warnings.append(
            "WEIGHT_ZERO: Weight_g=0 (chưa nhập cân nặng). "
            "Ước lượng hồi quy vẫn chạy nhưng độ chính xác có thể giảm."
        )

    return result


def feature_vector_to_ordered_list(
    features_dict: Dict[str, Optional[float]],
    allow_missing: bool = False,
) -> List[float]:
    """
    Chuyển dict sang list theo đúng thứ tự ALL_31_FEATURES.

    Parameters
    ----------
    features_dict : dict
        Vector đặc trưng đầu vào.
    allow_missing : bool
        Nếu False, sẽ raise ValueError nếu thiếu đặc trưng bắt buộc hoặc gặp giá trị None/non-finite.
        Tuyệt đối không tự ý gán 0.0 cho các biến bắt buộc bị thiếu.
    """
    ordered: List[float] = []
    for fdef in FEATURE_DEFS:
        val = features_dict.get(fdef.name)
        if val is None:
            if fdef.required and not allow_missing:
                raise ValueError(f"Thiếu đặc trưng bắt buộc '{fdef.name}' (giá trị là None/chưa xác định).")
            val = 0.0
        elif not math.isfinite(val):
            if not allow_missing:
                raise ValueError(f"Đặc trưng '{fdef.name}' có giá trị không hữu hạn ({val}).")
        ordered.append(float(val))
    return ordered
