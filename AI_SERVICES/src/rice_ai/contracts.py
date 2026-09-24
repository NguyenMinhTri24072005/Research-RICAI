"""Internal data contracts and exceptions for the Rice AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


import math


class PipelineError(Exception):
    """Lỗi xử lý trong pipeline nội bộ (độc lập với HTTP framework)."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or {}


@dataclass
class PredictionInput:
    """Tham số đầu vào vật lý cho pipeline ước lượng (toàn bộ tính theo đơn vị mm)."""
    diam: float  # mm
    height: float  # mm
    empty: float  # mm
    wall_thickness: float = 1.0  # mm
    weight_total: Optional[float] = None  # g
    sample_weight: Optional[float] = None  # g
    sample_count: Optional[int] = None
    estimator_mode: str = "auto"
    debug: bool = False

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của các tham số vật lý."""
        if not math.isfinite(self.diam) or self.diam <= 0:
            raise PipelineError(422, "INVALID_INPUT", f"Đường kính khay (diam) phải lớn hơn 0 (nhận: {self.diam})")
        if not math.isfinite(self.height) or self.height <= 0:
            raise PipelineError(422, "INVALID_INPUT", f"Chiều cao khay (height) phải lớn hơn 0 (nhận: {self.height})")
        if not math.isfinite(self.empty) or self.empty < 0 or self.empty > self.height:
            raise PipelineError(
                422,
                "INVALID_INPUT",
                f"Khoảng trống đỉnh khay (empty) phải trong khoảng [0, height]: empty={self.empty}, height={self.height}",
            )
        if not math.isfinite(self.wall_thickness) or self.wall_thickness < 0:
            raise PipelineError(422, "INVALID_INPUT", f"Độ dày thành khay không được âm: {self.wall_thickness}")
        if self.weight_total is not None:
            if not math.isfinite(self.weight_total) or self.weight_total < 0:
                raise PipelineError(422, "INVALID_INPUT", f"Khối lượng tổng (weight_total) không được âm: {self.weight_total}")
        if self.sample_weight is not None:
            if not math.isfinite(self.sample_weight) or self.sample_weight < 0:
                raise PipelineError(422, "INVALID_INPUT", f"Khối lượng mẫu (sample_weight) không được âm: {self.sample_weight}")
        if self.sample_count is not None:
            if not isinstance(self.sample_count, int) or self.sample_count < 0:
                raise PipelineError(422, "INVALID_INPUT", f"Số lượng hạt mẫu (sample_count) phải là số nguyên >= 0: {self.sample_count}")
        if self.estimator_mode not in ("auto", "regression", "geometry", "weight"):
            raise PipelineError(
                422,
                "INVALID_INPUT",
                f"estimator_mode '{self.estimator_mode}' không hợp lệ. Cho phép: auto, regression, geometry, weight",
            )


@dataclass
class ContainerResult:
    """Kết quả phân tích vật lý và hình học vật chứa."""
    inner_diam_mm: float
    container_height_mm: float
    empty_height_mm: float
    rice_height_mm: float
    bulk_volume_mm3: float
    pixels_per_mm: float
    raw_dict: Dict[str, Any] = field(default_factory=dict)
    visual_overlay: Optional[Any] = None


@dataclass
class GrainAnalysis:
    """Kết quả phân đoạn, phân loại và đo lường hạt lúa."""
    whole_grains: List[Dict[str, Any]]
    broken_grains: List[Dict[str, Any]]
    chalky_grains: List[Dict[str, Any]]
    foreign_objects: List[Dict[str, Any]]
    total_detected: int
    classified_counts: Dict[str, int]
    volumes_px3: List[float]
    skipped_measurement_count: int = 0
    uniformity_metrics: Dict[str, Any] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    # Regression consumes all valid CNN-whole grains; physical estimation can
    # use a separately filtered population, matching the research notebook.
    physical_grains: List[Dict[str, Any]] = field(default_factory=list)
    size_filter_stats: Dict[str, Any] = field(default_factory=dict)
    size_filter_rejected: List[Dict[str, Any]] = field(default_factory=list)
    raw_crops: List[Dict[str, Any]] = field(default_factory=list)
    cleaned_crops: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EstimateSet:
    """Tập hợp các kết quả ước lượng từ các phương pháp khác nhau."""
    geometry_est: Optional[int]
    weight_est: Optional[int]
    regression_est: Optional[float]
    final: int
    method_used: str


@dataclass
class PipelineResult:
    """Kết quả tổng thể từ phiên chạy RicePipeline."""
    request_id: str
    estimates: EstimateSet
    container: ContainerResult
    grain_analysis: GrainAnalysis
    features_31: Optional[Dict[str, Optional[float]]] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    debug_visuals: Dict[str, str] = field(default_factory=dict)
    artifact_manifest: Optional[Dict[str, Any]] = None
