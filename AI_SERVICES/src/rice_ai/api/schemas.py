"""API Request, Response and Error Schemas."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCode(str, Enum):
    """Mã lỗi chuẩn trả về cho client."""
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FEATURE = "MISSING_FEATURE"
    NON_FINITE_FEATURE = "NON_FINITE_FEATURE"
    NO_VALID_GRAINS = "NO_VALID_GRAINS"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    ARTIFACT_INCOMPATIBLE = "ARTIFACT_INCOMPATIBLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    SERVER_BUSY = "SERVER_BUSY"
    EMPTY_SAMPLE = "EMPTY_SAMPLE"


class EstimationMethod(str, Enum):
    """Phương pháp ước lượng thực sự được sử dụng."""
    REGRESSION_EXTRA_TREES = "regression_ExtraTrees"
    REGRESSION_OLS = "regression_OLS_Equation"
    GEOMETRY = "geometry"
    WEIGHT = "weight"
    HYBRID = "hybrid"
    NONE = "none"


class ComponentStatus(str, Enum):
    CONFIGURED = "configured"
    LOADED = "loaded"
    VERIFIED = "verified"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ServiceReadiness(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


@dataclass
class ErrorDetail:
    """Chi tiết lỗi trả về cho client."""
    code: str
    message: str
    field: Optional[str] = None
    stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"code": self.code, "message": self.message}
        if self.field:
            d["field"] = self.field
        if self.stage:
            d["stage"] = self.stage
        return d


@dataclass
class EstimationResult:
    """Kết quả ước lượng số hạt."""
    final: Optional[int] = None
    regression_est: Optional[float] = None
    geometry_est: Optional[int] = None
    weight_est: Optional[int] = None
    method_used: str = "none"
    feature_schema_version: str = "31v1"
    model_bundle: Optional[str] = None
    # Alias cho tương thích client cũ
    ai_est: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "final": self.final,
            "regression_est": self.regression_est,
            "geometry_est": self.geometry_est,
            "weight_est": self.weight_est,
            "method_used": self.method_used,
            "feature_schema_version": self.feature_schema_version,
        }
        if self.model_bundle:
            d["model_bundle"] = self.model_bundle
        # Giữ ai_est = geometry_est cho client cũ
        d["ai_est"] = self.ai_est if self.ai_est is not None else self.geometry_est
        return d


@dataclass
class TimingInfo:
    """Thời gian xử lý theo từng giai đoạn (ms)."""
    total_ms: float = 0.0
    decode_ms: float = 0.0
    container_ms: float = 0.0
    segmentation_ms: float = 0.0
    cleaning_ms: float = 0.0
    classification_ms: float = 0.0
    geometry_ms: float = 0.0
    regression_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 1) for k, v in asdict(self).items()}


@dataclass
class PredictResponse:
    """Response chuẩn của /predict."""
    status: str = "success"
    request_id: Optional[str] = None
    estimation: Optional[EstimationResult] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    features_used: Optional[Dict[str, Optional[float]]] = None
    warnings: List[str] = field(default_factory=list)
    timings_ms: Optional[TimingInfo] = None
    error: Optional[ErrorDetail] = None
    debug_info: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status}
        if self.request_id:
            d["request_id"] = self.request_id
        if self.estimation:
            d["estimation"] = self.estimation.to_dict()
        if self.metrics_summary:
            d["metrics_summary"] = self.metrics_summary
        if self.features_used:
            # Hỗ trợ an toàn giá trị None (không làm tròn None)
            d["features_used"] = {
                k: (round(v, 4) if v is not None else None)
                for k, v in self.features_used.items()
            }
        if self.warnings:
            d["warnings"] = self.warnings
        if self.timings_ms:
            d["timings_ms"] = self.timings_ms.to_dict()
        if self.error:
            d["error"] = self.error.to_dict()
        if self.debug_info is not None:
            d["debug_info"] = self.debug_info
        if self.artifacts is not None:
            d["artifacts"] = self.artifacts
        return d


@dataclass
class ComponentInfo:
    """Trạng thái một component."""
    status: str = "not_found"
    path: Optional[str] = None  # Chỉ trả tên file hoặc basename, không lộ absolute path

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status}
        if self.path:
            d["file"] = self.path
        return d


@dataclass
class StatusResponse:
    """Response chuẩn của /api/status."""
    service: str = "Rice Vision AI Inference API"
    readiness: str = "not_ready"
    version: str = "2.2.0"
    schema_version: str = "31v1"
    bundle_id: Optional[str] = None
    components: Dict[str, ComponentInfo] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "readiness": self.readiness,
            "version": self.version,
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "capabilities": self.capabilities,
        }
