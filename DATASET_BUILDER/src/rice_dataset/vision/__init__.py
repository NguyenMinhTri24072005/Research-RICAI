"""Vision modules vendored from CODE/modules for dataset extraction parity."""

from .container_detector import (
    detect_container_and_scale,
    draw_container_overlay,
    ContainerDetectionError,
)
from .grain_segmenter import (
    segment_grains_sahi,
)
from .grain_crop_cleaner import (
    clean_single_grain_crop,
)
from .grain_classifier import (
    GrainClassifier,
)
from .ellipsoid_geometry import (
    compute_single_grain_metrics,
    draw_grain_ellipse_overlay,
    THICKNESS_RATIO,
)
from .uniformity_evaluator import (
    evaluate_batch_uniformity,
)
from .grain_size_filter import (
    filter_grains_by_size,
)

__all__ = [
    "detect_container_and_scale",
    "draw_container_overlay",
    "ContainerDetectionError",
    "segment_grains_sahi",
    "clean_single_grain_crop",
    "GrainClassifier",
    "compute_single_grain_metrics",
    "draw_grain_ellipse_overlay",
    "THICKNESS_RATIO",
    "evaluate_batch_uniformity",
    "filter_grains_by_size",
]
