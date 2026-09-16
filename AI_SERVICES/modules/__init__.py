"""
Rice Vision AI — Production Modules (Self-Contained Package)
"""

from .container_detector import detect_container_and_scale, draw_container_overlay
from .grain_segmenter import segment_grains_sahi
from .grain_classifier import GrainClassifier
from .ellipsoid_geometry import (
    compute_single_grain_metrics,
    compute_folder_grains_summary,
    draw_grain_ellipse_overlay,
)
from .uniformity_evaluator import evaluate_batch_uniformity
from .grain_crop_cleaner import (
    GrainCropCleaner,
    clean_single_grain_crop,
    clean_dataset_cropped_grains,
)

__all__ = [
    "detect_container_and_scale",
    "draw_container_overlay",
    "segment_grains_sahi",
    "GrainClassifier",
    "compute_single_grain_metrics",
    "compute_folder_grains_summary",
    "draw_grain_ellipse_overlay",
    "evaluate_batch_uniformity",
    "GrainCropCleaner",
    "clean_single_grain_crop",
    "clean_dataset_cropped_grains",
]
