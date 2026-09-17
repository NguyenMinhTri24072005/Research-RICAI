"""Backward compatibility shim for feature_schema.

This module re-exports all symbols from `rice_ai.estimation.feature_schema`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from rice_ai.estimation.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    TRAINED_HYBRID_PACKING_FRACTION,
    compute_trained_hybrid_feature,
    compute_schema_hash,
    FeatureDef,
    FEATURE_DEFS,
    ALL_31_FEATURES,
    GRAIN_STAT_GROUPS,
    GRAIN_STAT_FEATURES,
    CONTAINER_FEATURES,
    FeatureValidationResult,
    validate_feature_vector,
    feature_vector_to_ordered_list,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "TARGET_COLUMN",
    "TRAINED_HYBRID_PACKING_FRACTION",
    "compute_trained_hybrid_feature",
    "compute_schema_hash",
    "FeatureDef",
    "FEATURE_DEFS",
    "ALL_31_FEATURES",
    "GRAIN_STAT_GROUPS",
    "GRAIN_STAT_FEATURES",
    "CONTAINER_FEATURES",
    "FeatureValidationResult",
    "validate_feature_vector",
    "feature_vector_to_ordered_list",
]
