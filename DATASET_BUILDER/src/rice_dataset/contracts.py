"""Feature, lineage, status and output schema contracts for Rice Dataset Extraction."""

from __future__ import annotations

from typing import Final, List, Set

SCHEMA_VERSION: Final[str] = "31v1"
CONTRACT_VERSION: Final[str] = "code-main-2026-09-22"

ORDERED_FEATURES: Final[List[str]] = [
    "Bulk_Rice_Volume_mm3",
    "Rice_Height_mm",
    "Weight_g",
    "Empty_Height_mm",
    "Pixels_Per_mm",
    "Container_Detected_Diam_px",
    "Inner_Diameter_mm",
    "Container_Height_mm",
    "Whole_Grains_Count",
    "Uniformity_Rate_Pct",
    "Estimated_Total_Seeds_Hybrid",
    "Grain_Length_mm_Mean",
    "Grain_Length_mm_Min",
    "Grain_Length_mm_Max",
    "Grain_Length_mm_Std",
    "Grain_Width_mm_Mean",
    "Grain_Width_mm_Min",
    "Grain_Width_mm_Max",
    "Grain_Width_mm_Std",
    "Grain_Thickness_mm_Mean",
    "Grain_Thickness_mm_Min",
    "Grain_Thickness_mm_Max",
    "Grain_Thickness_mm_Std",
    "Grain_Area_mm2_Mean",
    "Grain_Area_mm2_Min",
    "Grain_Area_mm2_Max",
    "Grain_Area_mm2_Std",
    "Grain_Volume_mm3_Mean",
    "Grain_Volume_mm3_Min",
    "Grain_Volume_mm3_Max",
    "Grain_Volume_mm3_Std",
]

LINEAGE_COLUMNS: Final[List[str]] = [
    "Sample_ID",
    "Original_Sample_ID",
    "Physical_Sample_ID",
    "Image_Filename",
    "Capture_Timestamp",
    "Pipeline_Contract_Version",
]

STATUS_COLUMNS: Final[List[str]] = [
    "Image_Status",
    "QC_Status",
    "QC_Reason",
]

DIAGNOSTIC_COLUMNS: Final[List[str]] = [
    "Physical_Estimated_Seeds",
    "Whole_Grains_Filtered_Count",
    "Whole_Grains_Raw_Count",
    "Mean_Clean_Grain_Volume_mm3",
    "Mean_Raw_Grain_Volume_mm3",
    "Scale_Pixels_Per_mm",
]

LABEL_COLUMN: Final[str] = "Actual_Count"

CANONICAL_COLUMNS: Final[List[str]] = (
    LINEAGE_COLUMNS
    + STATUS_COLUMNS
    + ORDERED_FEATURES
    + DIAGNOSTIC_COLUMNS
    + [LABEL_COLUMN]
)


class ImageStatus:
    FOUND: Final[str] = "FOUND"
    MISSING: Final[str] = "MISSING"


class QCStatus:
    PASS: Final[str] = "PASS"
    MISSING_IMAGE: Final[str] = "MISSING_IMAGE"
    DUPLICATE_IMAGE: Final[str] = "DUPLICATE_IMAGE"
    INVALID_MANUAL_DATA: Final[str] = "INVALID_MANUAL_DATA"
    IMAGE_DECODE_FAILED: Final[str] = "IMAGE_DECODE_FAILED"
    CONTAINER_DETECTION_FAILED: Final[str] = "CONTAINER_DETECTION_FAILED"
    INVALID_SCALE: Final[str] = "INVALID_SCALE"
    NO_SEGMENTS: Final[str] = "NO_SEGMENTS"
    NO_CNN_WHOLE_GRAINS: Final[str] = "NO_CNN_WHOLE_GRAINS"
    NO_VALID_GRAIN_MEASUREMENTS: Final[str] = "NO_VALID_GRAIN_MEASUREMENTS"
    FEATURE_VALIDATION_FAILED: Final[str] = "FEATURE_VALIDATION_FAILED"
    MODEL_LOAD_FAILED: Final[str] = "MODEL_LOAD_FAILED"
    UNEXPECTED_ERROR: Final[str] = "UNEXPECTED_ERROR"


ALL_QC_STATUSES: Final[Set[str]] = {
    QCStatus.PASS,
    QCStatus.MISSING_IMAGE,
    QCStatus.DUPLICATE_IMAGE,
    QCStatus.INVALID_MANUAL_DATA,
    QCStatus.IMAGE_DECODE_FAILED,
    QCStatus.CONTAINER_DETECTION_FAILED,
    QCStatus.INVALID_SCALE,
    QCStatus.NO_SEGMENTS,
    QCStatus.NO_CNN_WHOLE_GRAINS,
    QCStatus.NO_VALID_GRAIN_MEASUREMENTS,
    QCStatus.FEATURE_VALIDATION_FAILED,
    QCStatus.MODEL_LOAD_FAILED,
    QCStatus.UNEXPECTED_ERROR,
}
