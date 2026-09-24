"""Extraction pipeline configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CleanerConfig:
    open_ksize: int = 5
    min_neck_ratio: float = 0.15
    min_area: int = 35
    centrality_weight: float = 2.5
    fill_holes: bool = False
    sever_bridges: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open_ksize": self.open_ksize,
            "min_neck_ratio": self.min_neck_ratio,
            "min_area": self.min_area,
            "centrality_weight": self.centrality_weight,
            "fill_holes": self.fill_holes,
            "sever_bridges": self.sever_bridges,
        }


@dataclass
class ExtractionConfig:
    contract_version: str = "code-main-2026-09-22"
    id_prefix: str = "M"
    id_digits: int = 4
    project_root: Path = field(default_factory=lambda: Path("."))
    workbook_path: Path = field(default_factory=lambda: Path("DATASET_BUILDER/2_Manual_Records/manual_data.xlsx"))
    raw_images_dir: Path = field(default_factory=lambda: Path("DATASET_BUILDER/1_Raw_Images"))
    yolo_model_path: Path = field(default_factory=lambda: Path("RESULTS/all-new-data-v1.yolov8_yolov8s-seg_trained/weights/best.pt"))
    cnn_model_path: Path = field(default_factory=lambda: Path("RESULTS/CNN_DenseNet121_Trained/best_v3_step2.keras"))
    audit_dataset_output_path: Path = field(default_factory=lambda: Path("DATASET_BUILDER/3_AI_Extracted/ai_extracted_dataset.csv"))
    final_dataset_csv_path: Path = field(default_factory=lambda: Path("DATASET_BUILDER/4_Final_Dataset/final_regression_dataset.csv"))
    final_dataset_xlsx_path: Path = field(default_factory=lambda: Path("DATASET_BUILDER/4_Final_Dataset/final_regression_dataset.xlsx"))

    # SAHI / YOLO
    yolo_model_load_confidence: float = 0.70
    yolo_result_confidence: float = 0.50
    sahi_slice_size: int = 640
    sahi_overlap_ratio: float = 0.25
    sahi_min_area_px: int = 50

    # CNN
    cnn_whole_confidence: float = 0.90

    # Two-pass cleaner
    clean_step1: CleanerConfig = field(default_factory=lambda: CleanerConfig(
        open_ksize=5, min_neck_ratio=0.15, min_area=35, centrality_weight=2.5, fill_holes=False, sever_bridges=False
    ))
    clean_step2: CleanerConfig = field(default_factory=lambda: CleanerConfig(
        open_ksize=3, min_neck_ratio=0.15, min_area=25, centrality_weight=2.2, fill_holes=True, sever_bridges=False
    ))

    # Size filter
    size_filter_enabled: bool = True
    size_filter_k: float = 0.1
    size_filter_min_samples: int = 8

    # Physics / constants
    uniformity_threshold: float = 0.80
    physical_packing_fraction: float = 0.55
    trained_hybrid_packing_fraction: float = 0.62
    whole_grain_thickness_ratio: float = 0.80
    container_edge: str = "inner"
    apply_depth_correction: bool = False

    def resolve_path(self, path: Path | str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def get_resolved_workbook_path(self) -> Path:
        return self.resolve_path(self.workbook_path)

    def get_resolved_raw_images_dir(self) -> Path:
        return self.resolve_path(self.raw_images_dir)

    def get_resolved_yolo_model_path(self) -> Path:
        return self.resolve_path(self.yolo_model_path)

    def get_resolved_cnn_model_path(self) -> Path:
        return self.resolve_path(self.cnn_model_path)

    def get_resolved_audit_output_path(self) -> Path:
        return self.resolve_path(self.audit_dataset_output_path)

    def get_resolved_final_csv_path(self) -> Path:
        return self.resolve_path(self.final_dataset_csv_path)

    def get_resolved_final_xlsx_path(self) -> Path:
        return self.resolve_path(self.final_dataset_xlsx_path)

    def validate(self) -> None:
        """Validate all parameters against mandatory scientific constraints."""
        if self.yolo_model_load_confidence != 0.70:
            raise ValueError(f"Invalid yolo_model_load_confidence: {self.yolo_model_load_confidence}, expected 0.70")
        if self.yolo_result_confidence != 0.50:
            raise ValueError(f"Invalid yolo_result_confidence: {self.yolo_result_confidence}, expected 0.50")
        if self.sahi_slice_size != 640:
            raise ValueError(f"Invalid sahi_slice_size: {self.sahi_slice_size}, expected 640")
        if self.sahi_overlap_ratio != 0.25:
            raise ValueError(f"Invalid sahi_overlap_ratio: {self.sahi_overlap_ratio}, expected 0.25")
        if self.sahi_min_area_px != 50:
            raise ValueError(f"Invalid sahi_min_area_px: {self.sahi_min_area_px}, expected 50")
        if self.cnn_whole_confidence != 0.90:
            raise ValueError(f"Invalid cnn_whole_confidence: {self.cnn_whole_confidence}, expected 0.90")

        if self.clean_step1.sever_bridges is not False or self.clean_step2.sever_bridges is not False:
            raise ValueError("sever_bridges must be False for both cleaning steps")
        if self.clean_step1.min_neck_ratio != 0.15 or self.clean_step2.min_neck_ratio != 0.15:
            raise ValueError("min_neck_ratio must be 0.15 for both cleaning steps")

        if self.physical_packing_fraction != 0.55:
            raise ValueError(f"Invalid physical_packing_fraction: {self.physical_packing_fraction}, expected 0.55")
        if self.trained_hybrid_packing_fraction != 0.62:
            raise ValueError(f"Invalid trained_hybrid_packing_fraction: {self.trained_hybrid_packing_fraction}, expected 0.62")
        if self.whole_grain_thickness_ratio != 0.80:
            raise ValueError(f"Invalid whole_grain_thickness_ratio: {self.whole_grain_thickness_ratio}, expected 0.80")

    @classmethod
    def from_file(cls, config_path: Path | str, project_root_override: Optional[Path | str] = None) -> ExtractionConfig:
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        project_root = Path(project_root_override) if project_root_override is not None else Path(data.get("project_root", "."))

        clean1_data = data.get("clean_step1", {})
        clean1 = CleanerConfig(
            open_ksize=int(clean1_data.get("open_ksize", 5)),
            min_neck_ratio=float(clean1_data.get("min_neck_ratio", 0.15)),
            min_area=int(clean1_data.get("min_area", 35)),
            centrality_weight=float(clean1_data.get("centrality_weight", 2.5)),
            fill_holes=bool(clean1_data.get("fill_holes", False)),
            sever_bridges=bool(clean1_data.get("sever_bridges", False)),
        )

        clean2_data = data.get("clean_step2", {})
        clean2 = CleanerConfig(
            open_ksize=int(clean2_data.get("open_ksize", 3)),
            min_neck_ratio=float(clean2_data.get("min_neck_ratio", 0.15)),
            min_area=int(clean2_data.get("min_area", 25)),
            centrality_weight=float(clean2_data.get("centrality_weight", 2.2)),
            fill_holes=bool(clean2_data.get("fill_holes", True)),
            sever_bridges=bool(clean2_data.get("sever_bridges", False)),
        )

        cfg = cls(
            contract_version=str(data.get("contract_version", "code-main-2026-09-22")),
            id_prefix=str(data.get("id_prefix", "M")),
            id_digits=int(data.get("id_digits", 4)),
            project_root=project_root,
            workbook_path=Path(data.get("workbook_path", "DATASET_BUILDER/2_Manual_Records/manual_data.xlsx")),
            raw_images_dir=Path(data.get("raw_images_dir", "DATASET_BUILDER/1_Raw_Images")),
            yolo_model_path=Path(data.get("yolo_model_path", "RESULTS/all-new-data-v1.yolov8_yolov8s-seg_trained/weights/best.pt")),
            cnn_model_path=Path(data.get("cnn_model_path", "RESULTS/CNN_DenseNet121_Trained/best_v3_step2.keras")),
            audit_dataset_output_path=Path(data.get("audit_dataset_output_path", "DATASET_BUILDER/3_AI_Extracted/ai_extracted_dataset.csv")),
            final_dataset_csv_path=Path(data.get("final_dataset_csv_path", "DATASET_BUILDER/4_Final_Dataset/final_regression_dataset.csv")),
            final_dataset_xlsx_path=Path(data.get("final_dataset_xlsx_path", "DATASET_BUILDER/4_Final_Dataset/final_regression_dataset.xlsx")),
            yolo_model_load_confidence=float(data.get("yolo_model_load_confidence", 0.70)),
            yolo_result_confidence=float(data.get("yolo_result_confidence", 0.50)),
            sahi_slice_size=int(data.get("sahi_slice_size", 640)),
            sahi_overlap_ratio=float(data.get("sahi_overlap_ratio", 0.25)),
            sahi_min_area_px=int(data.get("sahi_min_area_px", 50)),
            cnn_whole_confidence=float(data.get("cnn_whole_confidence", 0.90)),
            clean_step1=clean1,
            clean_step2=clean2,
            size_filter_enabled=bool(data.get("size_filter_enabled", True)),
            size_filter_k=float(data.get("size_filter_k", 0.1)),
            size_filter_min_samples=int(data.get("size_filter_min_samples", 8)),
            uniformity_threshold=float(data.get("uniformity_threshold", 0.80)),
            physical_packing_fraction=float(data.get("physical_packing_fraction", 0.55)),
            trained_hybrid_packing_fraction=float(data.get("trained_hybrid_packing_fraction", 0.62)),
            whole_grain_thickness_ratio=float(data.get("whole_grain_thickness_ratio", 0.80)),
            container_edge=str(data.get("container_edge", "inner")),
            apply_depth_correction=bool(data.get("apply_depth_correction", False)),
        )

        cfg.validate()
        return cfg
