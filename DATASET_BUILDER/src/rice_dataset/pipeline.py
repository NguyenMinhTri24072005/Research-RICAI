"""End-to-end standalone batch dataset extraction pipeline."""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import ExtractionConfig
from .contracts import (
    CANONICAL_COLUMNS,
    CONTRACT_VERSION,
    ImageStatus,
    QCStatus,
)
from .feature_builder import build_physical_diagnostics, build_regression_features
from .io.dataset_writer import export_audit_and_training_datasets
from .io.image_index import ImageIndex
from .io.manual_records import ManualRecord, load_manual_records
from .vision.container_detector import ContainerDetectionError, detect_container_and_scale
from .vision.ellipsoid_geometry import compute_single_grain_metrics
from .vision.grain_classifier import GrainClassifier
from .vision.grain_crop_cleaner import clean_single_grain_crop
from .vision.grain_segmenter import segment_grains_sahi
from .vision.grain_size_filter import filter_grains_by_size


class DatasetExtractionPipeline:
    """Orchestrates container detection, SAHI segmentation, cleaning, CNN classification,

    geometry measurement and 31-feature generation for tabular dataset building.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.config.validate()

        self.sahi_model: Optional[Any] = None
        self.cnn_classifier: Optional[GrainClassifier] = None
        self.image_index: Optional[ImageIndex] = None

    def initialize_models(self) -> None:
        """Initialize YOLO and CNN models lazily."""
        if self.sahi_model is None:
            from sahi import AutoDetectionModel

            yolo_path = self.config.get_resolved_yolo_model_path()
            if not yolo_path.exists():
                raise FileNotFoundError(f"YOLO model weights not found: {yolo_path}")

            print(f"📦 Loading SAHI YOLO model from: {yolo_path}...")
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(yolo_path),
                confidence_threshold=self.config.yolo_model_load_confidence,
                device="cuda:0" if self._has_cuda() else "cpu",
            )
            print("✅ SAHI YOLO model loaded successfully.")

        if self.cnn_classifier is None:
            cnn_path = self.config.get_resolved_cnn_model_path()
            if not cnn_path.exists():
                raise FileNotFoundError(f"CNN model weights not found: {cnn_path}")

            print(f"📦 Loading DenseNet121 CNN model from: {cnn_path}...")
            self.cnn_classifier = GrainClassifier(model_path=cnn_path)
            print("✅ DenseNet121 CNN model loaded successfully.")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def get_image_index(self) -> ImageIndex:
        if self.image_index is None:
            raw_dir = self.config.get_resolved_raw_images_dir()
            self.image_index = ImageIndex(raw_dir, recursive=False)
        return self.image_index
    def validate_input_inventory(
        self,
        records: List[ManualRecord],
        require_exact_match: bool = True,
    ) -> Dict[str, int]:
        """Validate the canonical one-row/one-image flat input contract."""
        index = self.get_image_index()
        issues = index.validate_flat_layout(
            prefix=self.config.id_prefix,
            digits=self.config.id_digits,
        )

        expected_id = re.compile(
            rf"^{re.escape(self.config.id_prefix)}\d{{{self.config.id_digits}}}$",
            re.IGNORECASE,
        )
        record_ids = [record.sample_id.strip().upper() for record in records]
        invalid_record_ids = sorted({
            sample_id for sample_id in record_ids if expected_id.fullmatch(sample_id) is None
        })
        if invalid_record_ids:
            issues.append(
                f"Workbook contains {len(invalid_record_ids)} Sample_ID value(s) outside the "
                f"{self.config.id_prefix}{'0' * self.config.id_digits} contract: "
                + ", ".join(invalid_record_ids[:10])
            )

        record_counts: Dict[str, int] = {}
        for sample_id in record_ids:
            record_counts[sample_id] = record_counts.get(sample_id, 0) + 1
        duplicate_record_ids = sorted(
            sample_id for sample_id, count in record_counts.items() if count > 1
        )
        if duplicate_record_ids:
            issues.append(
                "Workbook contains duplicate Sample_ID value(s): "
                + ", ".join(duplicate_record_ids[:10])
            )

        image_ids = index.get_all_stems()
        record_id_set = set(record_ids)
        if require_exact_match:
            missing_images = sorted(record_id_set - image_ids)
            unreferenced_images = sorted(image_ids - record_id_set)
            if missing_images:
                issues.append(
                    f"{len(missing_images)} workbook row(s) have no matching flat image: "
                    + ", ".join(missing_images[:10])
                )
            if unreferenced_images:
                issues.append(
                    f"{len(unreferenced_images)} flat image(s) have no matching workbook row: "
                    + ", ".join(unreferenced_images[:10])
                )

        if issues:
            raise ValueError("Raw input inventory is invalid:\n - " + "\n - ".join(issues))

        return {
            "record_count": len(record_ids),
            "unique_record_count": len(record_id_set),
            "image_count": index.get_total_image_count(),
        }


    def process_single_sample(
        self,
        record: ManualRecord,
        image_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute the exact 15-step extraction pipeline for one manual record."""
        timestamp = datetime.now().isoformat()
        row: Dict[str, Any] = {
            "Sample_ID": record.sample_id,
            "Original_Sample_ID": record.original_sample_id,
            "Physical_Sample_ID": record.physical_sample_id,
            "Image_Filename": record.image_filename,
            "Capture_Timestamp": record.capture_timestamp if record.capture_timestamp else timestamp,
            "Pipeline_Contract_Version": CONTRACT_VERSION,
            "Image_Status": ImageStatus.FOUND,
            "QC_Status": QCStatus.PASS,
            "QC_Reason": None,
            "Actual_Count": record.actual_count,
        }

        # Step 1: Resolve image if not explicitly provided
        if image_path is None:
            idx = self.get_image_index()
            resolved_p, err = idx.resolve_image(record.sample_id)
            if err == "DUPLICATE_IMAGE":
                row["Image_Status"] = ImageStatus.FOUND
                row["QC_Status"] = QCStatus.DUPLICATE_IMAGE
                row["QC_Reason"] = "Multiple matching image files detected on disk"
                return row
            if err == "MISSING_IMAGE" or not resolved_p:
                row["Image_Status"] = ImageStatus.MISSING
                row["QC_Status"] = QCStatus.MISSING_IMAGE
                row["QC_Reason"] = f"Raw image not found on disk for Sample_ID: {record.sample_id}"
                return row
            image_path = resolved_p

        row["Image_Filename"] = image_path.name

        # Step 2: Decode image
        if not image_path.exists():
            row["Image_Status"] = ImageStatus.MISSING
            row["QC_Status"] = QCStatus.MISSING_IMAGE
            row["QC_Reason"] = f"Image path does not exist: {image_path}"
            return row

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None or img_bgr.size == 0:
            row["QC_Status"] = QCStatus.IMAGE_DECODE_FAILED
            row["QC_Reason"] = f"Failed to decode image file: {image_path.name}"
            return row

        # Ensure models are loaded
        try:
            self.initialize_models()
        except Exception as e:
            row["QC_Status"] = QCStatus.MODEL_LOAD_FAILED
            row["QC_Reason"] = f"Failed to initialize vision models: {e}"
            return row

        # Step 3: Detect container and scale
        try:
            container_info = detect_container_and_scale(
                image_input=img_bgr,
                inner_diam_mm=record.inner_diameter_mm,
                container_height_mm=record.container_height_mm,
                empty_height_mm=record.empty_height_mm,
                apply_depth_correction=self.config.apply_depth_correction,
                verbose=False,
            )
        except ContainerDetectionError as e:
            row["QC_Status"] = QCStatus.CONTAINER_DETECTION_FAILED
            row["QC_Reason"] = f"Container detection error: {e}"
            return row
        except Exception as e:
            row["QC_Status"] = QCStatus.CONTAINER_DETECTION_FAILED
            row["QC_Reason"] = f"Unexpected container detector failure: {e}"
            return row

        if container_info.get("status") == "failed":
            row["QC_Status"] = QCStatus.CONTAINER_DETECTION_FAILED
            row["QC_Reason"] = f"Container detection rejected: {container_info.get('reason', 'unknown')}"
            return row

        pixels_per_mm = container_info.get("pixels_per_mm")
        if pixels_per_mm is None or not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
            row["QC_Status"] = QCStatus.INVALID_SCALE
            row["QC_Reason"] = f"Invalid pixels_per_mm scale computed: {pixels_per_mm}"
            return row

        bulk_volume_mm3 = container_info.get("bulk_rice_volume_mm3")
        rice_height_mm = container_info.get("rice_height_mm")
        inner_w_px = container_info.get("inner_w_px")

        if bulk_volume_mm3 is None or rice_height_mm is None or inner_w_px is None:
            row["QC_Status"] = QCStatus.CONTAINER_DETECTION_FAILED
            row["QC_Reason"] = "Container detection succeeded but missing required geometric dimensions."
            return row

        # Step 4: Run SAHI YOLO Segmentation on full raw image
        try:
            raw_grains = segment_grains_sahi(
                detection_model=self.sahi_model,
                image_path=img_bgr,
                output_crop_dir=None,
                conf_threshold=self.config.yolo_result_confidence,
                slice_size=self.config.sahi_slice_size,
                overlap_ratio=self.config.sahi_overlap_ratio,
                min_area_px=self.config.sahi_min_area_px,
            )
        except Exception as e:
            row["QC_Status"] = QCStatus.UNEXPECTED_ERROR
            row["QC_Reason"] = f"SAHI segmentation failed: {e}"
            return row

        if not raw_grains or len(raw_grains) == 0:
            row["QC_Status"] = QCStatus.NO_SEGMENTS
            row["QC_Reason"] = "SAHI segmentation detected zero grain segments."
            return row

        # Step 5: Two-pass cleaner
        cleaned_grains: List[Dict[str, Any]] = []
        c1_cfg = self.config.clean_step1.to_dict()
        c2_cfg = self.config.clean_step2.to_dict()

        for g in raw_grains:
            crop_rgba = g.get("crop_rgba")
            if crop_rgba is None:
                continue

            try:
                cleaned1 = clean_single_grain_crop(crop_rgba, **c1_cfg)
                if (
                    cleaned1 is None
                    or not isinstance(cleaned1, np.ndarray)
                    or len(cleaned1.shape) != 3
                    or cleaned1.shape[2] != 4
                    or cv2.countNonZero(cleaned1[:, :, 3]) == 0
                ):
                    continue

                cleaned2 = clean_single_grain_crop(cleaned1, **c2_cfg)
                if (
                    cleaned2 is None
                    or not isinstance(cleaned2, np.ndarray)
                    or len(cleaned2.shape) != 3
                    or cleaned2.shape[2] != 4
                    or cv2.countNonZero(cleaned2[:, :, 3]) == 0
                ):
                    continue

                grain_copy = dict(g)
                grain_copy["crop_rgba"] = cleaned2
                cleaned_grains.append(grain_copy)
            except Exception:
                continue

        if not cleaned_grains:
            row["QC_Status"] = QCStatus.NO_SEGMENTS
            row["QC_Reason"] = "All grain segments were rejected during two-pass cleaning."
            return row

        # Step 6: CNN classification
        try:
            assert self.cnn_classifier is not None
            whole_grains, _ = self.cnn_classifier.filter_grains(
                grains_list=cleaned_grains,
                target_label="hat_nguyen",
                min_conf=self.config.cnn_whole_confidence,
            )
        except Exception as e:
            row["QC_Status"] = QCStatus.UNEXPECTED_ERROR
            row["QC_Reason"] = f"CNN classification failed: {e}"
            return row

        if not whole_grains or len(whole_grains) == 0:
            row["QC_Status"] = QCStatus.NO_CNN_WHOLE_GRAINS
            row["QC_Reason"] = f"CNN detected 0 whole grains at confidence threshold {self.config.cnn_whole_confidence}."
            return row

        # Step 7: Measure 3D geometry for CNN-whole grains
        raw_valid_metrics: List[Dict[str, Any]] = []
        raw_volumes_list: List[float] = []

        for g in whole_grains:
            try:
                m = compute_single_grain_metrics(
                    image_input=g["crop_rgba"],
                    pixels_per_mm=pixels_per_mm,
                    label="hat_nguyen",
                    scale_factor=4,
                    thickness_k=self.config.whole_grain_thickness_ratio,
                )
                if (
                    m["length_mm"] > 0
                    and m["width_mm"] > 0
                    and m["thickness_mm"] > 0
                    and m["area_mm2"] > 0
                    and m["volume_mm3"] > 0
                    and math.isfinite(m["volume_mm3"])
                ):
                    raw_valid_metrics.append(m)
                    raw_volumes_list.append(float(m["volume_mm3"]))
            except Exception:
                continue

        if not raw_valid_metrics or len(raw_valid_metrics) == 0:
            row["QC_Status"] = QCStatus.NO_VALID_GRAIN_MEASUREMENTS
            row["QC_Reason"] = "Failed to extract valid finite 3D geometry for any whole grain."
            return row

        # Step 8: Build physical diagnostic population (filtered subset)
        try:
            filtered_res = filter_grains_by_size(
                measurements=raw_valid_metrics,
                k=self.config.size_filter_k,
                min_samples=self.config.size_filter_min_samples,
                enabled=self.config.size_filter_enabled,
            )
            filtered_metrics = filtered_res.get("kept", [])
            filtered_volumes_list = [
                float(m["volume_mm3"]) for m in filtered_metrics if "volume_mm3" in m and m["volume_mm3"] > 0
            ]
        except Exception:
            filtered_volumes_list = list(raw_volumes_list)

        # Step 9: Assemble 31 features
        try:
            features = build_regression_features(
                raw_valid_metrics=raw_valid_metrics,
                bulk_volume_mm3=bulk_volume_mm3,
                rice_height_mm=rice_height_mm,
                weight_g=record.weight_g,
                empty_height_mm=record.empty_height_mm,
                pixels_per_mm=pixels_per_mm,
                container_detected_diam_px=inner_w_px,
                inner_diameter_mm=record.inner_diameter_mm,
                container_height_mm=record.container_height_mm,
                trained_hybrid_packing_fraction=self.config.trained_hybrid_packing_fraction,
            )
        except Exception as e:
            row["QC_Status"] = QCStatus.FEATURE_VALIDATION_FAILED
            row["QC_Reason"] = f"Feature assembly failed: {e}"
            return row

        # Step 10: Assemble physical diagnostics
        try:
            diagnostics = build_physical_diagnostics(
                filtered_volumes=filtered_volumes_list,
                raw_volumes=raw_volumes_list,
                bulk_volume_mm3=bulk_volume_mm3,
                pixels_per_mm=pixels_per_mm,
                container_detected_diam_px=inner_w_px,
                physical_packing_fraction=self.config.physical_packing_fraction,
            )
        except Exception as e:
            row["QC_Status"] = QCStatus.FEATURE_VALIDATION_FAILED
            row["QC_Reason"] = f"Physical diagnostic assembly failed: {e}"
            return row

        row.update(features)
        row.update(diagnostics)
        row["QC_Status"] = QCStatus.PASS
        row["QC_Reason"] = None
        return row

    def run_batch(
        self,
        records: Optional[List[ManualRecord]] = None,
        limit: Optional[int] = None,
        export: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run batch extraction over manual records."""
        loader_err_rows: List[Dict[str, Any]] = []
        if records is None:
            wb_path = self.config.get_resolved_workbook_path()
            records, errs = load_manual_records(wb_path)
            if errs:
                print(f"⚠️ Warning: Found {len(errs)} invalid manual record rows during loading.")
                for err in errs:
                    loader_err_rows.append({
                        "Sample_ID": err.get("sample_id", "UNKNOWN"),
                        "Original_Sample_ID": err.get("original_sample_id", err.get("sample_id", "UNKNOWN")),
                        "Physical_Sample_ID": err.get("physical_sample_id", err.get("sample_id", "UNKNOWN")),
                        "Image_Filename": err.get("image_filename"),
                        "Capture_Timestamp": err.get("capture_timestamp"),
                        "Pipeline_Contract_Version": CONTRACT_VERSION,
                        "Image_Status": ImageStatus.FOUND if err.get("image_filename") else ImageStatus.MISSING,
                        "QC_Status": QCStatus.INVALID_MANUAL_DATA,
                        "QC_Reason": f"Invalid manual record at row {err.get('row_index', '?')}: {err.get('error', 'unknown error')}",
                        "Actual_Count": None,
                    })

        self.validate_input_inventory(records, require_exact_match=True)

        if limit is not None and limit > 0:
            records = records[:limit]

        total = len(records)
        print(f"🚀 Starting batch extraction for {total} sample records...")
        results: List[Dict[str, Any]] = []

        start_time = time.time()
        for i, rec in enumerate(records, start=1):
            t0 = time.time()
            res = self.process_single_sample(rec)
            elapsed = time.time() - t0
            results.append(res)

            status = res.get("QC_Status")
            reason = f" ({res.get('QC_Reason')})" if res.get("QC_Reason") else ""
            print(f"[{i:03d}/{total:03d}] {rec.sample_id} -> {status}{reason} [{elapsed:.2f}s]")

            if progress_callback:
                progress_callback(i, total, res)

        # Include invalid manual rows in audit output
        results.extend(loader_err_rows)

        total_elapsed = time.time() - start_time
        pass_count = sum(1 for r in results if r.get("QC_Status") == QCStatus.PASS)
        print("=" * 60)
        print(f"🏁 Batch extraction complete in {total_elapsed:.1f}s.")
        print(f"   Total rows: {len(results)} | PASS: {pass_count} | Non-PASS: {len(results) - pass_count}")
        print("=" * 60)

        if export:
            self.export_results(results)

        return results

    def export_results(self, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Export results to canonical audit CSV and training-ready CSV/XLSX."""
        audit_path = self.config.get_resolved_audit_output_path()
        final_csv = self.config.get_resolved_final_csv_path()
        final_xlsx = self.config.get_resolved_final_xlsx_path()

        counts = export_audit_and_training_datasets(
            rows=rows,
            audit_csv_path=audit_path,
            final_csv_path=final_csv,
            final_xlsx_path=final_xlsx,
        )
        print(f"💾 Exported datasets:")
        print(f"   - Audit CSV ({counts['audit_total_rows']} rows): {audit_path}")
        print(f"   - Final CSV ({counts['training_csv_rows']} rows): {final_csv}")
        print(f"   - Final XLSX ({counts['training_xlsx_rows']} rows): {final_xlsx}")
        return counts
