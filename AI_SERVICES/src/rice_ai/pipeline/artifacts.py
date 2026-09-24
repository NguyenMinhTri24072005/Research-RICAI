"""Persistent, JSON-safe inference artifacts for Colab and local runs."""
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import cv2
import numpy as np

from rice_ai.contracts import PipelineResult


def _json_safe(value: Any) -> Any:
    """Convert pipeline values to JSON-safe primitives without hiding errors."""
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in {"crop_rgba", "crop_bgr", "mask_uint8"}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    safe_records = [_json_safe(record) for record in records]
    path.write_text(json.dumps(safe_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(safe_records)


def _write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    safe_records = [_json_safe(record) for record in records]
    keys = sorted({key for record in safe_records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["record"])
        writer.writeheader()
        for record in safe_records:
            writer.writerow(record if keys else {"record": ""})


def _write_image(path: Path, image: Any) -> bool:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


class InferenceArtifactWriter:
    """Write one complete, request-scoped result directory.

    The writer uses a temporary sibling directory and renames it only after
    all files are present. A result is therefore either complete or visibly
    incomplete; consumers never mistake a partially written request for a
    successful inference.
    """

    def __init__(self, root: Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def write(
        self,
        result: PipelineResult,
        source_image: np.ndarray,
        request_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "request_id": result.request_id}

        self.root.mkdir(parents=True, exist_ok=True)
        final_dir = self.root / result.request_id
        temp_dir = self.root / f".{result.request_id}.partial"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        try:
            input_dir = temp_dir / "input"
            reports_dir = temp_dir / "reports"
            input_dir.mkdir()
            reports_dir.mkdir()
            _write_image(input_dir / "original.jpg", source_image)
            (input_dir / "request.json").write_text(
                json.dumps(_json_safe(dict(request_payload or {})), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            container_overlay = result.container.raw_dict.get("overlay_bgr")
            if container_overlay is None:
                container_overlay = result.container.raw_dict.get("visual_overlay")
            _write_image(temp_dir / "1_container_detection.jpg", container_overlay)
            cropped_container = result.container.raw_dict.get("cropped_bgr")
            if not isinstance(cropped_container, np.ndarray):
                cropped_container = result.container.raw_dict.get("raw_cropped_bgr")
            _write_image(temp_dir / "1_cropped_container.jpg", cropped_container)

            grains = result.grain_analysis
            raw_dir = temp_dir / "1_raw_crops"
            clean_dir = temp_dir / "2_cleaned_crops"
            classified_root = temp_dir / "classified"
            raw_dir.mkdir()
            clean_dir.mkdir()
            (classified_root / "hat_nguyen").mkdir(parents=True)
            (classified_root / "hat_khuyet_tat").mkdir(parents=True)

            raw_count = 0
            for item in getattr(grains, "raw_crops", []) or []:
                image = item.get("crop_rgba")
                if _write_image(raw_dir / f"grain_{raw_count:04d}.png", image):
                    raw_count += 1
            clean_count = 0
            for item in getattr(grains, "cleaned_crops", []) or []:
                image = item.get("crop_rgba")
                label = str(item.get("predicted_label") or item.get("label") or "unknown")
                if _write_image(clean_dir / f"grain_{clean_count:04d}.png", image):
                    _write_image(classified_root / label / f"grain_{clean_count:04d}.png", image)
                    clean_count += 1

            _write_records(reports_dir / "whole_grains.json", grains.whole_grains)
            # Keep a compact classification chart when matplotlib is available.
            try:
                import matplotlib.pyplot as plt
                labels = list(grains.classified_counts.keys())
                values = [int(grains.classified_counts[label]) for label in labels]
                if sum(values) > 0:
                    figure, axis = plt.subplots(figsize=(5, 4))
                    axis.pie(values, labels=labels, autopct="%1.1f%%")
                    axis.set_title("Grain classification")
                    figure.savefig(temp_dir / "2_classification_pie_chart.png", dpi=140, bbox_inches="tight")
                    plt.close(figure)
            except Exception:
                pass

            detection = source_image.copy()
            for item in grains.broken_grains + grains.whole_grains:
                bbox = item.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255) if item in grains.broken_grains else (0, 180, 0)
                    cv2.rectangle(detection, (x1, y1), (x2, y2), color, 2)
            _write_image(temp_dir / "6_full_detection_result.jpg", detection)
            _write_csv(reports_dir / "grain_measurements.csv", grains.whole_grains)
            _write_csv(reports_dir / "grain_size_filter_decisions.csv", grains.size_filter_rejected)
            _write_csv(reports_dir / "physical_grains.csv", grains.physical_grains)
            _write_csv(reports_dir / "final_summary_report.csv", [{
                "request_id": result.request_id,
                "total_detected": grains.total_detected,
                "whole_grains": len(grains.whole_grains),
                "physical_grains": len(grains.physical_grains),
                "geometry_est": result.estimates.geometry_est,
                "regression_est": result.estimates.regression_est,
                "final": result.estimates.final,
            }])
            _write_csv(reports_dir / "regression_input_features.csv", [result.features_31 or {}])
            _write_csv(reports_dir / "final_prediction_summary.csv", [{
                "request_id": result.request_id,
                **_json_safe(result.estimates.__dict__),
            }])
            (reports_dir / "timings.json").write_text(json.dumps(_json_safe(result.timings_ms), indent=2), encoding="utf-8")
            (reports_dir / "result.json").write_text(json.dumps(_json_safe({
                "request_id": result.request_id,
                "estimates": result.estimates.__dict__,
                "container": result.container.__dict__,
                "warnings": result.warnings,
                "features_31": result.features_31,
            }), ensure_ascii=False, indent=2), encoding="utf-8")

            manifest = {
                "request_id": result.request_id,
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "relative_path": str(final_dir.relative_to(self.root)),
                "files": sorted(str(p.relative_to(temp_dir)).replace("\\", "/") for p in temp_dir.rglob("*") if p.is_file()),
                "counts": {
                    "raw_crops_saved": raw_count,
                    "cleaned_crops_saved": clean_count,
                    "total_detected": grains.total_detected,
                    "whole_grains": len(grains.whole_grains),
                    "physical_grains": len(grains.physical_grains),
                },
            }
            (reports_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (temp_dir / "COMPLETED").write_text("completed\\n", encoding="utf-8")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)
            manifest["path"] = str(final_dir)
            manifest["download_path"] = f"/api/results/{result.request_id}/download"
            (final_dir / "reports" / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return manifest
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def write_failure(self, request_id: str, error: Exception, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "request_id": request_id}
        directory = self.root / request_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "error.json").write_text(json.dumps({
            "request_id": request_id,
            "status": "failed",
            "error": str(error),
            "request": _json_safe(dict(payload or {})),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"request_id": request_id, "status": "failed", "path": str(directory)}
