"""Audit the supplied raw 450-grain image without modifying production code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
IMAGE = OUT / "actual_case_raw.jpg"

INNER_DIAM_MM = 32.7
CONTAINER_HEIGHT_MM = 48.7
EMPTY_HEIGHT_MM = 23.3
TRUE_RICE_DIAM_MM = 22.0
PACKING_FRACTION = 0.55
SAVED_NOTEBOOK_SCALE = 88.84
SAVED_MEAN_GRAIN_VOLUME_MM3 = 6.618
ACTUAL_COUNT = 450


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact(result: dict) -> dict:
    keep = (
        "pixels_per_mm", "inner_w_px", "outer_w_px", "container_w_px",
        "bulk_rice_volume_mm3", "rice_height_mm", "detect_type",
        "outer_confidence", "inner_confidence", "crop_bbox",
    )
    return {key: result.get(key) for key in keep}


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image for {path}")
    encoded.tofile(str(path))


def run_detector(label: str, detector, repetitions: int = 5) -> dict:
    runs = []
    first = None
    for _ in range(repetitions):
        result = detector(
            image_input=IMAGE,
            inner_diam_mm=INNER_DIAM_MM,
            container_height_mm=CONTAINER_HEIGHT_MM,
            empty_height_mm=EMPTY_HEIGHT_MM,
            detect_mode="inner",
        )
        if first is None:
            first = result
        runs.append(compact(result))

    assert first is not None
    write_image(OUT / f"actual_case_{label}_overlay.jpg", first["overlay_bgr"])
    write_image(OUT / f"actual_case_{label}_crop.jpg", first["raw_cropped_bgr"])
    scales = [float(item["pixels_per_mm"]) for item in runs]
    mean_scale = statistics.mean(scales)
    ratio_to_saved = mean_scale / SAVED_NOTEBOOK_SCALE
    corrected_volume = SAVED_MEAN_GRAIN_VOLUME_MM3 * ratio_to_saved**3
    corrected_count = round(
        math.pi * (TRUE_RICE_DIAM_MM / 2.0) ** 2
        * (CONTAINER_HEIGHT_MM - EMPTY_HEIGHT_MM)
        * PACKING_FRACTION
        / corrected_volume
    )
    return {
        "runs": runs,
        "scale_mean": mean_scale,
        "scale_range": max(scales) - min(scales),
        "scale_population_std": statistics.pstdev(scales),
        "ratio_to_saved_notebook_scale": ratio_to_saved,
        "grain_volume_counterfactual_from_scale_only_mm3": corrected_volume,
        "physical_count_counterfactual_from_scale_only": corrected_count,
    }


def main() -> None:
    image = read_image(IMAGE)
    if image is None:
        raise RuntimeError(f"Cannot read {IMAGE}")

    notebook_detector = load_module(
        "notebook_container_detector",
        ROOT / "CODE/modules/container_detector.py",
    )
    service_detector = load_module(
        "service_container_detector",
        ROOT / "AI_SERVICES/src/rice_ai/vision/container_detector.py",
    )

    with IMAGE.open("rb") as handle:
        image_hash = hashlib.sha256(handle.read()).hexdigest()

    bulk_volume = math.pi * (TRUE_RICE_DIAM_MM / 2.0) ** 2 * (
        CONTAINER_HEIGHT_MM - EMPTY_HEIGHT_MM
    )
    result = {
        "image": str(IMAGE.relative_to(ROOT)),
        "sha256": image_hash,
        "shape_hwc": list(image.shape),
        "inputs": {
            "inner_diameter_mm": INNER_DIAM_MM,
            "true_rice_diameter_mm": TRUE_RICE_DIAM_MM,
            "container_height_mm": CONTAINER_HEIGHT_MM,
            "empty_height_mm": EMPTY_HEIGHT_MM,
            "packing_fraction": PACKING_FRACTION,
            "actual_count": ACTUAL_COUNT,
        },
        "bulk_volume_recomputed_mm3": bulk_volume,
        "saved_notebook": {
            "pixels_per_mm": SAVED_NOTEBOOK_SCALE,
            "mean_grain_volume_mm3": SAVED_MEAN_GRAIN_VOLUME_MM3,
            "physical_count": round(
                bulk_volume * PACKING_FRACTION / SAVED_MEAN_GRAIN_VOLUME_MM3
            ),
        },
        "notebook_detector_current": run_detector(
            "notebook_detector", notebook_detector.detect_container_and_scale
        ),
        "ai_services_detector_current": run_detector(
            "ai_services_detector", service_detector.detect_container_and_scale
        ),
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / "actual_raw_audit.json").write_text(text, encoding="utf-8")
    (OUT / "actual_raw_audit.log").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
