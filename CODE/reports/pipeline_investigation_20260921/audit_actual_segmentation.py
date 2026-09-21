"""CPU audit of SAHI, cleaner and geometry for the supplied raw image.

CNN classification is deliberately not claimed here because the local TensorFlow
installation cannot import. Production sources and model artifacts are read-only.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
IMAGE = OUT / "actual_case_raw.jpg"
YOLO = ROOT / "RESULTS/all-new-data-v1.yolov8_yolov8s-seg_trained/weights/best.pt"
PIXELS_PER_MM = 88.83847481863837
SAVED_CNN_WHOLE_VOLUMES = [
    2.985, 3.173, 3.305, 3.332, 3.394, 3.394, 4.437, 4.631, 5.728,
    6.146, 6.441, 6.738, 6.833, 7.412, 8.110, 8.893, 8.963, 9.283,
    9.684, 11.200, 14.897,
]

CLEAN_STEP1 = dict(
    open_ksize=5, min_neck_ratio=0.20, min_area=35,
    centrality_weight=2.5, fill_holes=False, sever_bridges=True,
)
CLEAN_STEP2 = dict(
    open_ksize=3, min_neck_ratio=0.15, min_area=25,
    centrality_weight=2.2, fill_holes=True, sever_bridges=True,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image, max_dim: int = 1800) -> None:
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def describe(values) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "population_std": float(np.std(arr, ddof=0)),
    }


def main() -> None:
    image = read_image(IMAGE)
    if image is None:
        raise RuntimeError(f"Cannot read {IMAGE}")

    segmenter = load_module("audit_grain_segmenter", ROOT / "CODE/modules/grain_segmenter.py")
    cleaner = load_module("audit_grain_crop_cleaner", ROOT / "CODE/modules/grain_crop_cleaner.py")
    geometry = load_module("audit_ellipsoid_geometry", ROOT / "CODE/modules/ellipsoid_geometry.py")

    from sahi import AutoDetectionModel

    load_started = time.perf_counter()
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(YOLO),
        confidence_threshold=0.7,
        device="cpu",
    )
    load_seconds = time.perf_counter() - load_started

    infer_started = time.perf_counter()
    raw = segmenter.segment_grains_sahi(
        detection_model=model,
        image_path=image,
        output_crop_dir=None,
        conf_threshold=0.50,
        slice_size=640,
        overlap_ratio=0.25,
    )
    inference_seconds = time.perf_counter() - infer_started

    cleaned = []
    discarded = []
    for grain in raw:
        step1 = cleaner.clean_single_grain_crop(grain["crop_rgba"], **CLEAN_STEP1)
        step2 = cleaner.clean_single_grain_crop(step1, **CLEAN_STEP2)
        alpha = step2[:, :, 3] if step2.ndim == 3 and step2.shape[2] == 4 else None
        nonzero = int(cv2.countNonZero(alpha)) if alpha is not None else 0
        if nonzero >= 20:
            copy = dict(grain)
            copy["crop_rgba"] = step2
            copy["clean_alpha_area_px"] = nonzero
            cleaned.append(copy)
        else:
            discarded.append({"grain_id": grain.get("grain_id"), "area_px": nonzero})

    metrics = []
    errors = []
    for grain in cleaned:
        try:
            item = geometry.compute_single_grain_metrics(
                grain["crop_rgba"], pixels_per_mm=PIXELS_PER_MM,
                label="hat_nguyen",
            )
            numeric_keys = (
                "length_mm", "width_mm", "thickness_mm", "area_mm2",
                "volume_mm3", "a_px", "b_px", "c_px", "area_px2",
                "vol_px3", "k_factor",
            )
            item = {key: float(item[key]) for key in numeric_keys}
            item["grain_id"] = int(grain["grain_id"])
            item["score"] = float(grain["score"])
            metrics.append(item)
        except Exception as exc:
            errors.append({
                "grain_id": int(grain["grain_id"]),
                "error": f"{type(exc).__name__}: {exc}",
            })

    remaining = list(metrics)
    matched = []
    unmatched_targets = []
    for target in SAVED_CNN_WHOLE_VOLUMES:
        exact = [m for m in remaining if round(m["volume_mm3"], 3) == target]
        if not exact:
            unmatched_targets.append(target)
            continue
        chosen = min(exact, key=lambda m: abs(m["volume_mm3"] - target))
        matched.append(chosen)
        remaining.remove(chosen)

    selected_ids = {m["grain_id"] for m in matched}
    selected_areas = np.asarray([m["area_mm2"] for m in matched], dtype=float)
    if selected_areas.size:
        q1, q3 = np.quantile(selected_areas, [0.25, 0.75])
        area_iqr_lower = float(q1 - 1.5 * (q3 - q1))
    else:
        q1 = q3 = area_iqr_lower = float("nan")

    overlay = image.copy()
    for grain in raw:
        x1, y1, x2, y2 = map(int, grain["bbox"])
        selected = int(grain["grain_id"]) in selected_ids
        color = (0, 220, 0) if selected else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 8)
        cv2.putText(
            overlay, str(grain["grain_id"]), (x1, max(30, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA,
        )
    write_image(OUT / "actual_case_sahi_overlay.jpg", overlay)

    grain_by_id = {int(g["grain_id"]): g for g in cleaned}
    tiles = []
    for item in sorted(matched, key=lambda x: x["volume_mm3"]):
        grain = grain_by_id[item["grain_id"]]
        rgba = grain["crop_rgba"]
        bgr = rgba[:, :, :3].copy()
        bgr[rgba[:, :, 3] == 0] = 255
        h, w = bgr.shape[:2]
        scale = min(180 / max(h, 1), 180 / max(w, 1))
        resized = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
        tile = np.full((230, 220, 3), 255, dtype=np.uint8)
        y0 = (180 - resized.shape[0]) // 2
        x0 = (220 - resized.shape[1]) // 2
        tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.putText(tile, f"id={item['grain_id']} V={item['volume_mm3']:.3f}",
                    (5, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{item['length_mm']:.2f}x{item['width_mm']:.2f} mm",
                    (5, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        rows = []
        for start in range(0, len(tiles), 5):
            chunk = tiles[start:start + 5]
            while len(chunk) < 5:
                chunk.append(np.full_like(tiles[0], 255))
            rows.append(np.hstack(chunk))
        write_image(OUT / "actual_case_reconstructed_cnn_whole_gallery.jpg", np.vstack(rows), max_dim=2200)

    result = {
        "execution": {
            "device": "cpu",
            "model": str(YOLO.relative_to(ROOT)),
            "model_load_seconds": load_seconds,
            "sahi_inference_seconds": inference_seconds,
            "parameters": {
                "confidence_threshold_model": 0.7,
                "confidence_threshold_post": 0.50,
                "slice_size": 640,
                "overlap_ratio": 0.25,
            },
        },
        "counts": {
            "raw_sahi": len(raw),
            "saved_notebook_raw_sahi": 86,
            "after_two_stage_cleaner": len(cleaned),
            "discarded_by_cleaner": len(discarded),
            "geometry_valid_without_cnn_filter": len(metrics),
            "geometry_errors": len(errors),
            "cnn_stage": "NOT VERIFIED: local TensorFlow import fails",
        },
        "raw_detection_scores": describe([g["score"] for g in raw]),
        "raw_bbox_area_px2": describe([
            (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
            for g in raw
        ]),
        "clean_mask_area_px": describe([g["clean_alpha_area_px"] for g in cleaned]),
        "all_cleaned_geometry_without_cnn": {
            "length_mm": describe([m["length_mm"] for m in metrics]),
            "width_mm": describe([m["width_mm"] for m in metrics]),
            "thickness_mm": describe([m["thickness_mm"] for m in metrics]),
            "area_mm2": describe([m["area_mm2"] for m in metrics]),
            "volume_mm3": describe([m["volume_mm3"] for m in metrics]),
        },
        "reconstructed_saved_cnn_whole_selection": {
            "method": "Match the 20 rounded volumes printed by notebook Cell 11 plus inferred 21st candidate volume 6.441 against deterministic current results.",
            "target_count": len(SAVED_CNN_WHOLE_VOLUMES),
            "matched_count": len(matched),
            "unmatched_targets": unmatched_targets,
            "grain_ids": sorted(selected_ids),
            "volume_mm3": describe([m["volume_mm3"] for m in matched]),
            "length_mm": describe([m["length_mm"] for m in matched]),
            "width_mm": describe([m["width_mm"] for m in matched]),
            "area_mm2": describe([m["area_mm2"] for m in matched]),
            "area_iqr_q1": float(q1),
            "area_iqr_q3": float(q3),
            "area_iqr_lower_k_1_5": area_iqr_lower,
            "physical_count_from_exact_matched_mean_phi_0_55": (
                round(9655.370861542871 * 0.55 / statistics.mean([m["volume_mm3"] for m in matched]))
                if matched else None
            ),
            "records": sorted(matched, key=lambda x: x["volume_mm3"]),
        },
        "all_geometry_records": metrics,
        "smallest_20_by_volume_without_cnn": sorted(metrics, key=lambda x: x["volume_mm3"])[:20],
        "largest_10_by_volume_without_cnn": sorted(metrics, key=lambda x: x["volume_mm3"])[-10:],
        "discarded": discarded,
        "geometry_errors": errors,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / "actual_segmentation_audit.json").write_text(text, encoding="utf-8")
    (OUT / "actual_segmentation_audit.log").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
