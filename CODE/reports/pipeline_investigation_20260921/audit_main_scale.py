"""Read-only scale audit for the detector actually imported by the main notebook."""
from __future__ import annotations
import inspect, json, re, sys, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "CODE"))
from modules.container_detector import detect_container_and_scale
import modules.container_detector as detector_module

DATASET = ROOT / "DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv"
SAMPLES = ["M001a", "M010a", "M014a", "M020a", "M045a", "M056a"]
raw = pd.read_csv(DATASET)
images = {p.name.upper(): p for p in (ROOT / "DATASET_BUILDER/1_Raw_Images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
rows = []

def run_one(sample_id: str, run: int):
    src = raw[raw.Sample_ID.str.upper().eq(sample_id.upper())].iloc[0]
    path = images[str(src.Image_Filename).upper()]
    t0 = time.perf_counter()
    info = detect_container_and_scale(path, float(src.Inner_Diameter_mm), float(src.Container_Height_mm), float(src.Empty_Height_mm), detect_mode="inner")
    ratio = float(info["pixels_per_mm"] / src.Pixels_Per_mm)
    rec = {
        "sample_id": sample_id, "run": run, "image_filename": path.name,
        "actual_count": float(src.Actual_Count), "historical_pixels_per_mm": float(src.Pixels_Per_mm),
        "current_pixels_per_mm": float(info["pixels_per_mm"]), "current_inner_diameter_px": int(info["inner_w_px"]),
        "detect_type": info.get("detect_type"), "outer_confidence": info.get("outer_confidence"), "inner_confidence": info.get("inner_confidence"),
        "scale_ratio_current_over_historical": ratio,
        "conditional_geometry_count_multiplier_ratio_cubed": ratio ** 3,
        "conditional_pct_of_450_to_800_excess": 450.0 * (ratio ** 3 - 1.0) / 350.0 * 100.0,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    return rec, info

for sample in SAMPLES:
    rec, info = run_one(sample, 1)
    rows.append(rec)
    overlay = info["overlay_bgr"]
    if max(overlay.shape[:2]) > 1600:
        factor = 1600.0 / max(overlay.shape[:2])
        overlay = cv2.resize(overlay, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Failed to encode main detector overlay")
    encoded.tofile(OUT / f"main_scale_overlay_{sample}.jpg")
    print(json.dumps(rec, ensure_ascii=False), flush=True)
for run in range(2, 6):
    rec, _ = run_one(SAMPLES[1], run)
    rows.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

frame = pd.DataFrame(rows)
base = frame[frame.run.eq(1)].copy()
repeat = frame[frame.sample_id.eq(SAMPLES[1])]
source = inspect.getsource(detector_module)
summary = {
    "implementation": "CODE/modules/container_detector.py (main notebook import)",
    "samples": base.to_dict(orient="records"),
    "six_sample_relative_error_pct": {
        "min": float(((base.current_pixels_per_mm / base.historical_pixels_per_mm - 1) * 100).min()),
        "max": float(((base.current_pixels_per_mm / base.historical_pixels_per_mm - 1) * 100).max()),
        "median_abs": float(((base.current_pixels_per_mm / base.historical_pixels_per_mm - 1).abs() * 100).median()),
    },
    "repeatability": {
        "sample_id": SAMPLES[1], "runs": int(len(repeat)),
        "pixels_per_mm_values": repeat.current_pixels_per_mm.tolist(),
        "range": float(repeat.current_pixels_per_mm.max() - repeat.current_pixels_per_mm.min()),
        "std": float(repeat.current_pixels_per_mm.std(ddof=0)),
    },
    "source_randomness": {
        "contains_ransac": "ransac" in source.lower(),
        "uses_default_rng_without_explicit_seed": "np.random.default_rng()" in source,
        "rng_parameter_available": bool(re.search(r"rng:\s*Optional\[np\.random\.Generator\]", source)),
        "hough_present": "HoughCircles" in source,
    },
    "required_scale_ratio_if_all_450_to_800_error_is_scale": float((800.0 / 450.0) ** (1 / 3)),
}
frame.to_csv(OUT / "main_scale_runs.csv", index=False)
text = json.dumps(summary, ensure_ascii=False, indent=2)
(OUT / "main_scale_audit.json").write_text(text, encoding="utf-8")
(OUT / "main_scale_audit.log").write_text(text, encoding="utf-8")