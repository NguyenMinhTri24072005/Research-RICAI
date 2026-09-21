"""Reproduce arithmetic for the saved 450 -> 802 notebook run without rerunning GPU stages."""
from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NOTEBOOK = ROOT / "CODE/RICE_VISION_MAIN_PIPELINE.ipynb"
ACTUAL_COUNT_USER_REPORTED = 450.0
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def stream(cell_index: int) -> str:
    return "".join("".join(o.get("text", [])) for o in nb["cells"][cell_index].get("outputs", []) if o.get("output_type") == "stream")

def grab(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Pattern not found: {pattern}")
    return float(match.group(1).replace(",", ""))

c4, c5, c7, c10, c11, c12, c17 = map(stream, [4, 5, 7, 10, 11, 12, 17])
record = {
    "evidence_kind": "Inherited saved notebook output; arithmetic independently recomputed in this script",
    "notebook": str(NOTEBOOK.relative_to(ROOT)),
    "image_name": re.search(r"Ảnh phân tích\s*:\s*([^║]+)", c12).group(1).strip(),
    "actual_count_user_reported": ACTUAL_COUNT_USER_REPORTED,
    "pixels_per_mm": grab(r"Tỷ lệ quy đổi pixel\s*:\s*([0-9.]+)", c4),
    "inner_diameter_px": grab(r"MÉP TRONG phát hiện\s*:\s*([0-9.]+)", c4),
    "outer_diameter_px": grab(r"Mép ngoài:\s*([0-9.]+)", c4),
    "bulk_volume_mm3": grab(r"V_bulk\)\s*:\s*([0-9,.]+)", c4),
    "surface_detections": grab(r"Phát hiện\s*([0-9.]+) hạt", c5),
    "cnn_whole_count": grab(r"Hạt NGUYÊN[^:]*:\s*([0-9.]+)", c7),
    "mean_grain_volume_mm3": grab(r"mean_whole_grain_vol =\s*([0-9.]+)", c11),
    "physical_packing_fraction": grab(r"PACKING_FRACTION =\s*([0-9.]+)", c11),
    "physical_estimate": grab(r"estimated_seed_count = round\([^\n]+\) = ([0-9.]+)", c11),
    "regression_estimate": grab(r"DỰ ĐOÁN HỒI QUY \(AI\)\s*:\s*(?:\x1b\[[0-9;]*m)?([0-9.]+)", c17),
    "mean_length_mm": grab(r"Dài:\s*([0-9.]+)", c12),
    "mean_width_mm": grab(r"Rộng:\s*([0-9.]+)", c12),
}
r = record
r["formula_recomputed"] = round(r["bulk_volume_mm3"] * r["physical_packing_fraction"] / r["mean_grain_volume_mm3"])
r["observed_overestimate_count"] = r["physical_estimate"] - ACTUAL_COUNT_USER_REPORTED
r["observed_overestimate_pct"] = (r["physical_estimate"] / ACTUAL_COUNT_USER_REPORTED - 1) * 100
r["mean_grain_volume_needed_for_450_same_bulk_phi"] = r["bulk_volume_mm3"] * r["physical_packing_fraction"] / ACTUAL_COUNT_USER_REPORTED
r["mean_grain_volume_multiplier_needed"] = r["mean_grain_volume_needed_for_450_same_bulk_phi"] / r["mean_grain_volume_mm3"]
r["packing_fraction_needed_for_450_same_bulk_grain_volume"] = ACTUAL_COUNT_USER_REPORTED * r["mean_grain_volume_mm3"] / r["bulk_volume_mm3"]
r["scale_ratio_needed_if_only_px_per_mm_is_wrong"] = (r["physical_estimate"] / ACTUAL_COUNT_USER_REPORTED) ** (1 / 3)
r["implied_correct_pixels_per_mm_if_only_scale_is_wrong"] = r["pixels_per_mm"] / r["scale_ratio_needed_if_only_px_per_mm_is_wrong"]
r["implied_length_mm_if_only_scale_is_wrong"] = r["mean_length_mm"] * r["scale_ratio_needed_if_only_px_per_mm_is_wrong"]
r["implied_width_mm_if_only_scale_is_wrong"] = r["mean_width_mm"] * r["scale_ratio_needed_if_only_px_per_mm_is_wrong"]
r["counterfactual_estimate_phi_0_62"] = round(r["bulk_volume_mm3"] * 0.62 / r["mean_grain_volume_mm3"])
r["counterfactual_estimate_phi_0_82"] = round(r["bulk_volume_mm3"] * 0.82 / r["mean_grain_volume_mm3"])

candidate = Path(r"C:\Users\NGUYEN~1\AppData\Local\Temp\codex-clipboard-f3867fab-7cfa-4119-96ff-868da9814eb9.png")
if candidate.exists():
    with candidate.open("rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with Image.open(candidate) as im:
        size = list(im.size)
    r["candidate_attachment"] = {
        "path": str(candidate), "sha256": digest, "size_px": size,
        "raw_input_identity": "NOT VERIFIED: it contains drawn detections and current detector output did not match saved notebook scale",
    }

text = json.dumps(r, ensure_ascii=False, indent=2)
(OUT / "actual_case_arithmetic.json").write_text(text, encoding="utf-8")
(OUT / "actual_case_arithmetic.log").write_text(text, encoding="utf-8")
print(text)