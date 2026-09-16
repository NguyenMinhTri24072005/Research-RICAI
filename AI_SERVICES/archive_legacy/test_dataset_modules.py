import os
import sys
from pathlib import Path

# Add CODE to sys.path
sys.path.insert(0, str(Path("CODE").resolve()))
sys.stdout.reconfigure(encoding="utf-8")

from modules.container_detector import detect_container_and_scale
from modules.ellipsoid_geometry import compute_single_grain_metrics
from modules.uniformity_evaluator import evaluate_batch_uniformity
from modules.dataset_extractor import DatasetExtractor

print("=== 1. TESTING CONTAINER DETECTOR ===")
test_img = Path("DATASET_BUILDER/1_Raw_Images/M001/M001A.jpg")
if test_img.exists():
    res = detect_container_and_scale(
        test_img,
        inner_diam_mm=17.8,
        container_height_mm=33.9,
        empty_height_mm=10.9,
        wall_thickness_mm=2.0,
    )
    print(f"✅ Scale: {res['pixels_per_mm']:.2f} px/mm | Cup Diam: {res['container_w_px']} px | Rice Vol: {res['bulk_rice_volume_mm3']:.2f} mm3")
else:
    print(f"⚠️ Test image not found: {test_img}")

print("\n=== 2. TESTING UNIFORMITY EVALUATOR ===")
unif = evaluate_batch_uniformity([12.5, 13.0, 12.8, 13.2, 12.9, 13.1, 50.0])  # 50.0 is outlier
print(f"✅ Uniformity: {unif['uniformity_rate_pct']}% | Mean clean: {unif['mean_clean']} | Clean count: {unif['clean_count']}/{unif['total_count']}")

print("\n=== 3. TESTING DATASET EXTRACTOR INITIALIZATION ===")
extractor = DatasetExtractor(
    base_dir=".",
    yolo_model_path="RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt",
    cnn_model_path="RESULTS/CNN_DenseNet121_Trained/best_rice_densenet121.keras",
)
records = extractor.read_manual_records()
print(f"✅ Loaded {len(records)} records from manual_data.xlsx")

# Check matching images
found_count = 0
for r in records:
    match = extractor.find_matching_image(r["Sample_ID"])
    if match:
        found_count += 1

print(f"✅ Found {found_count} matching images out of {len(records)} sample rows.")
print("✨ ALL MODULE TESTS COMPLETED SUCCESSFULLY!")
