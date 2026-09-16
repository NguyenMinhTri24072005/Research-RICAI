import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def make_cell(cell_type, source):
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': [line + '\n' for line in source.strip().split('\n')]
    }

cells_main = [
    make_cell('markdown', r'''# 🌾 RICE VISION AI: PIPELINE CHÍNH ƯỚC LƯỢNG SỐ LƯỢNG & ĐÁNH GIÁ PHẨM CẤP (V2 - MODULAR)
**Đề tài Nghiên cứu Khoa học**: Hệ thống thị giác máy tính và học sâu phục vụ ước lượng số lượng và đánh giá phẩm cấp hạt giống lúa.

---

### 🌟 Kiến trúc Pipeline Tích hợp 6 Tầng:
1. **Tầng 1 (Vật chứa)**: Khử góc nghiêng miệng ly bằng Canny + `fitEllipse` bắt **Mép Trong**, tính tỷ lệ quy đổi $px/mm$ và thể tích khối lúa $V_{\text{bulk}}$ ($mm^3$).
2. **Tầng 2 (Bóc tách hạt)**: Bóc tách polygon từng hạt lúa bằng **SAHI + YOLO-seg** thành ảnh RGBA trong suốt.
3. **Tầng 3 (Phân loại phẩm cấp)**: Phân loại hạt nguyên vs hạt khuyết tật bằng mạng học sâu **DenseNet121 CNN**.
4. **Tầng 4 (Mô hình hóa 3D)**: Tính kích thước 2D và thể tích 3D Ellipsoid $V_{\text{grain}} = \frac{4}{3}\pi abc$ ($mm^3$).
5. **Tầng 5 (Lọc ngoại lai)**: Lọc bỏ hạt dị thường bằng phương pháp thống kê **IQR** và đánh giá độ đồng đều ($\ge 80\%$).
6. **Tầng 6 (Ước lượng số lượng)**: Ước lượng tổng số hạt lúa theo thể tích khối lúa và hệ số chèn lấp $\phi = 0.62$:
$$N_{\text{est}} = \frac{V_{\text{bulk}} \times \phi}{\bar{V}_{\text{grain}}}$$'''),

    make_cell('code', '''# Bước 1: Kết nối Google Drive & Cài đặt thư viện
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics sahi tensorflow matplotlib seaborn opencv-python numpy'''),

    make_cell('code', '''# Bước 2: Import Modules & Cấu hình Trọng số Model
import os
import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN TRÊN GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES"

code_dir = os.path.join(BASE_PATH, "CODE")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Import 5 module nghiệp vụ độc lập từ package modules
from modules.container_detector import detect_container_and_scale, draw_container_overlay
from modules.grain_segmenter import segment_grains_sahi
from modules.grain_classifier import GrainClassifier
from modules.ellipsoid_geometry import compute_single_grain_metrics, draw_grain_ellipse_overlay
from modules.uniformity_evaluator import evaluate_batch_uniformity

# Đường dẫn Trọng số Mô hình
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt")
CNN_MODEL_PATH  = os.path.join(BASE_PATH, "RESULTS/CNN_DenseNet121_Trained/best_rice_densenet121.keras")

# Khởi tạo mô hình AI
from sahi import AutoDetectionModel
print("🚀 Đang nạp mô hình YOLO-seg và DenseNet121 CNN...")
sahi_yolo_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=0.50,
    device="cuda:0"
)
cnn_classifier = GrainClassifier(model_path=CNN_MODEL_PATH, class_names=["hat_khuyet_tat", "hat_nguyen"])
print("✨ Nạp toàn bộ mô hình thành công!")'''),

    make_cell('code', '''# Bước 3: THIẾT LẬP THÔNG SỐ ĐẦU VÀO CHO ẢNH CẦN DỰ ĐOÁN
# ─────────────────────────────────────────────────────────────────────────────
# Bạn chỉ cần thay đổi các thông số ở ô này:
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_PATH          = os.path.join(BASE_PATH, "DATASET_BUILDER/1_Raw_Images/M001/M001A.jpg")
INNER_DIAM_MM       = 17.8    # Đường kính TRONG của ly (mm)
CONTAINER_HEIGHT_MM = 33.9    # Chiều cao toàn bộ thân ly (mm)
EMPTY_HEIGHT_MM     = 10.9    # Khoảng trống từ miệng ly đến mặt lúa (mm)
PACKING_FRACTION    = 0.62    # Hệ số chèn lấp khối lúa (~0.60 - 0.64)

# Thư mục lưu kết quả tạm
OUTPUT_DIR = "/content/pipeline_inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"🖼️ Ảnh đầu vào : {IMAGE_PATH}")
print(f"📏 Kích thước ly: ⌀ trong={INNER_DIAM_MM}mm | Cao={CONTAINER_HEIGHT_MM}mm | Hụt={EMPTY_HEIGHT_MM}mm")
print("=" * 60)'''),

    make_cell('code', '''# Bước 4: TẦNG 1 — Nhận diện Miệng Ly (Mép Trong) & Tính Thể tích Khối Lúa
container_info = detect_container_and_scale(
    image_input=IMAGE_PATH,
    inner_diam_mm=INNER_DIAM_MM,
    container_height_mm=CONTAINER_HEIGHT_MM,
    empty_height_mm=EMPTY_HEIGHT_MM,
    detect_mode="inner"
)

pixels_per_mm = container_info["pixels_per_mm"]
bulk_volume_mm3 = container_info["bulk_rice_volume_mm3"]

print("=" * 60)
print("  TẦNG 1: QUY ĐỔI KÍCH THƯỚC VẬT CHỨA (MÉP TRONG)")
print("=" * 60)
print(f"  • Đường kính MÉP TRONG phát hiện : {container_info['inner_w_px']} px (Mép ngoài: {container_info['outer_w_px']} px)")
print(f"  • Tỷ lệ quy đổi pixel           : {pixels_per_mm:.2f} px/mm")
print(f"  • Chiều cao khối lúa thực tế    : {container_info['rice_height_mm']:.1f} mm")
print(f"  • Thể tích khối lúa (V_bulk)    : {bulk_volume_mm3:,.2f} mm³")
print("=" * 60)

# Hiển thị ảnh khoanh Mép Trong (Xanh lá) và Mép Ngoài (Cam)
img_raw = cv2.imread(IMAGE_PATH)
vis_container = draw_container_overlay(img_raw, container_info)
plt.figure(figsize=(7, 5))
plt.imshow(cv2.cvtColor(vis_container, cv2.COLOR_BGR2RGB))
plt.title("Nhận diện Miệng Ly: Mép Trong (Xanh Lá) & Mép Ngoài (Cam)", fontweight='bold')
plt.axis('off')
plt.show()'''),

    make_cell('code', '''# Bước 5: TẦNG 2 & 3 — SAHI YOLO-seg Bóc Tách Hạt & CNN Phân Loại Phẩm Cấp
print("🚀 Đang chạy SAHI YOLO-seg bóc tách polygon từng hạt lúa...")
grains = segment_grains_sahi(
    detection_model=sahi_yolo_model,
    image_path=IMAGE_PATH,
    output_crop_dir=os.path.join(OUTPUT_DIR, "all_crops"),
    conf_threshold=0.50,
    slice_size=512,
    overlap_ratio=0.25
)

print(f"🔍 Đang phân loại {len(grains)} hạt lúa bằng DenseNet121 CNN...")
whole_grains, defective_grains = cnn_classifier.filter_grains(
    grains, target_label="hat_nguyen", min_conf=0.50
)

# Nếu tập hạt nguyên trống (do ảnh chụp hạt đặc thù), fallback lấy toàn bộ hạt để đo đạc
target_grains_for_measure = whole_grains if len(whole_grains) > 0 else grains
whole_ratio = (len(whole_grains) / len(grains) * 100) if grains else 0.0

print("=" * 60)
print("  TẦNG 2 & 3: KẾT QUẢ BÓC TÁCH & PHÂN LOẠI")
print("=" * 60)
print(f"  • Tổng số hạt phát hiện trên mặt ly : {len(grains)} hạt")
print(f"  • Hạt nguyên đạt chuẩn (🌟)          : {len(whole_grains)} hạt ({whole_ratio:.1f}%)")
print(f"  • Hạt khuyết tật / lép (⚠️)         : {len(defective_grains)} hạt ({100 - whole_ratio:.1f}%)")
print("=" * 60)

# Vẽ biểu đồ tròn phân loại phẩm cấp
plt.figure(figsize=(5, 5))
plt.pie(
    [max(1, len(whole_grains)), len(defective_grains)],
    labels=['Hạt nguyên', 'Hạt khuyết tật'],
    autopct='%.1f%%',
    colors=['#4CAF50', '#F44336'],
    startangle=140,
    explode=(0.05, 0)
)
plt.title("Tỷ lệ Phẩm cấp Hạt Lúa", fontweight='bold')
plt.show()'''),

    make_cell('code', '''# Bước 6: TẦNG 4 & 5 — Mô hình hóa 3D Ellipsoid & Đánh giá Độ đồng đều IQR
print(f"📐 Đang tính toán kích thước 2D và Thể tích 3D Ellipsoid cho {len(target_grains_for_measure)} hạt lúa...")
grain_metrics_list = []
volumes_list = []

for g in target_grains_for_measure:
    try:
        m = compute_single_grain_metrics(g["crop_rgba"], pixels_per_mm=pixels_per_mm, label="hat_nguyen")
        grain_metrics_list.append(m)
        volumes_list.append(m["volume_mm3"])
    except Exception:
        continue

# Lọc ngoại lai IQR và đánh giá độ đồng đều
uniformity_res = evaluate_batch_uniformity(volumes_list, threshold_rate=0.80)

# Tính toán ước lượng số lượng hạt theo công thức Hybrid
mean_grain_vol = uniformity_res["mean_clean"]
if mean_grain_vol > 0 and bulk_volume_mm3 > 0:
    estimated_seed_count = int(round((bulk_volume_mm3 * PACKING_FRACTION) / mean_grain_vol))
else:
    estimated_seed_count = 0

print("=" * 60)
print("  TẦNG 4 & 5: KÍCH THƯỚC HÌNH THÁI & ĐỘ ĐỒNG ĐỀU")
print("=" * 60)
lengths = [m["length_mm"] for m in grain_metrics_list] if grain_metrics_list else [0.0]
widths  = [m["width_mm"] for m in grain_metrics_list] if grain_metrics_list else [0.0]
areas   = [m["area_mm2"] for m in grain_metrics_list] if grain_metrics_list else [0.0]

print(f"  • Chiều dài hạt (2a)    : Mean={np.mean(lengths):.2f}mm | Min={np.min(lengths):.2f}mm | Max={np.max(lengths):.2f}mm")
print(f"  • Chiều rộng hạt (2b)   : Mean={np.mean(widths):.2f}mm | Min={np.min(widths):.2f}mm | Max={np.max(widths):.2f}mm")
print(f"  • Diện tích 2D (Area)   : Mean={np.mean(areas):.2f}mm² | Min={np.min(areas):.2f}mm² | Max={np.max(areas):.2f}mm²")
print(f"  • Thể tích 3D Ellipsoid : Mean={mean_grain_vol:.2f}mm³ (sau lọc IQR)")
print(f"  • Độ đồng đều lô hạt    : {uniformity_res['uniformity_rate_pct']}% -> {'✅ ĐẠT CHUẨN (>= 80%)' if uniformity_res['is_uniform'] else '❌ KHÔNG ĐỒNG ĐỀU'}")
print("=" * 60)'''),

    make_cell('code', '''# Bước 7: TẦNG 6 — BẢNG ĐIỀU KHIỂN TỔNG HỢP & DỰ BÁO SỐ LƯỢNG HẠT
print("\n" + "╔" + "═" * 68 + "╗")
print("║" + " " * 18 + "🏆 BẢNG KẾT QUẢ RICE VISION AI" + " " * 19 + "║")
print("╠" + "═" * 68 + "╣")
print(f"║  📸 Ảnh phân tích         : {os.path.basename(IMAGE_PATH):<42} ║")
print(f"║  📏 Thể tích khối lúa (V) : {bulk_volume_mm3:<10,.1f} mm³                           ║")
print(f"║  🌾 Số hạt thấy trên mặt ly: {len(grains):<4} hạt (Nguyên: {len(whole_grains)}, Khuyết: {len(defective_grains)})         ║")
print(f"║  🌟 Phẩm cấp hạt nguyên   : {whole_ratio:<5.1f}%                                   ║")
print(f"║  📐 Thể tích 1 hạt tb (v) : {mean_grain_vol:<6.2f} mm³                                 ║")
print(f"║  📊 Độ đồng đều hạt       : {uniformity_res['uniformity_rate_pct']:<5.1f}% ({'ĐẠT CHUẨN' if uniformity_res['is_uniform'] else 'CHƯA ĐẠT'})                       ║")
print("╠" + "═" * 68 + "╣")
print(f"║  🎯 DỰ ĐOÁN TỔNG SỐ HẠT   : {estimated_seed_count:<6} HẠT (Hệ số chèn lấp phi={PACKING_FRACTION})       ║")
print("╚" + "═" * 68 + "╝\n")

# Hiển thị 8 ảnh hạt tiêu biểu kèm viền Ellipse 3D
display_sample_grains = whole_grains if whole_grains else grains
if display_sample_grains and grain_metrics_list:
    num_show = min(8, len(display_sample_grains), len(grain_metrics_list))
    plt.figure(figsize=(14, 4))
    for i in range(num_show):
        plt.subplot(1, num_show, i + 1)
        m = grain_metrics_list[i]
        vis_g = draw_grain_ellipse_overlay(display_sample_grains[i]["crop_rgba"], m)
        plt.imshow(vis_g)
        plt.title(f"#{i+1}: {m['volume_mm3']:.1f}mm³\\n{m['length_mm']:.1f}x{m['width_mm']:.1f}mm", fontsize=9)
        plt.axis('off')
    plt.suptitle("Mẫu Hạt Lúa Được Khớp Mô Hình 3D Ellipsoid", fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()''')
]

nb_main = {
    'nbformat': 4,
    'nbformat_minor': 2,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells_main
}

save_path = "CODE/RICE_VISION_MAIN_PIPELINE.ipynb"
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(nb_main, f, ensure_ascii=False, indent=2)
print(f"✅ Đã cập nhật thành công Pipeline Chính Mới: {save_path}")
