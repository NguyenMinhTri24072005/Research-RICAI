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

# ─────────────────────────────────────────────────────────────
# 1. FILE 1: TRAIN_YOLO_SEGMENT.ipynb
# ─────────────────────────────────────────────────────────────
cells_train = [
    make_cell('markdown', '''# 🌾 MODULE 1: HUẤN LUYỆN MÔ HÌNH YOLO SEGMENTATION CHO HẠT LÚA
**Đề tài Nghiên cứu Khoa học**: Hệ thống thị giác máy tính và học sâu phục vụ ước lượng số lượng và đánh giá phẩm cấp hạt giống lúa.

### 🎯 Mục đích file này:
1. Kết nối Google Drive & cài đặt các thư viện cần thiết (`ultralytics`, `matplotlib`, `pyyaml`).
2. Cấu hình dataset và siêu tham số huấn luyện (Epochs, Batch size, Image size).
3. Huấn luyện mô hình phân đoạn cá thể hạt lúa (**YOLO Segmentation** - ví dụ `yolo26n-seg.pt` hoặc `yolov8n-seg.pt`).
4. Kiểm tra dự đoán thử nghiệm và tự động sao lưu toàn bộ trọng số (`best.pt`, `last.pt`) cùng biểu đồ mAP/Loss sang Google Drive.'''),

    make_cell('code', '''# Bước 1: Kết nối Google Drive
from google.colab import drive
drive.mount('/content/drive')'''),

    make_cell('code', '''# Bước 2: Cài đặt thư viện Ultralytics YOLO
!pip install ultralytics matplotlib pyyaml'''),

    make_cell('code', '''# Bước 3: Import các thư viện cần thiết
import os
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO'''),

    make_cell('code', '''# ─────────────────────────────────────────────────────────────────────────────
# Bước 4: CẤU HÌNH THAM SỐ HUẤN LUYỆN (Chỉnh sửa theo nhu cầu)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Đường dẫn thư mục gốc trong Google Drive
BASE_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/"

# 2. Tên model pretrained và dataset
MODEL_NAME   = "yolo26n-seg.pt"  # Các lựa chọn: yolo26n-seg.pt, yolov8n-seg.pt, yolov11n-seg.pt
DATASET_NAME = "35_special_images_segmentation.v1i.yolov8"

# 3. Siêu tham số huấn luyện
EPOCHS     = 100
BATCH_SIZE = 16
IMG_SIZE   = 640
DEVICE     = 0  # Sử dụng GPU 0 (hoặc 'cpu' nếu không có GPU)

# 4. Tự động thiết lập đường dẫn lưu trữ
MODELS_DIR  = os.path.join(BASE_PATH, "MODELS")
DATASET_DIR = os.path.join(BASE_PATH, "DATASETS", DATASET_NAME)
DATA_YAML   = os.path.join(DATASET_DIR, "data.yaml")

TRAIN_EXP_NAME = f"{DATASET_NAME}_{MODEL_NAME.replace('.pt', '')}_trained"
DRIVE_SAVE_DIR = os.path.join(BASE_PATH, "RESULTS", TRAIN_EXP_NAME)

print("=" * 60)
print(f"📦 Model ban đầu     : {os.path.join(MODELS_DIR, MODEL_NAME)}")
print(f"📊 File data.yaml    : {DATA_YAML}")
print(f"💾 Nơi lưu kết quả   : {DRIVE_SAVE_DIR}")
print(f"⚙️ Cấu hình          : Epochs={EPOCHS} | Batch={BATCH_SIZE} | ImgSize={IMG_SIZE}")
print("=" * 60)'''),

    make_cell('code', '''# Bước 5: Bắt đầu Huấn luyện YOLO Segmentation
model_path = os.path.join(MODELS_DIR, MODEL_NAME)

if not os.path.exists(model_path):
    print(f"⚠️ Không tìm thấy model tại {model_path}. Sẽ tự động tải {MODEL_NAME} từ Ultralytics...")
    model = YOLO(MODEL_NAME)
else:
    model = YOLO(model_path)

print("🚀 Bắt đầu quá trình huấn luyện mô hình YOLO Segmentation...")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project="rice_seed_segmentation",
    name=TRAIN_EXP_NAME,
    exist_ok=True
)
print("✅ Quá trình huấn luyện đã hoàn tất!")'''),

    make_cell('code', '''# Bước 6: Thử nghiệm Dự đoán nhanh trên 1 ảnh mẫu
best_weight_local = f"/content/rice_seed_segmentation/{TRAIN_EXP_NAME}/weights/best.pt"

if os.path.exists(best_weight_local):
    trained_model = YOLO(best_weight_local)
    test_demo_url = "https://media.istockphoto.com/id/186786216/photo/unmilled-rice-grains.jpg"
    res = trained_model.predict(source=test_demo_url, conf=0.4, save=True)
    print("✅ Đã chạy thử nghiệm dự đoán mẫu thành công!")
else:
    print("⚠️ Không tìm thấy trọng số best.pt cục bộ.")'''),

    make_cell('code', '''# Bước 7: Tự động sao lưu toàn bộ kết quả vào Google Drive
local_result_dir = f"/content/rice_seed_segmentation/{TRAIN_EXP_NAME}"

if os.path.exists(local_result_dir):
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    shutil.copytree(local_result_dir, DRIVE_SAVE_DIR, dirs_exist_ok=True)
    print("=" * 60)
    print(f"🎉 ĐÃ SAO LƯU THÀNH CÔNG TOÀN BỘ KẾT QUẢ VÀO GOOGLE DRIVE!")
    print(f"📂 Thư mục: {DRIVE_SAVE_DIR}")
    print(f"⭐ Trọng số tốt nhất: {os.path.join(DRIVE_SAVE_DIR, 'weights', 'best.pt')}")
    print("=" * 60)
else:
    print("❌ Không tìm thấy thư mục kết quả để sao lưu.")'''),

    make_cell('code', '''# Bước 8: Hiển thị Biểu đồ Kết quả và Ma trận Nhầm lẫn
results_img_path = os.path.join(DRIVE_SAVE_DIR, "results.png")
confusion_matrix_path = os.path.join(DRIVE_SAVE_DIR, "confusion_matrix_normalized.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

if os.path.exists(results_img_path):
    img_res = plt.imread(results_img_path)
    axes[0].imshow(img_res)
    axes[0].set_title("Quá trình Loss & mAP qua các Epochs", fontsize=12, fontweight='bold')
    axes[0].axis('off')

if os.path.exists(confusion_matrix_path):
    img_cm = plt.imread(confusion_matrix_path)
    axes[1].imshow(img_cm)
    axes[1].set_title("Ma trận nhầm lẫn chuẩn hóa (Confusion Matrix)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

plt.tight_layout()
plt.show()''')
]

# ─────────────────────────────────────────────────────────────
# 2. FILE 2: SAHI_CROP_GRAINS.ipynb
# ─────────────────────────────────────────────────────────────
cells_sahi = [
    make_cell('markdown', '''# 🌾 MODULE 2: TỰ ĐỘNG PHÂN ĐOẠN & BÓC TÁCH HẠT LÚA (SAHI + YOLO-SEG)
**Đề tài Nghiên cứu Khoa học**: Hệ thống thị giác máy tính và học sâu phục vụ ước lượng số lượng và đánh giá phẩm cấp hạt giống lúa.

### 🎯 Mục đích file này:
1. Nhận đường dẫn thư mục ảnh đầu vào (`INPUT_IMAGES_DIR`) và mô hình YOLO-seg (`MODEL_WEIGHTS_PATH`).
2. Sử dụng công nghệ **SAHI Sliced Prediction** cắt lát độ phân giải cao để phát hiện chính xác từng hạt lúa nhỏ.
3. Tự động bóc tách (crop) từng hạt lúa thành file ảnh PNG có **nền trong suốt (`RGBA`)**.
4. Xuất ảnh overlay trực quan (`_seg.jpg`) và lưu toàn bộ kết quả về các thư mục được chỉ định.'''),

    make_cell('code', '''# Bước 1: Kết nối Google Drive
from google.colab import drive
drive.mount('/content/drive')'''),

    make_cell('code', '''# Bước 2: Cài đặt thư viện SAHI và Ultralytics
!pip install ultralytics sahi opencv-python-headless matplotlib numpy'''),

    make_cell('code', '''# Bước 3: Import thư viện
import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction'''),

    make_cell('code', '''# ─────────────────────────────────────────────────────────────────────────────
# Bước 4: ⚙️ KHU VỰC CẤU HÌNH ĐƯỜNG DẪN ĐẦU VÀO / ĐẦU RA & THAM SỐ
# (Người dùng chỉ cần chỉnh sửa các biến trong cell này trước khi chạy)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Đường dẫn file trọng số YOLO-seg đã huấn luyện (best.pt)
MODEL_WEIGHTS_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt"

# 2. Đường dẫn folder chứa các ảnh gốc cần xử lý
INPUT_IMAGES_DIR   = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/TEST_IMAGES/DETECTABLE_TEST_IMAGES"

# 3. Đường dẫn folder đích để lưu các ảnh hạt lúa đã crop (PNG trong suốt RGBA)
OUTPUT_CROPPED_DIR = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/DETECTED_OBJECTS/OUTPUT_CROPPED_GRAINS"

# 4. Đường dẫn folder đích để lưu các ảnh tổng quan phủ màu mask (_seg.jpg)
OUTPUT_OVERLAY_DIR = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/DETECTED_OBJECTS/OUTPUT_OVERLAY_IMAGES"

# 5. Các tham số cắt lát SAHI
CONFIDENCE_THRESHOLD = 0.40   # Ngưỡng tin cậy (0.35 - 0.45)
SLICE_HEIGHT         = 512    # Chiều cao lát cắt (512 hoặc 640)
SLICE_WIDTH          = 512    # Chiều rộng lát cắt (512 hoặc 640)
OVERLAP_RATIO        = 0.25   # Tỷ lệ chồng lấn lát cắt (20% - 25%)
NMS_MATCH_THRESHOLD  = 0.65   # Ngưỡng khử trùng lặp NMS

print("=" * 60)
print(f"🎯 Model YOLO-seg      : {MODEL_WEIGHTS_PATH}")
print(f"📁 Folder ảnh đầu vào  : {INPUT_IMAGES_DIR}")
print(f"🌾 Folder lưu hạt crop : {OUTPUT_CROPPED_DIR}")
print(f"🖼️ Folder lưu overlay  : {OUTPUT_OVERLAY_DIR}")
print(f"⚙️ Tham số SAHI       : Slice={SLICE_WIDTH}x{SLICE_HEIGHT} | Overlap={OVERLAP_RATIO} | Conf={CONFIDENCE_THRESHOLD}")
print("=" * 60)'''),

    make_cell('code', '''# Bước 5: THỰC THI BÓC TÁCH HẠT LÚA TOÀN BỘ FOLDER
def run_grain_extraction_pipeline():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file model tại: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(INPUT_IMAGES_DIR):
        raise FileNotFoundError(f"❌ Không tìm thấy folder ảnh đầu vào tại: {INPUT_IMAGES_DIR}")

    os.makedirs(OUTPUT_CROPPED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)

    # 1. Khởi tạo mô hình SAHI
    print("⏳ Đang nạp mô hình vào SAHI...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_WEIGHTS_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cuda:0"
    )

    # 2. Lấy danh sách ảnh hợp lệ
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif')
    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(valid_exts)]
    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần xử lý trong: {INPUT_IMAGES_DIR}\n")

    total_grains_cropped = 0

    # 3. Duyệt qua từng ảnh để cắt lát & bóc tách hạt
    for img_idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(INPUT_IMAGES_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        print(f"[{img_idx}/{len(image_files)}] Đang xử lý: {img_name}...")

        # Dự đoán cắt lát SAHI
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            postprocess_match_threshold=NMS_MATCH_THRESHOLD
        )

        predictions = result.object_prediction_list
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ⚠️ Không thể đọc ảnh: {img_path}")
            continue

        global_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        img_grain_count = 0

        for idx, obj in enumerate(predictions):
            if obj.mask is None or not obj.mask.segmentation:
                continue

            local_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            for polygon in obj.mask.segmentation:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(local_mask, [pts], 255)
                cv2.fillPoly(global_mask, [pts], 255)

            ys, xs = np.where(local_mask == 255)
            if len(ys) > 0 and len(xs) > 0:
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1

                crop_img  = img_bgr[y1:y2, x1:x2]
                crop_mask = local_mask[y1:y2, x1:x2]

                # Chuyển sang định dạng RGBA (Kênh 4 là mặt nạ độ trong suốt Alpha)
                rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = crop_mask

                # Lưu file hạt lúa sạch nền
                crop_filename = f"{base_name}_grain_{idx:04d}.png"
                cv2.imwrite(os.path.join(OUTPUT_CROPPED_DIR, crop_filename), rgba)
                img_grain_count += 1

        # Tạo ảnh phủ màu xanh overlay trực quan
        color_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        color_overlay[global_mask == 255] = [0, 255, 0]
        img_with_masks = cv2.addWeighted(color_overlay, 0.4, img_bgr, 1.0, 0)
        cv2.imwrite(os.path.join(OUTPUT_OVERLAY_DIR, f"{base_name}_seg.jpg"), img_with_masks)

        total_grains_cropped += img_grain_count
        print(f"  👉 Đã bóc tách thành công: {img_grain_count} hạt.")

        # Giải phóng bộ nhớ tránh tràn RAM/VRAM
        del result, predictions, img_bgr, global_mask, local_mask, color_overlay, img_with_masks
        gc.collect()

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN THÀNH XỬ LÝ TOÀN BỘ FOLDER!")
    print(f"🌾 Tổng số hạt lúa đã bóc tách : {total_grains_cropped:,} hạt")
    print(f"📂 Thư mục lưu hạt sạch (RGBA) : {OUTPUT_CROPPED_DIR}")
    print(f"🖼️ Thư mục lưu ảnh trực quan   : {OUTPUT_OVERLAY_DIR}")
    print("=" * 60)

run_grain_extraction_pipeline()'''),

    make_cell('code', '''# Bước 6: Hiển thị Mẫu Trực Quan Các Hạt Lúa Đã Bóc Tách
cropped_files = [f for f in os.listdir(OUTPUT_CROPPED_DIR) if f.lower().endswith('.png')]

if len(cropped_files) > 0:
    sample_files = cropped_files[:min(8, len(cropped_files))]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, fname in enumerate(sample_files):
        fpath = os.path.join(OUTPUT_CROPPED_DIR, fname)
        img_bgra = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img_bgra is not None:
            # Chuyển BGRA sang RGBA để hiển thị bằng matplotlib
            img_rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
            axes[i].imshow(img_rgba)
            axes[i].set_title(f"Hạt {i+1}: {img_rgba.shape[1]}x{img_rgba.shape[0]}px", fontsize=10)
        axes[i].axis('off')

    plt.suptitle("MẪU HẠT LÚA ĐÃ BÓC TÁCH THÀNH CÔNG (NỀN TRONG SUỐT RGBA)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Chưa có ảnh hạt lúa nào trong thư mục đích.")''')
]

def save_nb(cells, path):
    nb = {
        'nbformat': 4,
        'nbformat_minor': 2,
        'metadata': {
            'colab': {'provenance': [], 'gpuType': 'T4'},
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU'
        },
        'cells': cells
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f'✅ Tạo thành công: {path}')

save_nb(cells_train, 'CODE/TRAIN_YOLO_SEGMENT.ipynb')
save_nb(cells_sahi, 'CODE/SAHI_CROP_GRAINS.ipynb')
