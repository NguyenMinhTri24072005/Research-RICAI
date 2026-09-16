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

cells_extraction = [
    make_cell('markdown', '''# 🌾 PIPELINE TỰ ĐỘNG TRÍCH XUẤT ĐẶC TRƯNG DATASET AI (RICE VISION AI)
**Đề tài Nghiên cứu Khoa học**: Hệ thống thị giác máy tính và học sâu phục vụ ước lượng số lượng và đánh giá phẩm cấp hạt giống lúa.

---

### 🎯 Quy trình Hoạt động & Mục tiêu:
1. **Duyệt qua từng mẫu** trong file Excel dữ liệu nhập tay `2_Manual_Records/manual_data.xlsx`.
2. **Đối chiếu ảnh chụp**: Tra cứu file ảnh tương ứng trong `1_Raw_Images/M###/M###X.jpg`.
   - ⚪ *Nếu chưa có ảnh*: Tự động để trống toàn bộ các trường trích xuất AI (NaN / blank).
   - 🟢 *Nếu đã có ảnh*: Tiến hành bóc tách và phân tích toàn diện qua 2 giai đoạn:
3. **Giai đoạn 1 (Bóc tách & Phân loại)**:
   - Nhận diện miệng ly (Canny + `fitEllipse`), khử nghiêng, tính tỷ lệ `pixels_per_mm`.
   - Bóc tách polygon hạt lúa bằng **SAHI + YOLO-seg**.
   - Phân loại phẩm cấp bằng **DenseNet121 CNN** $\to$ Lưu ảnh hạt nguyên vào `3_AI_Extracted/CROPPED_GRAINS/M###/M###X/`.
4. **Điểm Dừng Kiểm Duyệt Thủ Công (Manual Quality Control)**:
   - Bạn mở thư mục xem mắt thường các ảnh hạt nguyên, tự tay xóa hạt dính chùm hoặc mẩu vụn (nếu có).
5. **Giai đoạn 2 (Tính toán & Xuất Excel Hoàn chỉnh)**:
   - Đo đạc kích thước 2D (Dài, Rộng, Diện tích: Min, Max, Mean).
   - Mô hình hóa thể tích 3D Ellipsoid ($V = \\frac{4}{3}\\pi abc$: Min, Max, Mean).
   - Tính toán ước lượng số hạt theo thể tích và độ đồng đều.
   - Xuất file kết quả hoàn chỉnh `3_AI_Extracted/ai_extracted_dataset.xlsx`.'''),

    make_cell('code', '''# Bước 1: Kết nối Google Drive & Cài đặt thư viện
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics sahi tensorflow openpyxl matplotlib seaborn opencv-python'''),

    make_cell('code', '''# Bước 2: Khai báo Đường dẫn & Import Modules
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN TRÊN GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES"

# Thêm CODE vào sys.path để import các module
code_dir = os.path.join(BASE_PATH, "CODE")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Import các module độc lập
from modules.container_detector import detect_container_and_scale, draw_container_overlay
from modules.grain_segmenter import segment_grains_sahi
from modules.grain_classifier import GrainClassifier
from modules.ellipsoid_geometry import compute_single_grain_metrics, compute_folder_grains_summary
from modules.uniformity_evaluator import evaluate_batch_uniformity
from modules.dataset_extractor import DatasetExtractor

# Đường dẫn Model Trọng số
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt")
CNN_MODEL_PATH  = os.path.join(BASE_PATH, "RESULTS/CNN_DenseNet121_Trained/best_rice_densenet121.keras")

print("=" * 60)
print(f"📁 Thư mục gốc dự án  : {BASE_PATH}")
print(f"🎯 Trọng số YOLO-seg  : {YOLO_MODEL_PATH}")
print(f"🧠 Trọng số CNN       : {CNN_MODEL_PATH}")
print("=" * 60)'''),

    make_cell('code', '''# Bước 3: Khởi tạo Bộ điều phối Dataset Extractor
extractor = DatasetExtractor(
    base_dir=BASE_PATH,
    yolo_model_path=YOLO_MODEL_PATH,
    cnn_model_path=CNN_MODEL_PATH,
    packing_fraction=0.62  # Hệ số chèn lấp khối lúa (~0.60 - 0.64)
)

# Nạp model vào GPU
extractor.init_models(conf=0.50)'''),

    make_cell('code', '''# Bước 4: [GIAI ĐOẠN 1] — Bóc tách vật chứa & Lọc hạt nguyên lưu vào CROPPED_GRAINS
# Quá trình này sẽ tự động:
#  1. Duyệt qua 285 mẫu trong manual_data.xlsx
#  2. Đối chiếu ảnh trong 1_Raw_Images (nếu chưa có ảnh thì bỏ qua)
#  3. Khoanh miệng ly và bóc tách từng hạt lúa bằng SAHI YOLO-seg
#  4. Phân loại CNN và lưu riêng các hạt nguyên vào CROPPED_GRAINS/M###/M###X/
stage1_results = extractor.run_stage1_crop_and_classify(
    conf_yolo=0.50,
    slice_size=512,
    overlap=0.25,
    skip_existing=False  # Đổi True nếu muốn tiếp tục chạy nối tiếp
)'''),

    make_cell('markdown', '''# 🔍 BƯỚC 5: ĐIỂM DỪNG TƯƠNG TÁC — KIỂM DUYỆT THỦ CÔNG (TÙY CHỌN)
---
> **Hướng dẫn thực hiện**:
> 1. Bạn mở tab **Files** trên Google Colab hoặc truy cập trực tiếp Google Drive theo đường dẫn:
>    📁 `.../DATASET_BUILDER/3_AI_Extracted/CROPPED_GRAINS/`
> 2. Trong này chứa các thư mục con tương ứng với từng mẫu (ví dụ: `M001/M001A/`, `M001/M001B/`,...).
> 3. Bạn có thể mở xem các ảnh hạt nguyên:
>    - Nếu thấy ảnh nào bị **dính 2 hạt chùm** hoặc **mẩu vụn rác**, bạn chỉ cần **bấm chuột phải và Xóa (Delete) file ảnh đó**.
>    - Những hạt còn lại trong folder sẽ là tập mẫu chuẩn tuyệt đối để tính toán thể tích và kích thước.
> 4. Sau khi kiểm tra xong, hãy bấm chạy tiếp **Cell 6 bên dưới**!'''),

    make_cell('code', '''# Bước 6: [GIAI ĐOẠN 2] — Tính toán Kích thước 2D & Thể tích 3D Ellipsoid và Xuất Excel
# Quá trình này sẽ tự động:
#  1. Đọc lại các ảnh hạt nguyên đã chọn lọc trong CROPPED_GRAINS
#  2. Tính Min, Max, Mean của Chiều dài, Chiều rộng, Diện tích (mm²)
#  3. Tính Min, Max, Mean của Thể tích 3D Ellipsoid V = (4/3)*pi*a*b*c (mm³)
#  4. Ước lượng tổng số hạt theo thể tích khối lúa và tính Độ đồng đều (%)
#  5. Ghi vào bảng và xuất file Excel hoàn chỉnh.

excel_output_path = extractor.run_stage2_calculate_and_export(
    output_excel_filename="ai_extracted_dataset.xlsx"
)'''),

    make_cell('code', '''# Bước 7: Trực quan hóa Kết quả Trích xuất & Phân tích Tương quan
csv_path = os.path.join(BASE_PATH, "DATASET_BUILDER/3_AI_Extracted/ai_extracted_dataset.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df_valid = df[df['Image_Status'] == 'FOUND'].copy()
    
    print("=" * 80)
    print(f"📊 TỔNG KẾT BẢNG DỮ LIỆU ĐÃ TRÍCH XUẤT:")
    print(f"   • Tổng số mẫu trong danh sách : {len(df)} mẫu")
    print(f"   • Số mẫu đã trích xuất AI     : {len(df_valid)} mẫu")
    print(f"   • Số mẫu đang chờ chụp thêm   : {len(df) - len(df_valid)} mẫu")
    print("=" * 80)

    # Chuyển đổi các cột số
    df_valid['Actual_Count'] = pd.to_numeric(df_valid['Actual_Count'], errors='coerce')
    df_valid['Bulk_Rice_Volume_mm3'] = pd.to_numeric(df_valid['Bulk_Rice_Volume_mm3'], errors='coerce')
    df_valid['Grain_Volume_mm3_Mean'] = pd.to_numeric(df_valid['Grain_Volume_mm3_Mean'], errors='coerce')
    df_valid['Estimated_Total_Seeds_Hybrid'] = pd.to_numeric(df_valid['Estimated_Total_Seeds_Hybrid'], errors='coerce')

    plt.figure(figsize=(14, 5))

    # Biểu đồ 1: Tương quan giữa Thể tích khối lúa và Số hạt thực tế
    plt.subplot(1, 2, 1)
    sns.regplot(
        data=df_valid,
        x='Bulk_Rice_Volume_mm3',
        y='Actual_Count',
        color='#2E7D32',
        scatter_kws={'alpha': 0.6, 's': 30}
    )
    plt.title('Tương quan Thể tích Khối lúa vs Số hạt Thực tế (Actual Count)', fontweight='bold')
    plt.xlabel('Thể tích khối lúa ($mm^3$)')
    plt.ylabel('Số lượng hạt thực tế (hạt)')
    plt.grid(True, alpha=0.3)

    # Biểu đồ 2: Phân phối Thể tích 3D trung bình từng hạt lúa
    plt.subplot(1, 2, 2)
    sns.histplot(
        df_valid['Grain_Volume_mm3_Mean'].dropna(),
        kde=True,
        color='#1565C0',
        bins=20
    )
    plt.title('Phân phối Thể tích 3D Ellipsoid Trung bình ($mm^3$ / hạt)', fontweight='bold')
    plt.xlabel('Thể tích hạt trung bình ($mm^3$)')
    plt.ylabel('Tần suất')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(BASE_PATH, "DATASET_BUILDER/3_AI_Extracted/dataset_correlation_charts.png")
    plt.savefig(chart_path, dpi=300)
    print(f"🖼️ Đã lưu biểu đồ phân tích tương quan tại: {chart_path}")
    plt.show()
    
    # Hiển thị 5 dòng đầu tiên
    print("\n📋 5 Dòng đầu tiên của Dataset vừa trích xuất:")
    display_cols = ['Sample_ID', 'Actual_Count', 'Image_Status', 'Pixels_Per_mm', 'Grain_Length_mm_Mean', 'Grain_Volume_mm3_Mean', 'Estimated_Total_Seeds_Hybrid']
    print(df_valid[display_cols].head())
else:
    print(f"❌ Không tìm thấy file CSV tại: {csv_path}")''')
]

nb = {
    'nbformat': 4,
    'nbformat_minor': 2,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells_extraction
}

for path in ['DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb', 'CODE/AI_DATASET_EXTRACTION_PIPELINE.ipynb']:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã tạo thành công Notebook: {path}")
