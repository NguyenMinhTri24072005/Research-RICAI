# AI_SERVICES — Kho Lưu Trữ Mô Hình Triển Khai (Deployment Artifacts)

Thư mục này dùng để lưu trữ các trọng số (weights), cấu trúc phân loại, và tham số tiền xử lý được tuyển chọn phục vụ triển khai chính thức của hệ thống Rice Vision AI.

> **Lưu ý phân định kiến trúc:**
> - `AI_SERVICES/artifacts/`: Chỉ chứa các tệp nhị phân weights, checkpoint và metadata (`.pt`, `.keras`, `.h5`, `.joblib`, `.json`).
> - `AI_SERVICES/src/rice_ai/models/`: Chứa mã nguồn Python của các loader nạp mô hình (`vision_models.py`, `regression_loader.py`).

---

## 1. Cấu Trúc 3 Nhánh Model

```text
AI_SERVICES/artifacts/
├── yolo/
│   └── <tên_mô_hình>/
│       └── best.pt                  # Trọng số YOLO segmentation (SAHI compatible)
├── cnn/
│   └── <tên_mô_hình>/
│       └── best.keras (hoặc .h5)    # Phân loại hạt (GrainClassifier DenseNet)
├── regression/
│   └── <tên_mô_hình>/               # Thư mục bundle hồi quy (Regression Folder Contract)
│       ├── model.joblib             # Bắt buộc: Estimator scikit-learn
│       ├── scaler.joblib            # Bắt buộc: Scaler 31 chiều đồng bộ
│       ├── feature_schema.json      # Bắt buộc: Danh sách 31 đặc trưng và phiên bản
│       ├── config.json              # Tùy chọn: Siêu tham số và metadata
│       └── metrics.json             # Tùy chọn: Thống kê kiểm thử
├── manifest.json                    # Bằng chứng lịch sử của đợt tích hợp TASK-01 (không điều khiển runtime mới)
└── README.md
```

---

## 2. Quy Chuẩn Thư Mục Hồi Quy (Regression Folder Contract)

Khi triển khai mô hình hồi quy vào `artifacts/regression/<tên_mô_hình>/`, gói bundle phải tuân thủ hợp đồng:
- **`model.joblib`**: Đối tượng estimator đã fit (`predict` nhận mảng 31 chiều).
- **`scaler.joblib`**: Đối tượng tiền xử lý đã fit (`transform` mảng 31 chiều ra 31 chiều).
- **`feature_schema.json`**: Chứa danh sách đúng 31 đặc trưng theo thứ tự huấn luyện (`features`) và phiên bản (`schema_version`: `"31v1"`).
- **`config.json`** (tùy chọn): Nếu mô hình đặc biệt không cần chuẩn hóa (`"preprocessing": "none"`), loader sẽ không đòi hỏi `scaler.joblib`.
- **Tuyệt đối không dùng `pipeline.joblib` trong runtime**: Tránh rủi ro double-scaling hoặc che giấu thứ tự tiền xử lý.

---

## 3. Quy Trình Chuyển Từ Nghiên Cứu (R&D) Sang Triển Khai (Production)

1. **Giai đoạn R&D**:
   Lập trình viên có thể cấu hình trực tiếp đường dẫn trỏ đến thư mục xuất của quá trình huấn luyện tại `AI_SERVICES/.env`:
   ```dotenv
   YOLO_MODEL_PATH=../RESULTS/all-new-data-v1.yolov8_yolov8s-seg_trained/weights/best.pt
   CNN_MODEL_PATH=../RESULTS/CNN_DenseNet121_Trained/best_v3_step2.keras
   REGRESSION_MODEL_DIR=../LINEAR_REGRESSION_MODEL/models/ard/20260917_150243_1b4faf
   ```

2. **Giai đoạn Triển Khai (Deploy)**:
   - Sao chép tệp weights YOLO vào `artifacts/yolo/<tên_mô_hình>/best.pt`.
   - Sao chép tệp mô hình CNN vào `artifacts/cnn/<tên_mô_hình>/best.keras`.
   - Sao chép toàn bộ thư mục bundle hồi quy vào `artifacts/regression/<tên_mô_hình>/`.
   - Cập nhật biến môi trường tại `AI_SERVICES/.env`:
     ```dotenv
     YOLO_MODEL_PATH=artifacts/yolo/<tên_mô_hình>/best.pt
     CNN_MODEL_PATH=artifacts/cnn/<tên_mô_hình>/best.keras
     REGRESSION_MODEL_DIR=artifacts/regression/<tên_mô_hình>
     ```
   - Khởi động lại dịch vụ `app.py`. Hệ thống tự động nạp snapshot mới mà không cần sửa code, cập nhật SHA-256 thủ công hay chỉnh sửa manifest.
