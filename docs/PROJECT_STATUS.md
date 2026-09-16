# Project Status

Updated: 2026-09-17

## Current Phase

Dự án đang ở giai đoạn hoàn thiện nền tảng kỹ thuật và dịch vụ AI (hoàn tất TASK-00 và TASK-01), sẵn sàng bước vào giai đoạn giải trình mô hình (TASK-02) và thu thập mở rộng tập dữ liệu chuẩn thức (TASK-03). Kiến trúc cốt lõi đã được định hình với điện thoại làm camera node qua QR/HTTPS và máy tính/Colab làm host xử lý thị giác máy tính kết hợp hồi quy 31 biến.

## Completed

- **TASK-00 — Quy chuẩn thu thập dữ liệu, camera điện thoại và quản trị dữ liệu (Kỹ thuật)**
  - Ứng dụng `CAPTURE_APP` hoạt động ổn định với cơ chế kết nối điện thoại làm camera node qua QR Code và HTTPS tự ký.
  - Tự động sinh `Capture_Timestamp` cho từng mẫu; đã loại bỏ hai trường `Capture_Batch` và `Device_ID` khỏi luồng nhập liệu thực tế theo yêu cầu vận hành.
  - Hỗ trợ thao tác chạm lấy nét (tap-to-focus) và bật/tắt flash trực tiếp trên giao diện camera web di động.
  - Lưu trữ đồng bộ SQLite và Excel với cấu trúc ảnh phẳng (`M####.jpg`).
  - *Evidence:* 18/18 unit tests trong `DATASET_BUILDER/CAPTURE_APP/tests/` đạt PASS; tài liệu quy chuẩn `PROJECT_TASKS/TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md`.
  - *Ghi chú:* Về mặt kỹ thuật và phần mềm đã hoàn tất; việc nghiệm thu thực địa của nhóm trước khi thu thập quy mô lớn sẽ được tiến hành ở đầu TASK-03.

- **TASK-01 — Tích hợp mô hình hồi quy 31 biến vào AI_SERVICES**
  - Khóa contract và schema đặc trưng duy nhất tại `AI_SERVICES/feature_schema.py` với đúng 31 biến, thứ tự xác định, đơn vị rõ ràng, không chứa `Actual_Count` hay rò rỉ nhãn.
  - Tạo manifest `AI_SERVICES/artifacts/manifest.json` định danh bundle `rice_vision_extratrees_31v1_20260824`, liên kết model Extra Trees Regressor và `StandardScaler` có cùng nguồn gốc huấn luyện (24/08/2026).
  - Triển khai `AI_SERVICES/model_registry.py` nạp và xác thực model/scaler đồng bộ, kiểm tra $n\_features\_in\_ = 31$, smoke predict và loại bỏ lỗi cache vĩnh viễn.
  - Tinh chỉnh `AI_SERVICES/regression_engine.py` và `AI_SERVICES/app.py`: kiểm tra miền giá trị đầu vào (HTTP 422), bỏ fallback OLS ngầm, hỗ trợ `estimator_mode` (`auto`, `regression`, `geometry`, `weight`) và `debug=True`.
  - Endpoint `/api/status` phản ánh mức độ sẵn sàng thực tế không làm lộ đường dẫn ổ đĩa tuyệt đối; Express gateway và capture server bảo toàn mã lỗi và phát sự kiện SSE/WebSocket chống treo giao diện.
  - *Evidence:* 22/22 automated tests đạt 100% PASS trong `reports/task_01/test_report.json`; báo cáo nghiệm thu kỹ thuật `reports/task_01/verification.md`; báo cáo kiểm toán parity `docs/TASK_01_AUDIT.md`; CLI `verify_artifacts.py` đạt PASS; chạy thành công E2E trên fixture mẫu thực `M001A.jpg` đạt sai số 0 hạt (Ground Truth: 85, Dự đoán: 85, MAPE: 0.00%).

## In Progress

### TASK-03 — Thu thập và mở rộng dataset bằng CAPTURE_APP

- **Trạng thái:** In Progress — ứng dụng đã sẵn sàng, dữ liệu thử nghiệm lịch sử tồn tại nhưng dataset chính thức chưa khóa.
- **Hiện trạng dữ liệu thực tế:**
  - Có 57 thư mục mẫu ảnh gốc với 261 file ảnh trong `DATASET_BUILDER/1_Raw_Images/`.
  - File dataset trích xuất `DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv` chứa 285 dòng (254 mẫu hợp lệ `Image_Status == FOUND` tương ứng 57 cốc mẫu vật lý).
- **Chưa hoàn thành:**
  - Chưa hoàn tất 55 mẫu pilot có nhật ký thiết lập đối chiếu ảnh–Excel một-một độc lập.
  - Chưa đạt mục tiêu tối thiểu 100 mẫu vật lý có đo đếm `Actual_Count` độc lập cho benchmark chính thức.
  - **Official benchmark dataset not finalized.**

### TASK-04 — Benchmark 14 mô hình hồi quy (Provisional)

- **Trạng thái:** In Progress — có kết quả thăm dò sơ bộ trên dữ liệu chưa khóa; benchmark chính thức đang chờ TASK-03.
- **Hiện trạng:**
  - Có kết quả thăm dò sơ bộ trong `LINEAR_REGRESSION_MODEL/results/` (`training_evaluation_report.csv`, `tree_models_evaluation_report.csv`) từ ngày 24/08/2026.
  - Extra Trees Regressor đạt $R^2 = 0.9999$, $MAE = 0.45$ hạt trên tập test ngẫu nhiên 20%.
- **Chưa hoàn thành:**
  - Chưa có script `benchmark_all_regressors.py` chạy đúng danh mục 14 mô hình đã đăng ký.
  - Chưa áp dụng `Pipeline` lồng scaling vào từng fold (script cũ `train_linear_regression.py` chuẩn hóa toàn bộ dữ liệu trước khi split).
  - Chưa có phân chia dữ liệu độc lập theo ngày chụp và nested cross-validation.
  - **PROVISIONAL BENCHMARK ONLY (Dataset not yet finalized).**

### TASK-02 — Module giải trình kết quả AI / XAI

- **Trạng thái:** Not Started — đặc tả yêu cầu và công thức toán học đã xác định tại `PROJECT_TASKS/TASK_02_EXPLAINABLE_AI_XAI.md`, mã nguồn chưa triển khai.

### TASK-05 — Đánh giá Few-shot / Transfer Learning cho loại hạt khác

- **Trạng thái:** Not Started — mức ưu tiên P3, hoãn lại sau khi hoàn thành báo cáo đợt 1 và benchmark chính thức.

## Next Actions

1. **Triển khai TASK-02 (Explainability / XAI):**
   - Xây dựng module `AI_SERVICES/modules/explainability.py` tính toán đóng góp tuyến tính $Contribution_i = w_i \frac{x_i - \mu_i}{\sigma_i}$.
   - Trả về top 5 đặc trưng làm tăng/giảm số lượng hạt và sinh đoạn giải thích tiếng Việt.
   - Bổ sung trường giải trình vào response `/predict` hoặc endpoint `/explain`.
2. **Tiến hành thu thập dữ liệu TASK-03:**
   - Thực hiện phiên chụp pilot 55 mẫu vật lý có ghi nhật ký điều kiện chụp.
   - Mở rộng thu thập để đạt $\ge 100$ mẫu vật lý hợp lệ có `Actual_Count` đếm thủ công độc lập.
   - Khóa phiên bản dataset chính thức cho bài báo.
3. **Triển khai TASK-04 (Official Benchmark 14 Models):**
   - Xây dựng `benchmark_all_regressors.py` với `sklearn.Pipeline` tích hợp scaling trong từng fold cross-validation.
   - Chạy và ghi nhận kết quả trên tập test độc lập của dataset chính thức đã khóa.

## Blockers

- **None currently identified:** Không có blocker kỹ thuật nào trong hạ tầng suy luận của `AI_SERVICES` và `CAPTURE_APP`. Môi trường Windows local đã được cấu hình tương thích UTF-8 và đường dẫn tương đối độc lập.

## Latest Results

### TASK-01 Regression Integration (Verified)

- **Feature Schema:**
  - Số lượng biến: 31 biến (đúng thứ tự và đơn vị, `ddof = 0` cho toàn bộ độ lệch chuẩn).
  - Phiên bản schema: `31v1`.
  - Trạng thái: Verified (`AI_SERVICES/feature_schema.py`).
- **Model Bundle:**
  - Bundle ID: `rice_vision_extratrees_31v1_20260824`.
  - Estimator: `ExtraTreesRegressor` (n_features_in = 31).
  - Preprocessing: `StandardScaler` (n_features_in = 31).
  - Trạng thái: Verified (`AI_SERVICES/scripts/verify_artifacts.py` PASS).
- **API Endpoints:**
  - `GET /health`: PASS (HTTP 200).
  - `GET /api/status`: PASS (HTTP 200, readiness = "ready", component status verified, không leak path).
  - `POST /predict`: PASS (HTTP 422 khi dữ liệu sai; HTTP 200 khi hợp lệ kèm timings và metrics).
- **End-to-End Fixture Test (Sample M001a):**
  - File ảnh: `DATASET_BUILDER/1_Raw_Images/M001/M001A.jpg` ($D=1.78\text{ cm}, H=3.39\text{ cm}, H_{trống}=1.09\text{ cm}, W=2.69\text{ g}$).
  - Ground Truth (`Actual_Count`): **85 hạt**.
  - Kết quả dự đoán hệ thống: **85 hạt** (`regression_ExtraTrees`, raw output = 85.2).
  - Sai số tuyệt đối: **0 hạt** (MAPE: **0.00%**).
  - Thời gian xử lý CPU: 272.7s (48 lát cắt SAHI + YOLOv8s-seg + CNN DenseNet121 + Hồi quy).
- **Automated Test Suite:**
  - Tổng số test: 22 tests.
  - Kết quả: **22/22 PASS (100%)**, 0 failures, 0 errors (`reports/task_01/test_report.json`).

### TASK-00 Capture Protocol & Mobile Node (Verified Kỹ Thuật)

- **Automated Test Suite:** 18/18 PASS trong `DATASET_BUILDER/CAPTURE_APP/tests/`.
- **Khả năng:** Chạm lấy nét (tap-to-focus), bật/tắt flash, tự sinh `Capture_Timestamp`, đồng bộ QR/HTTPS.

### Exploratory Model Evaluation (Task 04 Provisional)

- **Tập dữ liệu thăm dò:** 55 mẫu trích xuất trước đây, split ngẫu nhiên 80/20.
- **Extra Trees Regressor:** $R^2 = 0.9999$, $MAE = 0.45$ hạt, $MAPE = 0.34\%$.
- *Lưu ý:* Đây là kết quả thăm dò sơ bộ trên tập chia ngẫu nhiên (chưa gom nhóm cốc vật lý); không phải kết quả benchmark chính thức.

## Important Files

- `docs/PROJECT_STATUS.md`: Báo cáo trạng thái tổng thể dự án.
- `docs/TASK_01_AUDIT.md`: Báo cáo kiểm toán P0 và ma trận đối chiếu 31 biến.
- `reports/task_01/verification.md`: Báo cáo nghiệm thu kỹ thuật TASK-01.
- `reports/task_01/test_report.json`: Nhật ký kết quả 22 bài kiểm thử tự động.
- `AI_SERVICES/app.py`: FastAPI inference service chính.
- `AI_SERVICES/feature_schema.py`: Single Source of Truth cho 31 đặc trưng hồi quy.
- `AI_SERVICES/model_registry.py`: Quản lý nạp và kiểm định model bundle.
- `AI_SERVICES/regression_engine.py`: Module suy luận hồi quy cây và phương trình.
- `AI_SERVICES/artifacts/manifest.json`: Metadata phiên bản hóa model và scaler.
- `AI_SERVICES/scripts/verify_artifacts.py`: CLI kiểm tra tính toàn vẹn của artifacts.
- `AI_SERVICES/scripts/run_tests.py`: CLI chạy toàn bộ bộ kiểm thử TASK-01.
- `PROJECT_TASKS/README.md`: Master task board và lộ trình nghiên cứu.
- `PROJECT_TASKS/TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md`: Quy chuẩn thu thập dữ liệu.
- `PROJECT_TASKS/TASK_01_AI_SERVICES_INTEGRATION.md`: Kế hoạch và tiến độ TASK-01.
- `PROJECT_TASKS/TASK_02_EXPLAINABLE_AI_XAI.md`: Đặc tả giải trình mô hình XAI.
- `PROJECT_TASKS/TASK_03_DATASET_EXPANSION.md`: Kế hoạch thu thập mẫu và QC.
- `PROJECT_TASKS/TASK_04_REGRESSION_BENCHMARK_PAPER.md`: Kế hoạch benchmark 14 mô hình.
- `PROJECT_TASKS/TASK_05_FEW_SHOT_TRANSFER_LEARNING.md`: Đặc tả nghiên cứu few-shot.

## Current Branch

- **Branch:** `main`
- **Latest commit:** `fca9e4b cập nhật các task`
- **Working tree:** Có các thay đổi chưa commit trong `AI_SERVICES/`, `DATASET_BUILDER/CAPTURE_APP/`, `PROJECT_TASKS/`, `docs/` và `reports/`.
