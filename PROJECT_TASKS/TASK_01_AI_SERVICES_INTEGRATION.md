# TASK-01: HOÀN TẤT TÍCH HỢP HỒI QUY 31 BIẾN VÀO AI_SERVICES

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Trạng thái kỹ thuật hiện tại

`AI_SERVICES/app.py` và `AI_SERVICES/regression_engine.py` đã được mở rộng:

- sử dụng `pathlib.Path`, tự dò trọng số local/Colab và dùng `tempfile`;
- đã tách các module xử lý vào `AI_SERVICES/modules/`;
- đã có `get_regression_artifacts()` để nạp model/scaler theo lazy loading.
- `/predict` dựng vector `ALL_31_FEATURES`, gọi `scaler.transform()` (khi có
  scaler) và `model.predict()` qua Extra Trees; OLS là fallback.
- frontend đã nhận `estimation.regression_est` và `metrics_summary`.

Các yêu cầu về manifest, validation bắt buộc và integration test vẫn chưa có
bằng chứng hoàn thành. Không có `AI_SERVICES/pipeline_logic/main_pipeline.py`;
không tạo task dựa trên đường dẫn này.

### 2. Mục tiêu

Đưa mô hình hồi quy đã được huấn luyện vào luồng suy luận thực tế, với schema
đặc trưng tái lập được và payload rõ ràng cho frontend.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể

- [ ] **3.1. Khóa contract model và schema đặc trưng**
  - [x] Dùng thứ tự `ALL_31_FEATURES` từ `LINEAR_REGRESSION_MODEL/train_linear_regression.py`.
  - [ ] Tạo một artifact manifest ghi rõ: tên model, scaler đi kèm, danh sách 31
    biến, phiên bản dataset và ngày huấn luyện.
  - [ ] Không ghép nhầm `best_linear_regression_model.joblib` với scaler sinh ra từ
    một lần train khác. Nếu model/scaler không cùng lần train, huấn luyện và lưu
    lại thành một cặp mới.

- [x] **3.2. Trích xuất đầy đủ 31 biến trong `/predict`**
  - [x] Dùng các module hiện có để tạo nhóm biến vật chứa, ảnh bề mặt, kích thước
    2D/3D và độ đồng đều.
  - [x] Đối chiếu tên biến và thứ tự với dataset cuối trước khi gọi model.
  - [x] Kiểm tra `Estimated_Total_Seeds_Hybrid` và mọi biến đầu vào để bảo đảm không
    dùng trực tiếp hoặc gián tiếp `Actual_Count` tại thời điểm suy luận.

- [ ] **3.3. Suy luận hồi quy và xử lý lỗi rõ ràng**
  - [ ] Nạp đúng model/scaler theo manifest.
  - [x] Gọi `scaler.transform([features])` rồi `model.predict(...)`.
  - [ ] Nếu thiếu model, thiếu feature hoặc giá trị không hợp lệ, trả lỗi có mã và
    thông báo cụ thể; không âm thầm thay 0 cho biến bắt buộc.

- [ ] **3.4. Chuẩn hóa payload API và frontend**
  - [x] Giữ `estimation.final` làm kết quả hiển thị chính.
  - [ ] Bổ sung `estimation.regression_est`, `estimation.geometry_est`,
    `estimation.weight_est`, cờ phương pháp đã dùng và `feature_schema_version`.
  - [x] Trả `metrics_summary` cho frontend; giao diện hiển thị tối thiểu số hạt
    bề mặt, kích thước trung bình, độ đồng đều và trạng thái model.

- [ ] **3.5. Kiểm thử tích hợp**
  - Viết fixture ảnh + form cố định, kiểm tra `/api/status` và `/predict` trên
    Windows.
  - Ghi runtime thực đo trên CPU/GPU thay vì cam kết thời gian không có baseline.

### Trạng thái kiểm toán (16/09/2026)

- [x] Schema 31 biến được khai báo tập trung và `/predict` tạo feature vector,
  suy luận Extra Trees/OLS và trả `regression_est`.
- [x] `/api/status` và frontend hiển thị trạng thái/kết quả hồi quy cơ bản.
- [ ] Chưa có manifest phiên bản dataset–model–scaler để chứng minh model và
  scaler là một cặp tương thích.
- [ ] Thiếu validation rõ ràng cho feature bắt buộc; luồng hiện tại có giá trị
  mặc định/fallback nên chưa đạt yêu cầu báo lỗi cụ thể của task.
- [ ] Chưa có integration test fixture tái lập được cho `/api/status` và
  `/predict` trên Windows, cũng chưa có số đo runtime.

### Tiêu chí hoàn thành toàn bộ task

1. [ ] `/api/status` xác nhận đủ YOLO, CNN, model hồi quy và scaler theo cùng
   manifest.
2. [ ] `/predict` tạo được vector 31 biến hợp lệ, chạy dự đoán hồi quy và trả JSON
   đúng schema đã công bố.
3. [ ] Có ít nhất một integration test tái lập được kết quả trên fixture.
4. [ ] Không còn tham chiếu runtime bắt buộc tới đường dẫn `/content/drive` khi chạy
   local Windows.
