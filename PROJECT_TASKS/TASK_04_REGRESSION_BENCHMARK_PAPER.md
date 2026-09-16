# TASK-04: BENCHMARK 14 MÔ HÌNH HỒI QUY VÀ KẾT QUẢ PHỤC VỤ BÀI BÁO

> **Căn cứ:** ghi chú “chạy nhiều regression, MSE” và “mỗi người có các mô hình khác nhau để làm bài toán regression, so sánh để làm báo.” trong `NOTES/note_24_08_2026.txt`.

**Phụ thuộc:** [TASK-00](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md) và [TASK-03](TASK_03_DATASET_EXPANSION.md). Chỉ bắt đầu đánh giá chính thức khi tập dữ liệu, schema đặc trưng và nhãn mục tiêu được chốt.

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Câu hỏi nghiên cứu

Trên cùng một tập đặc trưng và cùng một phép chia dữ liệu, họ mô hình nào dự đoán `Actual_Count` tốt nhất, ổn định nhất và đủ giải thích được? Mục tiêu là đối sánh công bằng, không phải tìm kết quả tốt nhất trên tập test bằng cách thử lặp lại.

Trước khi train, kiểm toán nguồn của 31 đặc trưng và nhãn. Đặc biệt không dùng một biến suy ra trực tiếp từ `Actual_Count` hoặc một target proxy như `Estimated_Total_Seeds_Hybrid` làm nhãn công bố nếu chưa nêu rõ bản chất và sai số của nó.

### 2. Danh mục cố định: 14 mô hình chính

| Họ mô hình | Mô hình | Số lượng |
| --- | --- | ---: |
| Tuyến tính/robust | OLS, Ridge, Lasso, ElasticNet, Huber | 5 |
| Cây/ensemble | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost | 5 |
| Phi tuyến | Linear SVR, RBF SVR, KNN Regressor, MLP Regressor | 4 |
| **Tổng** |  | **14** |

Bayesian Ridge, LightGBM và CatBoost có thể chạy như **mở rộng thăm dò**, nhưng phải báo cáo tách biệt; không được ghi là một phần của bảng xếp hạng 14 mô hình đã đăng ký.

### 3. Giao thức đánh giá tái lập được

1. **Chốt dữ liệu và test độc lập.** Giữ lại 20% dữ liệu làm test một lần, ưu tiên tách theo `Capture_Batch`/thiết bị/ngày. Nếu metadata chưa đủ để group split, dùng `random_state=42` và nêu rõ đây là giới hạn.
2. **Không rò rỉ dữ liệu.** Imputation, scaling, lựa chọn đặc trưng và tuning đều đặt trong `sklearn.Pipeline` và fit trong từng fold train. Không tính mean/std trên toàn bộ dữ liệu trước khi tách train/test.
3. **Tuning trên train.** Chạy nested cross-validation (hoặc GroupKFold khi có batch) trên 80% train; test độc lập chỉ dùng đúng một lần cho bảng kết quả cuối.
4. **Cùng điều kiện.** Cố định seed, feature set, ngân sách tuning, số fold và môi trường phiên bản cho mọi mô hình. Mỗi ANN do thành viên thực hiện cũng dùng chính split và ngân sách này.
5. **Báo cáo metric.** Bắt buộc: R², adjusted R² (khi phù hợp), MAE, MSE, RMSE, thời gian suy luận. Dùng sMAPE hoặc MAPE có điều kiện vì MAPE không ổn định khi `Actual_Count` gần 0. Báo cáo khoảng tin cậy bootstrap cho các metric chính.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể

- [ ] Tạo `benchmark_all_regressors.py` (hiện chưa có) và file cấu hình tái lập được: schema, target, split, seed, fold, không gian siêu tham số và phiên bản thư viện.
- [ ] Điều chỉnh [`train_linear_regression.py`](../LINEAR_REGRESSION_MODEL/train_linear_regression.py): chuyển scaler vào pipeline để chỉ fit bằng train/fold train.
- [ ] Xuất `benchmark_table.csv`, bảng LaTex, cấu hình chạy, dự đoán test và residual của từng mô hình vào `LINEAR_REGRESSION_MODEL/results/`.
- [ ] Vẽ hình 300 DPI: actual-vs-predicted, residual/Q-Q plot, metric comparison. Thêm feature importance bằng hệ số chuẩn hoá cho tuyến tính và permutation importance cho các mô hình khác.
- [ ] Ghi kiến trúc, số tham số, epoch, early stopping và seed của từng ANN do thành viên xây dựng; so sánh công bằng với các baseline.

### Trạng thái kiểm toán (16/09/2026)

- [x] Đã có báo cáo metric thăm dò cho một nhóm mô hình trong
  `LINEAR_REGRESSION_MODEL/results/`, gồm tree models và một bảng nhiều model.
- [x] Đã có artifact cấu hình/tầm quan trọng đặc trưng cho Extra Trees.
- [ ] Chưa có `benchmark_all_regressors.py`, cấu hình chạy tái lập hoặc bảng
  chính thức cho đúng 14 mô hình đã đăng ký.
- [ ] `train_linear_regression.py` chuẩn hóa toàn bộ dữ liệu trước khi tách
  train/test; chưa đạt yêu cầu pipeline fit trong train/fold train.
- [ ] Chưa có test độc lập khóa theo batch, nested CV, khoảng tin cậy bootstrap,
  residual đầy đủ hoặc artefact bản thảo theo Definition of Done.

### Tiêu chí hoàn thành toàn bộ task

- [ ] Tập test được khoá, sơ đồ chia dữ liệu và mọi bước tiền xử lý có thể chạy lại từ cấu hình.
- [ ] Có đúng 14 mô hình chính, kết quả CV và kết quả test độc lập; mọi mô hình mở rộng được ghi tách biệt.
- [ ] Có bảng metric, khoảng tin cậy, file dự đoán/residual và hình 300 DPI sẵn sàng dùng trong bản thảo.
- [ ] Bản thảo nêu rõ nguồn nhãn, giới hạn của thiết kế thu dữ liệu và nguy cơ tổng quát hoá theo batch/thiết bị.
