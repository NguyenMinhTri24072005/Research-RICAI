# 🌾 LINEAR_REGRESSION_MODEL — Module Huấn Luyện & Thử Nghiệm Hồi Quy

Tài liệu hướng dẫn kiến trúc, cấu hình, thực thi và quản lý artifacts cho module huấn luyện hồi quy ước lượng số hạt lúa giống (31 đặc trưng).

---

## 1. Điểm Vào Duy Nhất (Single Active Entrypoint)

Sau quá trình hợp nhất theo kế hoạch `NOTEBOOK_CONSOLIDATION_PLAN.md`, **điểm vào hoạt động duy nhất** để huấn luyện, so sánh và xuất bundle mô hình hồi quy là:

👉 **`RICE_SEED_REGRESSION_TRAINER.ipynb`**

### Trạng thái các notebook gốc (Legacy / Reference Only)
Ba notebook trước đây được giữ lại nguyên vẹn vì mục đích lưu trữ lịch sử và bảo toàn tính toàn vẹn chuỗi truy xuất (provenance) của các artifact đang triển khai:
* `RICE_SEED_LINEAR_REGRESSION_TRAINER.ipynb` *(Legacy - 6 mô hình OLS/Ridge/Bayesian/Huber/ElasticNet)*
* `RICE_SEED_LINEAR_REGRESSION_TRAINER_v2.ipynb` *(Legacy - bổ sung PLS, ARD, KernelRidge, Stacking, RF, GB, XGBoost)*
* `RICE_SEED_DECISION_TREE_TRAINER.ipynb` *(Legacy - 6 mô hình họ Tree: DT, RF, ET, GB, AdaBoost, HistGB)*

> [!WARNING]
> **Không di chuyển, đổi tên hoặc xóa 3 notebook gốc trên**, vì `AI_SERVICES/artifacts/manifest.json` đang tham chiếu trực tiếp đường dẫn và mã băm SHA-256 của chúng làm bằng chứng xuất xứ (provenance) cho mô hình sản xuất.

---

## 2. Môi Trường Thực Thi Được Xác Minh (Environment)

Môi trường huấn luyện chuẩn đã được thiết lập và kiểm chứng độc lập tại thư mục `LINEAR_REGRESSION_MODEL/.venv` (không can thiệp hoặc nâng cấp `AI_SERVICES/.venv`):

* **Python Version**: 3.12.10 (64-bit)
* **Jupyter Kernel**: `ricai-training`
* **File danh mục phụ thuộc**: [`requirements_training.txt`](requirements_training.txt)
  * `numpy==2.5.3`
  * `pandas==3.0.5`
  * `scikit-learn==1.9.1`
  * `scipy==1.18.1`
  * `joblib==1.6.0`
  * `matplotlib==3.11.2`
  * `seaborn==0.13.2`
  * `openpyxl==3.1.5`
  * `nbformat==5.11.1`
  * `nbconvert==7.17.1`
  * `ipykernel==7.3.0`

### Hướng dẫn kích hoạt & đăng ký Kernel:
```powershell
# Kích hoạt venv
.\LINEAR_REGRESSION_MODEL\.venv\Scripts\Activate.ps1

# Đăng ký kernel cho Jupyter / VS Code
python -m ipykernel install --user --name=ricai-training --display-name="Python (ricai-training)"
```

---

## 3. Danh Mục 19 Cấu Hình Ứng Viên (Model Inventory)

Trong `RICE_SEED_REGRESSION_TRAINER.ipynb`, mỗi mô hình được định nghĩa tại **một code cell riêng biệt** (Cell ID: `MODEL_<model_id>`), cho phép bật/tắt (`enabled = True/False`) và điều chỉnh siêu tham số độc lập:

| STT | model_id | Lớp Scikit-Learn / Thư viện | Tham số chính | Nguồn gốc | Trạng thái mặc định |
|:---:|:---|:---|:---|:---|:---:|
| 1 | `ridge_alpha_0_1` | `Ridge` | `alpha=0.1, random_state=42` | Trainer v1/v2 | Enabled |
| 2 | `ols` | `LinearRegression` | `fit_intercept=True` | Trainer v1/v2 | Enabled |
| 3 | `ridge_alpha_1` | `Ridge` | `alpha=1.0, random_state=42` | Trainer v1/v2 | Enabled |
| 4 | `bayesian_ridge` | `BayesianRidge` | Defaults scikit-learn | Trainer v1/v2 | Enabled |
| 5 | `huber` | `HuberRegressor` | `max_iter=1000` | Trainer v1/v2 | Enabled |
| 6 | `elastic_net` | `ElasticNet` | `alpha=0.01, l1_ratio=0.5, random_state=42` | Trainer v1/v2 | Enabled |
| 7 | `pls` | `PLSRegression` | `n_components=2, scale=True` *(Baseline cố định, không tuned bias)* | Trainer v2 | Enabled |
| 8 | `ard` | `ARDRegression` | Defaults scikit-learn | Trainer v2 | Enabled |
| 9 | `kernel_ridge_rbf` | `KernelRidge` | `kernel='rbf', alpha=1.0, gamma=None` | Trainer v2 | Enabled |
| 10 | `stacking_linear` | `StackingRegressor` | Base: OLS, Ridge(0.1), BayesianRidge; Final: RidgeCV | Trainer v2 | **Disabled** *(Chờ inner isolation)* |
| 11 | `random_forest_v2` | `RandomForestRegressor` | `n_estimators=500, min_samples_leaf=2, random_state=42` | Trainer v2 | Enabled |
| 12 | `gradient_boosting_v2`| `GradientBoostingRegressor` | `n_estimators=500, lr=0.03, max_depth=3, subsample=0.8, random_state=42` | Trainer v2 | Enabled |
| 13 | `xgboost_v2` | `XGBRegressor` | `n_estimators=600, lr=0.03, max_depth=4, subsample=0.8, random_state=42` | Trainer v2 | Optional *(Chỉ khi có xgboost)* |
| 14 | `decision_tree` | `DecisionTreeRegressor` | `random_state=42` | Decision Tree | Enabled |
| 15 | `random_forest_tree` | `RandomForestRegressor` | `n_estimators=100, random_state=42` | Decision Tree | Enabled |
| 16 | `extra_trees` | `ExtraTreesRegressor` | `n_estimators=100, random_state=42` | Decision Tree | Enabled |
| 17 | `gradient_boosting_tree`| `GradientBoostingRegressor` | `n_estimators=100, random_state=42` | Decision Tree | Enabled |
| 18 | `ada_boost` | `AdaBoostRegressor` | `n_estimators=50, random_state=42` | Decision Tree | Enabled |
| 19 | `hist_gradient_boosting`| `HistGradientBoostingRegressor` | `random_state=42` | Decision Tree | Enabled |

### Lưu ý kỹ thuật quan trọng về các ngoại lệ:
* **PLS**: Được cố định `n_components=2` làm baseline chuẩn, ghi rõ không phải siêu tham số tối ưu (nhằm tránh selection bias do tìm n_components trên toàn bộ tập train).
* **Stacking**: Mặc định `enabled = False` với lý do `inner CV preprocessing/group isolation not verified`. Nếu người dùng bật cell này, pipeline sẽ phát cảnh báo/chặn trước huấn luyện để tránh rò rỉ dữ liệu qua inner folds.
* **XGBoost**: Tùy chọn (Optional). Nếu máy chưa cài `xgboost`, mô hình sẽ tự động chuyển sang trạng thái `SKIPPED` với thông báo rõ ràng mà không làm gián đoạn các mô hình khác.
* **Biến thể RF & GB**: Giữ riêng biệt các biến thể từ Trainer v2 (n_estimators=500) và Decision Tree Trainer (n_estimators=100) để theo dõi và đối chiếu chính xác.

---

## 4. Tính Đúng Đắn Khoa Học & Chống Rò Rỉ Dữ Liệu (Scientific Correctness)

1. **Chuẩn 31 Đặc Trưng Trung Tâm**:
   * Khai báo tập trung qua `AI_SERVICES/feature_schema.py`. Không đưa `Actual_Count`, `Sample_ID` hoặc metadata nhóm vào ma trận $X$.
   * Kiểm tra tính toàn vẹn schema bằng `compute_schema_hash()`.
2. **Tiền xử lý trong từng Fold (Fold-Local Scaling)**:
   * Ma trận thô $X$ được đưa trực tiếp vào `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('model', estimator)])`.
   * `StandardScaler` được fit độc lập trong từng train fold của Cross-Validation. Tuyệt đối không fit scaler trước CV hoặc fit trên tập Test.
   * Scaler xuất xưởng (final scaler) được lấy trực tiếp từ pipeline cuối cùng huấn luyện trên toàn bộ outer train.
3. **Phân chia dữ liệu theo nhóm vật lý (Grouped Split)**:
   * Sử dụng parser chính quy `^(M\d+)[A-Za-z]+$` để gom các góc chụp của cùng một mẫu vật lý (ví dụ: `M001a`...`M001e` $\rightarrow$ Nhóm `M001`).
   * Sử dụng `GroupShuffleSplit` cho Outer Split (80% Train, 20% Test) và `GroupKFold(5)` cho Inner CV, đảm bảo giao thoa nhóm giữa Train và Test hoàn toàn bằng rỗng ($\text{Train} \cap \text{Test} = \emptyset$).
4. **Đóng băng mô hình theo CV (Freeze Winner by CV)**:
   * Mô hình được chọn dựa hoàn toàn trên `CV_MAE` trung bình trên tập Train trước khi nhìn vào tập Test.
   * Không dùng kết quả Test set để chọn mô hình hoặc tinh chỉnh siêu tham số.
5. **Đánh giá chỉ số minh bạch**:
   * Không làm tròn hoặc cắt cụt (clamp) dự đoán trước khi tính chỉ số sai số (MAE, MSE, RMSE, R2).
   * MAPE được tính loại trừ các mẫu $y=0$ và báo cáo rõ số lượng mẫu bị loại trừ; Adjusted $R^2$ chỉ hiển thị khi đủ bậc tự do ($n > p + 1$).
   * Không gọi dải sai số tham khảo $\pm 5\%$ là khoảng tin cậy (Confidence Interval).

---

## 5. Cấu Trúc Lưu Trữ & Bundle Độc Lập

Mỗi lần chạy ghi đè bundle của chính mô hình đó. Vì vậy, để mở hoặc thay một mô hình, chỉ cần vào đúng thư mục `<model_id>`; không cần tìm mã lần chạy. `run_id` vẫn được lưu bên trong metadata để truy vết lần huấn luyện đã tạo artifact hiện tại, nhưng **không** là một phần của đường dẫn.

```text
LINEAR_REGRESSION_MODEL/
  models/
    # === CÁC ARTIFACT ĐANG TRIỂN KHAI (BẢO TOÀN NGUYÊN VẸN) ===
    best_tree_ensemble_model.joblib
    scaler.joblib
    best_tree_model_info.json
    scaler_params.json

    # === BUNDLE HIỆN HÀNH, GHI ĐÈ THEO TỪNG MODEL ===
    <model_id>/
      pipeline.joblib          # Pipeline hoàn chỉnh (StandardScaler + Model)
      model.joblib             # Estimator trích xuất từ pipeline
      scaler.joblib            # Scaler trích xuất từ pipeline
      scaler_params.json       # Mean, scale, variance của 31 đặc trưng
      config.json              # Tham số model, source notebook, phiên bản thư viện
      feature_schema.json      # 31 tên đặc trưng, schema version, canonical hash
      metrics.json             # CV metrics, Train metrics, Test metrics
      cv_results.csv           # Chi tiết kết quả từng fold CV
      predictions_train.csv    # Kết quả dự đoán chi tiết tập Train
      predictions_test.csv     # Kết quả dự đoán chi tiết tập Test
      manifest.json            # Bảng mã băm SHA-256 kiểm định tính toàn vẹn
      equation.json            # Chỉ mô hình tuyến tính có parity PASS
      plots/
        parity_residual.png    # Đồ thị Parity và phân tích phần dư
  results/
    comparison.csv             # Bảng so sánh của lần chạy hiện hành
    comparison.png             # Đồ thị so sánh trực quan CV MAE vs Test MAE
    run_metadata.json          # Toàn bộ cấu hình và thông tin lần chạy hiện hành
    split_membership.csv       # Phân bổ từng mẫu vào Train/Test và từng CV Fold
    data_quality_report.json   # Báo cáo kiểm soát chất lượng dữ liệu đầu vào
    execution_report.json      # Báo cáo tiến trình và thời gian thực thi
```

## 6. Kiểm Thử Tự Động (Automated Testing)

Bộ kiểm thử đơn vị độc lập được đặt tại `LINEAR_REGRESSION_MODEL/tests/test_notebook_consolidation.py`:

```powershell
# Chạy bộ test hợp nhất
& '.\LINEAR_REGRESSION_MODEL\.venv\Scripts\python.exe' -m unittest discover -s 'LINEAR_REGRESSION_MODEL/tests' -p 'test_notebook_consolidation.py' -v
```

Bộ test bao gồm các kiểm định:
1. Tính hợp lệ của cấu trúc notebook (`nbformat` v4, đầy đủ cell IDs, compile cú pháp Python 0 lỗi).
2. Kiểm tra danh mục 19 mô hình đầy đủ, đúng siêu tham số nguồn.
3. Thẩm định Feature Contract (đúng 31 đặc trưng, không lẫn target hay Sample_ID).
4. Kiểm tra cô lập dữ liệu: không rò rỉ nhóm vật lý giữa Train/Test và giữa các Fold.
5. Kiểm tra tính đúng đắn của Preprocessing: scaler fit cục bộ trong từng Fold.
6. Xử lý biên cho metrics: dự đoán chiều $(n,)$, MAPE loại trừ $y=0$, Adjusted $R^2$.
7. Xác thực tính nhất quán khi Reload: `pipeline.predict(X) == model.predict(scaler.transform(X))`.
8. Phát hiện can thiệp mã băm (Tamper Detection).
9. Bảo toàn 100% mã băm của các artifact deployment đang hoạt động.
