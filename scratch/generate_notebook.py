import json
from pathlib import Path

def get_notebook_dict():
    cells = []

    def md_cell(text, cid):
        lines = [line + "\n" for line in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "markdown",
            "metadata": {"id": cid},
            "id": cid,
            "source": lines
        }

    def code_cell(text, cid):
        lines = [line + "\n" for line in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cid},
            "id": cid,
            "outputs": [],
            "source": lines
        }

    # Cell 0: Intro
    c0 = """# 🌾 RICE VISION AI — BỘ HUẤN LUYỆN & SO SÁNH HỒI QUY ĐA MÔ HÌNH (31 ĐẶC TRƯNG)

> **Trạng thái thực nghiệm**: PROVISIONAL (Pilot Benchmark).  
> **Mục tiêu**:
> 1. Huấn luyện và so sánh chuẩn mực 19 cấu hình mô hình hồi quy (tuyến tính, phi tuyến, cây quyết định, ensemble).
> 2. Đảm bảo tính đúng đắn khoa học: tiền xử lý chuẩn hóa (`StandardScaler`) độc lập trong từng fold Cross-Validation (Fold-Local Preprocessing), loại bỏ rò rỉ dữ liệu (Data Leakage).
> 3. Phân chia dữ liệu theo nhóm mẫu vật lý (`Grouped Split`) dựa trên mã mẫu thực tế (`Sample_ID`), ngăn chặn lặp mẫu giữa Train và Test.
> 4. Xuất gói triển khai độc lập (`Bundle`: Pipeline, Model, Scaler, Manifest, Metadata) cho từng mô hình thành công tại `models/<model_id>/<run_id>/`.

---

### Mục Lục (Table of Contents)
- **Khu 00**: Giới thiệu & Mục tiêu
- **Khu 01**: Môi trường & Khai báo thư viện (`ENV_SETUP`, `IMPORTS`)
- **Khu 02**: Cấu hình phiên chạy (`RUN_CONFIG`)
- **Khu 03**: Hợp đồng 31 đặc trưng & Nạp dữ liệu (`FEATURE_CONTRACT`, `DATA_LOAD`)
- **Khu 04**: Phân chia tập mẫu & K-Fold (`SPLIT_CONFIG`, `SPLIT_BUILD`)
- **Khu 05**: Tiền xử lý & Khung Pipeline (`PREPROCESSING`, `HELPERS`)
- **Khu 06**: Cấu hình 19 mô hình ứng viên (`MODEL_<model_id>`)
- **Khu 07**: Huấn luyện toàn bộ mô hình (`TRAIN_ALL`)
- **Khu 08**: Chọn lọc & Đánh giá Holdout Test (`SELECT_BEST`, `EVALUATE_ALL`, `COMPARE_ALL`)
- **Khu 09**: Chẩn đoán chi tiết mô hình được chọn (`DIAGNOSTICS`)
- **Khu 10**: Xuất Bundle độc lập cho mỗi mô hình (`EXPORT_ALL`)
- **Khu 11**: Kiểm thử nạp lại & Xác minh dự đoán (`RELOAD_SMOKE`)"""
    cells.append(md_cell(c0, "intro"))

    # Cell 1: ENV_SETUP
    c1 = """# Cell ID: ENV_SETUP
# ==============================================================================
# KHU 01: MÔI TRƯỜNG & KHAI BÁO THƯ VIỆN (ENV_SETUP & IMPORTS)
# ==============================================================================

import os
import sys
import json
import time
import math
import uuid
import hashlib
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Kiểm tra môi trường Google Colab (được bọc an toàn, không bắt buộc)
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print("Môi trường: Google Colab detected.")
    try:
        from google.colab import drive
        # drive.mount('/content/drive')  # Mở ghi chú nếu người dùng chạy trên Colab
    except Exception as e:
        print(f"Colab note: {e}")
else:
    print(f"Môi trường: Local Python {platform.python_version()} ({platform.architecture()[0]})")

# Import các thư viện lõi
import numpy as np
import pandas as pd
import scipy
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn
import sklearn
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
    cross_validate,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    max_error,
)

# In thông tin phiên bản kiểm chứng
print(f"numpy       : {np.__version__}")
print(f"pandas      : {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scipy       : {scipy.__version__}")
print(f"joblib      : {joblib.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")"""
    cells.append(code_cell(c1, "ENV_SETUP"))

    # Cell 2: RUN_CONFIG
    c2 = """# Cell ID: RUN_CONFIG
# ==============================================================================
# KHU 02: CẤU HÌNH DỮ LIỆU & THỰC THI (RUN_CONFIG)
# ==============================================================================

# Xác định đường dẫn thư mục gốc repo
try:
    _nb_dir = Path(__file__).resolve().parent
except NameError:
    _nb_dir = Path("LINEAR_REGRESSION_MODEL").resolve() if Path("LINEAR_REGRESSION_MODEL").exists() else Path(".").resolve()

REPO_ROOT = _nb_dir if (_nb_dir / "AI_SERVICES").exists() else _nb_dir.parent
if not (REPO_ROOT / "AI_SERVICES").exists():
    for p in [Path(".").resolve(), Path("..").resolve()]:
        if (p / "AI_SERVICES").exists():
            REPO_ROOT = p
            break

NOTEBOOK_DIR = REPO_ROOT / "LINEAR_REGRESSION_MODEL"
DATA_PATH = REPO_ROOT / "DATASET_BUILDER" / "4_Final_Dataset" / "final_linear_regression_dataset.csv"
OUTPUT_MODELS_ROOT = NOTEBOOK_DIR / "models"
OUTPUT_RESULTS_ROOT = NOTEBOOK_DIR / "results"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Chiến lược phân chia dữ liệu:
#   'GROUPED' (mặc định): Phân chia theo nhóm mẫu vật lý (Physical Sample ID)
#   'ROW_PROVISIONAL': Phân chia ngẫu nhiên từng dòng (chỉ dùng khi không có group ID đáng tin cậy)
SPLIT_MODE = "GROUPED"
GROUP_REGEX = r"^(M\\d+)[A-Za-z]+$"

# Danh sách model_id muốn chạy (None = chạy tất cả model có enabled=True)
ENABLED_MODEL_IDS: Optional[List[str]] = None

# Cấu hình đa luồng (CV_JOBS=1 và ESTIMATOR_JOBS=1 để tránh tranh chấp CPU)
CV_JOBS = 1
ESTIMATOR_JOBS = 1

# Cờ xuất bundle
EXPORT_BUNDLES = True

# Sinh run_id duy nhất cho phiên chạy (UTC timestamp + short hash)
RUN_ID = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

print(f"Run ID        : {RUN_ID}")
print(f"Repo Root     : {REPO_ROOT}")
print(f"Data Path     : {DATA_PATH} (Exists: {DATA_PATH.exists()})")
print(f"Split Mode    : {SPLIT_MODE}")
print(f"Seed / Folds  : {RANDOM_SEED} / {CV_FOLDS}")"""
    cells.append(code_cell(c2, "RUN_CONFIG"))

    # Cell 3: FEATURE_CONTRACT
    c3 = """# Cell ID: FEATURE_CONTRACT
# ==============================================================================
# KHU 03: HỢP ĐỒNG 31 ĐẶC TRƯNG TRUNG TÂM (FEATURE_CONTRACT)
# ==============================================================================

# Nạp schema tập trung từ AI_SERVICES/feature_schema.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "AI_SERVICES") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "AI_SERVICES"))

from feature_schema import (
    ALL_31_FEATURES,
    FEATURE_DEFS,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    compute_schema_hash,
)

# Khẳng định các bất biến bắt buộc
assert len(ALL_31_FEATURES) == 31, f"Yêu cầu đúng 31 đặc trưng, tìm thấy {len(ALL_31_FEATURES)}"
assert TARGET_COLUMN not in ALL_31_FEATURES, "Target không được xuất hiện trong ma trận X!"
assert "Sample_ID" not in ALL_31_FEATURES, "Sample_ID không được xuất hiện trong ma trận X!"
assert "Image_Status" not in ALL_31_FEATURES, "Image_Status không được xuất hiện trong ma trận X!"

SCHEMA_HASH = compute_schema_hash()
print(f"Schema Version: {FEATURE_SCHEMA_VERSION}")
print(f"Schema Hash   : {SCHEMA_HASH}")
print(f"Target Column : {TARGET_COLUMN}")
print(f"31 Features   : {ALL_31_FEATURES[:3]} ... {ALL_31_FEATURES[-3:]}")"""
    cells.append(code_cell(c3, "FEATURE_CONTRACT"))

    # Cell 4: DATA_LOAD
    c4 = """# Cell ID: DATA_LOAD
# ==============================================================================
# KHU 03 (Tiếp): NẠP DỮ LIỆU & KIỂM SOÁT CHẤT LƯỢNG (DATA_LOAD & QC)
# ==============================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy tập dữ liệu tại {DATA_PATH}")

# Tính mã băm SHA-256 của file dataset nguồn
with open(DATA_PATH, "rb") as f:
    DATASET_SHA256 = hashlib.sha256(f.read()).hexdigest()

raw_df = pd.read_csv(DATA_PATH)
initial_rows = len(raw_df)

# Lọc các dòng có ảnh hợp lệ (Image_Status == 'FOUND')
if "Image_Status" in raw_df.columns:
    df_found = raw_df[raw_df["Image_Status"] == "FOUND"].copy()
else:
    df_found = raw_df.copy()

found_rows = len(df_found)

# Kiểm tra các cột bắt buộc
missing_cols = [c for c in ALL_31_FEATURES + [TARGET_COLUMN, "Sample_ID"] if c not in df_found.columns]
if missing_cols:
    raise KeyError(f"Thiếu các cột bắt buộc trong dataset: {missing_cols}")

# Giữ row_id nguyên bản từ source dataset
df_found["source_row_id"] = df_found.index

# Ép kiểu numeric cho 31 features và Target
for col in ALL_31_FEATURES + [TARGET_COLUMN]:
    df_found[col] = pd.to_numeric(df_found[col], errors="coerce")

# Báo cáo các dòng có giá trị NaN / Inf
invalid_mask = df_found[ALL_31_FEATURES + [TARGET_COLUMN]].isna().any(axis=1) | \\
               np.isinf(df_found[ALL_31_FEATURES + [TARGET_COLUMN]]).any(axis=1)

if invalid_mask.any():
    rejected_indices = df_found[invalid_mask].index.tolist()
    print(f"CẢNH BÁO QC: Loại bỏ {len(rejected_indices)} dòng chứa NaN/Inf: {rejected_indices}")
    df_clean = df_found[~invalid_mask].copy()
else:
    df_clean = df_found.copy()

# Kiểm tra không có zero bất hợp lệ ở các cột hình học vật chứa bắt buộc > 0
strictly_positive_cols = [
    "Bulk_Rice_Volume_mm3", "Rice_Height_mm", "Pixels_Per_mm",
    "Container_Detected_Diam_px", "Inner_Diameter_mm", "Container_Height_mm"
]
for col in strictly_positive_cols:
    violating = (df_clean[col] <= 0).sum()
    if violating > 0:
        print(f"CẢNH BÁO QC: Cột {col} có {violating} dòng <= 0.")

# Tách riêng ma trận đặc trưng thô X và nhãn y
X_raw = df_clean[ALL_31_FEATURES].copy()
y_raw = df_clean[TARGET_COLUMN].to_numpy(dtype=np.float64)
metadata_df = df_clean[["source_row_id", "Sample_ID"]].copy()

DATA_QUALITY_REPORT = {
    "dataset_path": str(DATA_PATH.resolve()),
    "dataset_sha256": DATASET_SHA256,
    "initial_rows": initial_rows,
    "found_rows": found_rows,
    "clean_rows": len(df_clean),
    "features_count": len(ALL_31_FEATURES),
    "target_column": TARGET_COLUMN,
}

print(f"Dataset QC hoàn tất: {len(df_clean)}/{initial_rows} dòng hợp lệ (FOUND). Dataset SHA-256: {DATASET_SHA256[:12]}...")"""
    cells.append(code_cell(c4, "DATA_LOAD"))

    # Cell 5: SPLIT_CONFIG
    c5 = """# Cell ID: SPLIT_CONFIG
# ==============================================================================
# KHU 04: CẤU HÌNH PHÂN CHIA TẬP MẪU & NHÓM VẬT LÝ (SPLIT_CONFIG)
# ==============================================================================

import re

# Parser xác định nhóm mẫu vật lý từ Sample_ID (ví dụ: M001a, M001b -> nhóm M001)
group_pattern = re.compile(GROUP_REGEX, re.IGNORECASE)

groups_list = []
parse_failures = []
for idx, sid in enumerate(metadata_df["Sample_ID"]):
    match = group_pattern.match(str(sid).strip())
    if match:
        groups_list.append(match.group(1).upper())
    else:
        parse_failures.append((idx, sid))
        groups_list.append(f"UNKNOWN_{idx}")

metadata_df["group_id"] = groups_list

if parse_failures:
    print(f"LƯU Ý: Có {len(parse_failures)} mẫu không khớp regex nhóm: {parse_failures[:5]}")
else:
    print(f"Xác nhận: 100% ({len(metadata_df)}) mẫu khớp quy tắc nhóm vật lý. Tổng số nhóm: {metadata_df['group_id'].nunique()}")"""
    cells.append(code_cell(c5, "SPLIT_CONFIG"))

    # Cell 6: SPLIT_BUILD
    c6 = """# Cell ID: SPLIT_BUILD
# ==============================================================================
# KHU 04 (Tiếp): XÂY DỰNG TẬP HOLDOUT VÀ CV FOLDS (SPLIT_BUILD)
# ==============================================================================

if SPLIT_MODE == "GROUPED":
    n_groups = metadata_df["group_id"].nunique()
    if n_groups < CV_FOLDS:
        raise ValueError(f"Số lượng nhóm vật lý ({n_groups}) nhỏ hơn số fold CV ({CV_FOLDS})!")

    # Outer Split: GroupShuffleSplit (80% Train, 20% Test)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X_raw, y_raw, groups=metadata_df["group_id"]))

    # Khẳng định không có rò rỉ nhóm giữa Outer Train và Test
    train_groups = set(metadata_df.iloc[train_idx]["group_id"])
    test_groups = set(metadata_df.iloc[test_idx]["group_id"])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"RÒ RỈ DỮ LIỆU: Nhóm {overlap} xuất hiện ở cả Train và Test!"

    # Inner CV trên Outer Train: GroupKFold
    gkf = GroupKFold(n_splits=CV_FOLDS)
    cv_splits = []
    train_metadata_subset = metadata_df.iloc[train_idx].reset_index(drop=True)
    X_train_outer = X_raw.iloc[train_idx].reset_index(drop=True)
    y_train_outer = y_raw[train_idx]

    for fold_id, (fold_train_idx, fold_val_idx) in enumerate(gkf.split(X_train_outer, y_train_outer, groups=train_metadata_subset["group_id"])):
        # Kiểm tra không giao thoa nhóm trong từng fold
        fold_train_groups = set(train_metadata_subset.iloc[fold_train_idx]["group_id"])
        fold_val_groups = set(train_metadata_subset.iloc[fold_val_idx]["group_id"])
        fold_overlap = fold_train_groups.intersection(fold_val_groups)
        assert len(fold_overlap) == 0, f"RÒ RỈ FOLD {fold_id}: Nhóm {fold_overlap} giao thoa!"
        cv_splits.append((fold_train_idx, fold_val_idx))

    print(f"Phân chia GROUPED thành công:")
    print(f"  Train: {len(train_idx)} dòng ({len(train_groups)} nhóm)")
    print(f"  Test : {len(test_idx)} dòng ({len(test_groups)} nhóm)")
    print(f"  CV   : {CV_FOLDS} folds GroupKFold (0 rò rỉ nhóm)")

elif SPLIT_MODE == "ROW_PROVISIONAL":
    warnings.warn(
        "CẢNH BÁO: Đang dùng ROW_PROVISIONAL. Các ảnh chụp từ cùng một mẫu vật lý có thể bị "
        "phân vào cả Train và Test, dẫn đến nguy cơ overfit/rò rỉ lặp mẫu!"
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(X_raw)), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train_outer = X_raw.iloc[train_idx].reset_index(drop=True)
    y_train_outer = y_raw[train_idx]
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(kf.split(X_train_outer, y_train_outer))
else:
    raise ValueError(f"Chế độ SPLIT_MODE không hợp lệ: {SPLIT_MODE}")

X_test_outer = X_raw.iloc[test_idx].reset_index(drop=True)
y_test_outer = y_raw[test_idx]

# Ghi nhận phân bổ vào bảng metadata
split_records = []
for i, idx in enumerate(train_idx):
    split_records.append({
        "source_row_id": int(metadata_df.iloc[idx]["source_row_id"]),
        "Sample_ID": str(metadata_df.iloc[idx]["Sample_ID"]),
        "group_id": str(metadata_df.iloc[idx]["group_id"]),
        "split_role": "TRAIN",
        "fold": -1
    })
for i, idx in enumerate(test_idx):
    split_records.append({
        "source_row_id": int(metadata_df.iloc[idx]["source_row_id"]),
        "Sample_ID": str(metadata_df.iloc[idx]["Sample_ID"]),
        "group_id": str(metadata_df.iloc[idx]["group_id"]),
        "split_role": "TEST",
        "fold": -1
    })

# Cập nhật fold cho tập train
for fold_id, (_, fold_val_idx) in enumerate(cv_splits):
    for f_idx in fold_val_idx:
        split_records[f_idx]["fold"] = fold_id

SPLIT_MEMBERSHIP_DF = pd.DataFrame(split_records)"""
    cells.append(code_cell(c6, "SPLIT_BUILD"))

    # Cell 7: PREPROCESSING
    c7 = """# Cell ID: PREPROCESSING
# ==============================================================================
# KHU 05: TIỀN XỬ LÝ CHUẨN HÓA & KHUNG PIPELINE (PREPROCESSING)
# ==============================================================================

# Nguyên tắc bất biến:
#   1. Ma trận thô X được đưa vào sklearn.pipeline.Pipeline.
#   2. StandardScaler() được fit TRONG từng train fold của CV.
#   3. Không fit scaler trước CV, không fit trên Test, không double scaling.
#   4. Final scaler xuất xưởng được lấy từ final pipeline fit trên toàn bộ outer train.

def build_pipeline(estimator: Any) -> Pipeline:
    # Tạo một Pipeline mới chứa StandardScaler và estimator mục tiêu
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])"""
    cells.append(code_cell(c7, "PREPROCESSING"))

    # Cell 8: HELPERS
    c8 = """# Cell ID: HELPERS
# ==============================================================================
# KHU 05 (Tiếp): HÀM BỔ TRỢ HUẤN LUYỆN & TÍNH CHỈ SỐ (HELPERS)
# ==============================================================================

def train_one_model(
    model_id: str,
    spec: Dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    cv_folds_indices: List[Tuple[np.ndarray, np.ndarray]],
    cv_jobs: int = 1,
) -> Dict[str, Any]:
    # Huấn luyện 1 mô hình bằng Pipeline:
    # - Đo đạc hiệu năng qua CV 5-Fold (chuẩn hóa cục bộ trong từng fold).
    # - Huấn luyện Final Pipeline trên toàn bộ outer train.
    t0 = time.perf_counter()
    estimator = spec["factory"]()
    pipeline = build_pipeline(estimator)

    # Chạy cross-validation đa chỉ số
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    cv_raw = cross_validate(
        pipeline,
        X_tr,
        y_tr,
        cv=cv_folds_indices,
        scoring=scoring,
        n_jobs=cv_jobs,
        error_score="raise",
        return_estimator=True,
    )

    # Tính toán RMSE trên từng fold: sqrt(MSE)
    fold_mse = -cv_raw["test_neg_mse"]
    fold_rmse = np.sqrt(fold_mse)
    fold_mae = -cv_raw["test_neg_mae"]
    fold_r2 = cv_raw["test_r2"]

    cv_summary = {
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2, ddof=0)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae, ddof=0)),
        "cv_mse_mean": float(np.mean(fold_mse)),
        "cv_mse_std": float(np.std(fold_mse, ddof=0)),
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse, ddof=0)),
        "fold_details": [
            {
                "fold": int(i),
                "r2": float(fold_r2[i]),
                "mae": float(fold_mae[i]),
                "mse": float(fold_mse[i]),
                "rmse": float(fold_rmse[i]),
            }
            for i in range(len(fold_r2))
        ]
    }

    # Huấn luyện Final Pipeline trên toàn bộ Outer Train
    final_pipeline = clone(pipeline)
    final_pipeline.fit(X_tr, y_tr)

    elapsed = time.perf_counter() - t0

    return {
        "model_id": model_id,
        "spec": spec,
        "final_pipeline": final_pipeline,
        "cv_summary": cv_summary,
        "train_time_sec": float(elapsed),
        "status": "SUCCESS",
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int = 31,
    is_ols: bool = False,
) -> Dict[str, Any]:
    # Tính toán các chỉ số sai số chuẩn:
    # - Không làm tròn, không cắt cụt dự đoán trước khi tính.
    # - Xử lý shape (n,).
    # - MAPE loại trừ y=0 và báo cáo số mẫu loại trừ.
    # - Adjusted R2 chỉ tính cho OLS khi n > p + 1.
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    assert len(y_true_arr) == len(y_pred_arr), "Độ dài y_true và y_pred không khớp!"
    n = len(y_true_arr)
    p = n_features

    r2_val = float(r2_score(y_true_arr, y_pred_arr))
    mae_val = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse_val = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse_val = float(np.sqrt(mse_val))
    max_err_val = float(max_error(y_true_arr, y_pred_arr))

    # MAPE (loại trừ y == 0 để tránh chia cho 0)
    nonzero_mask = y_true_arr != 0
    excluded_zeros = int((~nonzero_mask).sum())
    if nonzero_mask.any():
        mape_val = float(np.mean(np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask])) * 100.0)
    else:
        mape_val = None

    # Adjusted R2
    if is_ols and n > (p + 1):
        adj_r2 = float(1.0 - (1.0 - r2_val) * (n - 1) / (n - p - 1))
        adj_r2_note = "OLS formula valid (n > p + 1)"
    else:
        adj_r2 = None
        adj_r2_note = "Not applicable (non-linear or n <= p + 1)"

    return {
        "n_samples": n,
        "r2": r2_val,
        "mae": mae_val,
        "mse": mse_val,
        "rmse": rmse_val,
        "max_error": max_err_val,
        "mape": mape_val,
        "mape_excluded_zeros_count": excluded_zeros,
        "adjusted_r2": adj_r2,
        "adjusted_r2_note": adj_r2_note,
    }"""
    cells.append(code_cell(c8, "HELPERS"))

    # Cell 9: Markdown Header for Models
    c9 = """## Khu 06: Cấu Hình 19 Mô Hình Ứng Viên (Model Inventory)

Mỗi cell dưới đây đăng ký cấu hình cho **một mô hình duy nhất**.  
Bạn có thể bật/tắt bằng cách sửa biến `enabled = True/False` trực tiếp trong từng cell tương ứng."""
    cells.append(md_cell(c9, "MODEL_CONFIGS_HEADER"))

    # Cell 10: MODEL_REGISTRY_INIT
    c10 = """# Cell ID: MODEL_REGISTRY_INIT
# ==============================================================================
# KHỞI TẠO TỪ ĐIỂN ĐĂNG KÝ MÔ HÌNH (MODEL REGISTRY)
# ==============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
print("Khởi tạo MODEL_REGISTRY rỗng.")"""
    cells.append(code_cell(c10, "MODEL_REGISTRY_INIT"))

    # Cells 11-29: 19 Model Spec Cells
    models_specs = [
        ("MODEL_ridge_alpha_0_1", "ridge_alpha_0_1", "1. Ridge Regression (alpha=0.1)", "Ridge", "Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import Ridge\nfactory = lambda: Ridge(alpha=0.1, random_state=RANDOM_SEED)",
         '{"alpha": 0.1, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Mô hình hồi quy tuyến tính chuẩn hóa L2 nhẹ"),

        ("MODEL_ols", "ols", "2. Ordinary Least Squares (OLS)", "LinearRegression", "Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import LinearRegression\nfactory = lambda: LinearRegression(fit_intercept=True)",
         '{"fit_intercept": True}', "StandardScaler in Pipeline", "Hồi quy tuyến tính cổ điển bình phương tối thiểu"),

        ("MODEL_ridge_alpha_1", "ridge_alpha_1", "3. Ridge Regularized (alpha=1.0)", "Ridge", "Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import Ridge\nfactory = lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED)",
         '{"alpha": 1.0, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Hồi quy Ridge với phạt L2 mạnh hơn"),

        ("MODEL_bayesian_ridge", "bayesian_ridge", "4. Bayesian Ridge Regression", "BayesianRidge", "Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import BayesianRidge\nfactory = lambda: BayesianRidge()",
         '{"n_iter": 300, "alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6}', "StandardScaler in Pipeline", "Hồi quy Bayes tự động suy biến siêu tham số"),

        ("MODEL_huber", "huber", "5. Huber Regressor", "HuberRegressor", "Robust Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import HuberRegressor\nfactory = lambda: HuberRegressor(max_iter=1000)",
         '{"max_iter": 1000, "epsilon": 1.35}', "StandardScaler in Pipeline", "Kháng ngoại lai mạnh mẽ dựa trên tổn thất Huber"),

        ("MODEL_elastic_net", "elastic_net", "6. ElasticNet (L1 + L2)", "ElasticNet", "Linear", "Trainer v1/v2", "True",
         "from sklearn.linear_model import ElasticNet\nfactory = lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED)",
         '{"alpha": 0.01, "l1_ratio": 0.5, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Kết hợp phạt Lasso (L1) và Ridge (L2)"),

        ("MODEL_pls", "pls", "7. PLS Regression (Baseline fixed n_components=2)", "PLSRegression", "Linear", "Trainer v2", "True",
         "from sklearn.cross_decomposition import PLSRegression\nfactory = lambda: PLSRegression(n_components=2, scale=True)",
         '{"n_components": 2, "scale": True}', "StandardScaler in Pipeline + internal PLS scaling", "Baseline cố định 2 thành phần tránh selection bias"),

        ("MODEL_ard", "ard", "8. ARD Regression (Bayesian Sparse)", "ARDRegression", "Linear", "Trainer v2", "True",
         "from sklearn.linear_model import ARDRegression\nfactory = lambda: ARDRegression()",
         '{"max_iter": 300}', "StandardScaler in Pipeline", "Automatic Relevance Determination tạo độ thưa thớt Bayes"),

        ("MODEL_kernel_ridge_rbf", "kernel_ridge_rbf", "9. Kernel Ridge (RBF Kernel)", "KernelRidge", "Kernel", "Trainer v2", "True",
         "from sklearn.kernel_ridge import KernelRidge\nfactory = lambda: KernelRidge(kernel='rbf', alpha=1.0, gamma=None)",
         '{"kernel": "rbf", "alpha": 1.0, "gamma": None}', "StandardScaler in Pipeline", "Hồi quy phi tuyến không tham số qua RBF Kernel"),

        ("MODEL_stacking_linear", "stacking_linear", "10. Stacking Regressor (Linear Ensemble)", "StackingRegressor", "Ensemble", "Trainer v2", "False",
         """def _make_stacking():
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, RidgeCV
    return StackingRegressor(
        estimators=[
            ('ols', LinearRegression()),
            ('ridge', Ridge(alpha=0.1, random_state=RANDOM_SEED)),
            ('bridge', BayesianRidge()),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        cv=5
    )
factory = _make_stacking""",
         '{"estimators": ["ols", "ridge", "bridge"], "final_estimator": "RidgeCV", "cv": 5}', "StandardScaler in outer pipeline", "Vô hiệu hóa mặc định: inner CV preprocessing/group isolation not verified"),

        ("MODEL_random_forest_v2", "random_forest_v2", "11. Random Forest (v2, 500 trees)", "RandomForestRegressor", "Tree Ensemble", "Trainer v2", "True",
         "from sklearn.ensemble import RandomForestRegressor\nfactory = lambda: RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=ESTIMATOR_JOBS)",
         '{"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Random Forest cấu hình sâu từ Trainer v2"),

        ("MODEL_gradient_boosting_v2", "gradient_boosting_v2", "12. Gradient Boosting (v2, 500 trees)", "GradientBoostingRegressor", "Tree Ensemble", "Trainer v2", "True",
         "from sklearn.ensemble import GradientBoostingRegressor\nfactory = lambda: GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=RANDOM_SEED)",
         '{"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Gradient Boosting 500 cây từ Trainer v2"),

        ("MODEL_xgboost_v2", "xgboost_v2", "13. XGBoost Regressor (v2, 600 trees)", "XGBRegressor", "Tree Ensemble", "Trainer v2", "True",
         """import importlib.util
HAS_XGB = importlib.util.find_spec("xgboost") is not None
if HAS_XGB:
    from xgboost import XGBRegressor
    factory = lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=ESTIMATOR_JOBS)
else:
    factory = None""",
         '{"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Tùy chọn: chỉ kích hoạt khi máy đã cài đặt package xgboost"),

        ("MODEL_decision_tree", "decision_tree", "14. Decision Tree (Cây quyết định gốc)", "DecisionTreeRegressor", "Tree", "Decision Tree Trainer", "True",
         "from sklearn.tree import DecisionTreeRegressor\nfactory = lambda: DecisionTreeRegressor(random_state=RANDOM_SEED)",
         '{"random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Cây quyết định đơn lẻ gốc (baseline họ cây)"),

        ("MODEL_random_forest_tree", "random_forest_tree", "15. Random Forest (Tree Trainer, 100 trees)", "RandomForestRegressor", "Tree Ensemble", "Decision Tree Trainer", "True",
         "from sklearn.ensemble import RandomForestRegressor\nfactory = lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=ESTIMATOR_JOBS)",
         '{"n_estimators": 100, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Random Forest 100 cây từ Decision Tree Trainer"),

        ("MODEL_extra_trees", "extra_trees", "16. Extra Trees (Cực kỳ ngẫu nhiên, 100 trees)", "ExtraTreesRegressor", "Tree Ensemble", "Decision Tree Trainer", "True",
         "from sklearn.ensemble import ExtraTreesRegressor\nfactory = lambda: ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=ESTIMATOR_JOBS)",
         '{"n_estimators": 100, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Mô hình Extra Trees hiện đang triển khai"),

        ("MODEL_gradient_boosting_tree", "gradient_boosting_tree", "17. Gradient Boosting (Tree Trainer, 100 trees)", "GradientBoostingRegressor", "Tree Ensemble", "Decision Tree Trainer", "True",
         "from sklearn.ensemble import GradientBoostingRegressor\nfactory = lambda: GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)",
         '{"n_estimators": 100, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Gradient Boosting 100 cây từ Decision Tree Trainer"),

        ("MODEL_ada_boost", "ada_boost", "18. AdaBoost (50 trees)", "AdaBoostRegressor", "Tree Ensemble", "Decision Tree Trainer", "True",
         "from sklearn.ensemble import AdaBoostRegressor\nfactory = lambda: AdaBoostRegressor(n_estimators=50, random_state=RANDOM_SEED)",
         '{"n_estimators": 50, "random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Adaptive Boosting với 50 DecisionTree gốc"),

        ("MODEL_hist_gradient_boosting", "hist_gradient_boosting", "19. HistGradientBoosting", "HistGradientBoostingRegressor", "Tree Ensemble", "Decision Tree Trainer", "True",
         "from sklearn.ensemble import HistGradientBoostingRegressor\nfactory = lambda: HistGradientBoostingRegressor(random_state=RANDOM_SEED)",
         '{"random_state": RANDOM_SEED}', "StandardScaler in Pipeline", "Thuật toán Gradient Boosting tối ưu hóa theo Histogram"),
    ]

    for cell_id, m_id, d_name, cls_name, fam, src_nb, default_en, factory_code, params_str, preproc, note in models_specs:
        c_code = f"""# Cell ID: {cell_id}
# ------------------------------------------------------------------------------
# Mô hình: {d_name}
# ------------------------------------------------------------------------------

model_id = "{m_id}"
display_name = "{d_name}"
model_class = "{cls_name}"
family = "{fam}"
source_notebook = "{src_nb}"
enabled = {default_en}

# Khởi tạo factory estimator
{factory_code}

params = {params_str}
preprocessing_note = "{preproc}"
notes = "{note}"

# Đăng ký vào bảng quản lý
MODEL_REGISTRY[model_id] = {{
    "model_id": model_id,
    "display_name": display_name,
    "model_class": model_class,
    "family": family,
    "source_notebook": source_notebook,
    "enabled": enabled,
    "factory": factory,
    "params": params,
    "preprocessing_note": preprocessing_note,
    "notes": notes,
}}
print(f"Registered: {{model_id:<25}} | Enabled: {{enabled}} | Source: {{source_notebook}}")"""
        cells.append(code_cell(c_code, cell_id))

    # Cell 30: TRAIN_ALL
    c30 = """# Cell ID: TRAIN_ALL
# ==============================================================================
# KHU 07: TIẾN HÀNH HUẤN LUYỆN TOÀN BỘ MÔ HÌNH (TRAIN_ALL)
# ==============================================================================

# Xác định danh sách model sẽ chạy
if ENABLED_MODEL_IDS is not None:
    # Người dùng ghi đè danh sách
    invalid_ids = [m for m in ENABLED_MODEL_IDS if m not in MODEL_REGISTRY]
    if invalid_ids:
        raise ValueError(f"Các model_id sau không tồn tại trong MODEL_REGISTRY: {invalid_ids}")
    active_models = {m: MODEL_REGISTRY[m] for m in ENABLED_MODEL_IDS}
else:
    active_models = {m: spec for m, spec in MODEL_REGISTRY.items() if spec.get("enabled", False)}

print(f"Chuẩn bị huấn luyện {len(active_models)} mô hình hợp lệ:")
for m_id, spec in active_models.items():
    print(f"  - {m_id:<25} ({spec['display_name']})")

# Kiểm tra an toàn bắt buộc: nếu stacking_linear được bật, từ chối cho đến khi inner isolation được chứng minh
if "stacking_linear" in active_models and active_models["stacking_linear"].get("enabled", False):
    raise RuntimeError(
        "LỖI BẢO VỆ: StackingRegressor hiện bị từ chối do chưa có inner fold group isolation và scaler cục bộ cho base models! "
        "Vui lòng tắt enabled=False ở cell MODEL_stacking_linear."
    )

TRAINED_RESULTS: Dict[str, Any] = {}

print("\\n" + "=" * 80)
print("BẮT ĐẦU HUẤN LUYỆN ĐA MÔ HÌNH (5-FOLD CV + OUTER TRAIN FIT)")
print("=" * 80)

for m_idx, (m_id, spec) in enumerate(active_models.items(), 1):
    print(f"[{m_idx:02d}/{len(active_models):02d}] Đang huấn luyện {m_id:<25} ... ", end="", flush=True)
    if spec.get("factory") is None:
        print("BỎ QUA (Factory là None, có thể do thiếu dependency)")
        TRAINED_RESULTS[m_id] = {"model_id": m_id, "spec": spec, "status": "SKIPPED", "error": "Factory is None"}
        continue

    try:
        res = train_one_model(
            model_id=m_id,
            spec=spec,
            X_tr=X_train_outer,
            y_tr=y_train_outer,
            cv_folds_indices=cv_splits,
            cv_jobs=CV_JOBS
        )
        TRAINED_RESULTS[m_id] = res
        mae_m = res["cv_summary"]["cv_mae_mean"]
        r2_m = res["cv_summary"]["cv_r2_mean"]
        t_sec = res["train_time_sec"]
        print(f"HOÀN TẤT ({t_sec:5.2f}s | CV MAE: {mae_m:6.4f} | CV R2: {r2_m:6.4f})")
    except Exception as ex:
        print(f"THẤT BẠI ({ex})")
        TRAINED_RESULTS[m_id] = {"model_id": m_id, "spec": spec, "status": "FAILED", "error": str(ex)}

print("=" * 80)
print(f"Huấn luyện hoàn tất: {sum(1 for r in TRAINED_RESULTS.values() if r.get('status') == 'SUCCESS')} thành công, "
      f"{sum(1 for r in TRAINED_RESULTS.values() if r.get('status') == 'FAILED')} thất bại.")"""
    cells.append(code_cell(c30, "TRAIN_ALL"))

    # Cell 31: SELECT_BEST
    c31 = """# Cell ID: SELECT_BEST
# ==============================================================================
# KHU 08: ĐÓNG BĂNG MÔ HÌNH TỐI ƯU DỰA TRÊN CROSS-VALIDATION (SELECT_BEST)
# ==============================================================================

# Quy tắc bất biến:
#   1. Đóng băng mô hình chiến thắng (Winner) TRƯỚC KHI đọc dữ liệu tập Test.
#   2. Tiêu chí chọn: min CV_MAE_mean -> min CV_RMSE_mean -> alphabetical model_id trên số thực thô (unrounded).
#   3. Tuyệt đối không dùng Test metrics để chọn mô hình!

successful_models = {k: v for k, v in TRAINED_RESULTS.items() if v.get("status") == "SUCCESS"}
if not successful_models:
    raise RuntimeError("Không có mô hình nào huấn luyện thành công!")

sorted_candidates = sorted(
    successful_models.values(),
    key=lambda x: (
        x["cv_summary"]["cv_mae_mean"],
        x["cv_summary"]["cv_rmse_mean"],
        x["model_id"]
    )
)

SELECTED_MODEL_ID = sorted_candidates[0]["model_id"]
best_cv = sorted_candidates[0]["cv_summary"]

print("=" * 80)
print(f"⭐ MÔ HÌNH ĐƯỢC CHỌN THEO ĐÁNH GIÁ CHÉO (FROZEN BY CV): {SELECTED_MODEL_ID}")
print(f"   Tên hiển thị : {sorted_candidates[0]['spec']['display_name']}")
print(f"   CV MAE Mean  : {best_cv['cv_mae_mean']:.4f} hạt (± {best_cv['cv_mae_std']:.4f})")
print(f"   CV RMSE Mean : {best_cv['cv_rmse_mean']:.4f} hạt (± {best_cv['cv_rmse_std']:.4f})")
print(f"   CV R2 Mean   : {best_cv['cv_r2_mean']:.4f} (± {best_cv['cv_r2_std']:.4f})")
print("=" * 80)"""
    cells.append(code_cell(c31, "SELECT_BEST"))

    # Cell 32: EVALUATE_ALL
    c32 = """# Cell ID: EVALUATE_ALL
# ==============================================================================
# KHU 08 (Tiếp): ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (EVALUATE_ALL)
# ==============================================================================

EVALUATION_RESULTS: Dict[str, Any] = {}

for m_id, res in successful_models.items():
    pipeline = res["final_pipeline"]
    is_ols = (m_id == "ols")

    # Dự đoán trên Train và Test bằng Pipeline hoàn chỉnh
    pred_train = np.asarray(pipeline.predict(X_train_outer), dtype=np.float64).reshape(-1)
    pred_test = np.asarray(pipeline.predict(X_test_outer), dtype=np.float64).reshape(-1)

    metrics_train = compute_metrics(y_train_outer, pred_train, n_features=31, is_ols=is_ols)
    metrics_test = compute_metrics(y_test_outer, pred_test, n_features=31, is_ols=is_ols)

    EVALUATION_RESULTS[m_id] = {
        "model_id": m_id,
        "spec": res["spec"],
        "cv_summary": res["cv_summary"],
        "final_pipeline": pipeline,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "predictions_train": pred_train,
        "predictions_test": pred_test,
        "train_time_sec": res["train_time_sec"],
    }

print(f"Đã hoàn thành đánh giá Holdout Test cho toàn bộ {len(EVALUATION_RESULTS)} mô hình thành công.")"""
    cells.append(code_cell(c32, "EVALUATE_ALL"))

    # Cell 33: COMPARE_ALL
    c33 = """# Cell ID: COMPARE_ALL
# ==============================================================================
# KHU 08 (Tiếp): BẢNG SO SÁNH VÀ TRỰC QUAN HÓA (COMPARE_ALL)
# ==============================================================================

comparison_rows = []
for m_id, ev in EVALUATION_RESULTS.items():
    cv = ev["cv_summary"]
    tr = ev["metrics_train"]
    te = ev["metrics_test"]
    spec = ev["spec"]

    comparison_rows.append({
        "Model ID": m_id,
        "Display Name": spec["display_name"],
        "Family": spec["family"],
        "CV_MAE (hạt)": cv["cv_mae_mean"],
        "CV_RMSE (hạt)": cv["cv_rmse_mean"],
        "CV_R2": cv["cv_r2_mean"],
        "Train_MAE (hạt)": tr["mae"],
        "Test_MAE (hạt)": te["mae"],
        "Test_RMSE (hạt)": te["rmse"],
        "Test_R2": te["r2"],
        "Test_MAPE (%)": te["mape"] if te["mape"] is not None else np.nan,
        "Train_Time (s)": round(ev["train_time_sec"], 2),
    })

COMPARISON_DF = pd.DataFrame(comparison_rows)
# Xếp hạng nghiêm ngặt theo CV_MAE
COMPARISON_DF.sort_values(by="CV_MAE (hạt)", ascending=True, inplace=True)
COMPARISON_DF.reset_index(drop=True, inplace=True)

print("\\n🏆 BẢNG XẾP HẠNG TỔNG HỢP HIỆU NĂNG HỒI QUY (SẮP THEO CV MAE):")
display_cols = ["Display Name", "CV_MAE (hạt)", "CV_RMSE (hạt)", "CV_R2", "Test_MAE (hạt)", "Test_RMSE (hạt)", "Test_R2", "Test_MAPE (%)"]
try:
    from IPython.display import display
    display(COMPARISON_DF[display_cols])
except Exception:
    print(COMPARISON_DF[display_cols].to_string(index=False))

# Vẽ đồ thị so sánh CV MAE vs Test MAE
plt.figure(figsize=(12, 6))
x = np.arange(len(COMPARISON_DF))
width = 0.35

plt.bar(x - width/2, COMPARISON_DF["CV_MAE (hạt)"], width, label="CV MAE (Train Fold Avg)", color="steelblue")
plt.bar(x + width/2, COMPARISON_DF["Test_MAE (hạt)"], width, label="Holdout Test MAE", color="coral")

plt.xticks(x, COMPARISON_DF["Model ID"], rotation=45, ha="right")
plt.ylabel("MAE (hạt lúa)")
plt.title("So sánh sai số trung bình (MAE) giữa Cross-Validation và Holdout Test")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()"""
    cells.append(code_cell(c33, "COMPARE_ALL"))

    # Cell 34: DIAGNOSTICS
    c34 = """# Cell ID: DIAGNOSTICS
# ==============================================================================
# KHU 09: CHẨN ĐOÁN CHI TIẾT MÔ HÌNH ĐƯỢC CHỌN (DIAGNOSTICS)
# ==============================================================================

diag_model_id = SELECTED_MODEL_ID
diag_data = EVALUATION_RESULTS[diag_model_id]

y_true = y_test_outer
y_pred = diag_data["predictions_test"]
residuals = y_true - y_pred

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1. Đồ thị Parity Plot (Actual vs Predicted)
axs[0].scatter(y_true, y_pred, alpha=0.7, edgecolors="k", color="royalblue")
min_val = min(y_true.min(), y_pred.min()) - 10
max_val = max(y_true.max(), y_pred.max()) + 10
axs[0].plot([min_val, max_val], [min_val, max_val], "r--", label="Đường lý tưởng (1:1)")

# Dải sai số tham khảo ±5% (Lưu ý: ĐÂY LÀ DẢI SAI SỐ THAM KHẢO, KHÔNG PHẢI KHOẢNG TIN CẬY THỐNG KÊ)
ref_line = np.linspace(min_val, max_val, 100)
axs[0].fill_between(ref_line, ref_line * 0.95, ref_line * 1.05, color="gray", alpha=0.2, label="Dải tham khảo ±5%")
axs[0].set_xlim(min_val, max_val)
axs[0].set_ylim(min_val, max_val)
axs[0].set_xlabel("Số hạt thực tế (Ground Truth)")
axs[0].set_ylabel("Số hạt dự đoán (Predicted)")
axs[0].set_title(f"Parity Plot: {diag_data['spec']['display_name']}")
axs[0].legend()
axs[0].grid(True, linestyle=":", alpha=0.6)

# 2. Đồ thị Phân tích Phần dư (Residuals vs Predicted)
axs[1].scatter(y_pred, residuals, alpha=0.7, edgecolors="k", color="darkorange")
axs[1].axhline(0, color="r", linestyle="--")
axs[1].set_xlabel("Số hạt dự đoán (Predicted)")
axs[1].set_ylabel("Phần dư: Actual - Predicted (hạt)")
axs[1].set_title("Biểu đồ phân tích phần dư (Residuals)")
axs[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# Phân tích Feature Attribution (Trọng số cho Linear, Feature Importances cho Tree)
model_obj = diag_data["final_pipeline"].named_steps["model"]
scaler_obj = diag_data["final_pipeline"].named_steps["scaler"]

if hasattr(model_obj, "feature_importances_"):
    importances = model_obj.feature_importances_
    feat_series = pd.Series(importances, index=ALL_31_FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    feat_series.head(15).plot(kind="barh", color="forestgreen").invert_yaxis()
    plt.xlabel("Mức độ quan trọng (Feature Importance)")
    plt.title(f"Top 15 đặc trưng quan trọng nhất ({diag_model_id})")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
elif hasattr(model_obj, "coef_"):
    # Trọng số thực sau khi tính đến scale: w_raw = coef / scale
    coef_1d = np.asarray(model_obj.coef_, dtype=np.float64).reshape(-1)
    raw_weights = coef_1d / scaler_obj.scale_
    weight_series = pd.Series(raw_weights, index=ALL_31_FEATURES).sort_values(key=abs, ascending=False)
    plt.figure(figsize=(10, 8))
    weight_series.head(15).plot(kind="barh", color="purple").invert_yaxis()
    plt.xlabel("Trọng số hồi quy thô (Raw Coefficient w_i)")
    plt.title(f"Top 15 đặc trưng ảnh hưởng lớn nhất ({diag_model_id})")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print(f"Mô hình {diag_model_id} không hỗ trợ trích xuất weights hoặc feature_importances.")"""
    cells.append(code_cell(c34, "DIAGNOSTICS"))

    # Cell 35: EXPORT_ALL
    c35 = """# Cell ID: EXPORT_ALL
# ==============================================================================
# KHU 10: XUẤT BUNDLE ĐỘC LẬP CHO TỪNG MÔ HÌNH (EXPORT_ALL)
# ==============================================================================

if not EXPORT_BUNDLES:
    print("Cờ EXPORT_BUNDLES = False. Bỏ qua xuất artifacts.")
else:
    # Bảo vệ các file deployment hiện hữu ở root models
    PROTECTED_ROOT_FILES = [
        OUTPUT_MODELS_ROOT / "best_tree_ensemble_model.joblib",
        OUTPUT_MODELS_ROOT / "scaler.joblib",
        OUTPUT_MODELS_ROOT / "best_tree_model_info.json",
        OUTPUT_MODELS_ROOT / "scaler_params.json",
        OUTPUT_MODELS_ROOT / "best_linear_regression_model.joblib",
        OUTPUT_MODELS_ROOT / "regression_equation.json",
    ]
    
    # 1. Xuất Bundle riêng cho TỪNG mô hình thành công
    for m_id, ev in EVALUATION_RESULTS.items():
        bundle_dir = OUTPUT_MODELS_ROOT / m_id / RUN_ID
        bundle_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = bundle_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        final_pipe = ev["final_pipeline"]
        scaler_inst = final_pipe.named_steps["scaler"]
        model_inst = final_pipe.named_steps["model"]

        # Lưu binary artifacts từ cùng một final pipeline (không refit!)
        pipeline_path = bundle_dir / "pipeline.joblib"
        model_path = bundle_dir / "model.joblib"
        scaler_path = bundle_dir / "scaler.joblib"
        
        joblib.dump(final_pipe, pipeline_path)
        joblib.dump(model_inst, model_path)
        joblib.dump(scaler_inst, scaler_path)

        # Lưu scaler params JSON
        scaler_params_path = bundle_dir / "scaler_params.json"
        with open(scaler_params_path, "w", encoding="utf-8") as f:
            json.dump({
                "mean": scaler_inst.mean_.tolist(),
                "scale": scaler_inst.scale_.tolist(),
                "var": scaler_inst.var_.tolist(),
                "features": ALL_31_FEATURES,
                "schema_version": FEATURE_SCHEMA_VERSION,
            }, f, indent=2)

        # Lưu config JSON
        config_path = bundle_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": m_id,
                "display_name": ev["spec"]["display_name"],
                "model_class": ev["spec"]["model_class"],
                "family": ev["spec"]["family"],
                "source_notebook": ev["spec"]["source_notebook"],
                "params": ev["spec"]["params"],
                "preprocessing": ev["spec"]["preprocessing_note"],
                "library_versions": {
                    "python": platform.python_version(),
                    "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                }
            }, f, indent=2)

        # Lưu feature schema JSON
        schema_path = bundle_dir / "feature_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": FEATURE_SCHEMA_VERSION,
                "schema_hash": SCHEMA_HASH,
                "target_column": TARGET_COLUMN,
                "features": ALL_31_FEATURES,
            }, f, indent=2)

        # Lưu metrics JSON
        metrics_path = bundle_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "cv_summary": ev["cv_summary"],
                "metrics_train": ev["metrics_train"],
                "metrics_test": ev["metrics_test"],
                "train_time_sec": ev["train_time_sec"],
            }, f, indent=2)

        # Lưu chi tiết các fold CV
        pd.DataFrame(ev["cv_summary"]["fold_details"]).to_csv(bundle_dir / "cv_results.csv", index=False)

        # Lưu file dự đoán Train và Test
        train_pred_df = pd.DataFrame({
            "source_row_id": metadata_df.iloc[train_idx]["source_row_id"].values,
            "Sample_ID": metadata_df.iloc[train_idx]["Sample_ID"].values,
            "Actual_Count": y_train_outer,
            "Predicted_Count": ev["predictions_train"],
            "Residual": y_train_outer - ev["predictions_train"],
        })
        train_pred_df.to_csv(bundle_dir / "predictions_train.csv", index=False)

        test_pred_df = pd.DataFrame({
            "source_row_id": metadata_df.iloc[test_idx]["source_row_id"].values,
            "Sample_ID": metadata_df.iloc[test_idx]["Sample_ID"].values,
            "Actual_Count": y_test_outer,
            "Predicted_Count": ev["predictions_test"],
            "Residual": y_test_outer - ev["predictions_test"],
        })
        test_pred_df.to_csv(bundle_dir / "predictions_test.csv", index=False)

        # Xuất phương trình toán học equation.json nếu là mô hình tuyến tính tương thích
        linear_whitelist = ["ols", "ridge_alpha_0_1", "ridge_alpha_1", "bayesian_ridge", "huber", "elastic_net", "ard"]
        if m_id in linear_whitelist and hasattr(model_inst, "coef_"):
            coef_1d = np.asarray(model_inst.coef_, dtype=np.float64).reshape(-1)
            raw_w = (coef_1d / scaler_inst.scale_).tolist()
            intercept_val = float(model_inst.intercept_[0]) if hasattr(model_inst.intercept_, "__len__") else float(model_inst.intercept_)
            raw_b = float(intercept_val - np.sum((coef_1d * scaler_inst.mean_) / scaler_inst.scale_))
            
            # Test equation parity
            eq_pred = raw_b + np.dot(X_test_outer.to_numpy(), np.array(raw_w))
            pipe_pred = ev["predictions_test"]
            if np.allclose(eq_pred, pipe_pred, rtol=1e-5, atol=1e-5):
                eq_path = bundle_dir / "equation.json"
                with open(eq_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model_id": m_id,
                        "intercept": raw_b,
                        "coefficients": {feat: raw_w[i] for i, feat in enumerate(ALL_31_FEATURES)},
                        "equation_parity_verified": True,
                    }, f, indent=2)

        # Lưu đồ thị parity_residual
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(y_test_outer, ev["predictions_test"], alpha=0.7, color="royalblue")
        ax[0].plot([y_test_outer.min(), y_test_outer.max()], [y_test_outer.min(), y_test_outer.max()], "r--")
        ax[0].set_title(f"Parity Plot: {m_id}")
        ax[1].scatter(ev["predictions_test"], y_test_outer - ev["predictions_test"], alpha=0.7, color="darkorange")
        ax[1].axhline(0, color="r", linestyle="--")
        ax[1].set_title("Residuals")
        plt.tight_layout()
        plt.savefig(plots_dir / "parity_residual.png", dpi=150)
        plt.close(fig)

        # Tạo manifest.json cho bundle
        def get_sha256(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        manifest_data = {
            "bundle_id": f"rice_vision_{m_id}_{RUN_ID}",
            "run_id": RUN_ID,
            "model_id": m_id,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": SCHEMA_HASH,
            "dataset_sha256": DATASET_SHA256,
            "fit_scope": "outer_train",
            "benchmark_type": "PROVISIONAL",
            "files": {
                "pipeline": {"path": "pipeline.joblib", "sha256": get_sha256(pipeline_path)},
                "model": {"path": "model.joblib", "sha256": get_sha256(model_path)},
                "scaler": {"path": "scaler.joblib", "sha256": get_sha256(scaler_path)},
                "config": {"path": "config.json", "sha256": get_sha256(config_path)},
                "metrics": {"path": "metrics.json", "sha256": get_sha256(metrics_path)},
                "feature_schema": {"path": "feature_schema.json", "sha256": get_sha256(schema_path)},
            }
        }
        with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    # 2. Xuất Global Comparison tại results/<RUN_ID>/
    global_res_dir = OUTPUT_RESULTS_ROOT / RUN_ID
    global_res_dir.mkdir(parents=True, exist_ok=True)

    COMPARISON_DF.to_csv(global_res_dir / "comparison.csv", index=False)
    SPLIT_MEMBERSHIP_DF.to_csv(global_res_dir / "split_membership.csv", index=False)
    
    with open(global_res_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(DATA_QUALITY_REPORT, f, indent=2)

    with open(global_res_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": RUN_ID,
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "dataset_sha256": DATASET_SHA256,
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "split_mode": SPLIT_MODE,
            "selected_model_id": SELECTED_MODEL_ID,
            "total_trained_models": len(EVALUATION_RESULTS),
        }, f, indent=2)

    print(f"\\n✅ ĐÃ XUẤT THÀNH CÔNG:")
    print(f"   - {len(EVALUATION_RESULTS)} model bundles tại: {OUTPUT_MODELS_ROOT}/<model_id>/{RUN_ID}/")
    print(f"   - Báo cáo đối sánh toàn cục tại: {global_res_dir}/")"""
    cells.append(code_cell(c35, "EXPORT_ALL"))

    # Cell 36: RELOAD_SMOKE
    c36 = """# Cell ID: RELOAD_SMOKE
# ==============================================================================
# KHU 11: KIỂM THỬ NẠP LẠI & XÁC MINH DỰ ĐOÁN (RELOAD_SMOKE)
# ==============================================================================

if not EXPORT_BUNDLES:
    print("EXPORT_BUNDLES là False. Bỏ qua bước kiểm tra reload.")
else:
    print("=" * 80)
    print("KIỂM THỬ XÁC MINH NẠP LẠI BUNDLE (RELOAD VERIFICATION & PREDICTION PARITY)")
    print("=" * 80)

    verification_records = []

    for m_id, ev in EVALUATION_RESULTS.items():
        bundle_dir = OUTPUT_MODELS_ROOT / m_id / RUN_ID
        
        # 1. Nạp lại artifacts từ đĩa
        reloaded_pipe = joblib.load(bundle_dir / "pipeline.joblib")
        reloaded_model = joblib.load(bundle_dir / "model.joblib")
        reloaded_scaler = joblib.load(bundle_dir / "scaler.joblib")

        # 2. Dự đoán thử nghiệm trên Holdout Test
        pred_pipeline = np.asarray(reloaded_pipe.predict(X_test_outer), dtype=np.float64).reshape(-1)
        pred_separate = np.asarray(reloaded_model.predict(reloaded_scaler.transform(X_test_outer)), dtype=np.float64).reshape(-1)
        in_memory_pred = ev["predictions_test"]

        # 3. Khẳng định đồng nhất dự đoán (Prediction Parity)
        parity_pipe_separate = np.allclose(pred_pipeline, pred_separate, rtol=1e-8, atol=1e-8)
        parity_pipe_inmem = np.allclose(pred_pipeline, in_memory_pred, rtol=1e-8, atol=1e-8)

        # 4. Kiểm tra mean của scaler phải bằng mean của Outer Train
        scaler_mean_match = np.allclose(reloaded_scaler.mean_, X_train_outer.mean(axis=0).values, rtol=1e-5, atol=1e-5)

        pass_all = parity_pipe_separate and parity_pipe_inmem and scaler_mean_match
        verification_records.append({
            "model_id": m_id,
            "pipe_vs_separate": parity_pipe_separate,
            "pipe_vs_inmem": parity_pipe_inmem,
            "scaler_fold_train_mean": scaler_mean_match,
            "overall_status": "PASS" if pass_all else "FAIL",
        })

    verif_df = pd.DataFrame(verification_records)
    print(verif_df.to_string(index=False))
    assert (verif_df["overall_status"] == "PASS").all(), "CÓ BUNDLE KHÔNG VƯỢT QUA KIỂM THỬ RELOAD!"
    print("\\n🎉 100% BUNDLE ĐÃ XÁC MINH NẠP THÀNH CÔNG VÀ DỰ ĐOÁN ĐỒNG NHẤT!")"""
    cells.append(code_cell(c36, "RELOAD_SMOKE"))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (ricai-training)",
                "language": "python",
                "name": "ricai-training"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

if __name__ == "__main__":
    nb = get_notebook_dict()
    out_path = Path("LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated {out_path} with {len(nb['cells'])} cells.")
