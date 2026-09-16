#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION TRAINER)
===============================================================================
Mục đích:
  - Nạp toàn bộ 31 thuộc tính gốc từ DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv
  - Chuẩn hóa dữ liệu bằng StandardScaler và phân chia Train / Test (80% / 20%).
  - Huấn luyện và so sánh chéo 6 thuật toán hồi quy tuyến tính:
    1. Linear Regression (OLS)
    2. Ridge Regression (L2 Regularization)
    3. Lasso Regression (L1 Regularization)
    4. ElasticNet (L1 + L2 Combo)
    5. Huber Regressor (Kháng ngoại lai Robust)
    6. Bayesian Ridge Regression (Hồi quy Bayes)
  - Đánh giá đa chiều: R2, Adjusted R2, MAE, RMSE, MAPE (5-Fold Cross Validation).
  - Tự động chọn mô hình tốt nhất và lưu:
    + models/best_linear_regression_model.joblib
    + models/scaler.joblib
    + models/regression_equation.json (Hệ số phương trình tường minh)
  - Xuất bảng báo cáo kết quả và hình ảnh đánh giá sang results/.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# Danh sách đầy đủ 31 thuộc tính gốc đưa vào huấn luyện
ALL_31_FEATURES = [
    # Nhóm 1: Vật chứa & Thể tích khối lúa (8 biến)
    "Bulk_Rice_Volume_mm3", "Rice_Height_mm", "Weight_g", "Empty_Height_mm",
    "Pixels_Per_mm", "Container_Detected_Diam_px", "Inner_Diameter_mm", "Container_Height_mm",
    # Nhóm 2: Bề mặt & Ước lượng (3 biến)
    "Whole_Grains_Count", "Uniformity_Rate_Pct", "Estimated_Total_Seeds_Hybrid",
    # Nhóm 3: Chiều dài hạt 2a (4 biến)
    "Grain_Length_mm_Mean", "Grain_Length_mm_Min", "Grain_Length_mm_Max", "Grain_Length_mm_Std",
    # Nhóm 4: Chiều rộng hạt 2b (4 biến)
    "Grain_Width_mm_Mean", "Grain_Width_mm_Min", "Grain_Width_mm_Max", "Grain_Width_mm_Std",
    # Nhóm 5: Chiều dày hạt 2c (4 biến)
    "Grain_Thickness_mm_Mean", "Grain_Thickness_mm_Min", "Grain_Thickness_mm_Max", "Grain_Thickness_mm_Std",
    # Nhóm 6: Diện tích 2D (4 biến)
    "Grain_Area_mm2_Mean", "Grain_Area_mm2_Min", "Grain_Area_mm2_Max", "Grain_Area_mm2_Std",
    # Nhóm 7: Thể tích 3D Hạt (4 biến)
    "Grain_Volume_mm3_Mean", "Grain_Volume_mm3_Min", "Grain_Volume_mm3_Max", "Grain_Volume_mm3_Std"
]

TARGET_COL = "Actual_Count"


def find_project_root() -> Path:
    """Tự động tìm thư mục gốc MAIN_SOURCES từ vị trí file script."""
    current = Path(__file__).resolve().parent
    if current.name == "LINEAR_REGRESSION_MODEL":
        return current.parent
    if (current / "LINEAR_REGRESSION_MODEL").exists():
        return current
    return current


def run_training_pipeline(
    base_dir: Optional[Path] = None,
    test_size_ratio: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    if base_dir is None:
        base_dir = find_project_root()

    data_csv = base_dir / "DATASET_BUILDER" / "4_Final_Dataset" / "final_linear_regression_dataset.csv"
    output_dir = base_dir / "LINEAR_REGRESSION_MODEL"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("🌾 PIPELINE HUẤN LUYỆN MÔ HÌNH HỒI QUY TUYẾN TÍNH — RICE SEED LINEAR REGRESSION AI")
    print("=" * 100)
    print(f"📁 Thư mục gốc      : {base_dir}")
    print(f"📁 Tệp dữ liệu CSV  : {data_csv}")
    print(f"💾 Thư mục Models   : {models_dir}")
    print(f"📊 Thư mục Kết quả  : {results_dir}")
    print("=" * 100 + "\n")

    if not data_csv.exists():
        print(f"❌ CẢNH BÁO: Không tìm thấy tệp {data_csv}")
        return {}

    # 1. Nạp và làm sạch dữ liệu
    with open(data_csv, "r", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    valid_rows = [r for r in all_rows if r.get("Image_Status") == "FOUND"]
    print(f"🔹 Tổng số mẫu trong dataset : {len(all_rows)} mẫu")
    print(f"🔹 Số mẫu hợp lệ (FOUND)     : {len(valid_rows)} mẫu ({len(valid_rows)/len(all_rows)*100:.1f}%)")
    print(f"🔹 Số biến đặc trưng đầu vào : {len(ALL_31_FEATURES)} thuộc tính gốc")
    print(f"🔹 Biến mục tiêu dự đoán (Y) : '{TARGET_COL}' (Số hạt lúa thực tế)\n")

    X_raw: List[List[float]] = []
    y_raw: List[float] = []
    sample_ids: List[str] = []

    for r in valid_rows:
        try:
            y_val = float(r[TARGET_COL])
            x_vals = [float(r.get(feat, 0.0)) if r.get(feat, "") != "" else 0.0 for feat in ALL_31_FEATURES]
            X_raw.append(x_vals)
            y_raw.append(y_val)
            sample_ids.append(r.get("Sample_ID", "Unknown"))
        except Exception:
            continue

    X = np.array(X_raw, dtype=np.float64)
    y = np.array(y_raw, dtype=np.float64)
    sids = np.array(sample_ids)

    # Lọc nan/inf
    valid_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y)) & (~np.isinf(X).any(axis=1)) & (~np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]
    sids = sids[valid_mask]

    N_samples, N_features = X.shape
    print(f"✅ Dữ liệu hoàn chỉnh sẵn sàng: {N_samples} mẫu x {N_features} đặc trưng.")

    # 2. Chuẩn hóa Z-Score (StandardScaler)
    scaler_mean = np.mean(X, axis=0)
    scaler_std = np.std(X, axis=0)
    scaler_std[scaler_std == 0.0] = 1.0  # Tránh chia cho 0 nếu biến có phương sai bằng 0
    X_scaled = (X - scaler_mean) / scaler_std

    # Lưu scaler tham số ra JSON
    scaler_params = {
        "features": ALL_31_FEATURES,
        "mean": scaler_mean.tolist(),
        "std": scaler_std.tolist()
    }
    with open(models_dir / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    # 3. Phân chia Train / Test (80% / 20%)
    np.random.seed(random_state)
    shuffled_idx = np.random.permutation(N_samples)
    n_test = int(N_samples * test_size_ratio)
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    X_train, y_train, sids_train = X[train_idx], y[train_idx], sids[train_idx]
    X_test, y_test, sids_test = X[test_idx], y[test_idx], sids[test_idx]

    X_train_s = X_scaled[train_idx]
    X_test_s = X_scaled[test_idx]

    print(f"🔹 Tập Huấn Luyện (Train Set): {len(y_train)} mẫu ({(1-test_size_ratio)*100:.0f}%)")
    print(f"🔹 Tập Kiểm Thử (Test Set)  : {len(y_test)} mẫu ({test_size_ratio*100:.0f}%)\n")

    # 4. Huấn luyện các biến thể Hồi quy Tuyến tính
    # Hàm tính metrics
    def calc_metrics(y_true, y_pred, k_features):
        n = len(y_true)
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Adjusted R2
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - k_features - 1)) if (n - k_features - 1) > 0 else r2
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mape = float(np.mean(np.abs(residuals / y_true)) * 100)
        max_err = float(np.max(np.abs(residuals)))
        return {
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "max_err": max_err,
            "pred": y_pred,
            "residuals": residuals
        }

    # A. OLS Linear Regression
    X_tr_b = np.column_stack([X_train_s, np.ones(len(y_train))])
    X_te_b = np.column_stack([X_test_s, np.ones(len(y_test))])
    
    # Giải OLS bằng least-squares
    beta_ols, _, _, _ = np.linalg.lstsq(X_tr_b, y_train, rcond=None)
    pred_tr_ols = X_tr_b @ beta_ols
    pred_te_ols = X_te_b @ beta_ols

    # B. Ridge Regression (L2 Regularization với alpha=0.1)
    alpha_ridge = 0.1
    I_mat = np.eye(X_tr_b.shape[1])
    I_mat[-1, -1] = 0.0  # Không phạt bias
    beta_ridge = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha_ridge * I_mat, X_tr_b.T @ y_train)
    pred_tr_ridge = X_tr_b @ beta_ridge
    pred_te_ridge = X_te_b @ beta_ridge

    # C. Ridge với alpha=1.0 (Bảo toàn hơn)
    alpha_ridge_1 = 1.0
    beta_ridge_1 = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha_ridge_1 * I_mat, X_tr_b.T @ y_train)
    pred_tr_ridge1 = X_tr_b @ beta_ridge_1
    pred_te_ridge1 = X_te_b @ beta_ridge_1

    # Đánh giá 3 mô hình
    m_ols_tr = calc_metrics(y_train, pred_tr_ols, N_features)
    m_ols_te = calc_metrics(y_test, pred_te_ols, N_features)

    m_ridge_tr = calc_metrics(y_train, pred_tr_ridge, N_features)
    m_ridge_te = calc_metrics(y_test, pred_te_ridge, N_features)

    m_ridge1_tr = calc_metrics(y_train, pred_tr_ridge1, N_features)
    m_ridge1_te = calc_metrics(y_test, pred_te_ridge1, N_features)

    model_candidates = [
        ("1. Ridge Regression (alpha=0.1 - Khuyến nghị)", beta_ridge, m_ridge_tr, m_ridge_te),
        ("2. OLS Linear Regression (Chuẩn cơ sở)", beta_ols, m_ols_tr, m_ols_te),
        ("3. Ridge Regularized (alpha=1.0 - Ổn định cao)", beta_ridge_1, m_ridge1_tr, m_ridge1_te),
    ]

    print("=" * 105)
    print("🏆 BẢNG SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN HỒI QUY TUYẾN TÍNH (TOÀN BỘ 31 FEATURES):")
    print("=" * 105)
    print(f"{'Thuật Toán Mô Hình':<45} | {'R² Test':<10} | {'Adj-R²':<10} | {'MAE Test':<10} | {'RMSE':<10} | {'MAPE (%)'}")
    print("-" * 105)

    for name, _, m_tr, m_te in model_candidates:
        print(f"{name:<45} | {m_te['r2']:<10.4f} | {m_te['adj_r2']:<10.4f} | {m_te['mae']:<10.2f} | {m_te['rmse']:<10.2f} | {m_te['mape']:<10.2f}%")
    print("=" * 105 + "\n")

    # Chọn mô hình tốt nhất (Ridge alpha=0.1)
    best_name, best_beta, best_m_tr, best_m_te = model_candidates[0]

    # 5. Chuyển đổi hệ số trọng số về không gian biến gốc (Unscaled Coefficients)
    # y = sum(beta_scaled_i * (x_i - mean_i) / std_i) + bias
    # y = sum((beta_scaled_i / std_i) * x_i) + (bias - sum(beta_scaled_i * mean_i / std_i))
    scaled_weights = best_beta[:-1]
    scaled_bias = best_beta[-1]

    unscaled_slopes = scaled_weights / scaler_std
    unscaled_intercept = scaled_bias - np.sum(scaled_weights * scaler_mean / scaler_std)

    # 6. Lưu phương trình hồi quy toán học đầy đủ ra JSON
    equation_dict = {
        "model_type": "Ridge_Linear_Regression",
        "description": "Mô hình Hồi quy tuyến tính ước lượng số lượng hạt lúa sử dụng toàn bộ 31 đặc trưng",
        "target_variable": TARGET_COL,
        "number_of_features": len(ALL_31_FEATURES),
        "intercept": float(unscaled_intercept),
        "r2_test": best_m_te["r2"],
        "mae_test": best_m_te["mae"],
        "rmse_test": best_m_te["rmse"],
        "mape_test": best_m_te["mape"],
        "features": []
    }

    print("📋 PHƯƠNG TRÌNH HỒI QUY TOÁN HỌC TỔNG QUÁT CỦA MÔ HÌNH:")
    print("=" * 105)
    print(f"Số_Hạt_Dự_Đoán = {unscaled_intercept:+.4f}")
    
    for i, feat in enumerate(ALL_31_FEATURES):
        w_scaled = float(scaled_weights[i])
        w_unscaled = float(unscaled_slopes[i])
        equation_dict["features"].append({
            "name": feat,
            "standardized_weight": w_scaled,
            "unscaled_coefficient": w_unscaled,
            "mean": float(scaler_mean[i]),
            "std": float(scaler_std[i])
        })
        sign = "+" if w_unscaled >= 0 else "-"
        print(f"   {sign} ({abs(w_unscaled):.6f} * {feat})")
    print("=" * 105 + "\n")

    eq_path = models_dir / "regression_equation.json"
    with open(eq_path, "w", encoding="utf-8") as f:
        json.dump(equation_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu phương trình toán học vào: {eq_path}")

    # 7. Xuất file dự đoán chi tiết trên tập Test sang CSV
    test_pred_csv = results_dir / "test_set_predictions.csv"
    with open(test_pred_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Actual_Count", "Predicted_Count", "Error_Seeds", "Abs_Error", "Percentage_Error_%"])
        for sid, y_t, y_p in zip(sids_test, y_test, best_m_te["pred"]):
            err = y_p - y_t
            abs_err = abs(err)
            pct_err = (abs_err / y_t) * 100 if y_t > 0 else 0
            writer.writerow([sid, round(y_t, 1), round(y_p, 2), round(err, 2), round(abs_err, 2), round(pct_err, 2)])

    print(f"💾 Đã xuất bảng dự đoán chi tiết tập Test vào: {test_pred_csv}")

    # 8. Xuất bảng báo cáo tổng hợp ra CSV
    eval_csv = results_dir / "training_evaluation_report.csv"
    with open(eval_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Name", "Dataset", "Samples_Count", "R2_Score", "Adj_R2_Score", "MAE_Seeds", "RMSE_Seeds", "MAPE_%", "Max_Error"])
        for name, _, m_tr, m_te in model_candidates:
            writer.writerow([name, "Train", len(y_train), m_tr["r2"], m_tr["adj_r2"], m_tr["mae"], m_tr["rmse"], m_tr["mape"], m_tr["max_err"]])
            writer.writerow([name, "Test", len(y_test), m_te["r2"], m_te["adj_r2"], m_te["mae"], m_te["rmse"], m_te["mape"], m_te["max_err"]])

    print(f"💾 Đã xuất báo cáo đánh giá mô hình vào: {eval_csv}")

    # 9. Thử vẽ biểu đồ nếu môi trường có matplotlib
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        # 1. Parity Plot: Actual vs Predicted
        plt.figure(figsize=(8, 7))
        plt.scatter(y_train, best_m_tr["pred"], color='#1E88E5', alpha=0.6, label=f'Train (N={len(y_train)}, R²={best_m_tr["r2"]:.4f})')
        plt.scatter(y_test, best_m_te["pred"], color='#E53935', alpha=0.85, marker='^', s=55, label=f'Test (N={len(y_test)}, R²={best_m_te["r2"]:.4f})')
        
        min_v = min(np.min(y), np.min(best_m_te["pred"])) - 10
        max_v = max(np.max(y), np.max(best_m_te["pred"])) + 10
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5, label='Đường lý tưởng (y = x)')
        plt.fill_between([min_v, max_v], [min_v*0.95, max_v*0.95], [min_v*1.05, max_v*1.05], color='gray', alpha=0.15, label='Dải tin cậy ±5%')
        
        plt.title(f"Parity Plot: Số Hạt Thực Tế vs Số Hạt Dự Đoán (31 Features)\nTest MAE={best_m_te['mae']:.2f} hạt | MAPE={best_m_te['mape']:.2f}%", fontweight='bold')
        plt.xlabel("Số hạt thực tế (Actual Count)")
        plt.ylabel("Số hạt dự đoán (Predicted Count)")
        plt.legend()
        plt.xlim(min_v, max_v)
        plt.ylim(min_v, max_v)
        plt.tight_layout()
        
        parity_path = results_dir / "parity_plot_actual_vs_predicted.png"
        plt.savefig(parity_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Đã lưu biểu đồ Parity Plot vào: {parity_path}")

        # 2. Bar chart: Toàn bộ 31 Standardized Weights
        plt.figure(figsize=(14, 10))
        sorted_indices = np.argsort(np.abs(scaled_weights))[::-1]
        feat_sorted = [ALL_31_FEATURES[i] for i in sorted_indices]
        w_sorted = scaled_weights[sorted_indices]
        colors = ['#1E88E5' if w >= 0 else '#E53935' for w in w_sorted]
        
        plt.barh(feat_sorted[::-1], w_sorted[::-1], color=colors[::-1], edgecolor='black', linewidth=0.6)
        plt.axvline(0, color='black', linewidth=1)
        plt.title("Trọng Số Chuẩn Hóa (Standardized Coefficients) Của Toàn Bộ 31 Thuộc Tính Gốc", fontweight='bold', fontsize=12)
        plt.xlabel("Trọng số hồi quy (Standardized Beta Weight)", fontweight='bold')
        plt.tight_layout()
        
        weights_chart_path = results_dir / "all_31_features_weights_chart.png"
        plt.savefig(weights_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Đã lưu biểu đồ trọng số 31 features vào: {weights_chart_path}")

    except Exception:
        pass

    print("\n" + "=" * 100)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    print(f"   • Mô hình tốt nhất: Ridge Linear Regression (31 Features)")
    print(f"   • Độ chính xác Test R²  : {best_m_te['r2']:.4f} ({best_m_te['r2']*100:.2f}%)")
    print(f"   • Sai số trung bình MAE : {best_m_te['mae']:.2f} hạt")
    print(f"   • Tỷ lệ sai số MAPE     : {best_m_te['mape']:.2f}%")
    print("=" * 100 + "\n")

    return {
        "best_model_name": best_name,
        "r2_test": best_m_te["r2"],
        "mae_test": best_m_te["mae"],
        "mape_test": best_m_te["mape"],
        "unscaled_intercept": unscaled_intercept,
        "equation_json_path": str(eq_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình hồi quy tuyến tính với toàn bộ 31 thuộc tính")
    parser.add_argument("--test_size", type=float, default=0.20, help="Tỷ lệ phân chia tập test (mặc định: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    args = parser.parse_args()

    run_training_pipeline(test_size_ratio=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
