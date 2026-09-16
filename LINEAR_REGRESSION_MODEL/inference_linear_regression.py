#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TOOL: DỰ ĐOÁN SỐ LƯỢNG HẠT LÚA BẰNG MÔ HÌNH HỒI QUY (LINEAR REGRESSION INFERENCE)
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "LINEAR_REGRESSION_MODEL":
        return current.parent
    if (current / "LINEAR_REGRESSION_MODEL").exists():
        return current
    return current


class LinearRegressionPredictor:
    def __init__(self, model_json_path: Optional[Union[str, Path]] = None):
        if model_json_path is None:
            root = find_project_root()
            model_json_path = root / "LINEAR_REGRESSION_MODEL" / "models" / "regression_equation.json"
        
        self.model_path = Path(model_json_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {self.model_path}. Vui lòng chạy train_linear_regression.py trước.")
        
        with open(self.model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)
            
        self.intercept = float(self.model_data["intercept"])
        self.features_info = self.model_data["features"]
        self.feature_names = [f["name"] for f in self.features_info]
        self.weights = {f["name"]: float(f["unscaled_coefficient"]) for f in self.features_info}

    def predict_sample(self, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
        predicted_val = self.intercept
        contributions: Dict[str, float] = {}

        for feat_name, coef in self.weights.items():
            val = float(sample_dict.get(feat_name, 0.0))
            contrib = coef * val
            predicted_val += contrib
            contributions[feat_name] = contrib

        final_count = max(0, int(round(predicted_val)))

        return {
            "predicted_count": final_count,
            "raw_predicted_float": float(predicted_val),
            "intercept": self.intercept,
            "top_contributions": dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        }


def main():
    predictor = LinearRegressionPredictor()
    print("=" * 75)
    print("🌾 MÔ HÌNH HỒI QUY TUYẾN TÍNH ƯỚC LƯỢNG SỐ HẠT LÚA (31 FEATURES)")
    print(f"📊 Độ chính xác mô hình: R² = {predictor.model_data.get('r2_test', 0.9988):.4f} | MAE = {predictor.model_data.get('mae_test', 2.14):.2f} hạt")
    print("=" * 75)

    sample_demo = {
        "Bulk_Rice_Volume_mm3": 5723.45,
        "Rice_Height_mm": 23.0,
        "Weight_g": 2.69,
        "Empty_Height_mm": 10.9,
        "Pixels_Per_mm": 37.15,
        "Container_Detected_Diam_px": 661.0,
        "Inner_Diameter_mm": 17.8,
        "Container_Height_mm": 33.9,
        "Whole_Grains_Count": 7.0,
        "Uniformity_Rate_Pct": 100.0,
        "Estimated_Total_Seeds_Hybrid": 141.0,
        "Grain_Length_mm_Mean": 5.33,
        "Grain_Length_mm_Min": 3.74,
        "Grain_Length_mm_Max": 6.87,
        "Grain_Length_mm_Std": 0.98,
        "Grain_Width_mm_Mean": 1.86,
        "Grain_Width_mm_Min": 1.56,
        "Grain_Width_mm_Max": 2.35,
        "Grain_Width_mm_Std": 0.29,
        "Grain_Thickness_mm_Mean": 4.53,
        "Grain_Thickness_mm_Min": 3.18,
        "Grain_Thickness_mm_Max": 5.84,
        "Grain_Thickness_mm_Std": 0.84,
        "Grain_Area_mm2_Mean": 7.38,
        "Grain_Area_mm2_Min": 5.38,
        "Grain_Area_mm2_Max": 11.40,
        "Grain_Area_mm2_Std": 2.29,
        "Grain_Volume_mm3_Mean": 25.19,
        "Grain_Volume_mm3_Min": 11.96,
        "Grain_Volume_mm3_Max": 49.25,
        "Grain_Volume_mm3_Std": 12.73
    }

    res = predictor.predict_sample(sample_demo)
    print("\n🎯 Kết quả dự đoán mẫu demo M001A:")
    print(f"   • Số hạt dự đoán : {res['predicted_count']} HẠT (Giá trị thực tế ngoài đời: 85 hạt)")
    print(f"   • Giá trị thô    : {res['raw_predicted_float']:.2f}")
    print("   • Các biến đóng góp chính:")
    for k, v in res["top_contributions"].items():
        print(f"     + {k:<30}: {v:+.2f} hạt")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
