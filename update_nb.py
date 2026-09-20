import json
import re

nb_path = 'CODE/RICE_VISION_MAIN_PIPELINE.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        new_source = re.sub(r'(!pip install.*)', r'\1 joblib scikit-learn==1.6.1', source)
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip('\n') # keep last line without \n if original had none

# Update Cell 3
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'PACKING_FRACTION' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Replace the comment
        new_source = source.replace('(~0.60 - 0.64)', '(~0.80 - 0.85)')
        
        # Insert validation if not exists
        if '0.80 <= PACKING_FRACTION <= 0.85' not in new_source:
            lines = new_source.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if line.strip().startswith('PACKING_FRACTION'):
                    new_lines.append('if not (0.80 <= PACKING_FRACTION <= 0.85):')
                    new_lines.append('    raise ValueError(f"PACKING_FRACTION={PACKING_FRACTION} không hợp lệ! Phải nằm trong khoảng 0.80-0.85")')
            new_source = '\n'.join(new_lines)
            
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip('\n')

# Find Cell 10 index
target_idx = -1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Bước 10:' in src or 'TẦNG 7' in src:
        target_idx = i
        break

if target_idx != -1:
    print(f"Found Cell 10 at index {target_idx}. Deleting and inserting new cells.")
    
    cell_config = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "REGRESSION_CONFIG"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# REGRESSION_CONFIG: CẤU HÌNH THƯ MỤC MÔ HÌNH VÀ THÔNG SỐ CÂN\n",
            "# ==============================================================================\n",
            "import os\n",
            "\n",
            "# Xóa state kết quả cũ nếu chạy lại\n",
            "if 'regression_bundle' in locals(): del regression_bundle\n",
            "if 'regression_features' in locals(): del regression_features\n",
            "if 'regression_result' in locals(): del regression_result\n",
            "\n",
            "# 1. ĐƯỜNG DẪN ĐẾN THƯ MỤC BUNDLE MÔ HÌNH\n",
            "# (Mặc định dùng ARD Regression thay vì Linear Regression mặc định cũ)\n",
            "REGRESSION_MODEL_DIR = \"LINEAR_REGRESSION_MODEL/models/ard\"\n",
            "\n",
            "# 2. KHỐI LƯỢNG MẪU THỰC TẾ (GAM)\n",
            "INPUT_WEIGHT_G = 3.71  # Vui lòng cân chính xác mẫu đang chụp để có kết quả tốt nhất\n",
            "\n",
            "# Xử lý đường dẫn\n",
            "resolved_model_dir = REGRESSION_MODEL_DIR\n",
            "if not os.path.isabs(resolved_model_dir):\n",
            "    resolved_model_dir = os.path.abspath(os.path.join(BASE_PATH, resolved_model_dir))\n",
            "\n",
            "print(f\"📂 Thư mục bundle mô hình được chọn: {resolved_model_dir}\")\n"
        ]
    }
    
    cell_loader = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "REGRESSION_BUNDLE_LOADER"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# REGRESSION_BUNDLE_LOADER: NẠP VÀ KIỂM TRA TÍNH TOÀN VẸN CỦA BUNDLE\n",
            "# ==============================================================================\n",
            "import json\n",
            "import joblib\n",
            "from sklearn.utils.validation import check_is_fitted\n",
            "\n",
            "def load_regression_bundle(model_dir):\n",
            "    schema_path = os.path.join(model_dir, \"feature_schema.json\")\n",
            "    model_path = os.path.join(model_dir, \"model.joblib\")\n",
            "    scaler_path = os.path.join(model_dir, \"scaler.joblib\")\n",
            "    config_path = os.path.join(model_dir, \"config.json\")\n",
            "    metrics_path = os.path.join(model_dir, \"metrics.json\")\n",
            "    \n",
            "    if not os.path.exists(model_dir):\n",
            "        raise FileNotFoundError(f\"Không tìm thấy thư mục mô hình: {model_dir}\")\n",
            "    if not os.path.exists(schema_path):\n",
            "        raise FileNotFoundError(f\"Thiếu feature_schema.json trong {model_dir}\")\n",
            "    if not os.path.exists(model_path):\n",
            "        raise FileNotFoundError(f\"Thiếu model.joblib trong {model_dir}\")\n",
            "        \n",
            "    with open(schema_path, \"r\", encoding=\"utf-8\") as f:\n",
            "        schema = json.load(f)\n",
            "        \n",
            "    if schema.get(\"schema_version\") != \"31v1\":\n",
            "        raise ValueError(\"Chỉ hỗ trợ schema_version '31v1'\")\n",
            "        \n",
            "    ordered_features = schema.get(\"features\", [])\n",
            "    if len(ordered_features) != 31 or \"Actual_Count\" in ordered_features:\n",
            "        raise ValueError(\"Schema phải chứa chính xác 31 features và không có Actual_Count.\")\n",
            "        \n",
            "    EXPECTED_31_FEATURES = [\n",
            "        \"Bulk_Rice_Volume_mm3\", \"Rice_Height_mm\", \"Weight_g\", \"Empty_Height_mm\",\n",
            "        \"Pixels_Per_mm\", \"Container_Detected_Diam_px\", \"Inner_Diameter_mm\", \"Container_Height_mm\",\n",
            "        \"Whole_Grains_Count\", \"Uniformity_Rate_Pct\", \"Estimated_Total_Seeds_Hybrid\",\n",
            "        \"Grain_Length_mm_Mean\", \"Grain_Length_mm_Min\", \"Grain_Length_mm_Max\", \"Grain_Length_mm_Std\",\n",
            "        \"Grain_Width_mm_Mean\", \"Grain_Width_mm_Min\", \"Grain_Width_mm_Max\", \"Grain_Width_mm_Std\",\n",
            "        \"Grain_Thickness_mm_Mean\", \"Grain_Thickness_mm_Min\", \"Grain_Thickness_mm_Max\", \"Grain_Thickness_mm_Std\",\n",
            "        \"Grain_Area_mm2_Mean\", \"Grain_Area_mm2_Min\", \"Grain_Area_mm2_Max\", \"Grain_Area_mm2_Std\",\n",
            "        \"Grain_Volume_mm3_Mean\", \"Grain_Volume_mm3_Min\", \"Grain_Volume_mm3_Max\", \"Grain_Volume_mm3_Std\"\n",
            "    ]\n",
            "    if ordered_features != EXPECTED_31_FEATURES:\n",
            "        raise ValueError(\"Danh sách và thứ tự feature không khớp với hợp đồng 31 features quy định.\")\n",
            "        \n",
            "    config = {}\n",
            "    if os.path.exists(config_path):\n",
            "        with open(config_path, \"r\", encoding=\"utf-8\") as f:\n",
            "            config = json.load(f)\n",
            "            \n",
            "    if config.get(\"preprocessing\") != \"none\" and not os.path.exists(scaler_path):\n",
            "        raise FileNotFoundError(f\"Mô hình yêu cầu preprocessing nhưng thiếu scaler.joblib\")\n",
            "    if config.get(\"preprocessing\") == \"none\" and os.path.exists(scaler_path):\n",
            "        raise ValueError(\"Bundle mâu thuẫn: có file scaler.joblib nhưng config khai báo preprocessing='none'\")\n",
            "        \n",
            "    try:\n",
            "        import warnings\n",
            "        with warnings.catch_warnings(record=True) as w:\n",
            "            warnings.simplefilter(\"always\")\n",
            "            model = joblib.load(model_path)\n",
            "            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None\n",
            "            if w:\n",
            "                for warn in w:\n",
            "                    print(f\"⚠️ Cảnh báo lúc load model/scaler: {warn.message}\")\n",
            "    except Exception as e:\n",
            "        raise RuntimeError(f\"Lỗi khi load model/scaler bằng joblib: {e}\")\n",
            "        \n",
            "    if hasattr(model, \"steps\") and hasattr(model, \"predict\"):\n",
            "        raise ValueError(\"model.joblib không được là Pipeline để tránh lặp preprocessing.\")\n",
            "        \n",
            "    check_is_fitted(model)\n",
            "    if hasattr(model, \"feature_names_in_\"):\n",
            "        if len(model.feature_names_in_) != 31:\n",
            "            raise ValueError(f\"Mô hình yêu cầu {len(model.feature_names_in_)} features, nhưng schema cung cấp 31.\")\n",
            "        if list(model.feature_names_in_) != ordered_features:\n",
            "            raise ValueError(\"Thứ tự features trong mô hình khác với schema.\")\n",
            "            \n",
            "    metrics = {\"unavailable\": True}\n",
            "    if os.path.exists(metrics_path):\n",
            "        try:\n",
            "            with open(metrics_path, \"r\", encoding=\"utf-8\") as f:\n",
            "                metrics = json.load(f)\n",
            "        except json.JSONDecodeError:\n",
            "            print(f\"⚠️ Không thể đọc {metrics_path} (File bị hỏng)\")\n",
            "            \n",
            "    return {\n",
            "        \"model\": model,\n",
            "        \"scaler\": scaler,\n",
            "        \"ordered_features\": ordered_features,\n",
            "        \"schema_version\": schema.get(\"schema_version\"),\n",
            "        \"folder\": model_dir,\n",
            "        \"config\": config,\n",
            "        \"metrics\": metrics\n",
            "    }\n",
            "\n",
            "print(\"🔄 Đang nạp bundle mô hình...\")\n",
            "regression_bundle = load_regression_bundle(resolved_model_dir)\n",
            "print(f\"✅ Nạp thành công mô hình: {regression_bundle['model'].__class__.__name__}\")\n"
        ]
    }
    
    cell_features = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "REGRESSION_FEATURES"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# REGRESSION_FEATURES: TRÍCH XUẤT ĐẶC TRƯNG TỪ QUY TRÌNH HÌNH HỌC\n",
            "# ==============================================================================\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import math\n",
            "\n",
            "def assemble_regression_features(bundle):\n",
            "    if len(whole_grains) == 0:\n",
            "        raise ValueError(\"Không tìm thấy hạt lúa nguyên nào (whole grains). Dừng suy luận regression.\")\n",
            "    \n",
            "    if INPUT_WEIGHT_G is None:\n",
            "        raise ValueError(\"Thiếu dữ liệu cân nặng thật (INPUT_WEIGHT_G=None). Vui lòng cân chính xác mẫu.\")\n",
            "    actual_weight_g = float(INPUT_WEIGHT_G)\n",
            "    if actual_weight_g <= 0:\n",
            "        raise ValueError(f\"Dữ liệu cân nặng không hợp lệ: {actual_weight_g} g\")\n",
            "        \n",
            "    lens_arr = np.array([float(r[\"length_mm_2a\"]) for r in whole_records])\n",
            "    wids_arr = np.array([float(r[\"width_mm_2b\"]) for r in whole_records])\n",
            "    thks_arr = np.array([float(r[\"thickness_mm_2c\"]) for r in whole_records])\n",
            "    area_arr = np.array([float(r[\"area_mm2\"]) for r in whole_records])\n",
            "    vols_arr = np.array([float(r[\"volume_3d_mm3\"]) for r in whole_records])\n",
            "    \n",
            "    if np.any(~np.isfinite(lens_arr)) or np.any(lens_arr <= 0) or np.any(vols_arr <= 0):\n",
            "        raise ValueError(\"Measurements chứa giá trị NaN, Inf hoặc <= 0.\")\n",
            "        \n",
            "    if \"rice_height_mm\" not in container_info or \"inner_w_px\" not in container_info:\n",
            "        raise ValueError(\"Thiếu thông số container_info (rice_height_mm hoặc inner_w_px).\")\n",
            "        \n",
            "    def calc_stats(arr):\n",
            "        return {\n",
            "            \"Mean\": round(float(np.mean(arr)), 3),\n",
            "            \"Min\": round(float(np.min(arr)), 3),\n",
            "            \"Max\": round(float(np.max(arr)), 3),\n",
            "            \"Std\": round(float(np.std(arr, ddof=0)), 3)\n",
            "        }\n",
            "        \n",
            "    lens_st = calc_stats(lens_arr)\n",
            "    wids_st = calc_stats(wids_arr)\n",
            "    thks_st = calc_stats(thks_arr)\n",
            "    area_st = calc_stats(area_arr)\n",
            "    vols_st = calc_stats(vols_arr)\n",
            "    \n",
            "    from modules.uniformity_evaluator import evaluate_batch_uniformity\n",
            "    batch_uniformity = evaluate_batch_uniformity(vols_arr.tolist())\n",
            "    unif_rate = round(float(batch_uniformity.get(\"uniformity_rate_pct\", 0.0)), 2)\n",
            "    \n",
            "    raw_bulk_volume = float(bulk_volume_mm3)\n",
            "    mean_volume = float(np.mean(vols_arr))\n",
            "    hybrid_estimate = int(round((raw_bulk_volume * 0.62) / mean_volume))\n",
            "    \n",
            "    feat_dict = {\n",
            "        \"Bulk_Rice_Volume_mm3\": round(raw_bulk_volume, 2),\n",
            "        \"Rice_Height_mm\": round(float(container_info[\"rice_height_mm\"]), 2),\n",
            "        \"Weight_g\": actual_weight_g,\n",
            "        \"Empty_Height_mm\": float(EMPTY_HEIGHT_MM),\n",
            "        \"Pixels_Per_mm\": round(float(pixels_per_mm), 2),\n",
            "        \"Container_Detected_Diam_px\": round(float(container_info[\"inner_w_px\"]), 2),\n",
            "        \"Inner_Diameter_mm\": float(INNER_DIAM_MM),\n",
            "        \"Container_Height_mm\": float(CONTAINER_HEIGHT_MM),\n",
            "        \"Whole_Grains_Count\": float(len(whole_records)),\n",
            "        \"Uniformity_Rate_Pct\": unif_rate,\n",
            "        \"Estimated_Total_Seeds_Hybrid\": float(hybrid_estimate),\n",
            "        \n",
            "        \"Grain_Length_mm_Mean\": lens_st[\"Mean\"],\n",
            "        \"Grain_Length_mm_Min\": lens_st[\"Min\"],\n",
            "        \"Grain_Length_mm_Max\": lens_st[\"Max\"],\n",
            "        \"Grain_Length_mm_Std\": lens_st[\"Std\"],\n",
            "        \n",
            "        \"Grain_Width_mm_Mean\": wids_st[\"Mean\"],\n",
            "        \"Grain_Width_mm_Min\": wids_st[\"Min\"],\n",
            "        \"Grain_Width_mm_Max\": wids_st[\"Max\"],\n",
            "        \"Grain_Width_mm_Std\": wids_st[\"Std\"],\n",
            "        \n",
            "        \"Grain_Thickness_mm_Mean\": thks_st[\"Mean\"],\n",
            "        \"Grain_Thickness_mm_Min\": thks_st[\"Min\"],\n",
            "        \"Grain_Thickness_mm_Max\": thks_st[\"Max\"],\n",
            "        \"Grain_Thickness_mm_Std\": thks_st[\"Std\"],\n",
            "        \n",
            "        \"Grain_Area_mm2_Mean\": area_st[\"Mean\"],\n",
            "        \"Grain_Area_mm2_Min\": area_st[\"Min\"],\n",
            "        \"Grain_Area_mm2_Max\": area_st[\"Max\"],\n",
            "        \"Grain_Area_mm2_Std\": area_st[\"Std\"],\n",
            "        \n",
            "        \"Grain_Volume_mm3_Mean\": vols_st[\"Mean\"],\n",
            "        \"Grain_Volume_mm3_Min\": vols_st[\"Min\"],\n",
            "        \"Grain_Volume_mm3_Max\": vols_st[\"Max\"],\n",
            "        \"Grain_Volume_mm3_Std\": vols_st[\"Std\"]\n",
            "    }\n",
            "    \n",
            "    for k, v in feat_dict.items():\n",
            "        if not math.isfinite(v):\n",
            "            raise ValueError(f\"Feature {k} không phải là số hữu hạn: {v}\")\n",
            "            \n",
            "    df = pd.DataFrame([feat_dict], columns=bundle[\"ordered_features\"])\n",
            "    return df\n",
            "\n",
            "print(\"🔄 Đang trích xuất 31 đặc trưng...\")\n",
            "regression_features = assemble_regression_features(regression_bundle)\n",
            "print(f\"✅ Đặc trưng hợp lệ. Hybrid (với 0.62): {regression_features['Estimated_Total_Seeds_Hybrid'].iloc[0]}\")\n"
        ]
    }
    
    cell_predict = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "REGRESSION_PREDICT"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# REGRESSION_PREDICT: THỰC THI SUY LUẬN SỐ LƯỢNG HẠT\n",
            "# ==============================================================================\n",
            "\n",
            "def predict_regression_bundle(bundle, features_df):\n",
            "    model = bundle[\"model\"]\n",
            "    scaler = bundle[\"scaler\"]\n",
            "    \n",
            "    if hasattr(model, \"feature_names_in_\"):\n",
            "        X_input = features_df\n",
            "    else:\n",
            "        X_input = features_df.values\n",
            "        \n",
            "    if scaler is not None:\n",
            "        X_scaled = scaler.transform(X_input)\n",
            "    else:\n",
            "        X_scaled = X_input\n",
            "        \n",
            "    if not np.all(np.isfinite(X_scaled)):\n",
            "        raise ValueError(\"Vector đặc trưng chứa NaN/Inf sau khi chuẩn hóa.\")\n",
            "        \n",
            "    raw_pred = model.predict(X_scaled)\n",
            "    if raw_pred.shape not in [(1,), (1, 1)]:\n",
            "        raise ValueError(f\"Mô hình trả về shape không hợp lệ: {raw_pred.shape}\")\n",
            "        \n",
            "    raw_val = float(raw_pred.item())\n",
            "    if not math.isfinite(raw_val):\n",
            "        raise ValueError(f\"Kết quả dự đoán không phải số hữu hạn: {raw_val}\")\n",
            "        \n",
            "    return raw_val\n",
            "\n",
            "print(\"🔄 Đang chạy suy luận...\")\n",
            "raw_predicted_count = predict_regression_bundle(regression_bundle, regression_features)\n",
            "final_predicted_seeds = max(0, int(round(raw_predicted_count)))\n",
            "regression_result = {\n",
            "    \"raw_count\": raw_predicted_count,\n",
            "    \"final_count\": final_predicted_seeds\n",
            "}\n",
            "print(f\"✅ Dự đoán thành công: {final_predicted_seeds} hạt\")\n"
        ]
    }
    
    cell_report = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "REGRESSION_REPORT"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# REGRESSION_REPORT: XUẤT KẾT QUẢ VÀ LƯU BÁO CÁO\n",
            "# ==============================================================================\n",
            "if 'regression_result' not in locals():\n",
            "    raise RuntimeError(\"Chưa có kết quả. Vui lòng chạy các Cell phía trên tuần tự.\")\n",
            "\n",
            "metrics = regression_bundle[\"metrics\"]\n",
            "cv_mae = metrics.get(\"cv_summary\", {}).get(\"cv_mae_mean\", \"N/A\")\n",
            "cv_r2 = metrics.get(\"cv_summary\", {}).get(\"cv_r2_mean\", \"N/A\")\n",
            "test_mae = metrics.get(\"metrics_test\", {}).get(\"mae\", \"N/A\")\n",
            "\n",
            "print(\"\\n\" + \"╔\" + \"═\" * 78 + \"╗\")\n",
            "print(\"║\" + \" \" * 22 + \"🤖 KẾT QUẢ DỰ ĐOÁN HỒI QUY (AI)\" + \" \" * 23 + \"║\")\n",
            "print(\"╠\" + \"═\" * 78 + \"╣\")\n",
            "print(f\"║  📸 Tệp ảnh phân tích      : {os.path.basename(IMAGE_PATH):<47} ║\")\n",
            "print(f\"║  ⚖️  Khối lượng mẫu         : {INPUT_WEIGHT_G:<6.2f} gam {'':<46} ║\")\n",
            "print(f\"║  🌾 Số hạt bề mặt ly       : {len(whole_grains):<3} hạt nguyên / {len(cleaned_grains):<3} hạt tổng{'':<18} ║\")\n",
            "print(f\"║  📏 Thể tích khối lúa (V)  : {bulk_volume_mm3:<10,.1f} mm³{'':<33} ║\")\n",
            "print(f\"║  📐 Thể tích 1 hạt nguyên  : {np.mean(regression_features['Grain_Volume_mm3_Mean'].iloc[0]):<6.2f} mm³ {'':<36} ║\")\n",
            "print(\"╠\" + \"═\" * 78 + \"╣\")\n",
            "print(f\"║  DỰ ĐOÁN VẬT LÝ (Cell 9): {estimated_seed_count:<5} HẠT (V_bulk * {PACKING_FRACTION} / V_grain_IQR){'':<6} ║\")\n",
            "print(f\"║  DỰ ĐOÁN HỒI QUY (AI)   : \\033[1;32m{regression_result['final_count']:<5}\\033[0m HẠT {'':<39} ║\")\n",
            "print(\"╠\" + \"═\" * 78 + \"╣\")\n",
            "print(f\"║  Thư mục Model: {REGRESSION_MODEL_DIR:<62} ║\")\n",
            "print(f\"║  Mô hình      : {regression_bundle['model'].__class__.__name__:<20} | Scaler: {regression_bundle['scaler'].__class__.__name__ if regression_bundle['scaler'] else 'None':<24} ║\")\n",
            "print(f\"║  Raw Predict  : {regression_result['raw_count']:<20.4f} | Training CV MAE: {cv_mae:<15} ║\")\n",
            "print(\"╚\" + \"═\" * 78 + \"╝\\n\")\n",
            "\n",
            "print(\"⚠️ Chưa tính đóng góp từng feature (Bỏ biểu đồ contributions).\")\n",
            "\n",
            "# 6. CẬP NHẬT KẾT QUẢ DỰ ĐOÁN HỒI QUY VÀO FILE BÁO CÁO TỔNG HỢP\n",
            "final_prediction_csv = os.path.join(OUTPUT_REPORT_DIR, \"final_prediction_summary.csv\")\n",
            "prediction_summary_record = {\n",
            "    \"Image_Name\": os.path.basename(IMAGE_PATH),\n",
            "    \"Bulk_Rice_Volume_mm3\": round(float(bulk_volume_mm3), 2),\n",
            "    \"Mean_Grain_Volume_mm3\": round(float(mean_whole_grain_vol), 2),\n",
            "    \"Whole_Grains_Count\": len(whole_records),\n",
            "    \"Physical_Formula_Estimate\": estimated_seed_count,\n",
            "    \"Linear_Regression_Predicted_Seeds\": regression_result[\"final_count\"],\n",
            "    \"Regression_Predicted_Seeds\": regression_result[\"final_count\"],\n",
            "    \"Regression_Raw_Prediction\": regression_result[\"raw_count\"],\n",
            "    \"Regression_Model\": regression_bundle[\"model\"].__class__.__name__,\n",
            "    \"Regression_Bundle_Dir\": REGRESSION_MODEL_DIR,\n",
            "    \"Regression_Scaler\": regression_bundle[\"scaler\"].__class__.__name__ if regression_bundle[\"scaler\"] else \"None\",\n",
            "    \"Feature_Schema_Version\": regression_bundle[\"schema_version\"]\n",
            "}\n",
            "\n",
            "import pandas as pd\n",
            "df_new = pd.DataFrame([prediction_summary_record])\n",
            "if os.path.exists(final_prediction_csv):\n",
            "    print(f\"⚠️ Cảnh báo: File {final_prediction_csv} đã tồn tại (kết quả cũ), ghi đè kết quả ảnh hiện tại.\")\n",
            "df_new.to_csv(final_prediction_csv, index=False, encoding=\"utf-8-sig\")\n",
            "\n",
            "feature_csv = os.path.join(OUTPUT_REPORT_DIR, \"regression_input_features.csv\")\n",
            "regression_features.to_csv(feature_csv, index=False, encoding=\"utf-8-sig\")\n",
            "\n",
            "print(f\"💾 Đã lưu summary vào: {final_prediction_csv}\")\n",
            "print(f\"💾 Đã lưu 31 features vào: {feature_csv}\")\n"
        ]
    }
    
    # Replace Cell 10 with the 5 new cells
    del nb['cells'][target_idx]
    nb['cells'].insert(target_idx, cell_report)
    nb['cells'].insert(target_idx, cell_predict)
    nb['cells'].insert(target_idx, cell_features)
    nb['cells'].insert(target_idx, cell_loader)
    nb['cells'].insert(target_idx, cell_config)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated successfully.")
