import json
import os
import re

nb_path = 'CODE/RICE_VISION_MAIN_PIPELINE.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 3 to include ENABLE_SIZE_FILTER
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Bước 3:' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        if 'ENABLE_SIZE_FILTER' not in src:
            addition = """
# 6. CẤU HÌNH BỘ LỌC KÍCH THƯỚC (OUTLIER BÉ)
ENABLE_SIZE_FILTER = True
SIZE_FILTER_K = 1.5
SIZE_FILTER_MIN_SAMPLES = 8
"""
            src += addition
            cell['source'] = [line + '\n' for line in src.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip('\n')

# Find step 8 index
step8_idx = -1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Bước 8:' in src or 'TẦNG 5' in src:
        step8_idx = i
        break

if step8_idx != -1:
    print(f"Replacing Cell 8 at index {step8_idx}")
    
    cell_8 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "GRAIN_MEASUREMENTS"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# BƯỚC 8: TẦNG 5 — GRAIN_MEASUREMENTS (ĐO ĐẠC HÌNH HỌC)\n",
            "# ==============================================================================\n",
            "if 'regression_bundle' in locals(): del regression_bundle\n",
            "if 'regression_features' in locals(): del regression_features\n",
            "if 'regression_result' in locals(): del regression_result\n",
            "if 'filtered_whole_grains' in locals(): del filtered_whole_grains\n",
            "if 'filtered_whole_records' in locals(): del filtered_whole_records\n",
            "\n",
            "whole_grain_metrics_list = []\n",
            "whole_records = []\n",
            "whole_volumes_list = []\n",
            "raw_valid_pairs = []\n",
            "\n",
            "grains_to_measure_3d = whole_grains if len(whole_grains) > 0 else []\n",
            "\n",
            "for idx, g in enumerate(grains_to_measure_3d):\n",
            "    try:\n",
            "        m = compute_single_grain_metrics(g[\"mask_binary\"], pixels_per_mm)\n",
            "        whole_grain_metrics_list.append(m)\n",
            "        whole_volumes_list.append(m[\"volume_mm3\"])\n",
            "        grain_idx = g.get(\"grain_id\", g.get(\"index\", idx))\n",
            "        rec = {\n",
            "            \"candidate_index\": idx,\n",
            "            \"grain_id\": grain_idx,\n",
            "            \"label\": \"hat_nguyen\",\n",
            "            \"confidence\": float(g.get(\"confidence\", 1.0)),\n",
            "            \"length_mm_2a\": round(float(m[\"length_mm\"]), 3),\n",
            "            \"width_mm_2b\": round(float(m[\"width_mm\"]), 3),\n",
            "            \"thickness_mm_2c\": round(float(m[\"thickness_mm\"]), 3),\n",
            "            \"area_mm2\": round(float(m[\"area_mm2\"]), 3),\n",
            "            \"volume_3d_mm3\": round(float(m[\"volume_mm3\"]), 3)\n",
            "        }\n",
            "        whole_records.append(rec)\n",
            "        raw_valid_pairs.append({\"grain\": g, \"record\": rec, \"metrics\": m})\n",
            "    except Exception as e:\n",
            "        print(f\"Lỗi đo đạc hạt {idx}: {type(e).__name__} - {e}\")\n",
            "\n",
            "if not whole_records:\n",
            "    print(\"⚠️ Không có hạt lúa nguyên nào hợp lệ.\")\n"
        ]
    }
    
    cell_8_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "GRAIN_SIZE_FILTER"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# BƯỚC 8.1: TẦNG 5.1 — GRAIN_SIZE_FILTER (BỘ LỌC KÍCH THƯỚC HẠT)\n",
            "# ==============================================================================\n",
            "import importlib\n",
            "try:\n",
            "    import modules.grain_size_filter\n",
            "    importlib.reload(modules.grain_size_filter)\n",
            "    from modules.grain_size_filter import filter_grains_by_size\n",
            "except ImportError:\n",
            "    from modules.grain_size_filter import filter_grains_by_size\n",
            "import pandas as pd\n",
            "import os\n",
            "\n",
            "ENABLE_SIZE_FILTER = ENABLE_SIZE_FILTER if 'ENABLE_SIZE_FILTER' in locals() else True\n",
            "SIZE_FILTER_K = SIZE_FILTER_K if 'SIZE_FILTER_K' in locals() else 1.5\n",
            "SIZE_FILTER_MIN_SAMPLES = SIZE_FILTER_MIN_SAMPLES if 'SIZE_FILTER_MIN_SAMPLES' in locals() else 8\n",
            "\n",
            "print(f\"🔄 Đang lọc hạt outlier diện tích nhỏ... (Enabled={ENABLE_SIZE_FILTER})\")\n",
            "filter_result = filter_grains_by_size(\n",
            "    measurements=whole_records, \n",
            "    k=SIZE_FILTER_K, \n",
            "    min_samples=SIZE_FILTER_MIN_SAMPLES,\n",
            "    enabled=ENABLE_SIZE_FILTER\n",
            ")\n",
            "\n",
            "filter_stats = filter_result[\"stats\"]\n",
            "kept_records = filter_result[\"kept\"]\n",
            "rejected_records = filter_result[\"rejected\"]\n",
            "\n",
            "filtered_whole_records = []\n",
            "filtered_whole_grains = []\n",
            "filtered_whole_grain_metrics_list = []\n",
            "filtered_whole_volumes_list = []\n",
            "filtered_pairs = []\n",
            "\n",
            "kept_indices = {rec[\"candidate_index\"] for rec in kept_records}\n",
            "for pair in raw_valid_pairs:\n",
            "    if pair[\"record\"][\"candidate_index\"] in kept_indices:\n",
            "        filtered_whole_records.append(pair[\"record\"])\n",
            "        filtered_whole_grains.append(pair[\"grain\"])\n",
            "        filtered_whole_grain_metrics_list.append(pair[\"metrics\"])\n",
            "        filtered_whole_volumes_list.append(pair[\"record\"][\"volume_3d_mm3\"])\n",
            "        filtered_pairs.append(pair)\n",
            "\n",
            "print(f\"📊 Trạng thái lọc: {filter_stats['status']}\")\n",
            "print(f\"📊 Kết quả Lọc: Giữ lại {filter_stats['count_after']}/{filter_stats['count_before']} hạt.\")\n",
            "if filter_stats.get('lower_bound') is not None:\n",
            "    print(f\"   (Area Lower Bound: {filter_stats['lower_bound']:.2f} mm2)\")\n",
            "\n",
            "df_decisions = pd.DataFrame(kept_records + rejected_records)\n",
            "if not df_decisions.empty:\n",
            "    csv_decision_path = os.path.join(OUTPUT_REPORT_DIR, \"grain_size_filter_decisions.csv\")\n",
            "    df_decisions.to_csv(csv_decision_path, index=False, encoding=\"utf-8-sig\")\n",
            "    \n",
            "if rejected_records:\n",
            "    print(\"\\n📋 [BẢNG HẠT BỊ LOẠI ĐẦU TIÊN]:\")\n",
            "    display(pd.DataFrame(rejected_records).head(5))\n",
            "    \n",
            "    # Gallery hạt bị loại\n",
            "    import matplotlib.pyplot as plt\n",
            "    num_show = min(12, len(rejected_records))\n",
            "    if num_show > 0:\n",
            "        plt.figure(figsize=(16, 4.5))\n",
            "        for i in range(num_show):\n",
            "            idx = rejected_records[i][\"candidate_index\"]\n",
            "            pair = raw_valid_pairs[idx]\n",
            "            plt.subplot(2, 6, i + 1)\n",
            "            vis_g = draw_grain_ellipse_overlay(pair[\"grain\"][\"crop_rgba\"], pair[\"metrics\"])\n",
            "            plt.imshow(vis_g)\n",
            "            plt.title(f\"ID: {rejected_records[i]['grain_id']}\\nArea: {rejected_records[i]['area_mm2']} mm2\", fontsize=8)\n",
            "            plt.axis('off')\n",
            "        plt.suptitle(\"Gallery các hạt bị loại do kích thước nhỏ\")\n",
            "        plt.show()\n"
        ]
    }
    
    cell_8_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "GRAIN_PHYSICAL_STATISTICS"},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# BƯỚC 8.2: TẦNG 5.2 — GRAIN_PHYSICAL_STATISTICS (TỔNG HỢP VÀ DỰ ĐOÁN VẬT LÝ)\n",
            "# ==============================================================================\n",
            "from modules.uniformity_evaluator import evaluate_batch_uniformity\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "if not filtered_whole_volumes_list:\n",
            "    estimated_seed_count = None\n",
            "    mean_whole_grain_vol = None\n",
            "    uniformity_res = {\"uniformity_rate_pct\": 0.0, \"is_uniform\": False, \"mean_clean\": 0.0}\n",
            "    print(\"⚠️ Không đủ dữ liệu hợp lệ sau lọc. Ước lượng vật lý = None.\")\n",
            "else:\n",
            "    uniformity_res_raw = evaluate_batch_uniformity(whole_volumes_list, threshold_rate=0.80)\n",
            "    uniformity_res = evaluate_batch_uniformity(filtered_whole_volumes_list, threshold_rate=0.80)\n",
            "    mean_whole_grain_vol = uniformity_res[\"mean_clean\"]\n",
            "    \n",
            "    if mean_whole_grain_vol > 0 and bulk_volume_mm3 > 0:\n",
            "        estimated_seed_count = int(round((bulk_volume_mm3 * PACKING_FRACTION) / mean_whole_grain_vol))\n",
            "    else:\n",
            "        estimated_seed_count = None\n",
            "\n",
            "    print(\"=\" * 75)\n",
            "    print(\"  TẦNG 5.2: KẾT QUẢ ĐO KÍCH THƯỚC 3D & ĐỘ ĐỒNG ĐỀU (SAU KHI LỌC)\")\n",
            "    print(\"=\" * 75)\n",
            "    lengths = [r[\"length_mm_2a\"] for r in filtered_whole_records]\n",
            "    widths  = [r[\"width_mm_2b\"] for r in filtered_whole_records]\n",
            "    thick   = [r[\"thickness_mm_2c\"] for r in filtered_whole_records]\n",
            "    areas   = [r[\"area_mm2\"] for r in filtered_whole_records]\n",
            "    vols    = [r[\"volume_3d_mm3\"] for r in filtered_whole_records]\n",
            "\n",
            "    print(f\"  • Số hạt sau bộ lọc               : {len(filtered_whole_records)} hạt (từ {len(whole_records)} hạt đo đạc hợp lệ)\")\n",
            "    print(f\"  • Thể tích 3D tb (V_tb raw)       : {uniformity_res_raw['mean_clean']:.2f} mm³\")\n",
            "    print(f\"  • Thể tích 3D tb (V_tb filtered)  : {mean_whole_grain_vol:.2f} mm³ (Loại hạt nhỏ + Lọc IQR)\")\n",
            "    print(f\"  • Độ đồng đều lô hạt sau lọc      : {uniformity_res['uniformity_rate_pct']}% -> {'✅ ĐẠT' if uniformity_res['is_uniform'] else '❌ KHÔNG ĐẠT'}\")\n",
            "    print(\"=\" * 75)\n",
            "\n",
            "    if len(lengths) >= 3:\n",
            "        fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "        axes[0, 0].hist(lengths, bins=15, color='#2196F3', edgecolor='black', alpha=0.7)\n",
            "        axes[0, 0].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2)\n",
            "        axes[0, 0].set_title(\"Chiều Dài (2a - mm)\")\n",
            "        \n",
            "        axes[0, 1].hist(widths, bins=15, color='#4CAF50', edgecolor='black', alpha=0.7)\n",
            "        axes[0, 1].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2)\n",
            "        axes[0, 1].set_title(\"Chiều Rộng (2b - mm)\")\n",
            "        \n",
            "        axes[1, 0].hist(areas, bins=15, color='#FF9800', edgecolor='black', alpha=0.7)\n",
            "        axes[1, 0].axvline(np.mean(areas), color='red', linestyle='--', linewidth=2)\n",
            "        axes[1, 0].set_title(\"Diện Tích (mm2)\")\n",
            "        \n",
            "        axes[1, 1].hist(vols, bins=15, color='#9C27B0', edgecolor='black', alpha=0.7)\n",
            "        axes[1, 1].axvline(mean_whole_grain_vol, color='red', linestyle='--', linewidth=2, label=f'Mean Clean={mean_whole_grain_vol:.2f}')\n",
            "        axes[1, 1].set_title(\"Thể Tích (mm3)\")\n",
            "        plt.legend()\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n"
        ]
    }
    
    # Delete old Cell 8 and insert 8, 8.1, 8.2
    del nb['cells'][step8_idx]
    nb['cells'].insert(step8_idx, cell_8_2)
    nb['cells'].insert(step8_idx, cell_8_1)
    nb['cells'].insert(step8_idx, cell_8)

# Update Cell 9 (BẢNG KẾT QUẢ)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'BẢNG KẾT QUẢ' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        # Change `whole_grain_metrics_list` to `filtered_whole_grain_metrics_list`
        src = src.replace("whole_grain_metrics_list", "filtered_whole_grain_metrics_list")
        # Change `estimated_seed_count:<6}` to check if it's None
        src = src.replace("estimated_seed_count:<6", "str(estimated_seed_count):<6")
        # Ensure it works when missing values
        src = src.replace("mean_whole_grain_vol:<6.2f", "str(mean_whole_grain_vol):<6")
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip('\n')

# Update REGRESSION_FEATURES (Cell 12 now shifted)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'REGRESSION_FEATURES' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        # Add support for `whole_records_input`
        src = src.replace("def assemble_regression_features(bundle):", "def assemble_regression_features(bundle, whole_records_input=None):\\n    records = whole_records_input if whole_records_input is not None else whole_records\\n    if not records:\\n        raise ValueError(\"Không tìm thấy hạt nào. Dừng suy luận regression.\")")
        src = src.replace("len(whole_grains) == 0", "len(records) == 0")
        src = src.replace("whole_records", "records")
        # Reset any output printing logic to support multiple calls
        src = src.replace("regression_features = assemble_regression_features(regression_bundle)", "regression_features = assemble_regression_features(regression_bundle)\\nif 'filtered_whole_records' in locals() and filtered_whole_records:\\n    filtered_regression_features = assemble_regression_features(regression_bundle, filtered_whole_records)\\nelse:\\n    filtered_regression_features = None")
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip('\n')

# Update REGRESSION_REPORT
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'REGRESSION_REPORT' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        src = src.replace("np.mean(regression_features['Grain_Volume_mm3_Mean'].iloc[0]):<6.2f", "regression_features['Grain_Volume_mm3_Mean'].iloc[0]:<6.2f")
        src = src.replace("estimated_seed_count:<5", "str(estimated_seed_count):<5")
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip('\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook modified for GRAIN_SIZE_FILTER plan.")
