import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def add_prediction_cell():
    notebook_path = 'CODE/CNN.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Cell Markdown
    cell_md = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# 🎯 BƯỚC BỔ SUNG: DỰ ĐOÁN PHÂN LOẠI CHO ẢNH TÙY CHỌN (INFERENCE TEST)\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            'Sử dụng cell code bên dưới để upload ảnh lên `/content` hoặc chỉ định đường dẫn 1 ảnh hạt lúa bất kỳ để xem mô hình dự đoán nhãn (`hat_nguyen` hay `hat_khuyet_tat`) kèm biểu đồ xác suất phần trăm trực quan.\n'
        ]
    }

    # 2. Cell Code
    cell_code = {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# 🖼️ DỰ ĐOÁN NHÃN CHO 1 ẢNH HẠT LÚA BẤT KỲ\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            'import cv2\n',
            'import numpy as np\n',
            'import matplotlib.pyplot as plt\n',
            'from google.colab import files\n',
            '\n',
            '# [TÙY CHỌN 1]: Bật True nếu muốn hiện hộp thoại tải ảnh trực tiếp từ máy tính\n',
            'UPLOAD_FROM_COMPUTER = False\n',
            '\n',
            '# [TÙY CHỌN 2]: Hoặc nhập đường dẫn ảnh đã có sẵn trên Colab / Google Drive\n',
            'CUSTOM_TEST_IMAGE_PATH = "/content/test_grain.png"\n',
            '\n',
            'def predict_single_grain(image_path, trained_model, labels):\n',
            '    if not os.path.exists(image_path):\n',
            '        print(f"❌ Không tìm thấy file ảnh tại: {image_path}")\n',
            '        print("💡 Gợi ý: Hãy upload ảnh lên Colab hoặc sửa lại biến CUSTOM_TEST_IMAGE_PATH.")\n',
            '        return\n',
            '\n',
            '    # 1. Đọc ảnh (hỗ trợ cả ảnh PNG trong suốt RGBA 4 kênh và ảnh màu 3 kênh)\n',
            '    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)\n',
            '    if img_raw is None:\n',
            '        print(f"❌ Không thể đọc file ảnh: {image_path}")\n',
            '        return\n',
            '\n',
            '    # 2. Xử lý kênh Alpha (nếu ảnh PNG nền trong suốt từ SAHI)\n',
            '    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:\n',
            '        crop_bgr = img_raw[:, :, :3].copy()\n',
            '        alpha = img_raw[:, :, 3]\n',
            '        crop_bgr[alpha == 0] = [0, 0, 0]  # Đổi nền trong suốt thành màu đen\n',
            '    else:\n',
            '        crop_bgr = img_raw.copy()\n',
            '\n',
            '    # 3. Tiền xử lý theo chuẩn kích thước mô hình (224x224)\n',
            '    resized = cv2.resize(crop_bgr, (IMG_SIZE[0], IMG_SIZE[1]))\n',
            '    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)\n',
            '    input_tensor = np.expand_dims(rgb_img, axis=0)\n',
            '\n',
            '    # 4. Dự đoán xác suất từ model\n',
            '    preds = trained_model.predict(input_tensor, verbose=0)[0]\n',
            '    pred_idx = int(np.argmax(preds))\n',
            '    pred_label = labels[pred_idx]\n',
            '    confidence = preds[pred_idx] * 100\n',
            '\n',
            '    # 5. Hiển thị kết quả trực quan\n',
            '    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n',
            '\n',
            '    display_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)\n',
            '    axes[0].imshow(display_rgb)\n',
            '    label_color = "#2E7D32" if pred_label == "hat_nguyen" else "#C62828"\n',
            '    axes[0].set_title(f"Dự đoán: {pred_label.upper()}\\nĐộ tự tin: {confidence:.2f}%", \n',
            '                      fontsize=13, fontweight="bold", color=label_color)\n',
            '    axes[0].axis("off")\n',
            '\n',
            '    y_pos = np.arange(len(labels))\n',
            '    bar_colors = ["#4CAF50" if c == "hat_nguyen" else "#F44336" for c in labels]\n',
            '    axes[1].barh(y_pos, preds * 100, color=bar_colors, alpha=0.85, edgecolor="black")\n',
            '    axes[1].set_yticks(y_pos)\n',
            '    axes[1].set_yticklabels([c.upper() for c in labels], fontsize=11, fontweight="bold")\n',
            '    axes[1].set_xlim(0, 100)\n',
            '    axes[1].set_xlabel("Xác suất (%)", fontsize=11)\n',
            '    axes[1].set_title("Phân phối xác suất 2 lớp", fontsize=12, fontweight="bold")\n',
            '    axes[1].grid(axis="x", linestyle="--", alpha=0.5)\n',
            '\n',
            '    for i, v in enumerate(preds * 100):\n',
            '        axes[1].text(v + 1.5, i, f"{v:.2f}%", va="center", fontweight="bold")\n',
            '\n',
            '    plt.tight_layout()\n',
            '    plt.show()\n',
            '\n',
            '    print("=" * 60)\n',
            '    print(f"🌾 KẾT QUẢ DỰ ĐOÁN : {pred_label.upper()}")\n',
            '    print(f"🎯 Độ tự tin       : {confidence:.2f}%")\n',
            '    for c, p in zip(labels, preds):\n',
            '        print(f"   - {c:15s}: {p*100:.2f}%")\n',
            '    print("=" * 60)\n',
            '\n',
            '# Thực thi\n',
            'if UPLOAD_FROM_COMPUTER:\n',
            '    print("📤 Hãy chọn 1 file ảnh hạt lúa từ máy tính để tải lên...")\n',
            '    uploaded = files.upload()\n',
            '    for fname in uploaded.keys():\n',
            '        print(f"\\n🚀 Đang phân tích ảnh: {fname}")\n',
            '        predict_single_grain(fname, model, class_names)\n',
            'else:\n',
            '    predict_single_grain(CUSTOM_TEST_IMAGE_PATH, model, class_names)\n'
        ]
    }

    nb['cells'].append(cell_md)
    nb['cells'].append(cell_code)

    for save_path in ['CODE/CNN.ipynb', 'AI_SERVICES/pipeline_logic/CNN.ipynb']:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã thêm cell dự đoán thành công vào: {save_path}")

if __name__ == '__main__':
    add_prediction_cell()
