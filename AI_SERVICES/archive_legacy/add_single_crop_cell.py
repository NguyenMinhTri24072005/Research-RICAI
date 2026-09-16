import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def update_notebook():
    notebook_path = 'CODE/SAHI_CROP_GRAINS.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Cell Markdown giải thích
    cell_md = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# 🎯 BƯỚC BỔ SUNG: BÓC TÁCH HẠT LÚA CHO 1 ẢNH ĐƠN LẺ TÙY CHỌN (SINGLE IMAGE)\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            'Cell bên dưới cho phép bạn cắt hạt lúa cho **đúng 1 bức ảnh chỉ định** từ đường dẫn tự nhập (file trong Drive), tự động lưu kết quả vào thư mục chỉ định và hiển thị ảnh overlay + các hạt crop trực tiếp trên màn hình.\n'
        ]
    }

    # 2. Cell Code xử lý ảnh đơn lẻ
    cell_code = {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# 🖼️ CẤU HÌNH & BÓC TÁCH CHO 1 ẢNH ĐƠN LẺ DUY NHẤT\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# 1. Nhập đường dẫn ảnh cần cắt (Thay đổi theo ý bạn)\n',
            'CUSTOM_IMAGE_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/TEST_IMAGES/DETECTABLE_TEST_IMAGES/test_sample.jpg"\n',
            '\n',
            '# 2. Nhập đường dẫn thư mục muốn lưu các hạt crop của ảnh này\n',
            'CUSTOM_OUTPUT_DIR = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/DETECTED_OBJECTS/SINGLE_IMAGE_CROPPED"\n',
            '\n',
            '# 3. Tham số tùy biến\n',
            'CONF_THRESH   = 0.40   # Ngưỡng tin cậy\n',
            'SLICE_SIZE    = 512    # Kích thước cắt lát (512x512)\n',
            'OVERLAP       = 0.25   # Tỷ lệ chồng lấn lát cắt (25%)\n',
            '\n',
            'def crop_single_image(image_path, output_dir, conf=0.4, slice_sz=512, overlap=0.25):\n',
            '    if not os.path.exists(image_path):\n',
            '        print(f"❌ Không tìm thấy file ảnh tại: {image_path}")\n',
            '        return\n',
            '\n',
            '    os.makedirs(output_dir, exist_ok=True)\n',
            '    base_name = os.path.splitext(os.path.basename(image_path))[0]\n',
            '\n',
            '    # 1. Nạp model nếu chưa có trong bộ nhớ\n',
            '    global detection_model\n',
            '    if \'detection_model\' not in globals() or detection_model is None:\n',
            '        print("⏳ Đang nạp mô hình SAHI...")\n',
            '        detection_model = AutoDetectionModel.from_pretrained(\n',
            '            model_type="ultralytics",\n',
            '            model_path=MODEL_WEIGHTS_PATH,\n',
            '            confidence_threshold=conf,\n',
            '            device="cuda:0"\n',
            '        )\n',
            '\n',
            '    print(f"🚀 Đang phân đoạn và bóc tách ảnh: {os.path.basename(image_path)}...")\n',
            '\n',
            '    # 2. Cắt lát dự đoán bằng SAHI\n',
            '    result = get_sliced_prediction(\n',
            '        image_path,\n',
            '        detection_model,\n',
            '        slice_height=slice_sz,\n',
            '        slice_width=slice_sz,\n',
            '        overlap_height_ratio=overlap,\n',
            '        overlap_width_ratio=overlap,\n',
            '        postprocess_match_threshold=0.65\n',
            '    )\n',
            '\n',
            '    predictions = result.object_prediction_list\n',
            '    img_bgr = cv2.imread(image_path)\n',
            '    if img_bgr is None:\n',
            '        print(f"❌ Không thể đọc nội dung ảnh: {image_path}")\n',
            '        return\n',
            '\n',
            '    global_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)\n',
            '    saved_crops = []\n',
            '\n',
            '    # 3. Duyệt và bóc tách từng hạt lúa\n',
            '    for idx, obj in enumerate(predictions):\n',
            '        if obj.mask is None or not obj.mask.segmentation:\n',
            '            continue\n',
            '\n',
            '        local_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)\n',
            '        for polygon in obj.mask.segmentation:\n',
            '            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))\n',
            '            cv2.fillPoly(local_mask, [pts], 255)\n',
            '            cv2.fillPoly(global_mask, [pts], 255)\n',
            '\n',
            '        ys, xs = np.where(local_mask == 255)\n',
            '        if len(ys) > 0 and len(xs) > 0:\n',
            '            y1, y2 = ys.min(), ys.max() + 1\n',
            '            x1, x2 = xs.min(), xs.max() + 1\n',
            '\n',
            '            crop_img  = img_bgr[y1:y2, x1:x2]\n',
            '            crop_mask = local_mask[y1:y2, x1:x2]\n',
            '\n',
            '            # Chuyển sang định dạng RGBA 4 kênh (Nền trong suốt)\n',
            '            rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)\n',
            '            rgba[:, :, 3] = crop_mask\n',
            '\n',
            '            crop_path = os.path.join(output_dir, f"{base_name}_grain_{idx:04d}.png")\n',
            '            cv2.imwrite(crop_path, rgba)\n',
            '            saved_crops.append(rgba)\n',
            '\n',
            '    # 4. Lưu ảnh overlay trực quan tổng thể\n',
            '    color_overlay = np.zeros_like(img_bgr, dtype=np.uint8)\n',
            '    color_overlay[global_mask == 255] = [0, 255, 0]\n',
            '    img_overlay = cv2.addWeighted(color_overlay, 0.4, img_bgr, 1.0, 0)\n',
            '    overlay_path = os.path.join(output_dir, f"{base_name}_overlay_result.jpg")\n',
            '    cv2.imwrite(overlay_path, img_overlay)\n',
            '\n',
            '    print("=" * 60)\n',
            '    print(f"✅ ĐÃ BÓC TÁCH XONG ẢNH: {os.path.basename(image_path)}")\n',
            '    print(f"🌾 Tổng số hạt lúa tìm thấy: {len(saved_crops)} hạt")\n',
            '    print(f"📂 Thư mục lưu hạt crop    : {output_dir}")\n',
            '    print(f"🖼️ Ảnh overlay trực quan   : {overlay_path}")\n',
            '    print("=" * 60)\n',
            '\n',
            '    # 5. Hiển thị kết quả trực tiếp trong Colab\n',
            '    plt.figure(figsize=(10, 6))\n',
            '    plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))\n',
            '    plt.title(f"Kết quả SAHI: Tìm thấy {len(saved_crops)} hạt lúa", fontsize=13, fontweight=\'bold\')\n',
            '    plt.axis(\'off\')\n',
            '    plt.show()\n',
            '\n',
            '    # Hiển thị 8 hạt mẫu đầu tiên\n',
            '    if len(saved_crops) > 0:\n',
            '        num_show = min(8, len(saved_crops))\n',
            '        fig, axes = plt.subplots(2, 4, figsize=(14, 6))\n',
            '        axes = axes.flatten()\n',
            '        for i in range(num_show):\n',
            '            axes[i].imshow(cv2.cvtColor(saved_crops[i], cv2.COLOR_BGRA2RGBA))\n',
            '            axes[i].set_title(f"Hạt {i+1}", fontsize=10)\n',
            '            axes[i].axis(\'off\')\n',
            '        for j in range(num_show, len(axes)):\n',
            '            axes[j].axis(\'off\')\n',
            '        plt.suptitle("MẪU CÁC HẠT LÚA ĐÃ BÓC TÁCH (NỀN TRONG SUỐT)", fontsize=12, fontweight=\'bold\')\n',
            '        plt.tight_layout()\n',
            '        plt.show()\n',
            '\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            '# GỌI HÀM BÓC TÁCH CHO ẢNH ĐƠN LẺ\n',
            '# ─────────────────────────────────────────────────────────────────────────────\n',
            'crop_single_image(\n',
            '    image_path=CUSTOM_IMAGE_PATH,\n',
            '    output_dir=CUSTOM_OUTPUT_DIR,\n',
            '    conf=CONF_THRESH,\n',
            '    slice_sz=SLICE_SIZE,\n',
            '    overlap=OVERLAP\n',
            ')\n'
        ]
    }

    # Thêm vào cuối notebook
    nb['cells'].append(cell_md)
    nb['cells'].append(cell_code)

    for save_path in ['CODE/SAHI_CROP_GRAINS.ipynb', 'AI_SERVICES/pipeline_logic/SAHI_CROP_GRAINS.ipynb']:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã thêm cell thành công vào: {save_path}")

if __name__ == '__main__':
    update_notebook()
