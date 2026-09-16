import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def make_cell(cell_type, source):
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': [line + '\n' for line in source.strip().split('\n')]
    }

cells_cnn = [
    make_cell('markdown', '''# 🌾 MODULE 3: HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI CHẤT LƯỢNG HẠT LÚA (DENSENET121) - BẢN TINH CHỈNH TỐI ƯU
**Đề tài Nghiên cứu Khoa học**: Hệ thống thị giác máy tính và học sâu phục vụ ước lượng số lượng và đánh giá phẩm cấp hạt giống lúa.

### 🎯 Các cải tiến tinh chỉnh chuyên sâu (Fine-Tuning v2):
1. **Nâng cấp độ phân giải**: Chuẩn $224 \times 224$ của DenseNet121.
2. **Tăng cường dữ liệu (Data Augmentation)**: Xoay tự do $360^\circ$, lật ngang/dọc, chỉnh sáng nhẹ.
3. **Tự động cân bằng trọng số lớp (Class Weights)**: Tránh thiên vị giữa hạt nguyên và hạt khuyết tật.
4. **Mở rộng phạm vi Fine-tuning**: Mở khóa **60 layers cuối cùng** của DenseNet121 (thay vì chỉ 30 layers) để học sâu hơn các đặc trưng vi mô của hạt lúa.
5. **Tối ưu tốc độ học (Learning Rate)**: Bắt đầu Giai đoạn 2 với `lr = 5e-5` kết hợp `patience=5` cho `ReduceLROnPlateau` để không bị tụt tốc độ học quá sớm.
6. **Kiểm soát EarlyStopping chuẩn xác**: Tăng `patience=12` để mô hình có đủ thời gian bứt phá vượt ngưỡng $88\% - 92\%+$.
7. **Đánh giá & Xuất báo cáo khoa học**: Xuất ma trận nhầm lẫn (Confusion Matrix) và bảng F1-Score trên tập Test độc lập.'''),

    make_cell('code', '''# Bước 1: Kết nối Google Drive & Cài đặt thư viện
from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow scikit-learn matplotlib seaborn numpy'''),

    make_cell('code', '''# Bước 2: Import thư viện & Cấu hình Tham số
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ HUẤN LUYỆN
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = "/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/"

# Đường dẫn folder chứa các thư mục nhãn: hat_nguyen và hat_khuyet_tat
DATASET_DIR = os.path.join(BASE_PATH, "DETECTED_OBJECTS/DATA1/OUTPUT_CROPPED_GRAINS")

# Thư mục lưu kết quả model sau khi train
SAVE_MODEL_DIR = os.path.join(BASE_PATH, "RESULTS/CNN_DenseNet121_Trained")
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# Siêu tham số
IMG_SIZE   = (224, 224)  # Kích thước chuẩn của DenseNet121
BATCH_SIZE = 16
SEED       = 42

print("=" * 60)
print(f"📁 Thư mục dữ liệu     : {DATASET_DIR}")
print(f"💾 Thư mục lưu kết quả : {SAVE_MODEL_DIR}")
print(f"⚙️ Kích thước ảnh      : {IMG_SIZE} | Batch Size: {BATCH_SIZE}")
print("=" * 60)'''),

    make_cell('code', '''# Bước 3: Load Dữ liệu & Chia tập Train (70%) / Validation (15%) / Test (15%)
# 1. Load tập Train (70%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 2. Load 30% còn lại để tách làm Validation (15%) và Test (15%)
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_val_batches = tf.data.experimental.cardinality(val_test_ds) // 2

val_ds  = val_test_ds.take(num_val_batches)
test_ds = val_test_ds.skip(num_val_batches)

print("=" * 60)
print(f"🎯 Các lớp nhận diện: {class_names}")
print(f"📊 Số batches: Train={len(train_ds)} | Validation={len(val_ds)} | Test={len(test_ds)}")
print("=" * 60)

# Tự động tính Class Weights cân bằng tỷ lệ mẫu
train_labels = []
for _, labels in train_ds.unbatch():
    train_labels.append(np.argmax(labels.numpy()))
train_labels = np.array(train_labels)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(weights))
print(f"⚖️ Trọng số phạt cân bằng lớp (Class Weights): {class_weights_dict}")

# Tối ưu hóa pipeline nạp dữ liệu vào GPU
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)'''),

    make_cell('code', '''# Bước 4: Thiết kế Khối Augmentation & Kiến trúc Mô hình DenseNet121
# 1. Khối tăng cường dữ liệu hạt lúa
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5), # Xoay góc tự do
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2)
], name="data_augmentation")

# 2. Khởi tạo Backbone DenseNet121
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # Giai đoạn 1: Đóng băng backbone

# 3. Ghép nối thành Model hoàn chỉnh
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input, name="densenet_preprocess")(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()'''),

    make_cell('code', '''# Bước 5: GIAI ĐOẠN 1 — Huấn luyện Warmup (Classifier Head)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

print("🚀 Bắt đầu Giai đoạn 1: Huấn luyện Classifier Head (25 Epochs)...")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    class_weight=class_weights_dict,
    callbacks=callbacks_phase1
)'''),

    make_cell('code', '''# Bước 6: GIAI ĐOẠN 2 — Fine-Tuning 60 Layers Cuối Cùng của DenseNet121
# [CẢI TIẾN 1]: Mở khóa 60 layers cuối cùng để học sâu các đặc trưng vi mô của hạt lúa
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

# [CẢI TIẾN 2]: Bắt đầu với Learning Rate tối ưu 5e-5 (thay vì 1e-5 quá nhỏ)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

best_model_path = os.path.join(SAVE_MODEL_DIR, "best_rice_densenet121.keras")

# [CẢI TIẾN 3]: Tăng patience của ReduceLR lên 5 và EarlyStopping lên 12
callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
]

print("🚀 Bắt đầu Giai đoạn 2: Fine-Tuning 60 layers sâu (50 Epochs)...")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=callbacks_phase2
)

# Lưu thêm 1 bản định dạng .h5 để tương thích ngược
legacy_model_path = os.path.join(SAVE_MODEL_DIR, "rice_grain_classifier_cnn.h5")
model.save(legacy_model_path)
print(f"🎉 Đã lưu model tối ưu tại: {best_model_path}")
print(f"📦 Đã lưu model tương thích ngược tại: {legacy_model_path}")'''),

    make_cell('code', '''# Bước 7: Vẽ Biểu đồ Loss & Accuracy qua 2 Giai đoạn Huấn luyện
acc  = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy', color='#2196F3', lw=2)
plt.plot(val_acc, label='Validation Accuracy', color='#4CAF50', lw=2)
plt.axvline(x=len(history_phase1.history['accuracy'])-1, color='red', linestyle='--', label='Bắt đầu Fine-Tuning')
plt.title('Độ chính xác qua các Epochs (Accuracy)', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss', color='#FF9800', lw=2)
plt.plot(val_loss, label='Validation Loss', color='#F44336', lw=2)
plt.axvline(x=len(history_phase1.history['loss'])-1, color='red', linestyle='--', label='Bắt đầu Fine-Tuning')
plt.title('Độ hao phí qua các Epochs (Loss)', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
chart_save_path = os.path.join(SAVE_MODEL_DIR, "training_history.png")
plt.savefig(chart_save_path, dpi=300)
print(f"📊 Đã lưu biểu đồ huấn luyện tại: {chart_save_path}")
plt.show()'''),

    make_cell('code', '''# Bước 8: Đánh giá trên Tập Test Độc lập & Vẽ Ma trận Nhầm lẫn (Confusion Matrix)
print("🔍 Đang đánh giá mô hình trên tập Test độc lập...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 1. In Báo cáo phân loại chi tiết (Precision, Recall, F1-Score)
print("\n" + "=" * 60)
print("  BÁO CÁO PHÂN LOẠI CHI TIẾT TRÊN TẬP TEST (CLASSIFICATION REPORT)")
print("=" * 60)
report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report_text)

# Lưu báo cáo vào file text
with open(os.path.join(SAVE_MODEL_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# 2. Vẽ Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix)
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Ma trận nhầm lẫn trên tập Test (Confusion Matrix)", fontweight='bold')
plt.xlabel("Nhãn dự đoán (Predicted)")
plt.ylabel("Nhãn thực tế (Ground Truth)")
plt.tight_layout()

cm_save_path = os.path.join(SAVE_MODEL_DIR, "test_confusion_matrix.png")
plt.savefig(cm_save_path, dpi=300)
print(f"🖼️ Đã lưu ma trận nhầm lẫn tại: {cm_save_path}")
plt.show()''')
]

nb = {
    'nbformat': 4,
    'nbformat_minor': 2,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells_cnn
}

for path in ['CODE/CNN.ipynb', 'AI_SERVICES/pipeline_logic/CNN.ipynb']:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã cập nhật thành công: {path}")
