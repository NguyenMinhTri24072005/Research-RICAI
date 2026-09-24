# 🌾 Rice Vision AI — AI Services & Production Deployment

Dịch vụ API ước lượng số lượng hạt lúa từ ảnh chụp. Đây là **package đóng gói sản phẩm hoàn chỉnh, độc lập và tự chứa (Self-Contained Package)**, sẵn sàng cho triển khai thực tế.

---

## 🏗️ 1. Kiến Trúc Sơ Đồ Luồng Dữ Liệu Của Hệ Thống (3 Tầng)

```text
                   [GIAO DIỆN NGƯỜI DÙNG]
               React 19 + Vite (Port 5173)
                            │
              (1) Gửi ảnh & thông số cốc lúa
                            ▼
               [API GATEWAY TRUNG GIAN]
               Node.js Express (Port 3000)
                            │
          ┌─────────────────┴─────────────────┐
          │ (Chế độ Local)                    │ (Chế độ Colab Runtime)
          ▼                                   ▼
 [AI SERVER LOCAL]                   [GOOGLE COLAB GPU RUNTIME]
Python FastAPI (Port 8000)           Tesla T4 GPU qua Ngrok Tunnel
• Nạp YOLOv8-seg (SAHI)              • Cắt lát SAHI siêu tốc (~1s)
• Nạp DenseNet121 Keras 3            • Phân loại CNN siêu tốc (~0.5s)
• Nạp Hồi quy 31 biến                • Trả kết quả về Node.js
```

### Cấu Trúc Thư Mục
```text
AI_SERVICES/
├── app.py                           # FastAPI compatibility entry point
├── src/
│   ├── rice_ai/                     # Backend theo các miền nghiệp vụ
│   └── notebooks/API_Server.ipynb    # Notebook Colab khởi chạy server qua Ngrok
├── artifacts/                       # Bundles CNN, YOLO và regression
├── .env                             # File cấu hình (NGROK_AUTH_TOKEN, NGROK_DOMAIN, AI_SERVER_URL)
├── capture_server/                  # [MÁY CHỦ BẮT HÌNH ẢNH CAMERA ĐIỆN THOẠI]
│   ├── app.py                       # FastAPI WebSocket Server cho Web Phone Camera
│   └── web/                         # Frontend giao diện chụp ảnh cho điện thoại
│
├── RICE_ESTIMATION_APPLICATION/     # [ỨNG DỤNG WEB FULLSTACK]
│   ├── frontend/                    # Giao diện người dùng (React 19 + Vite)
│   └── backend/                     # API Gateway trung gian (Node.js + Express)
│
└── archive_legacy/                  # [DI SẢN LƯU TRỮ]
```

---

## 🚀 2. Hướng Dẫn Vận Hành Toàn Diện

Tài liệu này hướng dẫn chi tiết **2 phương pháp vận hành hệ thống**:
1. **PHƯƠNG PHÁP 1: VẬN HÀNH THUẦN LOCAL (OFFLINE 100%)** — Toàn bộ Frontend, Backend Gateway và AI Server chạy trên máy tính cá nhân.
2. **PHƯƠNG PHÁP 2: VẬN HÀNH KẾT HỢP CLOUD GPU (GOOGLE COLAB RUNTIME)** — Web chạy trên máy bạn, còn việc tính toán nặng (YOLO + CNN) được chuyển lên GPU Tesla T4 miễn phí trên Google Colab.

### 💻 Phương Pháp 1: Chạy Thuần Local (Offline 100%)

> **Khi nào nên dùng?** Khi bạn muốn chạy offline không cần mạng, dữ liệu ảnh bảo mật tuyệt đối trên máy tính, hoặc máy có card đồ họa rời NVIDIA.

#### 📍 Bước 1: Khởi động AI Inference Server (Port 8000)
- Lần đầu tiên: Kích đúp vào file `setup_ai_service.bat` (Script tự động tạo môi trường ảo `.venv` và cài đặt thư viện).
- Các lần sau: Kích đúp vào file `run_ai_service.bat` (Hoặc mở PowerShell: `cd AI_SERVICES; .\run_ai_service.bat`)
- Màn hình terminal sẽ hiển thị `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`
- 👉 Giữ nguyên cửa sổ này để server tiếp tục chạy.

#### 📍 Bước 2: Khởi động Web Gateway Node.js (Port 3000)
- Mở cửa sổ Terminal (PowerShell hoặc Command Prompt) thứ 2:
  ```powershell
  cd "d:\GG_1\NGHIÊN CỨU KHOA HỌC\GROUP_MEMBERS\NGUYEN MINH TRI\MAIN_SOURCES\AI_SERVICES\RICE_ESTIMATION_APPLICATION\backend"
  npm install
  node server.js
  ```
- Terminal sẽ báo:
  ```text
  🚀 Backend đang chạy tại http://localhost:3000 | AI_SERVER_URL: http://localhost:8000
  ```

#### 📍 Bước 3: Khởi động Giao diện Web React (Port 5173)
- Mở cửa sổ Terminal thứ 3:
  ```powershell
  cd "d:\GG_1\NGHIÊN CỨU KHOA HỌC\GROUP_MEMBERS\NGUYEN MINH TRI\MAIN_SOURCES\AI_SERVICES\RICE_ESTIMATION_APPLICATION\frontend"
  npm install
  npm run dev
  ```
- Terminal sẽ báo:
  ```text
  VITE v8.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
  ```

#### 📍 Bước 4: Trải nghiệm trên trình duyệt
1. Mở trình duyệt truy cập: **`http://localhost:5173`**.
2. Nhập các thông số hình học của cốc lúa:
   - **Đường kính ly (mm)**: `17.8`
   - **Chiều cao ly (mm)**: `33.9`
   - **Mức hụt lúa (mm)**: `10.9`
   - **Độ dày thành ly (mm)**: `1.0`
3. Nhấp **Chọn ảnh mẫu lúa** và chọn một ảnh trong `DATASET_BUILDER/1_Raw_Images/M001/M001A.jpg`. (Hoặc sử dụng chế độ kết nối quét mã QR từ camera điện thoại)
4. Bấm **BẮT ĐẦU ƯỚC LƯỢNG** và chờ hệ thống phân tích (~8 - 14 giây trên CPU).

---

### ☁️ Phương Pháp 2: Chạy Kết Hợp Cloud GPU Runtime (Google Colab)

> **Khi nào nên dùng?** Khi bạn muốn tốc độ suy luận nhanh nhất (~1.5 – 3 giây/ảnh) mà không làm nóng CPU máy tính cá nhân nhờ tận dụng card đồ họa **Tesla T4 16GB VRAM** miễn phí của Google.

#### 📍 Bước 1: Khởi động GPU Server trên Google Colab
1. Mở file `.env` trong thư mục `AI_SERVICES` trên máy tính, điền thông tin của bạn vào:
   ```ini
   NGROK_AUTH_TOKEN=your_auth_token_here
   NGROK_DOMAIN=your-subdomain.ngrok-free.app   # (Nếu có static domain, để trống nếu không)
   ```
2. Đảm bảo file `.env` đã được đồng bộ lên thư mục `AI_SERVICES` trên Google Drive.
3. Mở trình duyệt, truy cập [Google Colab](https://colab.research.google.com/) và mở file notebook: `AI_SERVICES/src/notebooks/API_Server.ipynb` trên Drive.
4. Chọn menu **Runtime** $\to$ **Change runtime type** $\to$ Chọn **T4 GPU** $\to$ Bấm **Save**.
5. Chạy lần lượt các Cell:
   - **Cell 1**: Kết nối Google Drive & cài đặt thư viện cần thiết.
   - **Cell 2**: Tự động kiểm tra file mô hình và **đọc token từ file `.env`**.
   - **Cell 3**: Tự động nạp Auth Token, mở tunnel và khởi chạy FastAPI server.
6. Colab sẽ in ra đường link kết nối:
   ```text
   🚀 NGROK PUBLIC URL: https://your-subdomain.ngrok-free.app
   ```
   *(Link này cũng tự động được Colab lưu ngược vào file `.env` trên Google Drive. File `.env` đồng bộ về máy sẽ tự động cập nhật URL).*

#### 📍 Bước 2: Khởi động Web Gateway
Mở cửa sổ Terminal tại máy tính cá nhân của bạn:
```powershell
cd "d:\GG_1\NGHIÊN CỨU KHOA HỌC\GROUP_MEMBERS\NGUYEN MINH TRI\MAIN_SOURCES\AI_SERVICES\RICE_ESTIMATION_APPLICATION\backend"
node server.js
```
Terminal sẽ tự động đọc file `.env` và hiển thị:
```text
🚀 Backend đang chạy tại http://localhost:3000 | AI_SERVER_URL: https://your-subdomain.ngrok-free.app
```
*(Hệ thống sẽ tự động kết nối API với đường hầm Ngrok).*

#### 📍 Bước 3: Khởi động Giao diện Web React
Mở cửa sổ Terminal tiếp theo:
```powershell
cd "d:\GG_1\NGHIÊN CỨU KHOA HỌC\GROUP_MEMBERS\NGUYEN MINH TRI\MAIN_SOURCES\AI_SERVICES\RICE_ESTIMATION_APPLICATION\frontend"
npm run dev
```

#### 📍 Bước 4: Trải nghiệm siêu tốc với Cloud GPU
1. Mở trình duyệt tại **`http://localhost:5173`**.
2. Nhập thông số và tải ảnh lên (hoặc dùng chế độ kết nối quét mã QR từ camera điện thoại).
3. Bấm **BẮT ĐẦU ƯỚC LƯỢNG**.
4. Toàn bộ ảnh sẽ được gửi qua Ngrok lên GPU Google Colab, phân tích và trả về kết quả chỉ trong **~2 – 3 giây**!

---

## 🔍 3. Tổng Kết Bảng So Sánh Nhanh

| Tiêu chí | Chế độ 1: Thuần Local | Chế độ 2: Colab GPU Runtime |
| :--- | :--- | :--- |
| **Mạng Internet** | **Không cần (Offline 100%)** | Bắt buộc có Internet |
| **Thời gian khởi động** | **Kích đúp `run_ai_service.bat` là chạy ngay** | Mất ~2 phút chờ Colab nạp thư viện & cấp GPU |
| **Thời gian xử lý/ảnh** | ~8 – 14 giây (trên CPU Intel i3) | **~1.5 – 3 giây (trên GPU Tesla T4)** |
| **Độ bền vững** | Vĩnh viễn trên máy, không bao giờ hết hạn | Phiên Colab tự ngắt sau một thời gian không dùng |

---

## 📡 4. Đặc Tả Endpoint Chính: `POST /predict`

- **Content-Type**: `multipart/form-data`
- **Các trường đầu vào**:
  - `file`: File ảnh chụp cốc lúa (JPG/PNG).
  - `diam`: Đường kính trong của ly (mm).
  - `height`: Chiều cao toàn bộ thân ly (mm).
  - `empty`: Chiều cao khoảng trống từ miệng ly tới mặt lúa (mm).
  - `wall_thickness` (tùy chọn): Độ dày thành ly (mm, mặc định 1.0).
  - `weight_total` (tùy chọn): Tổng khối lượng mẫu (g).
  - `sample_count` (tùy chọn): Số hạt đem cân mẫu (hạt).
  - `sample_weight` (tùy chọn): Khối lượng số hạt mẫu (g).

- **Dữ liệu trả về (Response JSON)**:
  ```json
  {
    "status": "success",
    "estimation": {
      "final": 85,
      "regression_est": 85.2,
      "geometry_est": 282,
      "weight_est": null,
      "method_used": "regression_ExtraTrees",
      "feature_schema_version": "31v1",
      "model_bundle": "regression_ExtraTrees",
      "ai_est": 282
    },
    "metrics_summary": {
      "total_grains_detected": 42,
      "whole_grains_surface": 38,
      "uniformity_rate_pct": 94.7,
      "pixels_per_mm": 68.25,
      "bulk_rice_volume_mm3": 8450.0,
      "avg_length_mm": 6.82,
      "avg_width_mm": 2.15,
      "avg_thickness_mm": 1.83
    },
    "features_used": {
      "Container_Inner_Diameter_mm": 17.8,
      "Estimated_Total_Seeds_Hybrid": 167.0
    },
    "timings_ms": {
      "decode_ms": 15.2,
      "container_ms": 120.4,
      "sahi_ms": 1450.0,
      "features_ms": 45.1,
      "regression_ms": 2.5,
      "total_ms": 1820.0
    }
  }
  ```

---

## 🏛️ 5. Kiến Trúc Mô-đun Hóa Mới (`src/rice_ai`)

Hệ thống được tái cấu trúc thành thư viện Python hướng module sạch (`src/rice_ai`):

```text
src/rice_ai/
├── __init__.py
├── settings.py                 # Nguồn sự thật cấu hình (env -> fallback -> candidates)
├── contracts.py                # Data dataclasses & PipelineError
├── estimation/                 # Lớp logic ước lượng số lượng
│   ├── feature_schema.py       # Khế ước 31 đặc trưng (31v1), chuẩn hóa ddof=0, hybrid 0.62
│   ├── geometry.py             # Ước lượng hình học xếp chặt (0.82 packing fraction)
│   ├── weight.py               # Ước lượng mẫu theo khối lượng
│   ├── regression.py           # Suy luận mô hình hồi quy bất kỳ (scikit-learn / ensemble)
│   └── fusion.py               # Chiến lược ưu tiên tổng hợp (auto, regression, geometry, weight)
├── models/                     # Quản lý vòng đời và nạp mô hình
│   ├── regression_loader.py    # Nạp bundle thư mục (canonical pair hoặc legacy adapter)
│   └── vision_models.py        # Quản lý lazy load thread-safe cho YOLO SAHI & CNN
├── vision/                     # Xử lý ảnh và thị giác máy tính
│   ├── container_detector.py   # Nhận diện cốc & tính px/mm
│   ├── grain_segmenter.py      # Cắt lát SAHI & bóc tách hạt
│   ├── grain_crop_cleaner.py   # Làm sạch hạt, gỡ dính (Watershed/Morphology)
│   ├── grain_classifier.py     # Phân loại hạt nguyên/lỗi (DenseNet121 Keras 3)
│   ├── ellipsoid_geometry.py   # Khớp ellipsoid 3D tính thể tích
│   └── uniformity_evaluator.py # Tính toán phân phối kích thước hạt
├── pipeline/                   # Điều phối luồng xử lý
│   ├── image_io.py             # Giải mã ảnh & thư mục làm việc tạm
│   ├── container.py            # Bước 2: Đo đạc vật chứa
│   ├── grains.py               # Bước 3: Phát hiện, bóc tách và phân loại hạt
│   ├── features.py             # Bước 4: Trích xuất vector 31 đặc trưng
│   └── runner.py               # RicePipeline: Điều phối tuần tự từng bước
└── api/                        # Giao diện HTTP FastAPI
    ├── application.py          # Factory create_app() & Lifespan
    ├── routes.py               # /health, /api/status, /predict
    └── schemas.py              # Pydantic schemas cho request/response
```

---

## ⚙️ 6. Cấu Hình Đổi Mô Hình Không Cần Sửa Code

Bạn có thể thay đổi bất kỳ mô hình nào (YOLO, CNN hoặc Hồi quy) chỉ bằng cách cập nhật file `.env` và khởi động lại dịch vụ:

| Biến Môi Trường | Mô Tả | Ví Dụ Giá Trị |
| :--- | :--- | :--- |
| `YOLO_DEVICE` | Thiết bị tính toán cho YOLO | `auto` (cuda nếu có, ngược lại cpu)<br>`cpu`<br>`cuda`<br>`cuda:0` |
| `YOLO_MODEL_PATH` | Đường dẫn file trọng số YOLO (`.pt`) | `artifacts/yolo/best.pt`<br>`../YOLO_SEGMENTATION_TRAINING_WORKFLOW/runs/.../best.pt` |
| `CNN_MODEL_PATH` | Đường dẫn file trọng số CNN (`.keras`) | `artifacts/cnn/best_model.keras`<br>`../CNN_CLASSIFICATION_MODEL/runs/.../best_model.keras` |
| `REGRESSION_MODEL_DIR` | Thư mục chứa mô hình hồi quy | `artifacts/regression/production`<br>`../LINEAR_REGRESSION_MODEL/models/ard`<br>`../LINEAR_REGRESSION_MODEL/models` |
| `MAX_CONCURRENT_INFERENCES` | Số lượng request inference đồng thời tối đa | `1` (khuyến nghị cho GPU/CPU duy nhất) |

---

## 🚀 7. Vận Hành Trên Google Colab GPU (Tesla T4)

Notebook runtime chính thức của hệ thống được quản lý tại:
**`AI_SERVICES/src/notebooks/API_Server.ipynb`**

- Xem hướng dẫn chi tiết tại: [`src/notebooks/README.md`](src/notebooks/README.md).
- Gồm 8 cell có cấu trúc rõ ràng: Hướng dẫn → Mount & Root → Cài đặt thư viện (`requirements-colab.txt`) → Nạp Settings → Preflight kiểm tra model → Khởi chạy server & ngrok tunnel → Smoke test → Dừng/Restart an toàn.
- Đổi model chỉ cần sửa `.env` trên Drive và chạy lại từ Cell 4.

---

## 📦 8. Khế Ước Kho Mô Hình (`artifacts/`)

Thư mục `AI_SERVICES/artifacts/` đóng vai trò kho lưu trữ chuẩn hóa khi deploy độc lập:
- `artifacts/yolo/`: Chứa file `best.pt`.
- `artifacts/cnn/`: Chứa file `best_model.keras`.
- `artifacts/regression/`: Chứa các thư mục bundle hồi quy con:
  - **Canonical Bundle**: Gồm 3 tệp tách biệt:
    1. `model.joblib`: Model scikit-learn thuần (chỉ chứa `predict`, không bọc Pipeline).
    2. `scaler.joblib`: StandardScaler / RobustScaler / MinMaxScaler (hoặc `"preprocessing": "none"` trong config.json).
    3. `config.json`: Metadata chứa `features` (đúng 31 đặc trưng), `schema_version: "31v1"`.
  - **Legacy Adapter**: Tương thích tự động với thư mục cũ chứa `best_tree_ensemble_model.joblib` và `scaler_params.json`.

---

## 🛠️ 9. Kiểm Tra Tiền Trạm (Preflight Verification)

Trước khi khởi động server, bạn có thể kiểm tra tính toàn vẹn của bất kỳ bundle mô hình hồi quy nào:

```powershell
# Kiểm tra thư mục mặc định
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py

# Kiểm tra thư mục bundle cụ thể (ví dụ mô hình ARD Regression mới nhất)
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py --model-dir ../LINEAR_REGRESSION_MODEL/models/ard
```

---

## 🧪 10. Chạy Bộ Kiểm Thử Tự Động (Test Suites)

```powershell
# 1. Chạy toàn bộ unit test suite (75 tests, ~11 giây)
$env:PYTHONIOENCODING="utf-8"
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_settings AI_SERVICES.tests.test_regression_loader AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_estimators AI_SERVICES.tests.test_pipeline AI_SERVICES.tests.test_api_endpoints AI_SERVICES.tests.test_vision_models AI_SERVICES.tests.test_api_lifecycle AI_SERVICES.tests.test_colab_notebook -v

# 2. Chạy kiểm thử tích hợp đầy đủ với ảnh chụp thật M001A
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_real_pipeline -v
```

## Foreground Colab host and persisted results

The Colab notebook is `src/notebooks/API_Server.ipynb`. Run Cells 2–5 for mount, dependencies, settings, and preflight, then run Cell 6. Cell 6 intentionally remains running while Uvicorn and ngrok are available. Interrupt that cell to execute its `finally` cleanup and close both the server and tunnel; do not start a second background server from another cell.

`REGRESSION_MODEL_DIR` selects the regression bundle. The current research default is:

```text
../LINEAR_REGRESSION_MODEL/models/extra_trees
```

The selected folder must contain `model.joblib`, `scaler.joblib`, and `feature_schema.json`. The scaler is applied once before the estimator. `PACKING_FRACTION_GEOMETRY=0.55` is the physical cylinder estimate, while `PACKING_FRACTION_FEATURE_HYBRID=0.62` is the historical regression feature contract.

Every successful request is saved under `/content/pipeline_inference_results/<request_id>/` (or `RESULTS_ROOT` when overridden). The directory includes the input, container/crop artifacts, classification and filter records, regression features, timings, and `reports/artifact_manifest.json`. The API exposes the manifest and ZIP download endpoints. `/content` is ephemeral and is lost when the Colab runtime is reset, so download a ZIP when the result must be retained.