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
├── app.py                           # FastAPI Inference Server (Tầng Backend AI chính)
├── API_Server.ipynb                 # Notebook khởi chạy Server trên Google Colab qua Ngrok
├── .env                             # File cấu hình (NGROK_AUTH_TOKEN, NGROK_DOMAIN, AI_SERVER_URL)
├── modules/                         # [LÕI THUẬT TOÁN TỰ CHỨA]
│   ├── __init__.py                  # Export các module sản phẩm
│   ├── container_detector.py        # Module 1: Đo miệng ly, tính tỷ lệ px/mm & thể tích ly
│   ├── grain_segmenter.py           # Module 2: Bóc tách polygon hạt lúa (SAHI + YOLO-seg)
│   ├── grain_crop_cleaner.py        # Làm sạch ảnh crop, bẻ cầu dính (Watershed + Morphology)
│   ├── grain_classifier.py          # Module 3: Phân loại hạt nguyên / khuyết tật (DenseNet121)
│   ├── ellipsoid_geometry.py        # Module 4: Khớp elip 3D & thể tích hạt (V = 4/3 π a b c)
│   ├── regression_engine.py         # Module 5: Hồi quy dự đoán số lượng hạt (Extra Trees / OLS)
│   └── uniformity_evaluator.py      # Module 6: Đánh giá độ đồng đều mẻ lúa (chuẩn IQR)
│
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
   - **Đường kính ly (cm)**: `1.78`
   - **Chiều cao ly (cm)**: `3.39`
   - **Mức hụt lúa (cm)**: `1.09`
   - **Độ dày thành ly (cm)**: `0.1`
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
3. Mở trình duyệt, truy cập [Google Colab](https://colab.research.google.com/) và mở file notebook: `AI_SERVICES/API_Server.ipynb` trên Drive.
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
  - `diam`: Đường kính trong của ly (cm).
  - `height`: Chiều cao toàn bộ thân ly (cm).
  - `empty`: Chiều cao khoảng trống từ miệng ly tới mặt lúa (cm).
  - `wall_thickness` (tùy chọn): Độ dày thành ly (cm, mặc định 0.1).
  - `weight_total` (tùy chọn): Tổng khối lượng mẫu (g).
  - `sample_count` (tùy chọn): Số hạt đem cân mẫu (hạt).
  - `sample_weight` (tùy chọn): Khối lượng số hạt mẫu (g).

- **Dữ liệu trả về (Response JSON)**:
  ```json
  {
    "status": "success",
    "estimation": {
      "final": 182,
      "ai_est": 180,
      "weight_est": 185,
      "hybrid": true,
      "median_grain_vol_px3": 1450.2
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
    }
  }
  ```
