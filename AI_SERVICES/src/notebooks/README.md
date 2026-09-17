# 🌾 Rice Vision AI — Google Colab Runtime Guide

Thư mục này chứa notebook runtime chính thức và duy nhất của AI Services: API_Server.ipynb.

---

## 1. Yêu cầu & Cấu hình môi trường Google Colab

1. **Chọn phần cứng GPU**:
   - Menu Colab: **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** (Python 3).
2. **Mount Google Drive**:
   - Đảm bảo thư mục mã nguồn MAIN_SOURCES đã được đồng bộ lên Google Drive của bạn.
3. **Cấu hình Token & Model trong AI_SERVICES/.env**:
   - File cấu hình nằm tại: .../MAIN_SOURCES/AI_SERVICES/.env
   - Khai báo token:
     `ash
     NGROK_AUTH_TOKEN=your_token_here
     NGROK_DOMAIN=your-static-domain.ngrok-free.dev   # (Tùy chọn)
     PORT_AI=8000
     YOLO_DEVICE=auto                                 # Tự động nhận diện GPU cuda:0
     `
   - Cấu hình model paths (tính từ AI_SERVICES/ hoặc đường dẫn tuyệt đối):
     `ash
     REGRESSION_MODEL_DIR=artifacts/regression/ard    # Thư mục chứa model hồi quy
     YOLO_MODEL_PATH=artifacts/yolo/best.pt           # File trọng số YOLO
     CNN_MODEL_PATH=artifacts/cnn/best_v3_step2.keras # File trọng số DenseNet
     `

---

## 2. Trình tự thực thi các Cell trong Notebook

Notebook được tổ chức thành 8 cell tuần tự, tách biệt rõ ràng giữa cấu hình, khởi động và dọn dẹp:

| Thứ tự | Tên Cell | Mục đích |
|---|---|---|
| **Cell 1** | **Hướng dẫn vận hành** (Markdown) | Hướng dẫn tổng quát và quy tắc vận hành. |
| **Cell 2** | **Mount & PROJECT_ROOT** (Code) | Mount Google Drive, khai báo duy nhất PROJECT_ROOT bằng pathlib.Path. Kiểm tra tồn tại thư mục mã nguồn src/rice_ai. |
| **Cell 3** | **Dependencies & Environment** (Code) | Cài đặt gói từ equirements-colab.txt. Kiểm tra Python, sklearn, torch, CUDA và GPU device thực tế. |
| **Cell 4** | **Settings & Configuration** (Code) | Nạp Settings từ AI_SERVICES/.env qua src.rice_ai.settings. In ra đường dẫn mô hình đã resolve (không in token bí mật). |
| **Cell 5** | **Preflight Verification** (Code) | Nạp trước và kiểm tra toàn vẹn trọng số YOLO, CNN và bundle Regression. Phát hiện lỗi weights/device TRƯỚC KHI mở server. |
| **Cell 6** | **Start Server & ngrok Tunnel** (Code) | Khởi chạy FastAPI server qua Uvicorn asyncio task. Kiểm tra local /health và /api/status. Sau khi OK mới mở ngrok tunnel và cập nhật AI_SERVER_URL vào .env. |
| **Cell 7** | **Smoke Test / E2E (Tùy chọn)** (Code) | Gửi request kiểm tra /health, /api/status qua public URL. Cờ RUN_E2E = False mặc định để tránh tốn tài nguyên trừ khi cần test toàn diện. |
| **Cell 8** | **Stop & Restart Server** (Code) | Đóng ngrok tunnel, phát tín hiệu dừng Uvicorn server và dọn dẹp tài nguyên. An toàn khi chạy nhiều lần. |

---

## 3. Quy trình thay đổi Model hoặc Source Code

### A. Khi thay đổi Model (Chỉ đổi đường dẫn):
1. Chạy **Cell 8 (Stop & Restart Server)** để dừng server cũ và đóng tunnel.
2. Mở file AI_SERVICES/.env, sửa REGRESSION_MODEL_DIR hoặc YOLO_MODEL_PATH trỏ tới model mới.
3. Chạy lại tuần tự từ **Cell 4** -> **Cell 5** -> **Cell 6**. Không cần khởi động lại runtime Colab.

### B. Khi thay đổi Source Code trong src/rice_ai:
1. Chạy **Cell 8** để dừng server.
2. Chọn menu Colab: **Runtime** -> **Restart session** (hoặc nhấn phím tắt Ctrl + M .).
3. Chạy lại từ **Cell 2** đến **Cell 6**.
