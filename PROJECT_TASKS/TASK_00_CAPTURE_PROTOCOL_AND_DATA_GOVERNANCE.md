# TASK-00: QUY CHUẨN THU THẬP DỮ LIỆU, CAMERA ĐIỆN THOẠI VÀ KIỂM SOÁT CHẤT LƯỢNG

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Mục tiêu

Thiết lập một quy trình thu thập có thể lặp lại trước khi mở rộng dataset hay
so sánh mô hình. Kiến trúc chính thức của dự án là:

```text
Điện thoại = camera node
Máy tính   = host quản lý QR, mã ảnh, form, Excel, lưu trữ và xử lý AI
```

Không mua camera mạch, webcam giá rẻ hoặc vi điều khiển cho mục đích thay thế
camera điện thoại. `CAPTURE_APP` hiện đã đáp ứng kết nối điện thoại bằng QR qua
HTTPS và lưu ảnh phẳng theo `Sample_ID`.

### 2. Thiết lập phần cứng tối thiểu

- Điện thoại có camera sau, gắn trên **giá đỡ từ trên xuống**; cố định vuông góc
  với tâm miệng ly.
- Đặt camera ở độ cao/góc cho ảnh thấy rõ toàn bộ ly và hạt; có thể dùng nhiều
  độ cao trong khoảng đã kiểm thử, không bắt buộc một giá trị cố định.
- Bảo đảm ảnh đủ sáng, hạn chế bóng/chói trên vành ly; không cần lưu cấu hình
  ánh sáng thành metadata.
- Tấm chuẩn kích thước/ô caro nhỏ chỉ dùng để kiểm tra định kỳ tỷ lệ ảnh; không
  che miệng ly khi chụp mẫu.

Thiết bị cân điện tử được nhập tay vào form ở giai đoạn này. Tích hợp cân USB,
Bluetooth hoặc cổng COM chỉ được mở thành task riêng sau khi xác định rõ model
cân và giao thức dữ liệu của nó.

### 3. Quy trình chụp chuẩn

1. Có thể dùng nhiều điện thoại/camera sau; gán `Device_ID` hoặc `Device_Model`
   cho từng thiết bị sử dụng trong đợt.
2. Khóa lấy nét, phơi sáng và cân bằng trắng nếu camera hỗ trợ; không dùng zoom
   số hoặc flash tự động.
3. Đặt ly vào vị trí có dấu định tâm và bảo đảm góc chụp/độ cao cho ảnh thấy rõ
   toàn bộ ly cùng hạt.
4. Dùng `CAPTURE_APP`: máy tính khởi động host, điện thoại quét QR, điền form và
   chụp một ảnh cho mỗi `Sample_ID` (ví dụ `M0058.jpg`).
5. Đếm thực tế độc lập trước khi lưu `Actual_Count`.

### 4. Metadata và kiểm soát chất lượng

Ngoài các trường bắt buộc hiện có (`Sample_ID`, `Weight_g`,
`Container_Height_mm`, `Inner_Diameter_mm`, `Empty_Height_mm`,
`Actual_Count`), chỉ lưu metadata tối thiểu:

- `Capture_Batch`: mã phiên/đợt chụp để có thể chia train/test theo batch;
- `Device_ID` hoặc `Device_Model`: thiết bị tạo ảnh;
- `Capture_Timestamp`: thời điểm app lưu ảnh, sinh tự động.

`check_filtered_grains.py` là công cụ chẩn đoán kết quả segmentation. Ngưỡng
crop (ví dụ có ít hơn 3 hạt) không được dùng để tự động loại các mẫu ít hạt,
vì sẽ làm lệch phân bố vùng số lượng thấp.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể và trạng thái (kiểm toán 16/09/2026)

- [x] `CAPTURE_APP` hỗ trợ điện thoại làm camera node qua QR/HTTPS, lưu ảnh
  phẳng và các trường bản ghi thủ công cơ bản.
- [x] Đã có dữ liệu thủ công và ảnh gốc làm nền để kiểm tra quy trình.
- [ ] Schema workbook và `CAPTURE_APP` chưa lưu `Capture_Batch`, thông tin
  thiết bị và `Capture_Timestamp` tự động.
- [ ] Chưa có xác nhận protocol tối giản của nhóm trước khi thu hàng loạt.

### Tiêu chí hoàn thành toàn bộ task

- [ ] Workbook và `CAPTURE_APP` lưu được `Capture_Batch`, thông tin thiết bị và
  `Capture_Timestamp` cho từng mẫu.
- [ ] Mỗi mẫu có ảnh phẳng, bản ghi Excel và `Actual_Count` đối chiếu được.
- [ ] Có `Capture_Batch` để phục vụ đánh giá theo nhóm.
- [ ] Nhóm xác nhận quy trình trước khi bắt đầu thu thập quy mô lớn ở TASK-03.
