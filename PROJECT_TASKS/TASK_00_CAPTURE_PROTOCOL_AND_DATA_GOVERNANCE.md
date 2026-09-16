# TASK-00: QUY CHUẨN THU THẬP DỮ LIỆU, CAMERA ĐIỆN THOẠI VÀ KIỂM SOÁT CHẤT LƯỢNG

## 1. Mục tiêu

Thiết lập một quy trình thu thập có thể lặp lại trước khi mở rộng dataset hay
so sánh mô hình. Kiến trúc chính thức của dự án là:

```text
Điện thoại = camera node
Máy tính   = host quản lý QR, mã ảnh, form, Excel, lưu trữ và xử lý AI
```

Không mua camera mạch, webcam giá rẻ hoặc vi điều khiển cho mục đích thay thế
camera điện thoại. `CAPTURE_APP` hiện đã đáp ứng kết nối điện thoại bằng QR qua
HTTPS và lưu ảnh phẳng theo `Sample_ID`.

## 2. Thiết lập phần cứng tối thiểu

- Điện thoại có camera sau, gắn trên **giá đỡ từ trên xuống**; cố định vuông góc
  với tâm miệng ly.
- Chọn **một** chiều cao thực tế (ví dụ `18.0 cm`) và giữ nguyên cho cả đợt thu
  thập; không dùng một dải 15–20 cm.
- Nguồn sáng LED trắng tán xạ hoặc ring light có tản sáng để hạn chế bóng và
  chói trên vành ly.
- Tấm chuẩn kích thước/ô caro nhỏ chỉ dùng để kiểm tra định kỳ tỷ lệ ảnh; không
  che miệng ly khi chụp mẫu.

Thiết bị cân điện tử được nhập tay vào form ở giai đoạn này. Tích hợp cân USB,
Bluetooth hoặc cổng COM chỉ được mở thành task riêng sau khi xác định rõ model
cân và giao thức dữ liệu của nó.

## 3. Quy trình chụp chuẩn

1. Dùng cùng một điện thoại/camera sau trong một đợt; ghi nhận model điện thoại.
2. Khóa lấy nét, phơi sáng và cân bằng trắng nếu camera hỗ trợ; không dùng zoom
   số hoặc flash tự động.
3. Đặt ly vào vị trí có dấu định tâm, xác nhận góc chụp gần 90° (sai lệch dưới
   5°) và chiều cao cố định.
4. Dùng `CAPTURE_APP`: máy tính khởi động host, điện thoại quét QR, điền form và
   chụp một ảnh cho mỗi `Sample_ID` (ví dụ `M0058.jpg`).
5. Đếm thực tế độc lập trước khi lưu `Actual_Count`; ghi nhận người đếm hoặc ca
   thu thập khi có nhiều người tham gia.

## 4. Metadata và kiểm soát chất lượng

Ngoài các trường bắt buộc hiện có (`Sample_ID`, `Weight_g`,
`Container_Height_mm`, `Inner_Diameter_mm`, `Empty_Height_mm`,
`Actual_Count`), ghi nhận theo đợt thu thập:

- ngày/ca chụp, người thao tác, giống lúa hoặc loại hạt;
- model điện thoại, chiều cao chụp, cấu hình ánh sáng;
- mã đợt/batch để có thể chia train/test theo batch, tránh rò rỉ dữ liệu;
- cờ chất lượng ảnh: mờ, chói, lệch tâm, che khuất hoặc cần chụp lại.

`check_filtered_grains.py` là công cụ chẩn đoán kết quả segmentation. Ngưỡng
crop (ví dụ có ít hơn 3 hạt) không được dùng để tự động loại các mẫu ít hạt,
vì sẽ làm lệch phân bố vùng số lượng thấp.

## 5. Definition of Done

- Có ảnh kiểm tra đầu/cuối mỗi ca chụp cho thấy chiều cao và ánh sáng ổn định.
- Mỗi mẫu có ảnh phẳng, bản ghi Excel và `Actual_Count` đối chiếu được.
- Có `Capture_Batch` hoặc metadata tương đương để phục vụ đánh giá theo nhóm.
- Nhóm xác nhận quy trình trước khi bắt đầu thu thập quy mô lớn ở TASK-03.
