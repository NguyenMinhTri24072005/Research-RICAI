# TASK-03: MỞ RỘNG DỮ LIỆU VỚI CAPTURE_APP — ĐIỆN THOẠI LÀ CAMERA NODE, MÁY TÍNH LÀ HOST

> **Căn cứ:** ghi chú “thêm dữ liệu cho regression” và “thêm camera nhỏ, fix chiều cao chụp.” trong `NOTES/note_24_08_2026.txt`.

**Trạng thái:** CAPTURE_APP đã sẵn sàng để thu ảnh và nhập dữ liệu; cần thực hiện theo [TASK-00](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md) trước khi thu hàng loạt.

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Mục tiêu và phạm vi

Mục tiêu ngay trước mắt là tạo một tập dữ liệu thực, truy vết được, phục vụ hồi quy số lượng hạt:

- **Pilot:** 55 mẫu theo bảng lấy mẫu ở dưới, dùng để kiểm tra quy trình và chất lượng dữ liệu.
- **Mức tối thiểu cho benchmark hiện tại:** ít nhất 100 mẫu hợp lệ sau kiểm soát chất lượng.
- **Mở rộng 700–1.000 mẫu:** là giai đoạn tiếp theo sau khi quy trình pilot ổn định; không phải điều kiện hoàn thành của đợt thu đầu tiên.

Không mua mạch camera, webcam hay vi điều khiển để thay thế điện thoại. Kiến trúc được chốt là:

```text
Điện thoại = camera node (chụp ảnh, nhập form di động)
Máy tính   = host (CAPTURE_APP, QR, mã ảnh, Excel, lưu trữ, pipeline AI)
```

Chi tiết thiết lập và cách quản lý metadata thuộc [TASK-00](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md). Cân điện tử hiện được nhập tay; việc kết nối trực tiếp chỉ là một hạng mục tương lai sau khi xác định được model cân và giao thức truyền dữ liệu.

### 2. Thiết lập chụp bắt buộc

- Dùng điện thoại cố định trên giá, camera hướng gần vuông góc với miệng ly.
- Chọn độ cao/góc chụp trong khoảng đã kiểm thử để toàn bộ ly và hạt nhìn rõ;
  không yêu cầu cố định một độ cao hay ghi `Capture_Height_mm`.
- Bảo đảm ảnh đủ sáng và không chói/mờ; không yêu cầu lưu cấu hình ánh sáng.
- Khoá focus, exposure và white balance nếu ứng dụng điện thoại cho phép; kiểm tra ảnh không rung, không cháy sáng và ly nằm trọn khung.
- Khuyến nghị có bảng chuẩn kích thước/màu xuất hiện trong ảnh kiểm tra đầu mỗi phiên. Không bắt buộc đưa vào mọi ảnh sản xuất nếu ảnh hưởng pipeline.

### 3. Kế hoạch lấy mẫu pilot

| Phân tầng | Dải số hạt | Số mẫu | Mục đích |
| --- | ---: | ---: | --- |
| Cực thưa | 10–50 | 10 | Kiểm tra độ nhạy vùng ít hạt |
| Trung bình | 60–180 | 20 | Phân bố vận hành phổ biến |
| Đầy ly | 190–350 | 15 | Kiểm tra nén chặt và che khuất |
| Giống/hình dạng khác | Đa dạng | 10 | Kiểm tra độ bao phủ hình thái |
| **Tổng pilot** |  | **55** | Kiểm tra protocol trước khi mở rộng |

Mỗi phiên phải có `Capture_Batch` và `Device_ID` hoặc `Device_Model`.
`Capture_Timestamp` được app tạo khi lưu ảnh. Các nhóm này được giữ lại để tách
train/test theo batch hoặc thiết bị ở TASK-04, tránh ảnh cùng điều kiện xuất
hiện đồng thời ở train và test.

### 4. Quy trình thao tác

1. Chạy [`run_capture_app.bat`](../DATASET_BUILDER/CAPTURE_APP/run_capture_app.bat), chọn thư mục lưu ảnh phẳng và file Excel đích, rồi quét QR bằng điện thoại.
2. Khi mở phiên, chọn `Capture_Batch` và thiết bị. Với từng mẫu, nhập `Weight_g`, `Container_Height_mm`, `Inner_Diameter_mm`, `Empty_Height_mm` và `Actual_Count`. `Rice_Height_mm` là giá trị suy ra từ kích thước ly và khoảng trống, không nhập trùng.
3. Chụp/lưu. Ảnh nằm trực tiếp trong thư mục đã chọn với mã tăng dần, ví dụ `M0058.jpg`, `M0059.jpg`; không tạo cấu trúc con kiểu `M001/M001A`.
4. Chạy [`AI_DATASET_EXTRACTION_PIPELINE.ipynb`](../DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb) để bóc tách hạt và tạo đặc trưng.
5. Chạy `check_filtered_grains.py` để kiểm tra pipeline khi cần. Mốc “ít hơn 3 hạt crop” là **cảnh báo để kiểm tra ảnh/pipeline**, không phải quy tắc tự động loại mẫu ít hạt vì sẽ làm lệch phân bố dữ liệu.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể và trạng thái (kiểm toán 16/09/2026)

- [x] `CAPTURE_APP` có luồng desktop/mobile, lưu ảnh và workbook; 15 unit test
  đã PASS trong lần kiểm toán 16/09/2026.
- [x] Có 261 ảnh gốc trong 57 thư mục mẫu và workbook dataset cuối có 285 bản
  ghi tương ứng 57 mã mẫu gốc.
- [ ] Chưa chứng minh được 55 mẫu pilot có nhật ký thiết lập và đối chiếu
  ảnh–Excel một-một.
- [ ] Chưa đạt bằng chứng tối thiểu 100 mẫu vật lý hợp lệ: workbook thủ công
  hiện có 92 mã mẫu gốc, trong đó 35 bản ghi trống `Actual_Count`.
- [ ] Thiếu `Capture_Batch` và thông tin thiết bị, nên chưa thể xác nhận dữ liệu
  đủ điều kiện cho benchmark theo batch/thiết bị.

### Tiêu chí hoàn thành toàn bộ task

- [ ] Hoàn tất 55 mẫu pilot, có nhật ký thiết lập và ảnh/mã Excel khớp một-một.
- [ ] Từ pilot đã sửa các lỗi protocol trước khi thu tiếp; đạt ít nhất 100 mẫu hợp lệ, có nhãn `Actual_Count` đo thực.
- [ ] Ảnh lưu phẳng trong thư mục được chọn; file Excel và kết quả pipeline có đường dẫn/phiên bản rõ ràng.
- [ ] Pipeline không tự loại mẫu chỉ vì số hạt ít.
