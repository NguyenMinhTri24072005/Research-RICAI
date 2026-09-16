# TASK-03: MỞ RỘNG DỮ LIỆU VỚI CAPTURE_APP — ĐIỆN THOẠI LÀ CAMERA NODE, MÁY TÍNH LÀ HOST

> **Căn cứ:** ghi chú “thêm dữ liệu cho regression” và “thêm camera nhỏ, fix chiều cao chụp.” trong `NOTES/note_24_08_2026.txt`.

**Trạng thái:** CAPTURE_APP đã sẵn sàng để thu ảnh và nhập dữ liệu; cần thực hiện theo [TASK-00](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md) trước khi thu hàng loạt.

## 1. Mục tiêu và phạm vi

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

## 2. Thiết lập chụp bắt buộc

- Dùng điện thoại cố định trên giá, camera vuông góc với miệng ly (mục tiêu 90°).
- Chọn **một** chiều cao ống kính–mặt bàn, ví dụ 18,0 cm, đo thực tế và ghi trong `Capture_Height_mm`; không dùng một khoảng 15–20 cm.
- Dùng đèn LED tán xạ/ring light có diffuser, giữ cùng vị trí và cường độ khi thu một batch.
- Khoá focus, exposure và white balance nếu ứng dụng điện thoại cho phép; kiểm tra ảnh không rung, không cháy sáng và ly nằm trọn khung.
- Khuyến nghị có bảng chuẩn kích thước/màu xuất hiện trong ảnh kiểm tra đầu mỗi phiên. Không bắt buộc đưa vào mọi ảnh sản xuất nếu ảnh hưởng pipeline.

## 3. Kế hoạch lấy mẫu pilot

| Phân tầng | Dải số hạt | Số mẫu | Mục đích |
| --- | ---: | ---: | --- |
| Cực thưa | 10–50 | 10 | Kiểm tra độ nhạy vùng ít hạt |
| Trung bình | 60–180 | 20 | Phân bố vận hành phổ biến |
| Đầy ly | 190–350 | 15 | Kiểm tra nén chặt và che khuất |
| Giống/hình dạng khác | Đa dạng | 10 | Kiểm tra độ bao phủ hình thái |
| **Tổng pilot** |  | **55** | Kiểm tra protocol trước khi mở rộng |

Mỗi phiên phải ghi ít nhất `Capture_Batch`, ngày/giờ, thiết bị, người thao tác, giống/loại hạt và thiết lập đèn. Các nhóm này được giữ lại để tách train/test theo batch ở TASK-04, tránh ảnh cùng điều kiện xuất hiện đồng thời ở train và test.

## 4. Quy trình thao tác

1. Chạy [`run_capture_app.bat`](../DATASET_BUILDER/CAPTURE_APP/run_capture_app.bat), chọn thư mục lưu ảnh phẳng và file Excel đích, rồi quét QR bằng điện thoại.
2. Nhập các thông số thủ công của mẫu: `Weight_g`, `Container_Height_mm`, `Inner_Diameter_mm`, `Empty_Height_mm`, `Actual_Count` và các metadata phiên chụp. `Rice_Height_mm` là giá trị suy ra từ kích thước ly và khoảng trống, cần được kiểm tra công thức thay vì nhập trùng.
3. Chụp/lưu. Ảnh nằm trực tiếp trong thư mục đã chọn với mã tăng dần, ví dụ `M0058.jpg`, `M0059.jpg`; không tạo cấu trúc con kiểu `M001/M001A`.
4. Chạy [`AI_DATASET_EXTRACTION_PIPELINE.ipynb`](../DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb) để bóc tách hạt và tạo đặc trưng.
5. Chạy `check_filtered_grains.py` để tạo báo cáo QC. Mốc “ít hơn 3 hạt crop” là **cảnh báo để kiểm tra ảnh/pipeline**, không phải quy tắc tự động loại mẫu ít hạt vì sẽ làm lệch phân bố dữ liệu.
6. Chỉ đánh dấu loại trừ khi có lý do được ghi rõ (ảnh rung, lỗi nhãn, không thấy ly, lỗi phân đoạn không khắc phục được). Lưu lý do trong bảng dữ liệu.

## 5. Definition of Done

- Hoàn tất 55 mẫu pilot, có nhật ký thiết lập và ảnh/mã Excel khớp một-một.
- Từ pilot đã sửa các lỗi protocol trước khi thu tiếp; đạt ít nhất 100 mẫu hợp lệ, có nhãn `Actual_Count` đo thực.
- Ảnh lưu phẳng trong thư mục được chọn; file Excel, kết quả pipeline và báo cáo QC có đường dẫn/phiên bản rõ ràng.
- Không có mẫu nào được loại chỉ vì số hạt ít; mọi loại trừ đều có nguyên nhân lưu vết.
