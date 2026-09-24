# Rice Dataset Capture

Ứng dụng desktop điều khiển việc thu thập ảnh và dữ liệu thủ công. Ảnh có thể
đến từ webcam trên máy tính hoặc từ điện thoại Android/iPhone kết nối bằng QR.
Máy tính là nơi duy nhất cấp ID, lưu ảnh, lưu SQLite và đồng bộ Excel.

## Cài đặt lần đầu trên Windows

### Điều kiện tiên quyết

- Máy tính chạy Windows 10 hoặc Windows 11, 64-bit.
- Có kết nối Internet trong lần cài đầu tiên để tải thư viện Python.
- Đã cài **Python 3.12.x, 64-bit**. Khi cài từ python.org, bắt buộc chọn
  **Add python.exe to PATH** ở màn hình đầu tiên của trình cài đặt.
- Microsoft Excel là tùy chọn: ứng dụng vẫn tạo/lưu `.xlsx`, nhưng cần Excel
  hoặc ứng dụng tương thích nếu muốn mở và sửa bảng tính thủ công.

Ứng dụng không yêu cầu cài Anaconda, DroidCam, USB debugging hay phần mềm camera
ảo. Nếu đang dùng pyenv, có thể dùng Python 3.12.10; nếu không có pyenv, Python
3.12 cài thông thường vẫn hoạt động.

### Bước 1 — Giải nén ứng dụng

Giải nén thư mục `CAPTURE_APP` vào nơi bạn có quyền ghi, ví dụ:

    D:\RiceCapture\CAPTURE_APP

Không nên đặt trong `Program Files`, vì Windows có thể chặn ứng dụng tạo thư mục
`.venv` và `runtime`.

### Bước 2 — Kiểm tra Python

Mở **Command Prompt** hoặc **PowerShell**, chạy một trong hai lệnh:

    py -3.12 --version

hoặc:

    python --version

Kết quả phải bắt đầu bằng `Python 3.12`. Nếu báo không tìm thấy Python hoặc ra
phiên bản khác, hãy cài lại Python 3.12 x64 và chọn **Add python.exe to PATH**.

### Bước 3 — Tạo môi trường và cài thư viện

Chọn **một trong hai cách** sau.

**Cách A — Kích đúp file:** trong thư mục `CAPTURE_APP`, kích đúp:

    setup_capture_app.bat

**Cách B — Chạy trong PowerShell:** mở PowerShell, chuyển đến thư mục ứng dụng rồi chạy:

    cd "D:\RiceCapture\CAPTURE_APP"
    .\setup_capture_app.bat

Hoặc trong **Command Prompt**:

    cd /d "D:\RiceCapture\CAPTURE_APP"
    setup_capture_app.bat

Thay `D:\RiceCapture\CAPTURE_APP` bằng nơi bạn đã giải nén ứng dụng.

Script tự tạo môi trường ảo tại:

    CAPTURE_APP\.venv

Sau đó cài OpenCV, Pillow, FastAPI, OpenPyXL và các thư viện cần thiết khác.
Chờ đến khi cửa sổ hiện thông báo cài đặt hoàn tất. Chỉ cần làm bước này một lần
trên mỗi máy, hoặc chạy lại sau khi nhận phiên bản mã nguồn mới.

### Bước 4 — Mở ứng dụng

Chọn **một trong hai cách** sau.

**Cách A — Kích đúp file:**

    run_capture_app.bat

**Cách B — Chạy trong terminal:** nếu vẫn đang ở thư mục `CAPTURE_APP`, chạy:

    .\run_capture_app.bat

Trong Command Prompt, dùng:

    run_capture_app.bat

Không chạy trực tiếp các tệp `.py` hoặc dùng Python hệ thống; launcher luôn dùng
đúng môi trường `CAPTURE_APP\.venv` đã được tạo ở bước 3.

### Bước 5 — Thiết lập lần đầu trong ứng dụng

1. Chọn **thư mục lưu ảnh**. Ảnh được lưu phẳng như `M0001.jpg`, `M0002.jpg`.
2. Chọn hoặc tạo **file Excel** để lưu bảng dữ liệu. File Excel trống được tự thêm
   hàng tiêu đề khi lưu mẫu đầu tiên.
3. Mặc định ứng dụng dùng 4 chữ số sau `M` (`M0001`, `M0002`, ...).
4. Chọn nguồn webcam trên máy tính, hoặc mở tab **Thiết bị di động** để kết nối QR.
5. Nếu Windows Firewall hỏi khi bật kết nối điện thoại, cho phép trên mạng
   **Private**. Không cần cho phép mạng Public.

### Xử lý lỗi cài đặt thường gặp

- **Không tìm thấy Python 3.12:** cài Python 3.12 x64, tích `Add python.exe to PATH`,
  đóng/mở lại Command Prompt rồi chạy lại `setup_capture_app.bat`.
- **`.venv hiện tại không dùng Python 3.12`:** đóng ứng dụng, chỉ xóa thư mục
  `CAPTURE_APP\.venv`, sau đó chạy lại `setup_capture_app.bat`.
- **Không cài được thư viện:** kiểm tra Internet, tắt VPN/proxy nếu đang chặn
  Python Package Index, rồi chạy lại script.
- **Không mở được kết nối điện thoại:** xác nhận máy tính và điện thoại cùng Wi-Fi
  hoặc USB tethering; kiểm tra Windows Firewall đã cho phép mạng Private.

## Bàn giao cho người khác

Nén và gửi **toàn bộ thư mục `CAPTURE_APP`**, nhưng bỏ các thư mục/tệp phát sinh
trên máy cá nhân sau:

- `.venv` — môi trường Python phụ thuộc máy hiện tại và rất nặng.
- `runtime` — chứa đường dẫn Excel/ảnh cá nhân, thiết lập cục bộ và có thể có khóa HTTPS.
- `__pycache__` và `desktop.ini`.

Gói gửi tối thiểu phải gồm `src`, `requirements.txt`, `setup_capture_app.bat`,
`run_capture_app.bat`, `README.md` và `.python-version`. Có thể kèm `tests`,
`dataset_capture_app.py` và `capture_app_core.py` để kiểm thử/tương thích.

Trên máy nhận, làm theo đầy đủ mục **Cài đặt lần đầu trên Windows** ở trên.

Mỗi máy tự tạo `runtime` và chứng chỉ HTTPS riêng. Không gửi hay dùng chung các
file `*-key.pem` hoặc chứng chỉ CA giữa các máy.

## Chụp bằng điện thoại

### Cài chứng chỉ HTTPS một lần

Camera nhúng của trình duyệt yêu cầu HTTPS. Trên tab **Thiết bị di động**, bấm
**Xuất chứng chỉ…**, chuyển file `RiceCapture-CA.cer` sang điện thoại rồi cài.
Chỉ xuất file `.cer` công khai; không sao chép các file `*-key.pem` trong
`runtime\certificates` sang thiết bị khác.

> Lưu ý an toàn: chứng chỉ này là CA cục bộ. Chỉ cài trên thiết bị dành cho
> thu thập dữ liệu hoặc thiết bị bạn kiểm soát, và gỡ khi không còn dùng.

### Kết nối và chụp

1. Máy tính và điện thoại kết nối cùng Wi-Fi, hoặc dùng USB tethering.
2. Mở tab **Thiết bị di động**.
3. Lần đầu: bấm **Xuất chứng chỉ…**, chuyển và cài `RiceCapture-CA.cer` lên điện thoại.
4. Chọn địa chỉ mạng phù hợp và bấm **KHỞI ĐỘNG KẾT NỐI**.
5. Nếu Windows Firewall hỏi, cho phép ứng dụng trên mạng Private.
6. Quét QR HTTPS bằng Android hoặc iPhone đã tin cậy chứng chỉ.
7. Nhập thông số, chọn tỷ lệ 4:3, 16:9 hoặc theo camera; bấm **Bật camera**.
8. Khi camera có hỗ trợ, nút **Bật flash** sẽ xuất hiện. Căn ảnh, chụp và bấm **Lưu mẫu**.
9. Ảnh cùng form xuất hiện trên desktop và được ghi vào bảng Excel.

Camera được hiển thị trong một khung nằm dưới form; camera hệ thống toàn màn hình
vẫn có sẵn làm phương án dự phòng. Không cần DroidCam, USB debugging hoặc camera
ảo. QR chứa token ngẫu nhiên và hết hiệu lực khi ứng dụng dừng.

## Lưu dữ liệu

- Ảnh được lưu phẳng tại thư mục đã chọn: M0001.jpg, M0002.jpg, ...
- Số chữ số sau M được cố định là 4 để tương thích pipeline: M0001-M9999.
- SQLite nằm cạnh workbook dưới tên capture_data.sqlite3.
- Excel giữ bảy cột tương thích pipeline hiện tại.
- Nếu Excel đang bị khóa, mẫu vẫn an toàn trong SQLite và được đồng bộ lại sau.
- File Excel hoàn toàn trống sẽ được tự tạo hàng tiêu đề.

## Mã nguồn

    src/rice_capture/core       Quy tắc ID và kiểm tra form
    src/rice_capture/desktop    Giao diện Tkinter
    src/rice_capture/server     FastAPI, QR và WebSocket
    src/rice_capture/services   Điều phối và lưu mẫu
    src/rice_capture/storage    SQLite, Excel và ảnh
    src/rice_capture/web        Giao diện điện thoại
    tests                       Kiểm thử
    runtime                     Thiết lập và log cục bộ

Hai file dataset_capture_app.py và capture_app_core.py ở thư mục gốc chỉ là
lớp tương thích cho script và kiểm thử cũ.

## Chạy kiểm thử

    $env:PYTHONPATH = "$PWD\src"
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
