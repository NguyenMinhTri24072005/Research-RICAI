# TASK-02: GIẢI TRÌNH KẾT QUẢ AI/XAI — “TẠI SAO RA CON SỐ ĐÓ?”

> **Phụ thuộc:** TASK-01 phải tạo được schema 31 biến và dự đoán hồi quy thật.

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Mục tiêu

Khi hệ thống trả về số hạt dự đoán, người dùng và hội đồng cần xem được cả cơ
sở hình học lẫn các đặc trưng làm kết quả tăng/giảm. Giải trình phải là số liệu
truy vết được, không phải đoạn mô tả chung chung.

### 2. Hai lớp giải trình

#### 2.1. Hình học và vật lý

Hiển thị rõ các đại lượng đầu vào và công thức:

$$V_{bulk} = \pi \left(\frac{D_{inner}}{2}\right)^2(H_{ly} - H_{trống})$$

$$v_{grain} = \frac{4}{3}\pi abc$$

$$N_{geometry} = \frac{V_{bulk}\phi}{v_{grain}}, \quad \phi \approx 0.62$$

Kết quả cần ghi đơn vị, số hạt bề mặt được nhận diện, thể tích hạt trung vị và
mọi cảnh báo khi segmentation có chất lượng thấp.

#### 2.2. Đóng góp của hồi quy

Với hồi quy tuyến tính chuẩn hóa, dùng phân rã chính xác:

$$\hat y = w_0 + \sum_i w_i\frac{x_i - \mu_i}{\sigma_i}$$

$$Contribution_i = w_i\frac{x_i - \mu_i}{\sigma_i}$$

Đây là phương pháp chính; SHAP chỉ là tùy chọn nếu sau này dùng mô hình phi
tuyến. Gom 31 biến thành các nhóm: vật chứa/khối lượng, kích thước hạt, thể
tích–tỷ lệ ảnh và độ đồng đều–phẩm cấp.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể

- [ ] Viết `AI_SERVICES/modules/explainability.py` nhận feature vector, schema,
  scaler và model artifact manifest.
- [ ] Trả bảng đóng góp gồm `feature`, `raw_value`, `standardized_value`,
  `coefficient`, `contribution`, `category`; kèm top 5 tăng và top 5 giảm.
- [ ] Sinh phần giải thích tiếng Việt có số liệu, nêu rõ khi không đủ điều kiện
  để tin cậy kết quả.
- [ ] Bổ sung `/explain` hoặc trường `explanation` trong response `/predict`;
  endpoint phải dùng đúng feature vector đã suy luận, không tính lại khác schema.
- [ ] Frontend hiển thị công thức/tóm tắt, chỉ số hình học và waterfall/bar chart
  cho đóng góp hồi quy.

### Trạng thái kiểm toán (16/09/2026)

- [ ] Chưa có `AI_SERVICES/modules/explainability.py`.
- [ ] API chưa trả bảng đóng góp, giải thích tiếng Việt hoặc endpoint `/explain`.
- [ ] Frontend chưa có biểu đồ contribution/waterfall cho hồi quy.

### Tiêu chí hoàn thành toàn bộ task

1. [ ] Một request có thể tái tạo cùng `prediction`, feature vector và bảng đóng góp.
2. [ ] Tổng các đóng góp cộng intercept khớp với dự đoán hồi quy trong sai số làm tròn.
3. [ ] Người dùng xem được lời giải thích ngắn gọn bằng tiếng Việt và dữ liệu chi tiết
   phục vụ hội đồng kiểm tra.
