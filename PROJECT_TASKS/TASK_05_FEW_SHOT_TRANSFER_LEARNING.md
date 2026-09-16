# TASK-05: ĐÁNH GIÁ FEW-SHOT / TRANSFER LEARNING CHO LOẠI HẠT MỚI

> **Căn cứ:** ghi chú “mô hình RNN” và “few-shot learning → regression, transfer thành các loại hạt khác (khoảng vài chục hình để dự đoán được)” trong `NOTES/note_24_08_2026.txt`.

**Ưu tiên P3.** Chỉ thực hiện sau khi benchmark cơ sở ở [TASK-04](TASK_04_REGRESSION_BENCHMARK_PAPER.md) có protocol và baseline đáng tin cậy.

## Phần 1 — Mục tiêu và hướng dẫn

### 1. Câu hỏi nghiên cứu có thể kiểm chứng

Với một loại hạt mới (ví dụ đậu nành hoặc tiêu), mô hình/hệ đặc trưng đã học trên lúa có thể thích nghi đến đâu khi chỉ có 5, 10, 20 hoặc 30 mẫu **được gán nhãn thật**? Đây là thí nghiệm tính khả thi, không phải cam kết trước rằng 20–30 ảnh sẽ đạt một mức sai số cố định.

Tập ảnh tổng hợp hoặc augmentation chỉ được phép hỗ trợ train. Nó không thể thay thế mẫu thật độc lập để chứng minh mô hình hoạt động với loại hạt mới.

### 2. Thiết kế thử nghiệm

1. Chọn một loại hạt mới và chụp theo cùng [protocol](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md), với nhãn `Actual_Count` thực.
2. Kiểm tra trước chất lượng `container_detector` và `grain_segmenter`: tỷ lệ phát hiện ly, tỷ lệ mask đúng và các failure case. Nếu phân đoạn không đủ tốt, không diễn giải sai số hồi quy là năng lực transfer.
3. Chia **theo batch** thành tập thích nghi và tập test thật chưa nhìn thấy. Test phải gồm mẫu thật, ưu tiên nhiều phiên chụp/độ đầy ly, và không được dùng để sinh synthetic data hay chọn hyperparameter.
4. Trong tập thích nghi, chạy các mức 5/10/20/30 nhãn, nhiều seed/lần lấy mẫu. So sánh: không thích nghi, fine-tune/transfer, baseline hình học + hồi quy bảng, và mô hình huấn luyện từ đầu khi dữ liệu đủ.
5. Có thể dùng augment ảnh, nhiễu feature hợp lý hoặc synthetic tabular data như một ablation **chỉ trên train**. Báo cáo rõ nguồn, tỷ lệ synthetic/real và mức cải thiện so với không dùng synthetic; không đặt mục tiêu “sinh 1.000 mẫu” như một thước đo thành công.

### 3. RNN/GRU và mô hình tập hợp hạt

Mỗi hạt có thể biểu diễn bởi `[a, b, area, eccentricity, ...]`. RNN/GRU chỉ hợp lý như một baseline khi có quy tắc sắp thứ tự nhất quán (ví dụ theo diện tích hoặc vị trí góc quanh tâm); thứ tự tùy ý sẽ làm kết quả không ổn định. Song song, so sánh một mô hình không phụ thuộc thứ tự như Deep Sets/Set Transformer hoặc các thống kê phân bố. Các mô hình này kết hợp với đặc trưng vĩ mô của ly để dự đoán tổng số hạt.

### 4. Đầu ra và tiêu chí đánh giá

- Bảng MAE, RMSE, R² và sMAPE/MAPE phù hợp trên **test thật độc lập**, theo từng mức 5/10/20/30 shot; báo cáo trung bình và độ lệch chuẩn/khoảng tin cậy qua nhiều lần lấy mẫu.
- Báo cáo phân đoạn và các trường hợp thất bại bằng ảnh minh hoạ; tách sai số nhận diện khỏi sai số hồi quy.
- So sánh transfer với from-scratch và baseline hình học. Lưu seed, split, cấu hình, dữ liệu real/synthetic và phiên bản model.

Mục tiêu kỳ vọng `MAPE < 8%` với không quá 30 nhãn được giữ như một **giả thuyết thành công**. Chỉ được tuyên bố khi đạt trên tập test thật độc lập và có khoảng tin cậy; nếu không đạt, vẫn báo cáo trung thực đường cong hiệu năng theo số shot.

## Phần 2 — Công việc cụ thể và theo dõi

### Công việc cụ thể và trạng thái (kiểm toán 16/09/2026)

- [ ] Chưa có dataset hạt mới có nhãn thật, protocol thử nghiệm hay test độc lập.
- [ ] Chưa có baseline transfer/fine-tune, from-scratch hoặc thử nghiệm 5/10/20/30 shot.
- [ ] Chưa có triển khai hoặc đánh giá RNN/GRU, Deep Sets hay Set Transformer.

### Tiêu chí hoàn thành toàn bộ task

- [ ] Có protocol thử nghiệm, dữ liệu test thật độc lập và báo cáo kết quả cho 5/10/20/30 shot.
- [ ] Có tối thiểu baseline không transfer, transfer/fine-tune, và mô hình hình học/tabular; RNN/GRU được đối sánh công bằng nếu triển khai.
- [ ] Không có kết luận transfer chỉ dựa vào dữ liệu synthetic hoặc test đã tham gia tuning.
- [ ] Báo cáo nêu rõ loại hạt, điều kiện chụp, giới hạn và khả năng/không khả năng đạt mục tiêu MAPE 8%.
