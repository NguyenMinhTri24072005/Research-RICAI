# Legacy archive

Thư mục này chỉ lưu các tệp của quy trình cũ để tham chiếu; chúng không thuộc
luồng vận hành hiện tại của DATASET_BUILDER.

- `raw_image_renaming/`: script đổi tên ảnh theo cấu trúc cũ `M001/M001A.jpg`.
- `legacy_raw_survey/`: script và báo cáo khảo sát cũ, vốn giả định mỗi mẫu có
  năm ảnh `A` đến `E`.

Luồng hiện tại dùng Capture App để lưu ảnh phẳng theo `Sample_ID` (ví dụ
`M0001.jpg`), sau đó chạy `AI_DATASET_EXTRACTION_PIPELINE.ipynb` và kiểm tra
hạt bằng `check_filtered_grains.py`.
