# RICE VISION AI — MASTER TASK BOARD & ROADMAP

> **Cập nhật lần cuối:** 15/09/2026  
> **Trạng thái tổng thể:** Đang thực hiện  
> **Mục tiêu:** Xây dựng quy trình thu thập đáng tin cậy, benchmark có giá trị
> khoa học và sản phẩm AI có thể giải trình.

## 1. Bảng theo dõi nhiệm vụ

| Mã task | Nhiệm vụ | Ưu tiên | Trạng thái | Đặc tả |
| :--- | :--- | :---: | :--- | :--- |
| **TASK-00** | Quy chuẩn thu thập, điện thoại camera node và kiểm soát chất lượng | 🔴 P1 | ⏳ Mới | [TASK-00](TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md) |
| **TASK-01** | Hoàn tất tích hợp hồi quy 31 biến vào `AI_SERVICES` | 🔴 P1 | 🟡 Làm một phần | [TASK-01](TASK_01_AI_SERVICES_INTEGRATION.md) |
| **TASK-02** | Module giải trình kết quả AI/XAI | 🔴 P1 | ⏳ Chờ TASK-01 | [TASK-02](TASK_02_EXPLAINABLE_AI_XAI.md) |
| **TASK-03** | Thu thập/mở rộng dataset bằng `CAPTURE_APP` | 🔴 P1 | 🟡 Ứng dụng sẵn sàng, chờ thu thập | [TASK-03](TASK_03_DATASET_EXPANSION.md) |
| **TASK-04** | Benchmark 14 mô hình hồi quy và artefact bài báo | 🟡 P2 | ⏳ Chờ dataset ổn định | [TASK-04](TASK_04_REGRESSION_BENCHMARK_PAPER.md) |
| **TASK-05** | Nghiên cứu feasibility few-shot/transfer sang hạt khác | 🟢 P3 | ⏳ Định hướng nghiên cứu | [TASK-05](TASK_05_FEW_SHOT_TRANSFER_LEARNING.md) |

## 2. Thứ tự triển khai

```mermaid
gantt
    title Lộ trình dự án Rice Vision AI
    dateFormat  YYYY-MM-DD
    section Nền tảng dữ liệu
    TASK-00: Quy chuẩn chụp và metadata :active, t0, 2026-09-15, 3d
    TASK-03 Pilot 55 mẫu: after t0, 7d
    TASK-03 Mở rộng tối thiểu 100 mẫu hợp lệ: after t0, 21d
    section Dịch vụ AI
    TASK-01: Regression 31 biến trong API :active, t1, 2026-09-15, 5d
    TASK-02: Giải trình hình học và hồi quy : t2, after t1, 4d
    section Nghiên cứu và bài báo
    TASK-04: Benchmark trên dataset đã khóa : t4, after t1, 7d
    Phân tích và viết bài báo: after t4, 10d
    section Mở rộng
    TASK-05: Few-shot/transfer feasibility : t5, after t4, 10d
```

TASK-01 và TASK-03 có thể chạy song song: TASK-01 dùng tập dữ liệu hiện có để
hoàn thiện API; TASK-03 tạo dữ liệu chuẩn cho đánh giá và công bố sau cùng.

## 3. Nguyên tắc quyết định

- **Camera:** điện thoại là camera node; máy tính là host. Không mua camera
  mạch để thay thế điện thoại.
- **Dữ liệu trước chỉ số:** không công bố kết quả benchmark trước khi khóa giao
  thức chụp, metadata batch và tập test độc lập.
- **Tái lập:** mọi mô hình phải dùng cùng schema đặc trưng, split, seed, pipeline
  tiền xử lý và báo cáo metric.
- **Giải trình:** kết quả sản phẩm phải hiển thị được cơ sở hình học và đóng góp
  của đặc trưng hồi quy, thay vì chỉ trả một con số.

## 4. Cấu trúc thư mục

```text
PROJECT_TASKS/
├── README.md
├── TASK_00_CAPTURE_PROTOCOL_AND_DATA_GOVERNANCE.md
├── TASK_01_AI_SERVICES_INTEGRATION.md
├── TASK_02_EXPLAINABLE_AI_XAI.md
├── TASK_03_DATASET_EXPANSION.md
├── TASK_04_REGRESSION_BENCHMARK_PAPER.md
└── TASK_05_FEW_SHOT_TRANSFER_LEARNING.md
```
