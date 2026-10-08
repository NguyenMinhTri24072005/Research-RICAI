# Kiến trúc & Luồng hoạt động hệ thống đếm hạt gạo (RICAI Pipeline)

Dưới đây là sơ đồ luồng dữ liệu, trong đó làm nổi bật các **Mô hình Máy học (ML)** và **Học sâu (DL)** được sử dụng trong dự án, bao gồm đầu vào của chúng và thư mục chứa mã nguồn/trọng số.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "15px"}, "flowchart": {"htmlLabels": true, "padding": 24, "nodeSpacing": 60, "rankSpacing": 70, "wrappingWidth": 260}}}%%
flowchart TD
    %% Định nghĩa các Style
    classDef input fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef process fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#3b0764
    classDef ai_model_dl fill:#fecdd3,stroke:#be123c,stroke-width:3px,color:#881337
    classDef ai_model_ml fill:#fed7aa,stroke:#c2410c,stroke-width:3px,color:#7c2d12
    classDef postprocess fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#064e3b
    classDef math fill:#ffedd5,stroke:#b91c1c,stroke-width:2px,color:#7f1d1d
    classDef output fill:#fef08a,stroke:#a16207,stroke-width:2px,color:#422006

    %% 1. Input
    I1["Ảnh chụp từ trên xuống<br/>RGB Image"]:::input
    I2["Thông số ly chứa<br/>Đường kính, Chiều cao"]:::input
    I3["Trọng lượng ly<br/>Weight Sensor - Tuỳ chọn"]:::input

    %% 2. Tiền xử lý & Trích xuất ly
    subgraph S1["1. Hiệu chuẩn Hệ đo lường"]
        C1["Dò tìm viền ly<br/>Ellipse Fitting & RANSAC"]:::process
        C2["Tính tỷ lệ pixel/mm<br/>pixel_per_mm"]:::process
        C3["Tính Thể tích Khối<br/>Bulk Volume mm3"]:::process

        C1 --> C2
        C1 --> C3
    end

    %% 3. Phân đoạn hạt AI
    subgraph S2["2. Phân đoạn Hạt AI"]
        A1["Tách lưới ảnh SAHI<br/>Cắt ảnh thành patch nhỏ"]:::process

        A2["Deep Learning (CNN)<br/>YOLOv8-Seg / YOLOv26<br/><br/>Mã nguồn:<br/>CODE/modules/<br/>grain_segmenter.py<br/><br/>Trọng số:<br/>RESULTS/<br/>Segmentation Model Results"]:::ai_model_dl

        A3["Gộp kết quả NMS<br/>Tạo Mask thô"]:::process

        A1 -->|"Ảnh RGB cắt nhỏ (Patches)"| A2
        A2 -->|"Mask nhị phân phân đoạn"| A3
    end

    %% 4. Hậu xử lý & Đo lường
    subgraph S3["3. Làm sạch & Phân tích Hình học"]
        P1["Morphological & Watershed<br/>grain_crop_cleaner.py"]:::postprocess
        P2["Khớp Ellipse cho từng hạt<br/>Tìm Trục lớn, Trục bé"]:::process
        P3["Nội suy 3D Ellipsoid<br/>Chiều dày & Thể tích V_grain"]:::math

        P1 --> P2 --> P3
    end

    %% 5. Lọc hạt
    subgraph S4["4. Phân loại & Lọc hạt nguyên"]
        F1["Bộ lọc kích thước IQR<br/>Loại hạt vỡ/lép"]:::process

        F2["Deep Learning (CNN)<br/>DenseNet121 Classifier<br/>Phân loại hạt nứt vỡ<br/><br/>Mã nguồn:<br/>CODE/modules/<br/>grain_classifier.py"]:::ai_model_dl

        F3["Thống kê số liệu<br/>Mean V_grain, Uniformity Rate"]:::process

        F1 -.->|"Ảnh Crop từng hạt"| F2
        F1 -->|"Kích thước hình học hạt"| F3
    end

    %% 6. Dự đoán cuối cùng
    subgraph S5["5. Dự đoán Tổng số hạt"]
        M1["Vật lý cơ bản<br/>Physical Estimate = V_bulk / V_grain"]:::math
        M2["Trích xuất 31 Features<br/>Image & Geometry Features"]:::process

        M3["Machine Learning<br/>Bayesian Ridge (ARD Regression)<br/>Hồi quy tuyến tính<br/><br/>Thư mục:<br/>LINEAR_REGRESSION_MODEL/"]:::ai_model_ml

        M4["Machine Learning<br/>Bayesian Few-Shot Calibration<br/>Ridge Residual Model<br/><br/>Mã nguồn:<br/>FromThanh/<br/>bayes_fewshot.py"]:::ai_model_ml

        M2 -->|"Vector 31 đặc trưng"| M3
        M2 -.->|"V_bulk, V_grain, Diameter"| M4
    end

    %% Flow Connections
    I1 --> C1
    I2 --> C3
    I2 --> C1

    C2 --> A1
    I1 --> A1

    A3 --> P1

    P3 --> F1

    F3 --> M1
    F3 --> M2
    C3 --> M1
    C3 --> M2
    I3 -.->|"Dữ liệu cân nặng (g)"| M4

    M1 --> M3
    M1 -.-> M4

    %% Output
    O1["Dự đoán cuối cùng<br/>Tổng số hạt lúa"]:::output

    M3 -->|"Kết quả đếm hạt"| O1
    M4 -.->|"Kết quả đếm hạt<br/>+ Khoảng tin cậy 90%"| O1
```

### Các Mô Hình AI/ML Được Sử Dụng & Vị Trí:

1. **YOLO-Seg (YOLOv8 hoặc YOLOv26)**
   * **Phân loại:** Học sâu (Deep Learning - CNN)
   * **Nhiệm vụ:** Phân đoạn đối tượng hạt gạo (Instance Segmentation).
   * **Dữ liệu đầu vào:** Ảnh cắt nhỏ (Patches) sinh ra từ bộ cắt SAHI.
   * **Thư mục chứa mã nguồn:** `CODE/modules/grain_segmenter.py`
   * **Thư mục chứa kết quả/trọng số (Weights):** `RESULTS/Segmentation Model Results/`

2. **DenseNet121 Classifier (Tùy chọn)**
   * **Phân loại:** Học sâu (Deep Learning - CNN)
   * **Nhiệm vụ:** Phân loại nhị phân hạt nguyên / hạt nứt vỡ.
   * **Dữ liệu đầu vào:** Ảnh màu được cắt (crop) riêng lẻ từng hạt sau khi đã có mask phân đoạn.
   * **Thư mục chứa mã nguồn:** `CODE/modules/grain_classifier.py` và `CODE/TRAIN_CNN.ipynb`

3. **Bayesian Ridge / ARD Regression**
   * **Phân loại:** Máy học truyền thống (Machine Learning - Regression)
   * **Nhiệm vụ:** Dự đoán tổng số hạt gạo cuối cùng.
   * **Dữ liệu đầu vào:** Vector gồm 31 đặc trưng (Features) dạng bảng (Tabular) lấy từ kích thước vật lý của ly và các thông số đo lường hình học của hạt.
   * **Thư mục chứa mã nguồn:** `LINEAR_REGRESSION_MODEL/` (Dataset huấn luyện nằm ở `DATASET_BUILDER/`)

4. **Bayesian Few-Shot Calibration (Thuật toán của nhóm Thành)**
   * **Phân loại:** Máy học thống kê (Statistical Machine Learning)
   * **Nhiệm vụ:** Hiệu chuẩn (Calibrate) số lượng hạt dự đoán với khoảng tin cậy 90%, có khả năng tối ưu (fine-tune) theo từng loại ly mới chỉ bằng vài mẫu (Few-shot).
   * **Dữ liệu đầu vào:** Thể tích khối (V_bulk), Thể tích hạt (V_grain), Đặc trưng ly, và tuỳ chọn Cảm biến cân nặng (Weight Sensor).
   * **Thư mục chứa mã nguồn:** `FromThanh/bayes_fewshot.py` và notebook `Copy of rice_fewshot_v2_colab.ipynb`