# Báo cáo điều tra nguyên nhân sai số ước lượng số hạt lúa

Ngày điều tra: 2026-09-21  
Phạm vi: chỉ đọc mã nguồn/dữ liệu/artifact; không sửa production, không ghi đè model, không retrain artifact. Các model tạm trong audit baseline chỉ tồn tại trong RAM.  
Mẫu trọng tâm: người dùng báo `Actual ≈ 450`; ảnh raw đã được cung cấp, có SHA-256 `b51a23add18ca7302de6e4bb4aaa48e6385ab1984729588505e17409d10d1638`. Output đã lưu trong notebook cho mẫu này là physical `802`, regression `212`.

## 1. Tóm tắt kết luận

Sai số `450 → 802` nằm ở **nhánh vật lý**, không phải đầu ra hồi quy: notebook đã lưu đồng thời `physical=802` và `regression=212`. Người dùng xác nhận ly là **hình trụ có đường kính trong không đổi từ miệng đến đáy**; vì vậy counterfactual nón cụt trước đó bị vô hiệu và không phải hướng sửa. Trên chính ảnh raw, detector notebook tái tạo ổn định `88.8385 px/mm` trong 5/5 lần, gần như bằng `88.84 px/mm` của lần chạy gốc; do đó scale không phải nguyên nhân trực tiếp của sai số này. Với `V_bulk=9655.37 mm³` và `phi=0.55` đã xác nhận, để ra 450 phải có mean grain volume `11.801 mm³`, cao hơn mean của 21 hạt CNN chọn (`6.6181 mm³`) 78.3%. SAHI (`86`) và cleaner (`86`) tái tạo đúng; 21 hạt CNN đã chọn được ánh xạ ngược chính xác từ output cũ, còn IQR-area có cận dưới âm nên không loại hạt nhỏ nào. Vì vậy nghi vấn chính quay lại tập hạt được CNN chọn / mask-cleaner / mô hình ellipsoid làm `mean_grain_volume` quá thấp. Độc lập với case này, detector notebook và AI_SERVICES đã phân kỳ nghiêm trọng; AI_SERVICES bắt sai rim trên chính ảnh raw (`28.06 px/mm`). Regression 31-feature cũng không chứng minh được lợi ích từ vision: weight-only tốt hơn full-31 trong 5-fold group CV và mẫu 450 nằm ngoài miền train (`Actual_Count ≤ 305`, `Weight_g ≤ 8.08`).

## 2. Bảng xếp hạng nguyên nhân

Các tỷ lệ dưới đây là **counterfactual/sensitivity**, không cộng lại thành 100% vì scale, segmentation và ellipsoid đều tác động chung lên `mean_grain_volume`.

| Hạng | Nguyên nhân | Đóng góp ước tính vào phần dư 352 hạt | Tin cậy | Cơ sở |
| ---: | --- | ---: | --- | --- |
| 1 | Mean volume của 21 hạt CNN chọn quá thấp hoặc không đại diện | Cần tăng từ `6.6181` lên `11.801 mm³` (`+78.3%`) để 802 về 450, khi giữ hình trụ và phi=0.55 | Cao cho số học; Trung bình cho nguyên nhân upstream | Hình trụ được người dùng xác nhận; 21 record tái dựng exact từ output cũ; IQR hiện không loại hạt nhỏ |
| 2 | CNN/mask/cleaner chọn hoặc tạo hạt không đại diện; IQR không loại hạt nhỏ | Có thể giải thích một phần hoặc toàn bộ bias volume, nhưng chưa tách được theo stage | Trung bình | 21/21 map exact; volume `2.985–14.897`, lower bound area `−0.810 mm²`, nên giữ toàn bộ |
| 3 | Packing fraction vật lý chưa được đo | Không giải thích toàn bộ: cần `phi=0.308` để 450 với volume 6.618, thấp bất thường; vẫn cần đo thực nghiệm | Trung bình | Counterfactual `450×6.618/9655.37=0.308`; `0.62`/`0.82` làm lỗi lớn hơn |
| 4 | Divergence/sai scale giữa hai detector | `0%` cho notebook sample này; rủi ro rất lớn cho API vì AI_SERVICES trả `28.06` thay vì `88.84 px/mm` | Cao trên ảnh raw; Trung bình cho tổng hệ thống | 5 repeat notebook ổn định; audit AI_SERVICES trên chính raw bắt sai rim |
| 5 | Công thức lịch sử `c=0.85a` | 0% theo chiều gây overestimate; nó làm volume hạt lớn hơn và count nhỏ hơn, tức che bớt lỗi | Cao về chiều tác động | Dataset lịch sử có thickness≈0.85×length ở 100% dòng; công thức mới `c=0.80b` làm volume giảm median xuống 29.7% giá trị cũ |
| 6 | Regression | 0% của con số 802; là lỗi độc lập vì nó trả 212 trên mẫu 450 OOD | Cao | Output notebook Cell 17 và arithmetic audit |

## 3. Bằng chứng chi tiết

### 3.1. Scale / `pixels_per_mm`

#### 3.1.1. Có hai detector khác nhau

- API dùng `AI_SERVICES/src/rice_ai/vision/container_detector.py`. Bản này dùng contour/Hough, không chứa RNG/RANSAC; công thức `pixels_per_mm = inner_diam_px / inner_diam_mm` ở dòng 336.
- Notebook chính import `CODE/modules/container_detector.py` (Cell 2). Bản này resize về `working_max_dim=1600`, có `_ransac_fit_ellipse`, và tự tạo `np.random.default_rng()` không seed tại dòng 225.
- Bản notebook vẫn chấp nhận nhiều outer ellipse có `outer_confidence < 0.6`; sanity check chỉ chặn theo kích thước/tỷ lệ, không chặn confidence thấp.

#### 3.1.2. Audit detector AI_SERVICES trên 6 ảnh gốc

Lệnh:

```powershell
AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/investigate.py --mode scale
```

| Sample | Lịch sử px/mm | AI_SERVICES hiện tại | Sai lệch |
| --- | ---: | ---: | ---: |
| M001a | 37.15 | 37.15339 | +0.0091% |
| M010a | 73.59 | 73.59102 | +0.0014% |
| M014a | 70.71 | 70.71194 | +0.0027% |
| M020a | 73.55 | 73.54662 | −0.0046% |
| M045a | 71.82 | 71.81792 | −0.0029% |
| M056a | 70.31 | 70.30598 | −0.0057% |

M001a chạy 5 lần cho cùng `37.153387803297775`; range và population std đều bằng 0. Với implementation AI_SERVICES hiện tại, nghi vấn scale lệch hàng chục phần trăm bị **bác bỏ trên sáu ảnh này**. Đây chỉ là agreement với feature table lịch sử, không chứng minh feature table là ground truth vật lý.

Evidence: `scale_audit.json`, `scale_comparison.csv`, `scale_M*.jpg`.

#### 3.1.3. Audit detector mà notebook chính thực sự dùng

Lệnh:

```powershell
AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_main_scale.py
```

| Sample | Lịch sử px/mm | Notebook detector | Ratio | `ratio³` |
| --- | ---: | ---: | ---: | ---: |
| M001a | 37.15 | 137.0015 | 3.6878 | 50.1533 |
| M010a | 73.59 | 83.9866 | 1.1413 | 1.4865 |
| M014a | 70.71 | 83.9693 | 1.1875 | 1.6746 |
| M020a | 73.55 | 78.7676 | 1.0709 | 1.2283 |
| M045a | 71.82 | 72.6566 | 1.0116 | 1.0354 |
| M056a | 70.31 | 79.9213 | 1.1367 | 1.4687 |

Median absolute drift là `13.90%`. Nếu dùng ratio median như sensitivity cho output 802, sửa scale sẽ đưa `802 / 1.4776 ≈ 542.8`, loại khoảng `73.6%` phần dư 352 hạt. Các giá trị lịch sử không phải ground truth; bảng này chứng minh distribution drift, chưa tự xác định bên nào đúng. M001a ít nhất là một **QC failure**: `outer_confidence=0.0663` nhưng hàm vẫn trả thành công thay vì yêu cầu review.

M010a lặp 5 lần trong log cuối: `83.98660, 83.99910, 84.06916, 84.03984, 83.98633`; range `0.08283 px/mm`, std `0.03294`. RANSAC không tái lập tuyệt đối vì không seed, dù biến thiên ngẫu nhiên nhỏ hơn nhiều so với bias detector/historical.

Evidence: `main_scale_audit.json`, `main_scale_runs.csv`, `main_scale_overlay_*.jpg`.

#### 3.1.4. Tái xác minh độc lập ba scale lệch 9–11 lần

CSV hiện tại đã bỏ các dòng này, nhưng XLSX song song vẫn giữ 285 dòng. Đọc bằng runtime spreadsheet cho thấy:

- M029c: lịch sử `7.83 px/mm`, grain length mean `52.515 mm`.
- M037e: lịch sử `7.22 px/mm`, grain length mean `62.791 mm`.
- M038e: lịch sử `7.18 px/mm`, grain length mean `57.457 mm`.

Chạy detector notebook hiện tại lại trên raw images:

| Sample | Lịch sử XLSX | Hiện tại | Ratio |
| --- | ---: | ---: | ---: |
| M029c | 7.83 | 74.1541 | 9.4705× |
| M037e | 7.22 | 75.2447 | 10.4217× |
| M038e | 7.18 | 80.4185 | 11.2003× |

Vì vậy phát hiện 9–11× trong review cũ được **xác minh độc lập**, nhưng nó là lỗi feature lịch sử/XLSX và drift giữa detector notebook với dữ liệu cũ; không mô tả detector AI_SERVICES hiện tại. Ba scale 7.x tạo kích thước hạt dài 52–63 mm trong cốc 22.8 mm, vô lý vật lý.

Evidence: `historical_removed_rows.csv`, `removed_scale_audit.json`, `removed_scale_overlay_*.jpg`.

#### 3.1.5. Scale giải thích bao nhiêu cho 450→802?

Với count tỷ lệ xấp xỉ `pixels_per_mm³`, ratio cần thiết là:

```text
(802 / 450)^(1/3) = 1.21242
```

Tức `88.84 px/mm` của lần chạy thực tế chỉ cần cao hơn scale thật `21.24%`; scale tương ứng là `73.27 px/mm`. Khi nhân kích thước hạt hiện tại với 1.21242, mean length/width trở thành khoảng `6.09 × 2.09 mm`, hợp lý hơn nhiều so với `5.02 × 1.72 mm`. Đây là bằng chứng định lượng mạnh rằng scale **có khả năng** giải thích toàn bộ lỗi, nhưng chưa phải chứng minh cho ảnh cụ thể vì raw `20260920_182343[1].jpg` không còn local.

### 3.2. Giả định ellipsoid

- `AI_SERVICES/src/rice_ai/vision/ellipsoid_geometry.py:26-27,129-131` vẫn dùng `THICKNESS_RATIO['hat_nguyen']=0.85` và `c_px=a_px*k`.
- `CODE/modules/ellipsoid_geometry.py:26-27,131-133` hiện dùng `k=0.80` và `c_px=b_px*k`. Thay đổi từ `a` sang `b` xảy ra ở commit `ec22fe3`; k đổi 1.0→0.80 ở `593574c`.
- `CODE/modules/dataset_extractor.py:45,509` dùng module `CODE/modules/ellipsoid_geometry.py`, nhưng final dataset/artifact được tạo trước thay đổi.

Kết quả trên dữ liệu:

| Nguồn | Dòng hợp lệ | thickness > width | Median thickness/width | Median thickness/length |
| --- | ---: | ---: | ---: | ---: |
| CSV hiện tại | 245 | 245 (100%) | 2.6906 | 0.8500 |
| XLSX lịch sử | 254 | 254 (100%) | 2.6816 | 0.8500 |

Cả 245/245 CSV và 254/254 XLSX đều thỏa `thickness≈0.85×length`, chứng minh dữ liệu được tạo bằng công thức cũ. Với hạt thon dài, thickness phải nhỏ hơn breadth/width; nghiên cứu paddy miền nam Ấn Độ báo length 6.26–8.21 mm, breadth 2.24–3.37 mm, thickness 1.63–2.24 mm. Nghiên cứu tám giống rice khác báo length 5.29–6.99, width 2.52–3.10, thickness 1.88–2.13 mm. Nguồn: [Varietal distinctness in physical and engineering properties of paddy](https://pmc.ncbi.nlm.nih.gov/articles/PMC6423265/), [Evaluation of physical properties of rice](https://pmc.ncbi.nlm.nih.gov/articles/PMC6145214/).

Kết luận vật lý:

- `0.85` không nhất thiết sai nếu là tỷ lệ **thickness/width** được đo cho đúng giống; cái sai cấu trúc là nhân với trục dài `a`.
- Công thức cũ làm volume một hạt quá lớn, nên làm count vật lý **thấp đi**. Nó không gây 802; nó che bớt overestimate.
- Công thức notebook mới `c=0.80b` đúng hướng vật lý, nhưng dataset/scaler/model vẫn mang semantic cũ. Trên 245 dòng, volume mới ước tính chỉ bằng median `29.74%` volume cũ; nếu các thành phần khác giữ nguyên, count tăng khoảng `3.36×`. Đây là train/runtime feature drift nghiêm trọng.

### 3.3. Packing fraction 0.82, 0.62 và 0.55

Mã nguồn AI_SERVICES xác nhận hai mục đích khác nhau:

- `AI_SERVICES/src/rice_ai/estimation/geometry.py:14,36`: physical geometry dùng mặc định `0.82`.
- `AI_SERVICES/src/rice_ai/estimation/feature_schema.py:23,54`: feature 11 `Estimated_Total_Seeds_Hybrid` dùng contract `0.62`.
- `AI_SERVICES/src/rice_ai/settings.py:65-66` cấu hình riêng hai giá trị.

Tuy nhiên notebook chính hiện tại không còn dùng 0.82. Cell 3 và output Cell 11 dùng `PACKING_FRACTION=0.55`; commit `ec22fe3` xảy ra sau review cũ. Trường hợp 802 là `phi=0.55`, không phải 0.82.

Tái tính độc lập trên 51 dòng holdout lưu ở `models/extra_trees/predictions_test.csv`, join theo `Sample_ID` vì `source_row_id` đã stale:

| Phi | MAE | RMSE | Bias (pred−actual) | MAPE | R² |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.82 | 77.1373 | 86.8465 | +44.8627 | 57.0123% | −0.2511 |
| 0.62 | 47.2353 | 64.3738 | −6.4510 | 32.2031% | 0.3126 |
| 0.55 | 46.4314 | 66.5081 | −24.3529 | 27.9462% | 0.2662 |

Hai hàng 0.82/0.62 khớp report cũ, nên phát hiện đó được xác minh. Nhưng không được chọn phi chỉ vì metric tốt hơn trên bảng đã nhiễm lỗi scale/ellipsoid. Effective phi theo từng dòng có median `0.5398`, nhưng range `0.2173–9.0721`; range vô lý cho thấy chưa thể dùng bảng này để hiệu chuẩn vật lý.

Bằng chứng vật lý bên ngoài:

- Nghiên cứu paddy báo porosity khoảng 46–54%, tương ứng phần thể tích hạt ngoài khoảng 46–54%; nghiên cứu raw paddy khác báo porosity 58.01–62.84%, tương ứng packing khoảng 37–42%. Nguồn: [Some physical properties of paddy and rice](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/jsfa.2740230204), [Physical Properties of Raw and Parboiled Paddy](https://www.sciencedirect.com/science/article/pii/S1537511004000984).
- Một paper có hệ số `0.82`, nhưng đó là correction factor giữa volume hình học và volume theo density của **một hạt**, không phải packing fraction của cả cốc: [Mechanical and Processing Properties of Rice Grains](https://www.mdpi.com/2071-1050/12/2/552).

Vì vậy `0.55` nằm sát/nhỉnh hơn mép trên của một số dải công bố; `0.62` và đặc biệt `0.82` chưa có bằng chứng là packing fraction đúng cho cách đổ lúa của dự án. Phải đo bulk density/true density hoặc đo volume-count trên đúng giống, độ ẩm, cốc và quy trình lắc/đổ.

Counterfactual mẫu thực tế với cùng volume hạt 6.618:

- phi 0.55 → 802 (đã chạy)
- phi 0.62 → 905
- phi 0.82 → 1196
- phi cần để ra 450 → 0.3084, quá thấp để coi là cách sửa hợp lý; điều này chỉ ra mẫu số grain volume/scale cũng sai.

### 3.4. Regression học vision hay học cân nặng/container?

#### Model thực sự được nạp

- `AI_SERVICES/artifacts/manifest.json` trỏ model legacy root `LINEAR_REGRESSION_MODEL/models/best_tree_ensemble_model.joblib` + root `scaler.joblib`.
- `AI_SERVICES/src/rice_ai/settings.py:181-203` mặc định trỏ `LINEAR_REGRESSION_MODEL/models`; loader đi vào legacy adapter tại `regression_loader.py:272-321`.
- Notebook mẫu 802 lại chọn bundle khác: `LINEAR_REGRESSION_MODEL/models/extra_trees` (Cell 13). Đây là thêm một deployment divergence.

Feature importance tính lại trực tiếp:

| Model | Container first 8 | Surface 3 | Grain morphology 20 |
| --- | ---: | ---: | ---: |
| AI_SERVICES legacy production | 99.9796% | 0.0067% | 0.0136% |
| Notebook `extra_trees/model.joblib` | 99.9389% | 0.0577% | 0.00345% |

Trong notebook bundle, top importance là `Container_Height_mm 32.88%`, `Inner_Diameter_mm 27.26%`, `Weight_g 15.61%`, `Bulk_Rice_Volume_mm3 14.81%`. Pearson `Weight_g`–`Actual_Count` trên 245 dòng hiện tại là `0.998638`.

Baseline độc lập, ExtraTrees cùng random seed, 5-fold `GroupKFold` theo M###:

| Input | MAE | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: | ---: |
| Full 31 | 3.1341 | 4.8247 | 3.3335% | 0.99696 |
| Weight + 7 container | 3.0622 | 4.7578 | 3.2350% | 0.99705 |
| Weight only | **2.7157** | **4.3656** | **2.8766%** | **0.99751** |

Paired bootstrap theo 57 nhóm cho chênh lệch MAE `(baseline8 − full31)` là `−0.0508` hạt, 95% CI `[-0.2106, +0.1154]`, chứa 0. Full-31 không tốt hơn baseline8 có ý nghĩa; trong test này nó còn kém weight-only.

Hai loại cốc trong 245 dòng:

- 17.8×33.9 mm: 121 dòng, count 25–145.
- 22.8×37.1 mm: 124 dòng, count 150–305.
- Overlap count range: 0.

Do đó kích thước cốc là proxy gần như hoàn hảo cho miền target. Mẫu thực tế có cốc `32.7×48.7 mm`, count≈450 và weight 10.7 g—đều ngoài miền train (max count 305, max weight 8.08). ExtraTrees không ngoại suy theo quy luật tuyến tính; output 212 là OOD failure hợp lý, không phải bằng chứng pipeline tốt.

### 3.5. Chất lượng dữ liệu training và split

| Kiểm tra | CSV hiện tại | XLSX/artifact lịch sử |
| --- | ---: | ---: |
| Tổng dòng | 276 | XLSX 285 |
| FOUND / complete 31+target | 245 | XLSX 254 |
| MISSING | 31 | 31 |
| Duplicate Sample_ID | 0 | Artifact 254 rows có M017d lặp 2 lần (253 unique) |
| Whole_Grains_Count < 8 | 99/245 (40.4%) | Không tính lại trên artifact-only table |

Tám Sample_ID có trong artifact/XLSX nhưng không còn ở CSV hiện tại: `M027c, M029a, M029c, M029d, M029e, M037e, M038c, M038e`. Dataset CSV bị `.gitignore`, nên Git không cung cấp lịch sử/hash để giải thích ai/đợt nào xóa các dòng này. Chỉ 25/51 `source_row_id` trong saved holdout còn trỏ đúng vị trí hiện tại; vì vậy audit phải join bằng `Sample_ID`.

Split verification:

- Trainer hiện tại đặt `SPLIT_MODE="GROUPED"`, outer `GroupShuffleSplit`, inner `GroupKFold` trong notebook (cells quanh dòng JSON 531–547).
- Tự chạy lại GroupShuffleSplit: 45 train groups, 12 test groups, overlap 0.
- Saved bundle `extra_trees`: 45 train groups, 12 test groups, overlap 0.
- AI_SERVICES legacy production dùng row split: 56 train groups, 33 test groups, **32 group overlap**. Metric MAE≈0.34 của split này bị leakage giữa các ảnh a/b/c/d/e cùng mẫu vật lý và không được dùng làm bằng chứng generalization.

### 3.6. Đối chiếu mẫu thực tế 450→802

Output đã lưu trong `CODE/RICE_VISION_MAIN_PIPELINE.ipynb` cung cấp:

| Stage | Giá trị |
| --- | ---: |
| Image | `20260920_182343[1].jpg` |
| Inputs | inner diam 32.7 mm; true rice diam 22.0 mm; container height 48.7 mm; empty 23.3 mm; weight 10.7 g; phi 0.55 |
| Container | inner 2905 px; outer 3172 px; 88.84 px/mm; rice height 25.4 mm |
| Bulk volume | 9655.37 mm³ |
| SAHI | 86 detections |
| Cleaner | 86/86 retained |
| CNN | 21 whole, 65 defective |
| Size filter | 21/21 retained; lower bound −0.81 mm² (`non_positive_lower_bound`) |
| Mean dimensions | 5.02×1.72 mm; thickness implied 1.376 mm with k=0.80 |
| Mean grain volume | 6.618 mm³ |
| Physical | 802 |
| Hybrid feature phi=0.62 | 905 |
| Regression ExtraTrees | 212 |

Từ output lưu sẵn, số học chỉ cho thấy có hai khả năng: tử số `V_bulk` quá lớn hoặc mẫu số `V_grain` quá nhỏ:

```text
round(9655.37 × 0.55 / 6.618) = 802
error = 802 − 450 = 352 = +78.22%
Nếu giữ V_bulk hình trụ: V_grain cần = 11.801 mm³
```

Khi chưa có raw, audit đầu tiên xếp `V_grain` là nghi vấn chính. Bằng chứng raw ở mục kế tiếp bác bỏ kết luận đó cho **mẫu cụ thể này**: `6.618 mm³` kết hợp với thể tích nón cụt vẫn cho 452; lỗi chi phối là tử số hình trụ. Giá trị volume hạt và CNN vẫn cần calibration độc lập, nhưng không được dùng 11.801 như một target để tune.

Giới hạn còn lại cho mẫu cụ thể:

- Raw đã có; `pipeline_inference_results/{crops,reports}` gốc từ Colab vẫn không có, nhưng audit CPU tái tạo được detector/SAHI/cleaner/geometry và ánh xạ 21 record từ output lưu.
- Local venv vẫn không import được TensorFlow (`ModuleNotFoundError: tensorflow.python`), nên chưa có **CNN prediction E2E mới**; không claim CNN local pass.

#### 3.6.1. Đối chiếu trực tiếp với ảnh raw mẫu 450 → 802

##### 3.6.1.1. Danh tính fixture và detector

Ảnh raw người dùng cung cấp được lưu read-only trong thư mục evidence dưới tên `actual_case_raw.jpg`, kích thước `8160×6120`, SHA-256 `b51a23add18ca7302de6e4bb4aaa48e6385ab1984729588505e17409d10d1638`.

Lệnh:

```powershell
AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_actual_raw.py
```

| Detector | 5 lần chạy px/mm | Rim trong px | Kết luận |
| --- | ---: | ---: | --- |
| `CODE/modules/container_detector.py` (notebook thật) | `88.8384748` mỗi lần | `2905` | Khớp output notebook gốc `88.84`; range/std = `0` |
| `AI_SERVICES/src/rice_ai/vision/container_detector.py` | `28.0597232` mỗi lần | `918` | Bắt nhầm rim/vùng trong; không thể thay thế detector notebook |

Detector notebook có `outer_confidence=0.1888–0.2642` trong 5 lần nhưng vẫn trả success; đây là lỗi QC/API riêng. Tuy nhiên không làm sai scale của lần chạy notebook đang điều tra: `88.8385 / 88.84 = 0.999983`, và counterfactual count từ scale-only vẫn là `802`.

##### 3.6.1.2. SAHI, cleaner và 21 hạt CNN đã lưu

Lệnh:

```powershell
AI_SERVICES/.venv/Scripts/python.exe CODE/reports/pipeline_investigation_20260921/audit_actual_segmentation.py
```

CPU chạy `221` SAHI slices với đúng `slice_size=640`, `overlap=0.25`, `confidence model=0.7`, `post=0.5`; mất `143.43 s`. Kết quả tái tạo:

- SAHI raw: `86`, bằng output notebook đã lưu `86`.
- Sau hai bước cleaner: `86/86`, không hạt nào bị loại.
- Geometry: `86/86` hợp lệ khi dùng scale notebook vừa tái tạo.
- CNN không thể rerun local do `ModuleNotFoundError: tensorflow.python`; vì thế không claim một CNN prediction mới.
- Tuy nhiên 20 giá trị volume nhỏ nhất/lớn nhất Cell 11, cộng với giá trị trung tâm suy ra `6.440988`, ánh xạ được **đúng 21/21** record deterministic của lần rerun. Mean tái dựng `6.618097 mm³`, cho physical `802`, khớp notebook.
- Tập 21 này có volume `2.985–14.897 mm³`, length `3.321–6.999 mm`, width `1.400–2.667 mm`. IQR-area cho `Q1=4.236`, `Q3=7.601`, lower bound `−0.810 mm²`; chính vì cận âm, size filter giữ toàn bộ. Điều này xác minh trạng thái `non_positive_lower_bound` đã lưu, không chứng minh mọi hạt nhỏ là sai.

Evidence trực quan: `actual_case_sahi_overlay.jpg` (xanh: 21 record được tái dựng từ output CNN cũ; đỏ: record khác) và `actual_case_reconstructed_cnn_whole_gallery.jpg`.

##### 3.6.1.3. Counterfactual nón cụt — **bị bác bỏ bởi xác nhận vật chứa**

Script `audit_actual_geometry.py` từng kiểm tra một giả định: nếu khoang trong thuôn tuyến tính, nội suy từ `32.7 mm` tại rim, `22.0 mm` tại mặt lúa và chiều cao `48.7 mm` sẽ suy ra đáy `10.3356 mm`, cho `V=5440.85 mm³` và `N≈452`. Sự khớp này là một **counterfactual**, không phải đo trực tiếp.

Người dùng xác nhận ly là hình trụ, có đường kính giữ nguyên từ miệng đến đáy. Vì vậy giả định nón cụt là sai cho vật chứa này; kết quả `452` không được dùng để giải thích lỗi, không được triển khai trong code và chỉ được giữ lại như evidence về một giả định đã bị loại bỏ.

Với hình trụ được xác nhận, `V_bulk=9655.37 mm³` là đúng theo các inputs hiện có. Giữ `phi=0.55`, count 450 đòi hỏi `mean_grain_volume=11.801 mm³`, trong khi 21 record CNN map từ output cũ có `6.618097 mm³`. Hướng điều tra/sửa đúng chuyển sang: kiểm chứng selection CNN, mask/cleaner, phép đo ellipsoid và calibration volume hạt; không đổi mô hình thể tích ly.

Evidence: `actual_geometry_counterfactual.json` (giữ nguyên với trạng thái counterfactual invalidated), `actual_raw_audit.json`, `actual_segmentation_audit.json` và các overlay/gallery.
### 3.7. Đối chiếu trực tiếp với REVIEW.md cũ

| Claim cũ | Kết luận độc lập |
| --- | --- |
| phi 0.82 MAE 77.14, bias +44.86, MAPE 57.01% | **VERIFIED**, tái tính từ holdout; nhưng notebook hiện dùng 0.55 |
| scale M029c/M037e/M038e lệch 9–11× | **VERIFIED** bằng XLSX 285 dòng + raw images + detector notebook hiện tại |
| scale drift nói chung | **VERIFIED cho `CODE/modules`** trên sample historical, nhưng trên raw 450 notebook tái tạo đúng `88.84`; **AI_SERVICES sai trực tiếp** (`28.06 px/mm`); hai bản đã phân kỳ |
| thickness > width toàn bộ dữ liệu | **VERIFIED**: 245/245 CSV và 254/254 XLSX |
| 99.89% importance từ metadata | Hướng kết luận **VERIFIED**, số mới là 99.9796% production legacy và 99.9389% notebook bundle |
| 285 rows / 254 valid / duplicate M017d | **VERIFIED cho XLSX/artifact lịch sử**; CSV hiện tại đã drift thành 276/245 và không còn duplicate |

## 4. Câu hỏi còn bỏ ngỏ

1. Cần đo trực tiếp profile/đường kính đáy **bên trong** của loại ly này (hoặc ảnh chuẩn cạnh thước) để xác nhận giả định nón cụt tuyến tính. Counterfactual 452 là bằng chứng mạnh nhưng không thay thế calibration hình học.
2. Cần lưu `/content/pipeline_inference_results/` cho lần 802 hoặc rerun CNN trong Colab đã pin để kiểm chứng độc lập 21 whole / 65 defective, confidence, crop trước/sau cleaner và tác động từng stage.
3. Cần đo bằng thước kẹp tối thiểu 30 hạt cùng giống: length, width, thickness; đồng thời cân 100/1000 hạt. Không có ground truth này thì chưa thể đánh giá bias còn lại của mask/cleaner/CNN/ellipsoid.
4. Cần ghi lại cách đổ/lắc, độ ẩm, bulk density và true density để xác định packing fraction; chưa có bằng chứng nội bộ nào chứng minh 0.82 là packing của cốc.
5. Cần quyết định source-of-truth dữ liệu: CSV 276 dòng hay XLSX 285 dòng; phải version/hash dataset thay vì ignore toàn bộ final dataset.
6. Cần xác định deployment đích dùng model legacy root hay bundle `models/extra_trees`; hiện AI_SERVICES và notebook dùng hai bundle khác nhau. Full local CNN E2E vẫn NOT VERIFIED vì TensorFlow hỏng.

## 5. Đề xuất sửa lỗi cụ thể theo ưu tiên

### P0 — Audit và hiệu chuẩn nguồn `mean_grain_volume` trước khi đổi hằng số

Files:

- `CODE/modules/grain_segmenter.py`
- `CODE/modules/grain_crop_cleaner.py`
- `CODE/modules/grain_classifier.py`
- `CODE/modules/grain_size_filter.py`
- `CODE/modules/ellipsoid_geometry.py`
- các module tương ứng trong `AI_SERVICES/src/rice_ai/vision/`
- `CODE/RICE_VISION_MAIN_PIPELINE.ipynb`, Cell 5–11

Thay đổi đề xuất:

- Giữ mô hình `V_bulk` hình trụ cho loại ly đã được xác nhận; không triển khai nón cụt.
- Lưu artifact từng stage cho mỗi inference: polygon/mask SAHI, crop raw, crop sau clean step 1/2, nhãn và confidence CNN, ellipse và volume. Có thể review một record bằng `grain_id`.
- Thay bộ lọc small-outlier hiện chỉ dùng lower IQR âm bằng QC hai phía, với ngưỡng physical/domain đã được hiệu chuẩn hoặc rule `min_area`/`min_length` rõ ràng. Không tự chọn threshold để ép case 450 đúng.
- Tạo protocol đánh nhãn thủ công một tập crop: whole/defective/fragment/merged/background. Từ đó đánh giá precision/recall CNN và lỗi segment/clean, thay vì chỉ tin label CNN.
- So sánh volume ellipsoid từ crop với đo thước kẹp/ảnh chuẩn của >=30 hạt cùng giống. `k=0.80` chỉ được chốt từ đo physical, không tune theo Actual_Count.

Kiểm chứng:

- Với fixture raw này, tái tạo deterministic `86` SAHI và 21 selected records, nhưng không hardcode count 450.
- Báo distribution mean/median volume trước/sau từng gate, số hạt reject theo từng lý do và ảnh gallery audit.
- Trên tập calibration độc lập, đo MAE/bias/MAPE physical estimate; chỉ sửa threshold/ellipsoid sau khi metric cải thiện trên holdout.
### P1 — Hợp nhất và kiểm định scale trước mọi tuning khác

Files:

- `CODE/modules/container_detector.py`
- `AI_SERVICES/src/rice_ai/vision/container_detector.py`
- notebook Cell 4 / API container stage

Thay đổi đề xuất:

- Chỉ giữ một implementation dùng chung; không copy hai bản.
- Truyền seed cố định/RNG từ config cho RANSAC.
- Fail explicit khi `outer_confidence`/`inner_confidence` thấp; không trả prediction khi rim chưa được nghiệm thu.
- Dùng marker/ruler/đường kính chuẩn cùng mặt phẳng hạt để có ground-truth px/mm; lưu overlay + expected/observed diameter.
- Thêm OOD/QC gate cho scale theo setup camera, không dùng giá trị lịch sử làm ground truth tuyệt đối.

Kiểm chứng:

- Bộ fixture ≥20 ảnh, nhiều cốc/ánh sáng/camera height; error `|px/mm_pred−px/mm_manual|/manual` báo median, P95 và max.
- Cùng ảnh 20 lần phải cho cùng output bitwise hoặc tolerance <0.01%.
- M001a/M029c/M037e/M038e phải fail QC hoặc đạt tolerance, không silent success.

### P2 — Hiệu chuẩn ellipsoid bằng đo trực tiếp rồi tái sinh dataset/model

Files:

- `CODE/modules/ellipsoid_geometry.py`
- `AI_SERVICES/src/rice_ai/vision/ellipsoid_geometry.py`
- `CODE/modules/dataset_extractor.py`
- `DATASET_BUILDER/4_Final_Dataset/*`
- `LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb`

Thay đổi đề xuất:

- Đồng nhất `c = k × b`, trong đó k lấy từ caliper của đúng giống/độ ẩm; không tune k theo Actual_Count.
- Sửa docstring `c/a` còn sót thành semantic đúng.
- Regenerate toàn bộ 31 features bằng cùng commit/module với runtime, sau đó retrain/re-export model+scaler.
- Lưu version/hash geometry contract trong bundle.

Kiểm chứng:

- MAE length/width/thickness và volume so với caliper/khối lượng-density trên tập calibration.
- Invariant `length ≥ width ≥ thickness > 0` đạt gần 100%, ngoại lệ phải review ảnh.
- Train/runtime parity test chạy cùng crop phải cho 31 features giống nhau.

### P3 — Tách rõ packing, shape correction và calibration

Files:

- `AI_SERVICES/src/rice_ai/estimation/geometry.py`
- `AI_SERVICES/src/rice_ai/estimation/feature_schema.py`
- notebook configuration/report cells

Thay đổi đề xuất:

- Đặt tên riêng `bulk_packing_fraction` và `single_grain_shape_correction`; không dùng 0.82 cho hai khái niệm.
- Chỉ chốt phi sau khi scale/volume đã đúng; lưu giống, độ ẩm và protocol đổ/lắc kèm calibration.
- Không chọn phi bằng cách minimize target error trên cùng tập dùng đánh giá.

Kiểm chứng:

- Holdout theo physical sample và leave-one-cup-out; báo MAE, bias, MAPE và interval theo cốc/protocol.
- Phi đo từ `bulk_density/true_density` phải nhất quán với phi suy ra từ known count trong sai số định trước.

### P3 — Chặn regression ngoài miền và kiểm tra giá trị vision

Files:

- `AI_SERVICES/src/rice_ai/models/regression_loader.py`
- `AI_SERVICES/src/rice_ai/estimation/regression.py`
- notebook regression cells
- trainer notebook

Thay đổi đề xuất:

- Lưu min/max hoặc distribution của 31 features, cup IDs, target/weight range trong bundle; reject/warn OOD trước `predict`.
- Với ExtraTrees, không quảng bá extrapolation cho count/weight/cup ngoài train.
- Dùng GroupKFold + leave-one-cup/configuration-out; luôn so full31 với weight-only, container8 và vision-only.
- Chỉ giữ morphology features nếu cải thiện có CI loại 0 trên dữ liệu ngoài nhóm/cốc.

Kiểm chứng:

- Mẫu cốc 32.7×48.7, weight10.7 phải trả OOD rõ thay vì một số 212 không cảnh báo.
- Full31 phải thắng baseline theo MAE với paired group bootstrap CI; nếu không, không claim vision contribution.

### P3 — Khóa data governance

Files:

- `DATASET_BUILDER` manifest/schema
- trainer data loader
- `.gitignore`/artifact storage policy phù hợp

Thay đổi đề xuất:

- Chọn một dataset canonical, có content hash/version; XLSX chỉ export, không phải source song song.
- Không dùng `source_row_id` làm identity; dùng unique immutable `Sample_ID` + capture ID.
- QC explicit cho MISSING, duplicate, `<8 whole grains`, impossible dimensions, scale confidence và feature-contract version.

Kiểm chứng:

- CI/audit fail nếu CSV/XLSX row count/hash khác, duplicate ID, group overlap hoặc feature generator version mismatch.

### P4 — Tạo fixture E2E cho chính mẫu lỗi

Files/artifacts:

- raw fixture 450 hạt (không dùng ảnh đã annotate)
- expected input JSON
- stage outputs/overlays/crops/31-feature vector
- test runner CPU-bounded hoặc Colab pinned

Thay đổi đề xuất:

- Sửa/pin TensorFlow/Keras và scikit-learn đúng version; không dùng môi trường 1.9.1 cho artifact tạo bằng 1.6.1 nếu chưa xác nhận compatibility.
- Lưu log từng stage và cả physical/regression/fusion method; không gọi một trong hai là “final” mơ hồ.

Kiểm chứng:

- Test regression của chính bug: raw image + inputs phải tái tạo stable stage outputs.
- Báo error budget từng stage; acceptance đầu tiên nên là scale và physical volume trước khi yêu cầu final count tolerance.

## 6. Tệp tái tạo trong thư mục này

- `investigate.py`: tabular/model/packing/data-quality audit và AI_SERVICES scale audit.
- `audit_main_scale.py`: detector thật của notebook, 6 ảnh + repeatability.
- `audit_removed_scale.py`: ba historical scale outlier từ XLSX.
- `audit_actual_case.py`: trích output notebook và tái tính 450→802.
- `audit_actual_raw.py`: detector notebook/API trên raw fixture, 5 repeat mỗi bản.
- `audit_actual_segmentation.py`: SAHI/cleaner/geometry CPU và tái dựng 21 record CNN từ output lưu.
- `audit_actual_geometry.py`: counterfactual hình trụ/nón cụt dùng đúng sample inputs.
- `actual_case_raw.jpg`: raw fixture có checksum; `actual_case_*overlay*.jpg` và `*gallery*.jpg`: evidence trực quan.
- `COMMANDS.md`: lệnh, pass/fail và giới hạn môi trường.
- `tabular_audit.json`, `tabular_audit.log`: model/baseline/data-quality evidence.
- `scale_audit.json`, `main_scale_audit.json`, `removed_scale_audit.json`: scale evidence.
- `actual_case_arithmetic.json`: arithmetic mẫu cụ thể.
- CSV predictions/comparisons và các overlay JPG tương ứng.

## Kết luận cuối

Không có bằng chứng rằng “phi 0.82” là nguyên nhân trực tiếp của output 802, vì lần chạy đó thực tế dùng phi 0.55. Người dùng xác nhận ly hình trụ; vì vậy giả thuyết nón cụt và kết quả 452 trước đó bị loại bỏ. Scale detector notebook cũng tái tạo đúng cho case này. Với `V_bulk=9655.37 mm³` và phi 0.55, mismatch còn lại nằm ở mean volume hạt: 21 record CNN chọn cho `6.6181 mm³`, trong khi count 450 đòi `11.801 mm³`. Cần điều tra/hiệu chuẩn segment → cleaner → CNN selection → ellipsoid bằng crop có nhãn và đo physical, không tune theo target 450. Regression trả 212 vì OOD và metadata-dominated; AI_SERVICES bắt nhầm rim trên raw fixture, nên detector/runtime vẫn phải hợp nhất và có QC.