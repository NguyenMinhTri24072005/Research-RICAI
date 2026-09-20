# Kế hoạch module lọc hạt nhỏ bất thường sau CNN

Status: READY FOR IMPLEMENTATION — tham số thử nghiệm, chưa xác nhận cải thiện độ chính xác
Created: 2026-09-20
Target agent: Gemini 3.1 Pro High
Target notebook: CODE/RICE_VISION_MAIN_PIPELINE.ipynb
Scope: module lọc, cell đo/lọc và báo cáo liên quan; không train model, không đổi đầu vào SAHI.
Các đường dẫn trong tài liệu tính từ repository root. Dùng tiêu đề/marker để tìm cell, không dùng index cố định.

## 1. Mục tiêu và baseline đã kiểm tra

Người dùng đã khôi phục SAHI chạy trên ảnh gốc: USE_CROPPED_CONTAINER=False, slice_size=640, overlap_ratio=0.25. Giữ cấu hình này cùng các thông số mẫu do người dùng nhập.

Vấn đề cần xử lý: một số mảnh nhỏ bị CNN nhận là hạt nguyên và đi vào phép đo. Lọc các trường hợp nhỏ bất thường khỏi tập dùng ước lượng vật lý, đồng thời giữ đủ bằng chứng để kiểm tra việc loại nhầm. Bộ lọc không sửa nhãn CNN và không khẳng định mọi vùng bị loại đều là hạt khuyết tật.

Bằng chứng repository tại lúc lập plan:

- Notebook có 15 cells. Bước 7 tạo whole_grains, defective_grains, all_classified_grains và whole_ratio.
- Bước 8 đo bằng compute_single_grain_metrics, lưu whole_grain_metrics_list, whole_records, whole_volumes_list. Các record hiển thị đã được round 2 chữ số.
- Bước 8 đang fallback sang cleaned_grains nếu không có whole_grains. Nhánh này có thể dùng hạt khuyết tật như hạt nguyên trong tính vật lý; cần bỏ trong luồng đo mới.
- Bước 8 đang bỏ qua lỗi đo bằng except Exception: continue, không ghi ID/lý do.
- evaluate_batch_uniformity trong CODE/modules/uniformity_evaluator.py dùng IQR HAI PHÍA trên volume, trả mean_clean và tỷ lệ đồng đều; không trả danh sách ID/mask giữ/loại.
- Bước 9 đang ghép whole_grain_metrics_list[i] với grains_to_measure_3d[i]. Nếu một hạt đo lỗi, index có thể lệch.
- REGRESSION_FEATURES::assemble_regression_features sử dụng measurements chưa round, whole_records để lấy count, và mean volume trước lọc cho hybrid.
- CODE/modules/ellipsoid_geometry.py::compute_single_grain_metrics trả area_mm2 từ diện tích contour lớn nhất, đổi đơn vị bằng pixels_per_mm². Tái sử dụng đúng phép đo này; không tự thay bằng bbox area.
- Hợp đồng regression hiện có chưa chứa bước area-IQR mới. Không âm thầm đổi tập hạt tạo 31 features rồi gọi đó là đúng contract train.
- Working tree có nhiều sửa đổi trước phiên này. Giữ nguyên, không reset/restore/stash hoặc ghi đè notebook từ HEAD.

## 2. Quyết định thiết kế

Luồng mới:

SAHI ảnh gốc → clean → CNN → đo từng hạt được CNN chọn nguyên → area-IQR một phía → thống kê vật lý.

Từ phép đo tạo hai tập rõ ràng:

| Tập | Ý nghĩa | Nơi sử dụng |
| --- | --- | --- |
| Raw measured whole grains | CNN chọn nguyên, đo hợp lệ, chưa qua area-IQR mới | Đặc trưng/hồi quy legacy; baseline đối chiếu |
| Filtered measured whole grains | Tập raw trừ các outlier nhỏ theo diện tích | Thống kê/ước lượng vật lý sau lọc, hình minh họa |
| Measurement invalid | Đo lỗi hoặc kích thước không hợp lệ | Báo cáo lỗi; không dùng tính toán |
| Small outlier | Đo hợp lệ nhưng diện tích dưới ngưỡng | Báo cáo review; không đổi sang defective_grains |

Mặc định bật bộ lọc cho nhánh vật lý. Nhánh regression legacy vẫn dùng raw để giữ semantic đã train. Báo cáo phải ghi rõ điều này, không tuyên bố regression cũng đã được sửa nhờ bộ lọc.

Không đưa bảng features sau lọc vào scaler/model cũ trong implementation này. Chỉ xuất bảng đối chiếu features. Sau khi người dùng kiểm chứng bộ lọc, việc sinh lại dataset/train và áp dụng đồng nhất cho regression là công việc riêng.

## 3. Files dự kiến tạo/sửa

- Tạo CODE/modules/grain_size_filter.py: hàm lọc thuần, chỉ phụ thuộc NumPy/thư viện chuẩn.
- Sửa CODE/RICE_VISION_MAIN_PIPELINE.ipynb: import/reload module, config, tách cell đo và cell filter, báo cáo.
- Tạo CODE/tests/test_grain_size_filter.py: unit test thuật toán.
- Tạo CODE/tests/test_main_pipeline_size_filter.py: kiểm thử kết nối notebook, ID và regression parity.
- Chỉ cập nhật CODE/tests/test_main_pipeline_regression.py nếu chữ ký hàm thêm dữ liệu đầu vào tường minh; giữ các phép kiểm model/scaler đang có.
- Tạo CODE/reports/grain_size_filter/verification.md sau khi test: lệnh, kết quả thật, giới hạn kiểm chứng.
- Không sửa AI_SERVICES, CAPTURE_APP, training notebook, model/scaler/config bundle hoặc PROJECT_STATUS.
- Không thay thuật toán của uniformity_evaluator.py hay ellipsoid_geometry.py vì chúng được các luồng khác dùng chung.

## 4. API module và hợp đồng dữ liệu

Hàm public:

```python
def filter_small_grain_outliers(
    measurements,
    *,
    enabled=True,
    iqr_multiplier=1.5,
    min_samples=8,
):
    ...
```

measurements là sequence dict theo đúng thứ tự ứng viên CNN. Mỗi dict chứa:
- candidate_index: chỉ số duy nhất trong danh sách whole_grains ở lần đo này.
- grain_id: ID gốc (grain_id → index → candidate_index). Không renumber sau loại.
- length_mm, width_mm, thickness_mm, area_mm2, volume_mm3: số đo chưa round.
- measurement_error: None khi hợp lệ; chuỗi mô tả khi đo thất bại. Khi thất bại, số đo thiếu dùng None, không dùng 0.

Module không nhận ảnh, không gọi CNN/YOLO, không đọc file, không vẽ hình, không đo lại hạt và không mutate input. Ảnh và bbox giữ trong bảng ghép ở notebook bằng candidate_index; không dùng grain_id làm khóa duy nhất vì ID nguồn có thể trùng.

Kết quả là dict:
- kept_indices, rejected_indices, invalid_indices: index vị trí trong measurements, giữ thứ tự đầu vào.
- decisions: một dict cho mỗi hàng, có candidate_index, grain_id, decision, reason, area_mm2.
- summary: enabled, method='lower_area_iqr_v1', status, iqr_multiplier, min_samples, total_count, valid_count, kept_count, rejected_count, invalid_count, q1, q3, iqr, lower_bound, warnings.
- q1/q3/iqr/lower_bound là float chưa round hoặc None khi không tính; không ghi NaN vào JSON.

decision gồm keep / reject / invalid. Reason gồm kept, disabled, insufficient_samples, degenerate_iqr, nonpositive_lower_bound, small_area_outlier, measurement_error, invalid_metrics.
Giữ reason cụ thể của measurement_error ở field riêng để truy vết.

## Phase 0 — Ghi nhận baseline

1. Chạy git status --short và git diff --stat; đọc source notebook hiện tại.
2. Kiểm tra AGENTS.md nếu có. Định vị cell bằng Bước 7, Bước 8, REGRESSION_FEATURES, REGRESSION_REPORT.
3. Xác nhận USE_CROPPED_CONTAINER=False và giữ nguyên IMAGE_PATH, cân nặng, kích thước ly, PACKING_FRACTION hiện tại.
4. Chụp baseline trong bộ nhớ khi test: source cell SAHI/CNN và 31 features/prediction trên fixture synthetic đã có.
5. Dùng AI_SERVICES/.venv/Scripts/python.exe nếu tồn tại và import được NumPy. Không mặc định python của hệ thống. Ghi phiên bản môi trường khi chạy kiểm tra artifact.
6. Không chạy lại plan regression folder cũ; chỉ dùng nó làm bối cảnh contract.

Acceptance: có baseline để so sánh; không ghi đè các chỉnh sửa sẵn có.
Stop: nếu notebook thực tế thay đổi luồng lớn so với plan, báo khác biệt trước khi tự đổi thiết kế.

## Phase 1 — Thuật toán IQR một phía

1. Validate enabled là bool, iqr_multiplier hữu hạn > 0, min_samples là số nguyên >= 4, candidate_index duy nhất. Cấu hình sai phải raise ValueError.
2. Tách measurement_error và các hàng có bất kỳ số đo bắt buộc nào thiếu, không phải số hữu hạn hoặc <= 0 vào invalid_indices. Không đưa chúng vào percentile.
3. Khi disabled: giữ toàn bộ hàng hợp lệ; hàng invalid vẫn ghi rõ. Không fallback về measurements không hợp lệ.
4. Khi số hàng hợp lệ < min_samples: bỏ qua lọc thống kê, giữ toàn bộ hàng hợp lệ; status='insufficient_samples'.
5. Tính q1,q3 bằng np.percentile(areas,[25,75],method='linear') trên area_mm2 chưa round; iqr=q3-q1.
6. Nếu iqr <= 1e-12 * max(abs(q1),abs(q3),1.0): bỏ qua lọc thống kê; status='degenerate_iqr'. Không dùng ngưỡng bằng Q1 để loại các sai khác đo rất nhỏ.
7. Tính lower_bound=q1-iqr_multiplier*iqr. Nếu lower_bound <= 0: giữ mọi hàng hợp lệ, status='nonpositive_lower_bound'; thông báo IQR không đủ phân biệt các mảnh nhỏ trong mẫu này.
8. Nếu lower_bound > 0: reject chỉ khi area_mm2 < lower_bound; bằng ngưỡng vẫn giữ. Không tạo upper fence, không loại hạt lớn bằng module mới.
9. Chỉ tính một lượt trên tập hợp lệ ban đầu. Không lọc lặp hoặc cập nhật Q1/Q3 sau mỗi lần loại.
10. Nếu input rỗng/toàn invalid: trả các list đúng và status='empty'/'no_valid_measurements'; không crash percentile, không tạo số đo giả.

Defaults thử nghiệm: enabled=True, multiplier=1.5, min_samples=8. Số 8 là lựa chọn kỹ thuật khởi đầu, không phải ngưỡng khoa học đã được xác nhận trên dataset.
Không bổ sung median-ratio, ngưỡng mm² cố định, upper-IQR, điều kiện chiều dài hoặc lọc ROI trong phase này. Chúng cần thí nghiệm riêng.

Acceptance: kept/rejected/invalid là ba tập rời nhau và phủ đủ input; không mutate; giữ đúng ID và thứ tự; không làm tròn trước so sánh.

## Phase 2 — Cell cấu hình, đo và lọc

Bước 2: import/reload modules.grain_size_filter theo cách notebook đang reload các module.

Bước 3: thêm duy nhất:
```python
GRAIN_SIZE_FILTER_ENABLED = True
GRAIN_SIZE_FILTER_IQR_MULTIPLIER = 1.5
GRAIN_SIZE_FILTER_MIN_SAMPLES = 8
```
Kèm chú thích đây là lower-area-IQR sau CNN, chưa áp dụng vào regression legacy.

Bước 7: giữ whole_grains/defective_grains và nhãn CNN nguyên trạng. Chỉnh nhãn báo cáo 'CNN chọn nguyên' để không đồng nhất với 'đủ điều kiện đo sau lọc'.

Tách Bước 8 thành các cell có marker:
1. GRAIN_MEASUREMENTS: chỉ đo, lập bảng ID/metrics/ảnh gốc.
2. GRAIN_SIZE_FILTER: cell mới gọi module, tạo tập raw/filtered và hiển thị quyết định.
3. GRAIN_PHYSICAL_STATISTICS: thống kê, IQR volume cũ, ước lượng vật lý và các biểu đồ đang có.

Yêu cầu GRAIN_MEASUREMENTS:
- Mỗi whole_grains có đúng một hàng measurements, dù phép đo lỗi.
- Chỉ gọi compute_single_grain_metrics một lần/hạt, label='hat_nguyen'; lưu full precision.
- Lỗi đo ghi candidate_index, grain_id, exception type/message; không except-pass.
- Không có whole_grains: tạo kết quả rỗng và trạng thái không đủ dữ liệu; bỏ fallback cleaned_grains.
- Ghép ảnh/metrics bằng candidate_index; không ghép hai list có thể khác độ dài bằng cùng i.
- Đầu cell xóa/khởi tạo lại state của filter, physical estimate, regression_features/regression_result và bảng đối chiếu để không dùng kết quả lần trước khi chạy lỗi.

Yêu cầu GRAIN_SIZE_FILTER:
- Gọi filter_small_grain_outliers; tạo raw_valid_pairs và filtered_pairs từ index kết quả.
- Duy trì aliases whole_grain_metrics_list, whole_records, whole_volumes_list là tập RAW HỢP LỆ cho regression legacy.
- Tạo filtered_whole_grains, filtered_whole_grain_metrics_list, filtered_whole_records, filtered_whole_volumes_list cho vật lý.
- whole_records/filtered_whole_records có thể round để xuất/hiển thị; tính toán luôn từ metrics chưa round.
- Không xóa ảnh nguồn, không chuyển hạt bị loại sang defective_grains.
- Chạy lại cell filter với config mới phải dựng lại từ measurements raw, không lọc tiếp từ filtered lần trước. Reset kết quả phía sau trước khi tính.

GRAIN_PHYSICAL_STATISTICS:
- Tính uniformity_res_raw=evaluate_batch_uniformity(raw volumes) để đối chiếu.
- Tính uniformity_res_filtered=evaluate_batch_uniformity(filtered volumes). Giữ IQR volume HAI PHÍA cũ và công thức vật lý hiện tại để chỉ thay một yếu tố nghiên cứu.
- estimated_seed_count dùng mean_clean từ filtered volumes và PACKING_FRACTION hiện tại; baseline dùng raw mean_clean.
- Báo cáo rõ hai bước: lower-area-IQR mới rồi volume-IQR cũ; không gọi count sau area-IQR là count sau volume-IQR.
- Không có hạt hợp lệ: physical estimate=None, hiển thị 'Không đủ dữ liệu', CSV để trống và có status. Không báo 0 hạt hoặc dùng kết quả cũ.
- Min_samples chỉ quyết định có áp dụng area-IQR hay không, không phải bằng chứng đủ hạt để ước lượng chính xác.

Acceptance: bước SAHI/CNN không đổi, raw/filtered không trộn lẫn, lỗi đo không gây lệch ảnh/ID, chạy lại filter không cần chạy lại YOLO/CNN.

## Phase 3 — Hiển thị, báo cáo và hợp đồng regression

Cell filter hiển thị:
- Số CNN chọn nguyên, đo hợp lệ, lỗi đo, loại do nhỏ và giữ lại.
- Q1, Q3, IQR, lower bound, status thực tế và các thông số cấu hình.
- Bảng ID/area_mm2/decision/reason và gallery tối đa 12 hạt bị loại hoặc invalid có ảnh; hiển thị riêng lý do.
- Biểu đồ phân bố diện tích trước/sau, đường lower bound nếu tồn tại; xử lý input rỗng.
- Overlay ảnh gốc: màu giữ, loại do nhỏ, đo lỗi, CNN khuyết tật; có chú giải. Dùng global_bbox; không suy luận ID theo thứ tự sau lọc.
- Ghi rõ đây là bộ lọc độ phù hợp cho phép đo; giảm số phát hiện không tự chứng minh tăng độ chính xác.

Bước 9:
- Gallery hình học phải dùng filtered_pairs để ghép đúng ảnh với số đo.
- Nhãn số CNN nguyên và số dùng đo sau lọc phải riêng biệt.
- Không dùng tỷ lệ giữ lại để thay whole_ratio hoặc gọi là độ chính xác CNN.
- Sửa định dạng giá trị None cho các trường vật lý, không crash với :f/:d.

REGRESSION_FEATURES:
- Nhánh dự đoán legacy vẫn dùng raw_valid_pairs, giữ chính xác schema/order 31 features, rounding, uniformity và công thức hybrid của bundle.
- Để xuất features so sánh, cho assemble_regression_features nhận optional keyword metrics/records; mặc định resolve None về raw aliases ở thời điểm gọi (không capture globals trong default argument).
- Tạo vector filtered từ cùng hàm với metrics/records filtered; mọi feature phụ thuộc hạt, gồm Whole_Grains_Count, Uniformity_Rate_Pct, 20 statistics và hybrid phải cùng dùng tập filtered. Không trộn count raw với mean filtered.
- Chỉ vector raw được truyền vào predict_regression_bundle; filtered chỉ để audit, không gọi scaler/model trên vector này.
- Nếu raw không hợp lệ/rỗng: dừng regression với lý do tường minh và reset result, không dùng zero-vector.
- Không tự thêm mode 'filtered regression' hoặc giả metadata khớp với model cũ.

REGRESSION_REPORT và CSV:
- Ghi Regression_Feature_Population='raw_valid_cnn_whole' và Physical_Feature_Population='area_iqr_filtered_then_volume_iqr'.
- Ghi riêng số lượng từng tập, thông số/status filter, estimates raw/filtered vật lý.
- Giữ nghĩa các cột legacy hồi quy; thêm cột rõ ràng cho nhánh filtered. Không đổi hệ số vật lý hay hybrid.
- Xuất dưới OUTPUT_REPORT_DIR: grain_size_filter_decisions.csv, grain_size_filter_summary.json, regression_feature_comparison.csv.
- Bảng comparison có feature/raw_value/filtered_value/delta; để trống nếu không tính được.
- Đầu cell filter vô hiệu hóa comparison cũ; report chỉ dùng kết quả lần hiện tại. JSON dùng null, không NaN.
- Export images/gallery vào OUTPUT_DIR, không copy ảnh dataset/model vào Git; không xóa thư mục nguồn hay file người dùng.

Acceptance: kết quả vật lý trước/sau được đối chiếu; regression legacy không đổi với cùng measurements raw; báo cáo không trộn hai quần thể.

## Phase 4 — Kiểm thử

Unit tests trong test_grain_size_filter.py:
1. [1,11,12,12,13,13,14,14] có Q1=11.75, Q3=13.25, lower=9.5: chỉ loại phần tử 1. Tạo các metrics khác hữu hạn dương.
2. Diện tích bằng lower bound phải giữ; kiểm tra bằng dữ liệu reference tính percentile độc lập hoặc fixture phù hợp.
3. Hạt diện tích lớn không bị module mới loại do upper-outlier.
4. n<8 giữ toàn bộ hợp lệ, có status; empty/toàn invalid không crash.
5. IQR=0 và IQR gần zero: giữ hợp lệ, báo degenerate; không tự thay bằng median cutoff.
6. lower<=0: không loại, có status; không tự ép ngưỡng dương.
7. NaN/Inf/None/<=0 và measurement_error: invalid, không đi vào percentile hoặc regression raw.
8. enabled=False: không loại outlier thống kê; dữ liệu invalid vẫn được phân biệt.
9. IDs/thứ tự/input không đổi; index partition đầy đủ, không trùng; config sai bị reject.
10. Nhân tất cả area với cùng hệ số dương thông thường: quyết định IQR giữ nguyên; không phụ thuộc độ cao chụp đơn thuần.

Notebook integration trong test_main_pipeline_size_filter.py:
- Trích AST/hàm/cell thật theo marker; không copy lại thuật toán filter trong test.
- Một hạt giữa danh sách đo lỗi: gallery/record còn lại phải giữ đúng grain_id, bbox, ảnh và metrics.
- Chạy enabled True rồi False: phải phục hồi toàn bộ raw valid, không dùng tập đã lọc.
- Không CNN whole: không fallback defective; physical None và regression không có result hợp lệ.
- Tham số mẫu/SAHI/CNN trước-sau giữ nguyên; không bật lại crop.
- So sánh 31-vector raw trước/sau integration với fixture đang có: count, order và từng giá trị phải bằng nhau. Kiểm model/scaler parity bằng existing regression tests.
- Vector filtered có count/mean/std/uniformity/hybrid đồng bộ; spy model xác nhận không nhận vector filtered.
- Đo một lần/hạt; rerun filter không gọi YOLO/CNN/geometry lần nữa.
- Rerun cell lỗi không được hiển thị result cũ.

Lệnh Windows từ repo root:
```powershell
& "AI_SERVICES/.venv/Scripts/python.exe" -m py_compile CODE/modules/grain_size_filter.py
& "AI_SERVICES/.venv/Scripts/python.exe" -m unittest CODE.tests.test_grain_size_filter CODE.tests.test_main_pipeline_size_filter -v
& "AI_SERVICES/.venv/Scripts/python.exe" -m unittest CODE.tests.test_main_pipeline_regression -v
git diff --check
```

Ghi rõ nếu test model gặp cảnh báo sklearn khác phiên bản artifact. Không đổi expected prediction chỉ để pass và không tự nâng/hạ môi trường chung.

test_main_pipeline_crop.py là test từ thử nghiệm crop trước đã bị người dùng hoàn tác; không phục hồi code crop để làm test cũ pass. Ghi rõ test đó ngoài phạm vi kế hoạch này nếu không còn áp dụng.

Colab/manual:
1. Reload module bằng Bước 2 khi code mới; chạy lại từ cấu hình tới báo cáo.
2. Dùng ít nhất 3 ảnh đang có: mẫu thường, mẫu có mảnh nhỏ bị CNN nhận nguyên, mẫu ít hạt nguyên. Nếu thiếu loại ảnh, ghi NOT VERIFIED.
3. Trên cùng ảnh và cùng kết quả CNN, so sánh enabled=False/True; không đổi đồng thời confidence, slice_size, overlap hay model.
4. Xem từng vùng bị loại và vài vùng được giữ; ghi quyết định người review nếu có. Nếu hạt nguyên thật bị loại nhiều, chưa chấp nhận defaults.
5. Ghi số hạt từng nhóm, trung bình thể tích raw/filtered, physical estimate trước/sau; regression legacy phải không đổi do bật/tắt area filter.
6. Chạy CPU khi khả thi; không giả lập test CNN/YOLO thật bằng kết quả mock rồi ghi E2E PASS.

Acceptance kỹ thuật: unit/integration pass, raw-regression parity pass, export đọc được.
Acceptance nghiên cứu: cần review ảnh thật. Nếu chưa có review ảnh, ghi 'implementation verified; filter efficacy pending', không gọi cải thiện độ chính xác.

## Phase 5 — Bàn giao

- Lưu CODE/reports/grain_size_filter/verification.md: file sửa, lệnh/test thực sự chạy, ảnh đã dùng, kết quả before/after, bước chưa verify.
- Hướng dẫn bật/tắt và chạy lại cell GRAIN_SIZE_FILTER rồi các cell downstream; không bắt chạy lại SAHI/CNN khi chỉ đổi k/min_samples.
- Nếu đổi ảnh, kích thước ly hoặc model CNN, phải chạy lại các bước upstream tương ứng.
- Kiểm git diff; chỉ scope cho phép; không commit/push khi chưa được yêu cầu.
- Không update PROJECT_STATUS hoặc khẳng định task nghiên cứu đã hoàn thành.

## Giới hạn và stop conditions

- IQR không bảo đảm tìm mọi mảnh nhỏ; khi phân bố rộng, ít mẫu hoặc IQR=0, thiết kế chủ động không tự đặt ngưỡng thay thế.
- Hạt bị che khuất hoặc nhìn nghiêng có thể nhỏ; kết quả loại là không dùng đo, không phải ground-truth class.
- Bộ lọc diện tích không sửa over-segmentation, duplicated masks hay dự đoán sai kích thước tương tự hạt thật.
- Nếu cần ngưỡng hỗn hợp, lọc theo chiều dài, ROI hoặc thay model: báo hạng mục kế tiếp, không tự mở rộng.
- Nếu cần áp dụng filtered features cho model regression hiện tại: dừng nhánh đó; cần thỏa thuận sinh lại dữ liệu/train hoặc thí nghiệm riêng có nhãn rõ. Không fake metadata/provenance.
- Không thể ghép ID/ảnh/metrics an toàn, test raw feature parity sai, hoặc chưa hiểu measurement schema: sửa mapping/contract trước khi tiếp tục; không bỏ assertion.
- Không nhận diện được ảnh thật/hardware thiếu: ghi NOT VERIFIED; không bịa test và metric.
- Không có hạt hợp lệ: không dùng hạt CNN khuyết tật hoặc kết quả lần trước để cứu prediction.

## Checklist giao Gemini 3.1 Pro High

- [ ] Đọc plan này và trạng thái working tree; giữ mọi chỉnh sửa hiện có.
- [ ] Giữ SAHI trên ảnh gốc và các tham số sample/CNN hiện tại.
- [ ] Tạo grain_size_filter.py đúng API, index partition và lower-area-IQR một lượt.
- [ ] Implement n<8, degenerate IQR, lower<=0, disabled, invalid và empty đúng quy định.
- [ ] Import/reload module và đặt ba config ở Bước 3.
- [ ] Tách cell measurement → filter → physical statistics.
- [ ] Bỏ fallback cleaned_grains khi không có CNN whole; log mọi lỗi đo.
- [ ] Dùng candidate_index ghép ảnh/metrics; giữ raw aliases cho regression.
- [ ] Áp dụng filtered population cho vật lý rồi volume-IQR cũ; hiển thị hai bước.
- [ ] Bổ sung bảng/gallery/overlay và report CSV/JSON theo ID.
- [ ] Tạo raw/filtered feature comparison, chỉ predict trên vector raw legacy.
- [ ] Guard empty và state cũ; rerun filter không chạy lại vision/measurement.
- [ ] Chạy unit, notebook integration và existing regression tests theo plan.
- [ ] Thử ảnh thật khi có điều kiện; phân biệt test mock với E2E.
- [ ] Viết verification.md trung thực, chỉ ra efficacy còn pending nếu chưa review ảnh.
- [ ] Kiểm diff, không commit/push, không mở rộng sang training/AI_SERVICES.

