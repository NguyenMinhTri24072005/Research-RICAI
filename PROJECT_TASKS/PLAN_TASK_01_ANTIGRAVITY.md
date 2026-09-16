# TASK-01 — Kế hoạch thực thi dành cho Antigravity

Ngày đánh giá: 16/09/2026.

## 1. Phạm vi và độ tin cậy của đánh giá

Mục tiêu: hoàn thành tích hợp hồi quy 31 biến trong AI_SERVICES, ưu tiên backend
Colab runtime, đồng thời chạy được local Windows. Đây là kế hoạch, chưa phải
bằng chứng Task 01 hoàn thành.

Đã đọc mã nguồn AI_SERVICES/app.py, regression_engine.py, các cell code của
API_Server.ipynb, yêu cầu task, danh mục modules/frontend/backend, manifest
thăm dò best_tree_model_info.json và danh sách features trong scaler_params.json.
Đã xem README, requirements và tài liệu trạng thái/quyết định của dự án.

Sau khi công cụ đọc file hoạt động trở lại, đã đọc thêm gateway Express,
capture_server/app.py và controller.py, các phần request/result/SSE của React,
các điểm hiển thị kết quả mobile, stage 1/2 của dataset_extractor.py, cấu hình
notebook extraction và các đoạn split/scaler/train/export của trainer tree.
Đã đối chiếu đoạn chuẩn hóa/chia dữ liệu trong train_linear_regression.py.
Chưa kiểm tra model binary, chưa chạy suy luận hay benchmark; chưa kiểm toán
từng dòng mọi module/notebook hoặc dữ liệu gốc. Antigravity phải hoàn tất P0
trước khi kết luận artifact trên đĩa tương thích với source train hiện tại.
Không dùng kết quả test CAPTURE_APP trước đây làm bằng chứng cho AI_SERVICES.

## 2. Đánh giá dự án và các vấn đề đã xác nhận

| Thành phần | Hiện trạng | Ý nghĩa đối với Task 01 |
| --- | --- | --- |
| DATASET_BUILDER/CAPTURE_APP | Có luồng điện thoại–host–SQLite–Excel; đã bỏ batch/device theo yêu cầu người dùng | Giữ trải nghiệm nhập gọn, không thêm lại hai trường |
| CODE/modules và AI_SERVICES/modules | Cùng tồn tại các module container, segmentation, cleaner, classifier, geometry, uniformity | Cần đối chiếu công thức/config; chưa khẳng định hai bản giống nhau |
| AI_SERVICES/app.py | Có pipeline và hồi quy trong /predict | Cần validation, model contract, lỗi có mã, readiness thực |
| API_Server.ipynb | Import app từ AI_SERVICES; mở ngrok và khởi động uvicorn | Dùng cùng pipeline với local; notebook chỉ bootstrap |
| LINEAR_REGRESSION_MODEL | Có artifact Extra Trees, scaler, OLS, JSON kết quả | Chưa có bằng chứng model/scaler cùng lần train |
| React + Express gateway + capture_server | Đã tồn tại; cần kiểm toán request/response từng đường đi | Tránh backend mới đúng nhưng một giao diện vẫn nuốt lỗi/không hiện kết quả |
| PROJECT_TASKS | Một số checkbox/schema đã lạc hậu so với code | Chỉ cập nhật theo bằng chứng, không dựa trên tick cũ |
| docs/PROJECT_STATUS.md | Chủ yếu mô tả khởi tạo Git | Cần bổ sung trạng thái kỹ thuật sau nghiệm thu |

Các phát hiện cụ thể từ code hiện tại:

1. /api/status luôn trả status=ready, kiểm tra file tồn tại mà chưa xác minh
   nạp model; trả cả đường dẫn filesystem. Cần phân biệt liveness/readiness.
2. load_tree_model() ghép tree với scaler theo tên file, scaler có thể vắng mặt;
   lỗi load bị đổi thành (None, None), cache loaded=True ngay cả khi thất bại.
3. regression_engine.py tắt InconsistentVersionWarning toàn cục. Cần đối chiếu
   phiên bản và thể hiện lỗi tương thích, không che cảnh báo quan trọng.
4. assemble_31_features() và predict_from_tree() dùng .get(..., 0); thống kê
   rỗng cũng thành 0. Chưa phân biệt dữ liệu thiếu với số 0 hợp lệ.
5. weight_total mặc định 0 nhưng Weight_g là feature hồi quy; thiếu cân nặng
   hiện vẫn tạo vector để predict.
6. Estimated_Total_Seeds_Hybrid nhận estimates.final từ hình học/cân mẫu.
   Cần kiểm chứng có cùng công thức với dữ liệu train hay không.
7. PACKING_FRACTION=0.82 trong app.py, còn tài liệu Task 02 nêu xấp xỉ 0.62.
   Đây là khác biệt giữa code và tài liệu, chưa phải bằng chứng giá trị nào đúng.
8. predict_regression() tự chuyển Extra Trees sang OLS nhúng trong code; lỗi
   cuối cùng trả 0. Giá trị hồi quy bằng 0 bị điều kiện >0 coi như thất bại.
9. Lỗi /predict trả JSON status=error nhưng HTTP 200; upload đọc toàn bộ vào RAM;
   chưa có validation hữu hạn/miền giá trị cho thông số vật lý.
10. CNN/geometry có except Exception: pass ở từng hạt, không trả số hạt lỗi;
    raw_crops, crop và artifacts chẩn đoán bị xóa ở cuối request.
11. /predict là async nhưng chạy phần suy luận đồng bộ trực tiếp; cần đưa xử lý
    nặng khỏi event loop và giới hạn truy cập model dùng chung.
12. API đã trả regression_est, ai_est, weight_est, method_used và
    feature_schema_version bên trong estimation. Task 01 mô tả một phần này
    chưa có; còn thiếu geometry_est, version gắn contract thật và error schema.
13. Notebook cài dependency không pin, in kiểm tra tồn tại model, import app
    rồi mở tunnel. Chưa có preflight load/checksum/schema trước khi công bố ready.
14. README nêu tốc độ CPU/T4 nhưng chưa thấy bằng chứng benchmark tái lập trong
    các nguồn đã đọc. Không dùng các con số này làm mục tiêu nghiệm thu.
15. Sai lệch extractor/runtime đã xác nhận; chưa xác nhận dataset đã train
    artifact hiện tại được tạo với phiên bản nào:

    | Điểm đối chiếu | Extractor/notebook hiện tại | Runtime app.py |
    | --- | --- | --- |
    | Packing fraction | Class mặc định 0.62; notebook truyền 0.80 | 0.82 |
    | Volume dùng ước lượng | Trung bình mm³ | Trung vị px³, có đổi đơn vị |
    | Feature Estimated_Total_Seeds_Hybrid | Round(bulk × packing / volume_mean), không trộn cân mẫu | estimates.final, có thể trung bình hình học và cân mẫu |
    | Cleaner | Hai lượt với tham số riêng, lọc alpha | Một lượt, bộ tham số khác |
    | SAHI overlap | Mặc định 0.25; cell gọi stage 1 đang comment | 0.20 |
    | Precision export | ppm/bulk 2 chữ số, thống kê 3 chữ số | Assemble trước rounding hiển thị |

    Cùng tên và đủ 31 cột chưa bảo đảm inference đúng phân phối train. Không
    chọn hệ số theo trực giác. Chốt contract theo nguồn train thực; nếu đổi
    công thức, tạo schema/bundle mới, không đổi âm thầm model cũ.
16. Trainer tree split trước rồi fit StandardScaler trên train; Extra Trees
    nhận X_train_scaled, export model/scaler vào cùng thư mục. Đây là bằng
    chứng về source, chưa chứng minh hai binary hiện có cùng run. Script
    linear riêng tính mean/std trên toàn bộ X trước split, còn trainer tree
    CV dùng X_train_scaled đã fit trước các fold. Phân biệt các đường train;
    khi tái huấn luyện/đánh giá dùng preprocessing trong từng fold.
17. Gateway /api/status chỉ báo gateway/config, không hỏi readiness AI;
    lỗi HTTP upstream bị đổi thành 500/string. Mobile bọc lỗi vào detail;
    React catch báo chung lỗi kết nối/timeout nên mất thông tin validation.
18. Express broadcast SSE tới mọi dashboard; NodeHub broadcast tới mọi mobile,
    không dùng node_id để định tuyến kết quả. React chỉ xử lý SSE success,
    nên lỗi chụp sau có thể để kết quả cũ còn hiển thị.
19. Hai UI đã có bảng hồi quy/hình học/cân mẫu, không cần xây lại; cần thêm
    warnings/timing/debug. React ghi GPU cố định khi loading dù có thể dùng CPU.
20. Gateway upload.any() không giới hạn số lượng/kích thước, chỉ xóa file đầu;
    capture_server đọc toàn bộ ảnh vào RAM. /ws/host không kiểm tra token như
    /ws/node. Sửa giới hạn/cleanup và bảo vệ kênh kết quả trong phạm vi luồng
    đang dùng; không mở rộng thành hệ quản lý tài khoản.

## 3. Những quyết định phải giữ

- Điện thoại là camera node; máy tính quản lý giao diện/host; Colab có thể làm
  backend suy luận. Không thay kiến trúc thành native app trong Task 01.
- Không thêm Capture_Batch/Device_ID, không bắt người dùng nhập metadata thiết bị.
- Giữ Capture_Timestamp tự sinh và mã ảnh phẳng của CAPTURE_APP.
- Không chỉnh dataset gốc, model cũ, .env/token hoặc chứng chỉ người dùng.
- Không tự chạy lại toàn bộ benchmark 14 model, XAI, few-shot hay mở rộng dataset.
- Không reset/revert worktree. Đã có thay đổi chưa commit ở hai app camera,
  CAPTURE_APP và PROJECT_TASKS; ghi nhận trước khi sửa.
- Không đọc/quét .venv. Khi cần test chỉ kiểm tra đường dẫn executable phù hợp.
- Không cần duy trì một bản pipeline riêng trong notebook Colab.

## 4. P0 — Kiểm toán nguồn và khóa các lựa chọn trước triển khai

Đọc toàn bộ các file sau, xác định tên hàm/call site thực tế rồi cập nhật plan
nếu có khác biệt; không tạo pipeline_logic/main_pipeline.py vì đường dẫn đó
không tồn tại trong kiến trúc hiện tại:

- AI_SERVICES/app.py; regression_engine.py; modules/*.py.
- AI_SERVICES/API_Server.ipynb (code cells, bỏ qua outputs chứa thông tin runtime).
- AI_SERVICES/capture_server/app.py, controller.py, start.py.
- AI_SERVICES/RICE_ESTIMATION_APPLICATION/backend/server.js.
- AI_SERVICES/RICE_ESTIMATION_APPLICATION/frontend/src/App.jsx.
- AI_SERVICES/capture_server/web/static/capture.js.
- CODE/modules/dataset_extractor.py và các module hình học/segmentation/classifier.
- DATASET_BUILDER/AI_DATASET_EXTRACTION_PIPELINE.ipynb (code/config).
- LINEAR_REGRESSION_MODEL/train_linear_regression.py,
  inference_linear_regression.py, notebook train tree nếu có.
- JSON model/scaler/equation và schema/header dataset được train thực tế.
- README, requirements, launcher .bat, package.json của backend/frontend.

Deliverables P0:

1. docs/TASK_01_AUDIT.md ghi call graph cả đường React và điện thoại, dependency
   cần thiết, endpoint upload/status/result thật và artifact được chọn.
2. Bảng 31 dòng: tên/thứ tự feature, đơn vị, công thức train, công thức runtime,
   nguồn dữ liệu, quy tắc missing/zero, precision, ddof thống kê, kết quả parity.
3. Kiểm chứng định nghĩa đường kính trong/ngoài, wall thickness, cm→mm,
   container diameter px, packing fraction, median/mean, crop cleaner, class
   order/preprocessing CNN, pixels/mm, feature Hybrid và nguồn nhãn Actual_Count.
4. Kiểm tra notebook train tree nhận raw X hay scaled X, scaler fit ở đâu và
   artifact nào được export cùng run. Không suy ra điều này chỉ từ số 31.
5. Kiểm tra dữ liệu nhiều ảnh của cùng mẫu vật lý trước khi làm fixture/holdout.
   Timestamp/ngày chụp không bảo đảm chia đúng nhóm mẫu vật lý.
6. Xác nhận hai đường đi:
   - Desktop: React → Express /api/predict → AI /predict; trả HTTP và SSE.
   - Mobile: HTTPS capture_server /api/capture → Express /api/predict → AI;
     trả mobile qua HTTP/NodeHub WebSocket, React qua SSE /api/capture/stream.
   Giữ transport hiện có; không chuyển React sang WebSocket chỉ để thống nhất.
   Dùng request_id/session tạm tự sinh để routing/deduplicate, không bắt nhập
   Device_ID hoặc lưu metadata thiết bị vào dataset.

Gate P0: sai tên/thứ tự/đơn vị/công thức hoặc không xác minh được model/scaler
thì chưa được ghi verified. Tên file, mtime, JSON điểm số và feature count không
chứng minh cùng lần train. Metric R² cao không chứng minh leakage, cũng không
thể coi là độ chính xác sản phẩm đã xác nhận.

## 5. P1 — Contract đặc trưng và bundle artifact

Đề xuất thêm:

- AI_SERVICES/feature_schema.py: schema duy nhất cho 31 biến và validator.
- AI_SERVICES/model_registry.py: đọc manifest, validate/load/cache bundle.
- AI_SERVICES/artifacts/manifest.json: metadata nhỏ có thể version-control.
- AI_SERVICES/scripts/verify_artifacts.py: preflight CLI, exit code khác 0 khi lỗi.
- AI_SERVICES/scripts/export_artifact_bundle.py: export sang thư mục mới.

Manifest phải có schema_version, bundle_id, model_family, model path/checksum,
preprocessing type (none/scaler/pipeline), scaler path/checksum khi cần, danh
sách feature có thứ tự và đơn vị, target, dataset fingerprint/version,
training_run_id, trained_at có nguồn chứng minh, phiên bản Python/numpy/sklearn,
pipeline feature config, YOLO/CNN identifiers/checksums, CNN class order và
preprocessing. Đường dẫn tương đối với manifest/project root; env override
được validate. Không gắn timestamp audit làm ngày train giả.

Validation: file đủ, checksum đúng, đúng estimator, n_features_in_=31, kiểm tra
feature_names_in_ khi có, scaler đúng feature order/dimension, preprocessing
không áp dụng hai lần, smoke predict hữu hạn trên vector fixture hợp lệ.
Bundle load là một đơn vị; tránh model nạp thành công nhưng scaler thất bại.
Cache keyed theo bundle identity; không cache lỗi vĩnh viễn; load có lock.

Nếu provenance cũ không xác minh được: ghi trạng thái unverified; thử tìm code
export/run log. Nếu phải train lại để tạo cặp có nguồn gốc rõ, tạo artifact mới
với một estimator đã chọn, pipeline tiền xử lý fit chỉ trên train, seed cố định
và split có căn cứ. Không ghi đè model cũ, không biến bước này thành Task 04.
Nếu chưa đủ dữ liệu/nguồn nhãn để train đúng, báo blocker cụ thể; không xuất
manifest verified giả để vượt kiểm thử.

OLS fallback chỉ được giữ nếu có artifact/equation có provenance, đơn vị và
preprocessing đã kiểm tra parity. Không dùng hằng số OLS nhúng âm thầm.

Gate P1: verify_artifacts CLI pass cho bundle thực hoặc có blocker chính xác;
ready chỉ khi model và preprocessing thực sự tương thích. Scaler không bắt
buộc nếu bundle được chứng minh dùng raw features hoặc Pipeline tự xử lý.

## 6. P2 — Pipeline dùng chung, validation và chính sách suy luận

Giữ các module CV hiện tại; tách điều phối khỏi endpoint vào inference_pipeline.py
hoặc services/inference.py, chọn một vị trí và ghi trong audit. app.py tập trung
parse input, validate, điều phối và serialize response.

Input validation trước suy luận:

- Kiểm tra loại/giải mã ảnh, giới hạn byte/pixel cấu hình và xử lý ảnh sai định dạng.
- diam>0, height>0, 0<=empty<=height; wall_thickness không âm và quan hệ đường
  kính đúng theo định nghĩa đã chốt ở P0; toàn bộ số phải finite.
- weight_total không âm; phân biệt không nhập (None) với 0; sample_count và
  sample_weight phải cùng đủ, count là số nguyên dương khi dùng cân mẫu.
- Không chấp nhận Actual_Count làm feature request. Ground truth phục vụ test
  chỉ nằm trong fixture/benchmark, không truyền vào pipeline suy luận.

Feature validation:

- Tách feature hồi quy khỏi estimate UI: Hybrid phải đúng định nghĩa train,
  không lấy tùy tiện final của UI. Khóa cleaner/SAHI/packing/precision theo
  bundle; parity test cùng ảnh+form, so vector và tập hạt được chấp nhận.
  Tính đến bước duyệt crop thủ công của extractor, không giả định runtime
  tự tái hiện được quyết định của người kiểm duyệt.
- Không thay feature bắt buộc bằng 0 khi thiếu/NaN/Inf.
- Không có hạt nguyên đo được là lỗi/no-result có mã; không tạo 20 thống kê 0
  rồi trả hồi quy bình thường. Một hạt hợp lệ có std=0 là hợp lệ.
- Dùng cùng tập hạt hợp lệ để tạo năm nhóm thống kê, ghi rejected counts/lý do.
- Case ly trống phải có policy rõ: lỗi EMPTY_SAMPLE hoặc kết quả 0 qua đường
  được kiểm chứng; không coi model.predict()=0 là lỗi chỉ vì điều kiện >0.
- Vector truyền model và vector trace phải là cùng dữ liệu đầy đủ precision.

Policy đề xuất phù hợp UI đang cho bỏ trống cân nặng:

- estimator_mode=auto (mặc định): ưu tiên regression nếu bundle và đủ 31
  feature hợp lệ; nếu thiếu điều kiện hồi quy, cho phép geometry/weight đã đủ
  điều kiện với warning/code rõ. Không tự đổi sang OLS không verified.
- estimator_mode=regression: thiếu Weight_g/feature → 422; thiếu model/scaler
  hoặc bundle không hợp lệ → 503. Không trả hình học như kết quả regression.
- estimator_mode=geometry hoặc weight: chạy đúng phương pháp yêu cầu.
- Mọi trường ước lượng không có giá trị dùng null; final/method_used phải khớp
  phương pháp thực sự được chọn, không ghi hybrid khi chỉ dùng weight.

Đưa suy luận đồng bộ ra thread worker có giới hạn; giữ status/health responsive.
Mặc định một inference đồng thời trên model cache, queue có giới hạn và lỗi busy
rõ. Tránh sửa confidence/config của model chung giữa các request. Chưa cần
Celery/Redis hoặc hệ thống job phân tán trong Task 01.

## 7. P3 — API contract và tích hợp các giao diện

Thêm schemas.py/error types và tài liệu docs/API_CONTRACT.md.

- GET /health: liveness nhẹ.
- GET /api/status: overall ready/degraded/not_ready, version/schema/bundle,
  từng component configured/loaded/verified/error và khả năng chạy mỗi method.
  Không trả absolute filesystem path hoặc secret cho client.
- POST /predict: giữ multipart và field cũ; mode/debug là tùy chọn có mặc định.
- Response success: request_id, estimation.final/regression_est/geometry_est/
  weight_est/method_used, feature_schema_version, model bundle/version,
  metrics_summary, warnings, timings_ms; ai_est giữ alias của geometry_est để
  tương thích client cũ trong giai đoạn chuyển đổi.
- Error: HTTP 4xx/5xx đúng, {status:error, request_id, error:{code,message,
  field?,stage?}}; không trả traceback cho client. Nội dung log gắn request_id.
- Codes tối thiểu: INVALID_IMAGE, INVALID_INPUT, MISSING_FEATURE,
  NON_FINITE_FEATURE, NO_VALID_GRAINS, MODEL_UNAVAILABLE,
  ARTIFACT_INCOMPATIBLE, INFERENCE_FAILED, SERVER_BUSY.

Gateway/capture_server phải bảo toàn status code/error/warnings/request_id;
không đổi error thành success hay 0. Có timeout cấu hình và thông báo runtime
không kết nối được. SSE/WebSocket/HTTP mang cùng request_id; định tuyến mobile
đúng session/request, dashboard chỉ nhận phiên được phép. Deduplicate HTTP và
stream; xử lý started/error/completed và hai request liên tiếp để tránh kết
quả cũ. Xác thực /ws/host; đánh giá bind address/origin/auth gateway trước khi
expose ngoài localhost. Giới hạn một file tại mọi hop, cleanup toàn bộ file
tạm trong finally.

React và điện thoại: hiển thị final, phương pháp thực, trạng thái hồi quy,
thiếu input nào khi yêu cầu regression; null là chưa có kết quả, không thành 0.
Giữ form/camera và thay đổi touch focus/flash hiện có. Không thêm lại metadata.
Không hardcode GPU/Extra Trees khi chưa biết device/model thực.

## 8. P4 — Quan sát quá trình test và ảnh chẩn đoán tối thiểu

Đo bằng perf_counter theo stage: decode, container, segmentation, cleaning,
classification, geometry/features, regression, total. Tách cold model load và
warm inference; GPU phải đo phần công việc đã hoàn tất. Giao diện có elapsed
time/end-to-end, không tự gán các mốc phần trăm tiến trình không có telemetry.

debug=true trả/giữ có giới hạn: ảnh gốc có ellipse miệng ly/tỷ lệ đo, overlay
segmentation, bảng crop raw/cleaned + nhãn/confidence, accepted/rejected count,
feature vector full precision và config/version. Mặc định debug=false. Artifact
phải copy ra nơi có request_id trước khi xóa temp; endpoint đọc artifact chống
path traversal, có TTL/cleanup và giới hạn dung lượng, không dùng URL trỏ temp
đã xóa. Gateway phải chuyển tiếp URL/data an toàn qua đúng backend runtime.

UI desktop ưu tiên panel chẩn đoán gọn có thể mở rộng; mobile chỉ tóm tắt và
ảnh overlay khi bật test. Không triển khai SHAP/waterfall/XAI trong Task 01.
Chỉ tính sai số khi fixture có Actual_Count thật; confidence CNN không phải độ
chính xác đếm hạt. Phân biệt lỗi container, segmentation, classifier và regression.

## 9. P5 — Local Windows và Colab runtime

- Config project root + manifest path; bỏ dependency bắt buộc vào Drive của
  cá nhân trong code service. Colab có cấu hình DRIVE_PROJECT ở bootstrap.
- Không tự chọn model YOLO nền trong MODELS làm model hạt production khi thiếu
  trọng số trained; chỉ dùng artifact đúng manifest.
- Notebook: mount/config → install môi trường xác nhận → verify bundle → warmup
  service → mở tunnel khi sẵn sàng. Rerun cell không để nhiều server/tunnel/model
  cache cũ; log run_id/bundle_id và hướng dẫn restart khi đổi artifact.
- Soát requirements theo imports thật; tách dependency runtime Colab/capture
  host nếu cần, pin theo artifact để tái lập. Không ép wheel GPU lên Windows.
- Bảo toàn .env/token; chỉ báo tên cấu hình thiếu, không in hoặc commit secret.
- README cập nhật lệnh chạy/test cho Windows và notebook; gỡ cam kết tốc độ
  không có benchmark, ghi số đo thực kèm điều kiện.

## 10. P6 — Kiểm thử và nghiệm thu

Đề xuất AI_SERVICES/tests và fixtures có license/nguồn rõ, không dùng dummy.jpg
làm bằng chứng ML. Một ảnh có form hợp lệ cố định để kiểm tra tái lập; nếu có
ground truth đếm thật thì dùng đánh giá sai số, tuyệt đối không tạo nhãn giả.

| Nhóm | Case bắt buộc | Bằng chứng |
| --- | --- | --- |
| Artifact | Thiếu file, checksum sai, nhầm scaler/order/version, preprocessing none/scaler/pipeline | Registry báo đúng code, không fallback OLS âm thầm |
| Features | Đủ 31, thiếu key, NaN/Inf, một hạt std=0, không hạt, cm/mm, hybrid parity | Vector đúng công thức/dữ liệu train |
| API nhanh | Happy path bằng fake dependencies có chủ đích, validation 422, unavailable 503, busy | HTTP/schema/error/warnings nhất quán |
| Real integration | Ảnh+form cố định, model thật trên Windows CPU và Colab GPU nếu có | /status loaded verified, /predict full path, fixture repeat trong tolerance |
| Clients | React upload, mobile upload, gateway errors, WebSocket routing | Kết quả đúng request, null/error không hiển thị 0 |
| Debug | overlay/ellipse/crops, TTL, lỗi vẫn cleanup | Ảnh còn truy cập được trong TTL, không leak đường dẫn |
| Concurrency | Hai request, health khi inference đang chạy, timeout | Không lẫn config/result; service còn phản hồi |
| Compatibility | CAPTURE_APP regression tests và extractor fixture | Không thêm batch/device, không hỏng dữ liệu cũ |

Fast suite không cần GPU/model lớn; real-model suite đánh dấu riêng. Skipped
real-model test không được tính là đã nghiệm thu ML. Không trộn import modules
từ CODE với AI_SERVICES trong một interpreter mà không xác minh module origin.

Benchmark tối thiểu: ghi hardware/OS/Python/lib versions, bundle/checksum,
input resolution/số hạt/slice settings; đo cold start riêng, 2 warmup và ít nhất
10 warm requests trên cùng fixture; báo median/p95/min/max từng stage và tổng.
Local/network/Colab tách rõ server inference với upload+ngrok+gateway+render.
Không cam kết T4 nhanh hơn CPU trước khi có số đo. Nếu chưa có runtime GPU,
ghi unmeasured và để gate tương ứng mở.

Lệnh dự kiến sau khi tạo test/scripts (chưa phải lệnh đã tồn tại):

```powershell
# Từ AI_SERVICES, dùng interpreter môi trường dự án hiện có phù hợp
python scripts/verify_artifacts.py --manifest artifacts/manifest.json
python -m pytest tests -m "not real_models"
python -m pytest tests -m real_models
python scripts/benchmark_inference.py --manifest artifacts/manifest.json --warmup 2 --runs 10
```

## 11. File bàn giao và tiêu chí đóng Task 01

Thứ tự bắt buộc: P0 → P1 → P2 → P3 → P4/P5 → P6. Khi provenance/model contract
bị chặn, vẫn hoàn thành validation/API/tests fake được nhưng không đánh verified.

Deliverables:

- Audit + bảng provenance/parity 31 feature.
- Schema, manifest, registry, verifier/exporter và inference pipeline chung.
- API contract, cập nhật gateway/React/mobile, diagnostics/timing.
- Notebook bootstrap/config và hướng dẫn local/Colab.
- Unit/contract/integration tests; fixtures thật có metadata form.
- reports/task_01/verification.md, benchmark.json hoặc .csv, environment.json,
  example_success.json, example_errors.json; artifact ảnh ngoài Git nếu lớn.

Task 01 chỉ đóng khi:

- [ ] Bundle thực có provenance/checksum/preprocessing đúng, verifier pass.
- [ ] /api/status báo loaded/verified thật; scaler chỉ cần khi contract yêu cầu.
- [ ] Runtime và extractor/training tạo cùng 31 feature theo công thức đã chốt.
- [ ] Invalid/missing feature trả lỗi hoặc degraded result theo mode công khai.
- [ ] API trả schema/method đúng, các giao diện không nuốt lỗi hay hiển thị sai 0.
- [ ] Real-model integration Windows và smoke test Colab có bằng chứng. Nếu
      chưa có runtime, ghi blocked/unverified và giữ gate mở; không kết luận
      Task 01 hoàn thành hay runtime pass từ test giả.
- [ ] Có số đo tốc độ và ảnh chẩn đoán để kiểm tra segmentation/classifier.
- [ ] Không dependency bắt buộc vào đường dẫn Drive cá nhân ở service local.
- [ ] README/task board cập nhật theo bằng chứng, bảo toàn mọi dữ liệu/artifact cũ.

## 12. Prompt bàn giao cho Antigravity

Đọc PROJECT_TASKS/PLAN_TASK_01_ANTIGRAVITY.md và TASK_01_AI_SERVICES_INTEGRATION.md.
Thực hiện theo P0–P6, ưu tiên backend Colab runtime dùng chung app.py với local.
Trước tiên kiểm chứng những điểm audit còn thiếu; tôn trọng code và thay đổi chưa
commit của người dùng. Hoàn thành artifact contract, feature parity/validation,
API semantics, tích hợp các client, timing/debug images và test tái lập. Không
thêm lại Capture_Batch/Device_ID và không sửa dataset/model gốc. Nếu không chứng
minh được provenance model/scaler, ghi unverified và giải quyết nguồn gốc trước
khi đánh ready. Sau triển khai chạy test, ghi bằng chứng, tự review và cập nhật
checkbox Task 01 theo kết quả thực tế; chỉ báo hoàn thành khi các gate đã đạt.
