# AI Services — Modular Inference and Folder-based Regression Plan

Status: READY FOR IMPLEMENTATION  
Created: 2026-09-18  
Planner: Codex  
Implementation agent: Gemini 3.8 Flash  
Inspected baseline: `7e7063d`, clean working tree before this plan.

## 1. Mục tiêu và phạm vi

Người dùng muốn đổi regression bằng một đường dẫn thư mục: lúc nghiên cứu trỏ đến thư mục output training; khi triển khai copy bundle vào AI_SERVICES rồi đổi cùng đường dẫn đó. Không phải nhập hash, class, hệ số hoặc sửa manifest trung tâm mỗi lần đổi model.

Tách mã Python inference vào một package nguồn riêng. FastAPI là lớp giao tiếp; pipeline và từng giai đoạn xử lý độc lập với HTTP, React, Node.js và camera điện thoại. Giữ cách chạy Windows và notebook Colab. Đây là tái cấu trúc backend AI, không phải viết lại backend MERN.

Phiên lập plan chỉ đọc code và tạo tài liệu này, chưa chạy test hay sửa implementation. Evidence dưới đây là static inspection; không coi là xác nhận runtime mới.

Phạm vi implementation: cấu hình, loader, tách module, migration imports, tests, README và cell launcher Colab. Không training lại, không đổi dataset, không sửa CAPTURE_APP, không triển khai TASK-02+, không tự chọn model tốt nhất. Không tự commit/push nếu chưa được yêu cầu.

## 2. Đánh giá hiện trạng và evidence

| Vấn đề / điểm đã có | Evidence hiện tại | Quyết định thiết kế |
|---|---|---|
| Đã có thuật toán CV tách module | `modules/container_detector.py`, `grain_segmenter.py`, `grain_crop_cleaner.py`, `grain_classifier.py`, `ellipsoid_geometry.py`, `uniformity_evaluator.py` | Di chuyển và tái sử dụng; không viết lại thuật toán. |
| App chứa quá nhiều trách nhiệm | `app.py`: `resolve_model_paths`, model cache, `execute_*`, `compute_final_estimates`, `/predict` chứa toàn bộ workflow | App chỉ bootstrap; route chỉ HTTP; pipeline điều phối; stage xử lý riêng. |
| Registry buộc cấu hình nhiều chi tiết | `model_registry.py:ModelRegistry.load_bundle`, `artifacts/manifest.json` | Runtime mới chọn folder; không bắt buộc hash/manifest hoặc class name do người dùng nhập. |
| Đổi model còn bị khóa ở công cụ | `scripts/verify_artifacts.py` reject model khác ExtraTreesRegressor và scaler khác StandardScaler | Preflight kiểm tra giao diện, feature contract và prediction, không khóa họ model/scaler. |
| Nhãn model bị hardcode | `regression_engine.py:predict_regression` luôn trả `ExtraTrees` | Lấy tên từ đối tượng đã load. Giữ alias nhãn ExtraTrees cũ cho đúng model đó. |
| Có hệ số OLS cứng | `_OLS_INTERCEPT`, `_OLS_COEFFICIENTS`, `predict_from_equation` | Không đưa vào inference mới. Không fallback sang phương trình hệ số cũ. |
| Output training dùng được trực tiếp | `LINEAR_REGRESSION_MODEL/models/ard/`: model/scaler/pipeline joblib, feature_schema, config, metrics, manifest | Dùng `model.joblib` + `scaler.joblib` làm định dạng chính; không cần đổi training notebook. |
| Training manifest khác deployment manifest | Training dùng `files.model.path`; runtime hiện dùng `model.relative_path` | Không yêu cầu người dùng chuyển đổi hai manifest. Đọc file chuẩn trong folder. |
| Ba output chưa được trung bình chung | `app.py:compute_final_estimates`, bước 9 `/predict` | `auto` hiện ưu tiên regression; fallback trung bình geometry/weight. Giữ chính sách và tách vào `fusion.py`. |
| Hai công thức hình học khác nhau có chủ ý | Geometry: 0.82 / median pixel volume; feature hybrid: 0.62 / mean mm³ | Không hợp nhất constant hoặc đổi semantic khi di chuyển code. |
| Missing bị che trong debug | `app.py:debug_info` đổi None thành 0; `schemas.py:PredictResponse.to_dict` round cả None | Debug phải giữ null; serializer phải hỗ trợ null. |
| Lỗi đo từng hạt bị nuốt | `execute_grain_classification_and_metrics`: `except Exception: pass` | Vẫn bỏ qua hạt đo lỗi như hiện tại nhưng ghi warning và số lượng bị loại. |
| API async đang chạy CV đồng bộ | `/predict` gọi trực tiếp CV/TensorFlow/SAHI | Đưa toàn pipeline đồng bộ sang worker thread; giới hạn inference đồng thời, giữ health responsive. |
| Colab dùng lại app | `API_Server.ipynb` có `from app import app`, nhưng preflight hardcode file regression cũ | Giữ import launcher; dùng chung Settings và loader cho notebook. |
| Client hiện có cần tương thích | Node gateway gọi `/predict`; phone gọi gateway; tests import `app` và patch `app.execute_*` | Giữ wire contract, cập nhật test patch vào dependency mới. |

Lưu ý: hash chỉ chứng minh tính toàn vẹn so với hash được ghi, không tự chứng minh model/scaler cùng lần train. Đặt hai file cùng folder cũng không tự chứng minh điều đó. Export hiện tại cung cấp pair từ cùng pipeline; người dùng phải copy cả folder từ cùng lần export. Runtime mới báo tương thích kỹ thuật, không gọi đó là chứng minh nguồn gốc hay độ chính xác khoa học.

## 3. Trải nghiệm cấu hình cuối cùng

Thêm `REGRESSION_MODEL_DIR` vào `AI_SERVICES/.env.example`; nạp `.env` bằng `python-dotenv` với `override=False`. Biến môi trường tiến trình có ưu tiên cao hơn `.env`. Nếu dependency chưa có thì thêm vào requirements/setup dùng chung.

Đường dẫn tương đối luôn tính từ AI_SERVICES, không từ working directory. Hỗ trợ đường dẫn tuyệt đối Windows/Colab, dấu cách và Unicode. Đường dẫn đã cấu hình sai phải báo lỗi rõ, không tự chọn model khác.

Ví dụ nghiên cứu (minh họa, không tự chuyển active model sang ARD):

```dotenv
REGRESSION_MODEL_DIR=../LINEAR_REGRESSION_MODEL/models/ard
```

Ví dụ deploy sau khi người dùng copy cùng bundle:

```dotenv
REGRESSION_MODEL_DIR=artifacts/regression/ard
```

Đổi đường dẫn hoặc train ghi đè folder: restart service để nạp snapshot mới. Không hot reload theo từng request và không ghi ngược vào folder training. Notebook Colab cho phép đặt `os.environ['REGRESSION_MODEL_DIR']` trước import app, rồi khởi động lại server/runtime khi đổi model.

### Folder contract

```text
<selected_model_folder>/
  model.joblib           # fitted estimator, bắt buộc
  scaler.joblib          # fitted scaler cùng export, bắt buộc mặc định
  feature_schema.json    # ordered features + schema_version, bắt buộc
  config.json            # metadata training/version, nếu có thì đọc
  pipeline.joblib        # có thể tồn tại nhưng runtime KHÔNG dùng
  metrics.json           # optional, không quyết định prediction
  manifest.json          # optional, không gate bằng hash
  scaler_params.json    # optional, không tái dựng scaler từ JSON
```

Scope chính là estimator sklearn-compatible đã serialize bằng joblib và scaler transform giữ nguyên 31 chiều. Không hứa hỗ trợ tùy ý JSON trọng số, ONNX, PCA hoặc custom model không có package cần thiết.

Model thực sự train không scaler: chỉ chấp nhận `config.json` có trường máy đọc rõ `"preprocessing": "none"` và không có scaler file. Không suy luận từ việc file scaler bị thiếu hay từ họ decision tree. Config hiện tại có chuỗi `StandardScaler in Pipeline` vẫn dùng pair bình thường; chưa cần sửa export.

Không tự chuyển sang `pipeline.joblib` khi pair lỗi, không scale hai lần, không `fit`/`fit_transform` trong inference. Với folder chỉ có pipeline: báo định dạng chưa được hỗ trợ và hướng dẫn export model/scaler riêng. Điều này làm thứ tự lựa chọn dễ hiểu dù các folder hiện tại có cả ba joblib.

### Giữ active model hiện tại trong migration

Default khi chưa cấu hình: `../LINEAR_REGRESSION_MODEL/models` để vẫn dùng Extra Trees đang deploy, không đổi sang folder `extra_trees/` vì đó có thể là model train khác.

Loader có adapter legacy nhỏ, chỉ khi folder có `best_tree_ensemble_model.joblib` + `scaler.joblib` + `scaler_params.json`, không có `model.joblib`: lấy ordered features từ `scaler_params.json.features`, so với contract hiện tại, cảnh báo legacy layout. Không đọc hash của manifest trung tâm. Nếu có canonical `model.joblib` nhưng bundle thiếu/hỏng thì fail, không fallback legacy. Không search đệ quy hoặc chọn file mới nhất.

Adapter này giữ file model/scaler cũ nguyên vẹn. Khi người dùng chọn folder chuẩn, adapter không tham gia. Không tự copy, xóa hay ghi đè weights thật trong quá trình refactor.

## 4. Cấu trúc nguồn đích

```text
AI_SERVICES/
  app.py                         # bootstrap tương thích python app.py / app:app
  src/
    rice_ai/
      __init__.py
      settings.py                # Settings; path và inference constants
      contracts.py               # dữ liệu nội bộ + PipelineError
      api/
        __init__.py
        application.py           # create_app, lifespan, dependencies, admission lock
        routes.py                # /health, /api/status, /predict
        schemas.py               # response wire contract hiện tại
      pipeline/
        __init__.py
        runner.py                # RicePipeline.run; điều phối, không chứa thuật toán
        image_io.py              # decode + request temporary workspace
        container.py             # cm -> mm, gọi detector
        grains.py                # orchestration cleaning/classification/measurement
        features.py              # assemble_31_features
      vision/                    # chuyển 6 modules CV hiện tại vào đây
        __init__.py
        container_detector.py
        grain_segmenter.py
        grain_crop_cleaner.py
        grain_classifier.py
        ellipsoid_geometry.py
        uniformity_evaluator.py
      models/
        __init__.py
        vision_models.py         # lazy load YOLO/CNN, cache + lock
        regression_loader.py     # folder loader + LoadedRegression
      estimation/
        __init__.py
        feature_schema.py        # 31-feature definitions, validation, hybrid
        regression.py            # validate -> transform -> predict
        geometry.py              # geometry estimate
        weight.py                # weight estimate
        fusion.py                # select_final_estimate, mode/fallback policy
  tests/
  scripts/
  artifacts/                     # weights deploy; src/rice_ai/models chứa code loader
    README.md
    yolo/                        # <model_name>/best.pt
    cnn/                         # <model_name>/<classifier>.keras hoặc .h5
    regression/                  # <model_name>/model.joblib + scaler + schema
  API_Server.ipynb                # launcher, không duplicate inference
  capture_server/                 # giữ riêng
  RICE_ESTIMATION_APPLICATION/    # giữ riêng
  .venv/                         # giữ tại AI_SERVICES
```

Mỗi package có `__init__.py`. `app.py` thêm duy nhất AI_SERVICES/src vào sys.path tại bootstrap rồi gọi `create_app()`. Tests/scripts dùng cùng bootstrap rõ ràng; không rải fallback sys.path trong từng module. Nội bộ import bằng `rice_ai.*`.

Giữ `feature_schema.py` ở root như shim re-export ngắn nếu cần cho consumer bên ngoài (training consolidation test hiện import tên này); shim không chứa business logic. Root `model_registry.py`, `regression_engine.py`, `schemas.py` và `modules/` chỉ xóa sau khi chuyển hết consumer active; shim tạm chỉ được giữ khi có consumer thực tế, có comment migration. Archive không phải runtime consumer và không cần rewrite hàng loạt.

Dependency một chiều: API -> pipeline -> vision / estimation / model providers. Các module dưới không import FastAPI, UploadFile, app hoặc gateway. Model loader không import thuật toán ảnh. Không thêm microservice, queue server, plugin framework, abstract class cho mỗi stage hay DI framework.

## 5. Hợp đồng dữ liệu và thuật toán giữ nguyên

`contracts.py` dùng dataclass/TypedDict đơn giản cho `PredictionInput`, `ContainerResult`, `GrainAnalysis`, `EstimateSet`, `PipelineResult`; tên field phản ánh đơn vị. Có thể giữ dictionary do CV trả về tại adapter boundary để tránh rewrite 6 thuật toán.

- Input HTTP: `file`, `diam`, `height`, `empty`, `wall_thickness` (cm), `weight_total`, `sample_weight` (g), `sample_count`, `estimator_mode`, `debug`.
- Sau container stage: mm, mm³, pixels_per_mm; không chuyển đơn vị lại ở nhiều nơi.
- GrainAnalysis: whole_grains, volumes_px3, total_detected, classified counts, skipped measurement count. Giữ nguyên logic chỉ hạt nguyên được dùng đo thống kê.
- Feature stage: đúng ALL_31_FEATURES; không Actual_Count; mean/min/max/std giữ ddof=0; hybrid `round(bulk_mm3 * 0.62 / mean(valid grain volume mm3))` độc lập weight/final.
- `hybrid_estimate` fallback tùy ý trong assembler không còn dùng ở workflow mới. Chỉ giữ tham số tương thích nếu consumer thật sự cần, nhưng không nhận giá trị khác semantic để lấp missing.
- Geometry: `round(bulk_mm3 * pixels_per_mm**3 * 0.82 / median(volumes_px3))`; thiếu mẫu hợp lệ -> None.
- Weight: `round(weight_total / sample_weight * sample_count)` khi các input cần thiết > 0; còn lại None.
- Regression: cùng thứ tự 31 feature -> scaler.transform một lần -> model.predict một lần; lấy một scalar hữu hạn (accept shape (1,) hoặc (1,1), reject multi-output). Check NaN/Inf trước clamp; giữ quy tắc nonnegative và round một chữ số hiện tại.
- Preserve hiện tại `Weight_g` optional/zero với warning; không mở rộng việc impute sang feature khác. Missing required -> explicit error; std=0 ở một hạt và empty_height=0 là hợp lệ. Debug giữ None/null.
- `fusion.select_final_estimate(mode, estimates)`: regression mode bắt buộc regression; geometry/weight mode chọn output tương ứng; auto ưu tiên regression rồi geometry+weight mean rồi một output còn lại. Không tự lấy trung bình ba nhánh vì regression đã dùng feature từ hình học và weight, nên chúng không độc lập.
- Tách final policy để sau này nghiên cứu fusion mới trong một file. Fusion ba nhánh/chọn trọng số tối ưu không thuộc implementation này; chưa có công thức được xác nhận.

Wire contract giữ `/health`, `/api/status`, `/predict`, existing estimation keys, `ai_est=geometry_est`, metrics_summary, warnings, timings, debug. Giữ các error 422/503/500 theo hiện tại. Nhãn model lấy từ loaded estimator; ExtraTreesRegressor giữ alias `ExtraTrees`, model khác dùng tên class thật. Thêm metadata/status theo hướng additive, không bắt client đổi cùng lúc.

## 6. Loader và lifecycle cụ thể

`load_regression_folder(folder: Path) -> LoadedRegression`:

1. Resolve folder, chọn canonical pair hoặc adapter legacy theo quy tắc trên.
2. Đọc schema; so sánh version + list ordered features chính xác. Không yêu cầu tính/hash schema. Nếu có feature_names_in_ trên scaler/model cũng phải khớp, không chỉ kiểm tra count.
3. Nạp bằng joblib; phát hiện estimator đã là Pipeline trong model.joblib thì reject để tránh double preprocessing.
4. Kiểm tra callable predict/transform, fitted state bằng API sklearn thích hợp, 31 input và output scaler shape (1,31); không khóa StandardScaler/ExtraTrees. Model có dependency như xgboost thiếu -> báo dependency cụ thể; không tự pip install.
5. Với objects fit bằng named DataFrame, truyền DataFrame có tên đúng khi cần ở boundary tương ứng; thêm pandas dependency rõ nếu dùng. Với estimator fit bằng array, truyền array. Không che cảnh báo tên feature bằng suppress toàn cục.
6. Smoke prediction bằng vector hợp lệ từ fixture synthetic test contract; finite, scalar. Zero-vector chỉ là smoke API toán học, không thay thế feature validation/domain check.
7. Capture version warnings vào diagnostics. Incompatible load -> error; load/smoke được nhưng version khác -> cảnh báo, không tuyên bố tương thích đã được chứng minh cho mọi dữ liệu.
8. Chỉ publish LoadedRegression sau khi load và checks thành công. Trạng thái configured/loaded/error + compatibility_checked; không dùng provenance.verified để gate. `verified` cũ không được hiểu là scientific validity.

Một instance loader/provider cho mỗi app. Regression nạp trong lifespan để `/api/status` phản ánh đúng trước request đầu tiên; lỗi regression cho phép service degraded và auto fallback, regression mode trả 503. YOLO/CNN vẫn lazy, lock khi load. Không load lại mỗi request; restart để đổi folder. /health chỉ liveness.

Pipeline chạy ở worker thread; một admission lock/gate cho inference mỗi app để model mutable không bị dùng đồng thời. Request đang chạy không chặn /health; request inference thứ hai trả 503 SERVER_BUSY rõ ràng, không chờ vô hạn. Giữ 1 worker là cấu hình khuyến nghị trong đợt này. TemporaryDirectory sống trong worker đến khi pipeline kết thúc; cancellation HTTP không được xóa workspace đang được worker dùng.

Mỗi stage đo timing và gắn request_id. Chỉ bắt lỗi per-grain tại chỗ có thể skip; lỗi stage fatal đi qua PipelineError rồi route map HTTP. Giữ diagnostics nhẹ, không thêm dashboard/overlay/XAI trong đợt này.

## 7. Phases thực thi

### Phase 0 — Baseline và characterization

- Đọc plan, git status/diff và các file liên quan; giữ mọi thay đổi người dùng đang có.
- Dùng `AI_SERVICES/.venv/Scripts/python.exe`; ghi phiên bản Python/sklearn và bundle đang dùng. Không tự thay interpreter hoặc install lại cả environment.
- Chạy test feature/regression/registry/API hiện tại; phân biệt lỗi baseline với regression mới.
- Chạy fixture M001A baseline một lần nếu thiếu evidence hiện tại; lưu feature vector, predictions, methods, counts, timings và warnings vào `reports/ai_services_restructure/`. Không ghi đè reports TASK-01 cũ.
- Viết characterization tests cho geometry/weight/fusion và schema response trước khi di chuyển logic. Golden values lấy output thực tế hoặc reference toán học độc lập, không fix expected=85.
- Pass: có baseline hoặc blocker được ghi chính xác. Nếu môi trường/fixture không đủ, tiếp tục unit work độc lập nhưng không tuyên bố full E2E pass.

### Phase 1 — Settings và folder loader

- Tạo settings.py, models/regression_loader.py; thêm REGRESSION_MODEL_DIR example/default/precedence; dùng loader contract mục 3 và 6.
- Thay script `verify_artifacts.py` bằng preflight folder: `--model-dir` optional override; không yêu cầu manifest/hashes/class thủ công; không ghi file vào training folder.
- Giữ chọn active model cũ bằng adapter legacy; chưa đổi thuật toán prediction.
- Tests: folder thật legacy + ARD/extra_trees chuẩn nếu dependency đủ; temp bundles nhỏ cho Ridge + StandardScaler và DecisionTree + MinMaxScaler; missing/corrupt scaler/model; schema reorder/count; optional no-scaler explicit; Pipeline bị đặt sai; invalid path và Unicode; no fit/one transform; negative/non-finite/multioutput checks.
- Pass: đổi một path và restart đổi đúng model/scaler; không phải sửa source, hash, class; file lỗi không silently chọn model khác.

### Phase 2 — Tách các giai đoạn tính toán

- Move 6 modules vào vision; giữ thuật toán, sửa relative imports. Không đổi ngưỡng SAHI 640/0.20/0.5 hay cleaner 0.15.
- `execute_container_analysis` -> pipeline/container.py.
- `execute_sahi_crops` gọi vision segmenter với model từ vision_models; không giữ global model ở app.
- `execute_grain_classification_and_metrics` -> pipeline/grains.py, gọi cleaner/classifier/geometry modules riêng; thêm diagnostic skip nhưng giữ cách lấy mẫu.
- `assemble_31_features` -> pipeline/features.py; feature_schema -> estimation/feature_schema.py.
- `compute_final_estimates` -> geometry.py, weight.py, fusion.py; tách final selection ở bước 9 vào cùng fusion.py.
- Regression -> estimation/regression.py; remove hardcoded OLS path khỏi active code và sửa model name thực tế.
- Pass: stage tests bằng synthetic fixture giữ numerical parity với baseline, hybrid 0.62/mean và geometry 0.82/median không bị trộn; không có HTTP dependency ở stage.

### Phase 3 — Runner, API và launchers

- Tạo RicePipeline nhận providers/dependencies qua constructor; run(input, image_bytes, request_id) trả PipelineResult. Runner thể hiện thứ tự stage, không chứa công thức hoặc đọc env.
- Tạo create_app(settings=None, pipeline=None), app.state lifecycle, route dependency để test inject. Move schemas, HTTP validation và response assembly; xử lý null serializer.
- Giữ multipart API, status/error contract. Move CV vào worker và admission gate như mục 6. Test cleanup success/error/cancellation.
- Root app.py chỉ bootstrap/re-export app và uvicorn main; batch script vẫn chạy được. Notebook chỉ dùng Settings/loader preflight, app: không hardcode regression filenames cũ và không duplicate pipeline.
- Tests đổi patch target `app.execute_*` thành stage/provider dependency thật; dùng TestClient context manager để lifespan chạy. Không sửa assertion để che lỗi.
- Consumer shim: giữ feature_schema import cũ nếu external test cần; update internal imports sang rice_ai. Rà scripts/export_artifact_bundle.py: không còn mutate manifest active; sửa thành wrapper preflight hoặc đánh deprecated rõ nếu không còn cần, không tự xóa artifacts nó từng tạo.
- Pass: local launcher và Colab bootstrap dùng cùng package/cấu hình; health responsive khi inference fake đang chạy; API client cũ đọc được response.

### Phase 4 — Verification tích hợp và model switching

- Chạy layers test theo mục 8; chạy E2E cùng active legacy bundle sau refactor để so parity.
- Test folder switching bằng direct regression fixture cho ít nhất 2 model family; khi đã có per-stage parity, chỉ cần một image E2E cuối cho active model, không chạy ảnh chậm cho mọi model.
- Copy bundle test vào temporary deploy folder, trỏ path mới: cùng feature vector phải cho raw prediction bằng bundle gốc (rtol=1e-9, atol=1e-9 trong cùng process/environment).
- E2E phải assert regression_est hữu hạn và method thực sự là regression trong regression mode; auto-only success có thể che fallback. Nếu trước/sau cùng weights khác numerical output vượt tolerance, dừng để điều tra; không đổi expected theo code mới.
- Không yêu cầu prediction bằng Actual_Count 85; báo sai số như measurement, không dùng một ảnh làm chứng minh model tốt.
- Pass: tests critical thực thi pass, skip ghi NOT VERIFIED, API và same-model E2E parity có evidence.

### Phase 5 — Hướng dẫn và cleanup có giới hạn

- README: folder contract, cấu hình R&D/deploy, restart, environment/dependency versions, preflight và test commands, sơ đồ trách nhiệm module, model/scaler từ cùng export.
- .env.example chứa một dòng đường dẫn regression dễ sửa; không commit .env thật.
- Loại code implementation cũ chỉ sau import search/tests; giữ shim có lý do rõ. Giữ artifacts manifest cũ như historical evidence nếu docs/report tham chiếu; đánh dấu không còn điều khiển runtime. Không xóa model đang dùng, notebook training, report cũ hay archive ngoài scope.
- Viết `reports/ai_services_restructure/verification.md`: command/output thực tế, trước/sau, limitation và skips. PROJECT_STATUS chỉ cập nhật factual khi có yêu cầu hoặc nằm trong prompt execution; không gọi refactor này scientifically validated.
- Pass: từ fresh shell user chỉ sửa folder path, preflight, restart và predict; runtime source có một canonical implementation cho mỗi trách nhiệm.

## 8. Test matrix và lệnh

Chạy từ project root, đúng interpreter. Các test mới là deliverables của implementation, chưa tồn tại trong session lập plan.

```powershell
# Baseline (trước migration)
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_model_registry AI_SERVICES.tests.test_api_endpoints -v

# Sau implementation
& AI_SERVICES/.venv/Scripts/python.exe -m compileall -q AI_SERVICES/src AI_SERVICES/app.py AI_SERVICES/scripts AI_SERVICES/tests
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_settings AI_SERVICES.tests.test_regression_loader AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_estimators AI_SERVICES.tests.test_pipeline AI_SERVICES.tests.test_api_endpoints -v
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py --model-dir ../LINEAR_REGRESSION_MODEL/models/ard
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_real_pipeline -v
```

CLI relative model-dir cũng tính từ AI_SERVICES (ghi help rõ); absolute luôn dùng nguyên vị trí. Test legacy registry cũ chuyển thành loader tests, không giữ assertions hash-only như gate mới. Có thể giữ test filename cũ nếu tiện nhưng sửa danh sách command trong README/report tương ứng.

| Layer | Trường hợp bắt buộc | Pass condition |
|---|---|---|
| Settings/import | process env > .env > default, cwd khác, Unicode Windows, POSIX Colab | Resolve cùng folder, import không load CV weights |
| Loader | canonical/legacy, hai họ model, hai scaler, missing/corrupt/schema mismatch | Error rõ; không fallback ngầm; chỉ publish bundle hợp lệ |
| Preprocessing | transform exactly once, no fit, names/order, explicit no-scaler | Raw output khớp gọi trực tiếp pair |
| Feature | 31/order/target excluded, hybrid reference 0.62, valid zero, missing/nonfinite | Semantic parity; không thêm imputation |
| Estimation | geometry, weight, all auto fallback combinations, rounding | Independent reference cases đúng; method_used trung thực |
| API | health/status, invalid image/input, all four modes, regression unavailable | Existing fields/status preserved; debug null không 500 |
| Lifecycle | cache once, failed load không partial publish, concurrency, cleanup | Health responsive; second inference SERVER_BUSY; no temp leak |
| Relocation | cùng bundle ở training folder và temp deploy folder | Cùng raw prediction |
| E2E | M001A với weights hiện tại, strict regression mode | Regression thực sự chạy; compare baseline cùng model |

Fixture hiện có: `DATASET_BUILDER/1_Raw_Images/M001/M001A.jpg`; form diam=1.78, height=3.39, empty=1.09, weight_total=2.69. Ground truth 85 chỉ dùng báo sai số. E2E cũ dùng auto và chỉ kiểm tra final>0 nên phải tăng assertion để không nhầm fallback với regression pass.

Kiểm tra manual Colab dùng đúng DRIVE_PROJECT thực tế người dùng đang chạy; không hardcode lại GROUP_MEMBERS/MEMBERS vào core package. Không có Colab session thì ghi Colab runtime NOT VERIFIED; static import/bootstrap test không thay cho runtime test.

## 9. Rủi ro và giới hạn đã biết

- Version training ARD ghi Python 3.13.15/sklearn 1.6.1; runtime requirements hiện chỉ giới hạn sklearn>=1.3.0. Ghi phiên bản load thực tế và test direct pair; không sửa version toàn môi trường theo phỏng đoán.
- Folder có pair sai cùng dimension có thể vẫn predict được; bỏ hash không giải quyết provenance và hash cũng không chứng minh pairing. Dùng whole-folder export, parity với pipeline.joblib trong test kiểm chứng nếu có (chỉ test, không runtime fallback).
- Training đang override trực tiếp folder: không retrain ghi đè đồng thời lúc server khởi động. Model cached giữ snapshot; restart sau khi export hoàn tất.
- Gateway hiện timeout 120s và không forward estimator_mode/debug. Đợt này test Python API trực tiếp; không hứa mọi CPU E2E dài sẽ qua gateway. Nếu cần sửa gateway UX/timeout, báo follow-up riêng.
- Weight mode hiện cũng đi qua vision. Giữ behavior trong migration, không tự thêm bypass thay đổi response metrics; tối ưu riêng sau baseline.
- Không sửa lỗi thuật toán nhận diện trong lúc di chuyển. Mọi thay đổi numerical ngoài fixes rõ trong plan cần giải thích và test độc lập.

## 10. Definition of Done và stop conditions

- [ ] Một REGRESSION_MODEL_DIR chọn bundle, không cập nhật hash/class/hệ số thủ công.
- [ ] Training folder và deploy folder chạy cùng code; pair được load trực tiếp; không fit và không double-scale.
- [ ] Active model cũ giữ nguyên trong migration; thử model khác không hardcode nhãn ExtraTrees.
- [ ] Feature count/order/unit/hybrid semantics giữ nguyên và được test.
- [ ] Có module riêng cho IO, container, grains, features, geometry, weight, regression và final selection.
- [ ] App/route không chứa CV hoặc công thức estimation; modules độc lập với HTTP.
- [ ] Windows launcher, Colab launcher và API compatibility được kiểm tra đúng mức; thiếu môi trường ghi rõ.
- [ ] Test layers và same-model E2E pass; skips không bị tính thành verification.
- [ ] README hướng dẫn đủ để người khác đổi model bằng folder path.

Dừng phần phụ thuộc nếu schema không rõ, scaler thiếu mà không explicit none, artifact load fail/dependency missing, parity khác cùng weights, hoặc legacy artifact không còn. Không tự sửa feature order cho hợp model, không train model thay thế, không fake provenance, không hardcode prediction hay sửa expected cho test pass. Vẫn hoàn tất phần độc lập và báo blocker.

Nếu người dùng muốn thay công thức fusion ba output, cần công thức/tiêu chí đánh giá riêng; plan hiện hoàn tất việc tách fusion và giữ current policy, không suy đoán trọng số.

## 11. Execution checklist for Gemini 3.8 Flash

- [ ] Đọc toàn bộ plan; git status/diff; bảo toàn thay đổi có sẵn; không commit/push tự động.
- [ ] Dùng AI_SERVICES/.venv/Scripts/python.exe; ghi baseline tests và same-model fixture evidence.
- [ ] Thêm src/rice_ai và package imports; tạo settings.py với REGRESSION_MODEL_DIR theo mục 3.
- [ ] Viết regression_loader.py: canonical pair, schema list, legacy adapter, atomic publish, no mandatory hash.
- [ ] Cho phép scaler khác StandardScaler qua transform interface; no-scaler chỉ explicit none.
- [ ] Không dùng pipeline.joblib làm runtime fallback; không dựng trọng số/scaler bằng tay.
- [ ] Sửa verify_artifacts.py theo loader; bỏ hardcoded ExtraTrees/StandardScaler gate.
- [ ] Test hai họ model, hai scaler, folder relocation, lỗi schema/file, no fit và one transform.
- [ ] Move 6 CV modules vào vision; giữ thresholds/algorithm và sửa imports.
- [ ] Tách container/grain orchestration, features và model cache ra khỏi app.
- [ ] Move feature_schema, giữ 31v1, 0.62/mean, ddof=0; giữ root shim nếu consumer cần.
- [ ] Tách geometry 0.82/median, weight và final selection thành ba modules.
- [ ] Regression dùng actual model name, finite scalar validation, giữ rounding/clamp; bỏ active OLS constants.
- [ ] Tạo RicePipeline.run và data contracts; stage không import HTTP.
- [ ] Tạo create_app/lifespan/routes; root app.py chỉ launcher tương thích.
- [ ] Worker inference + admission gate; status readiness thật; giữ errors/response fields.
- [ ] Giữ null trong debug và serialization; báo skipped grain measurements.
- [ ] Sửa test patch targets và TestClient lifespan; không bỏ negative tests.
- [ ] Colab preflight dùng chung settings/loader; không duplicate business logic.
- [ ] Chạy commands/test matrix; strict regression E2E M001A; so baseline cùng weights.
- [ ] Đổi folder deployment temporary chỉ bằng path và xác nhận parity.
- [ ] README + .env.example + verification report; liệt kê những gì NOT VERIFIED.
- [ ] Rà imports trước cleanup; không xóa model thật, dataset, training notebook, evidence cũ.
- [ ] Báo file changed, tests, model active, limitation/blocker và cách chạy; dừng.

## 12. Bổ sung bắt buộc — Kho model triển khai cho toàn hệ thống

Yêu cầu bổ sung của người dùng: chuẩn bị riêng một folder chung chứa YOLO, CNN và regression khi chốt hệ thống. Mục này mở rộng cấu hình regression-only ở các phase trên; implementation agent phải thực hiện cùng plan.

### Cấu trúc và cấu hình

```text
AI_SERVICES/artifacts/
  README.md
  yolo/
    rice_segmentation/
      best.pt
  cnn/
    grain_classifier/
      best.keras
  regression/
    ard/
      model.joblib
      scaler.joblib
      feature_schema.json
      config.json                # nếu export có
```

Tên model là tên dễ đọc, không thêm cấp run-ID/timestamp. artifacts chứa weights/metadata; src/rice_ai/models chứa mã loader. File .h5 cũng dùng được nếu tương thích CNN loader hiện tại. Tên model và tên file trong ví dụ chỉ minh họa, không phải yêu cầu tự chọn model mới.

Lập trình viên chỉnh AI_SERVICES/.env; settings.py đọc và validate tập trung, providers nhận Settings:

```dotenv
YOLO_MODEL_PATH=artifacts/yolo/rice_segmentation/best.pt
CNN_MODEL_PATH=artifacts/cnn/grain_classifier/best.keras
REGRESSION_MODEL_DIR=artifacts/regression/ard
```

YOLO/CNN trỏ file cụ thể để giữ nguyên tên weights đã export; regression trỏ folder chứa pair và schema. Áp dụng cùng precedence process env > .env > default, relative path tính từ AI_SERVICES, hỗ trợ absolute/Unicode Windows và Colab. Khi nghiên cứu, hai file vision có thể ở RESULTS hoặc ngoài repository, regression ở folder training. Khi deploy, copy weights/bundle cùng metadata cần thiết vào artifacts, đổi ba đường dẫn và restart. Không cần sửa source hoặc hash thủ công.

Đổi vị trí không đổi preprocessing: CNN phải phù hợp class mapping, target size và preprocessing của GrainClassifier; YOLO phải là segmentation model tương thích SAHI. Không coi mọi file cùng phần mở rộng là thay thế được. Giữ metadata từ export nếu có, không bắt tạo manifest mới.

Khi chưa đặt YOLO_MODEL_PATH/CNN_MODEL_PATH, giữ thứ tự candidates hiện tại để baseline không đổi, nhưng gom resolve logic vào Settings. Nếu explicit path thiếu hoặc load lỗi thì báo đúng component, không fallback sang weights khác. Folder deployment rỗng không được ưu tiên hơn model đang dùng. Không thêm hardcoded Drive path cá nhân mới vào core.

### Bổ sung vào các phase và nghiệm thu

- Phase 1: thêm YOLO_MODEL_PATH/CNN_MODEL_PATH vào Settings và .env.example, nối vision_models provider đến resolved paths. Tạo skeleton artifacts/yolo, artifacts/cnn, artifacts/regression bằng .gitkeep và artifacts/README.md; chưa tạo folder model minh họa có weights giả.
- Không tự copy, di chuyển hoặc ghi đè weights thật trong refactor; không tự đưa checkpoint lớn vào Git. Giữ artifacts/manifest.json cũ như evidence theo plan.
- Phase 3: launcher local và Colab đều lấy ba cấu hình từ Settings; không hardcode weights lần nữa trong notebook.
- Phase 4: test cả hai vision paths ở training/deploy, relative/absolute/Unicode; explicit missing path không fallback; candidates mặc định giữ behavior cũ. Mock loader để assert đúng file resolved, tránh copy checkpoint lớn chỉ để test đường dẫn. Baseline E2E phải vẫn dùng cùng weights.
- Phase 5: .env.example và README hướng dẫn đủ ba đường dẫn, cách copy model chốt, giữ metadata/preprocessing và restart. README artifacts phân biệt weights với code loader.
- DoD bổ sung: có ba nhánh folder và tài liệu; đổi vị trí YOLO/CNN/regression chỉ bằng cấu hình, không sửa source; test resolution/provider pass; weights đang chạy được giữ nguyên.

### Checklist bổ sung cho implementation agent

- [ ] Tạo skeleton artifacts/{yolo,cnn,regression} và README, không di chuyển model thật.
- [ ] Thêm YOLO_MODEL_PATH/CNN_MODEL_PATH bên cạnh REGRESSION_MODEL_DIR trong Settings/.env.example.
- [ ] Sửa vision_models provider dùng Settings; explicit path lỗi không tự chọn weights khác.
- [ ] Đồng bộ local/Colab, test path resolution và baseline weights như Phase 4 bổ sung.
- [ ] Ghi hướng dẫn R&D -> deploy cho cả ba loại model và kiểm tra DoD bổ sung.
