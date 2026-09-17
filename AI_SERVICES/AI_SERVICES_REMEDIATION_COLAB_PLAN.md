# AI Services — Remediation and Colab Runtime Plan

Status: READY FOR IMPLEMENTATION  
Created: 2026-09-18  
Scope: sửa lỗi sau tái cấu trúc, quản lý notebook runtime trong src, ưu tiên Colab.  
Implementation target: Gemini 3.8 Flash.

## 1. Mục tiêu và quan hệ với plan cũ

Đây là plan follow-up riêng cho implementation đã có tại `src/rice_ai`. Không chạy lại toàn bộ AI_SERVICES_RESTRUCTURING_PLAN.md, không thiết kế lại kiến trúc. Với phần sửa lỗi và vị trí notebook, tài liệu này thay thế chỉ dẫn tương ứng của plan cũ; các invariants khoa học và folder model vẫn giữ nguyên.

Mục tiêu nghiệm thu: đổi model bằng đường dẫn hoạt động đúng; API tương thích client; loader từ chối bundle không hợp lệ; runtime ưu tiên GPU khi có; notebook Colab được quản lý gọn trong src và restart được rõ ràng; chỉ còn một implementation active cho mỗi công đoạn.

Session tạo plan chỉ thêm tài liệu này. Không implement, move notebook, xóa source, chạy training hoặc commit/push trong session planning.

## 2. Baseline và bằng chứng review kế thừa

Working tree chứa implementation chưa commit của Gemini, plan cũ và reports untracked. Base commit đã review: `7e7063d`. Trước khi triển khai phải kiểm tra lại git status/diff, không giả định trạng thái vẫn giống session review.

Inherited evidence từ review trước, KHÔNG phải tests vừa chạy trong session tạo plan:

- 62 tests pass với AI_SERVICES/.venv/Scripts/python.exe và PYTHONIOENCODING=utf-8. Lần không đặt UTF-8 có lỗi print tiếng Việt trong regression_engine legacy.
- Preflight legacy Extra Trees và ARD pass; ARD có warning về feature names.
- M001A strict regression: HTTP 200, regression=85.2, final=85, geometry=282, hybrid feature=167, 90 detected/6 whole, khoảng 127 giây CPU.
- Probe trong bộ nhớ: loader chấp nhận schema_version sai, scaler.feature_names_in_ đảo thứ tự, model hai output; preprocessing=none bỏ qua scaler tồn tại.
- Probe route: cân/số mẫu âm biến thành None và tới pipeline; hủy coroutine request thả admission gate trong khi worker vẫn chạy.

Không dùng một ảnh đúng 85 để tuyên bố độ chính xác chung 100%. Không có evidence Colab runtime mới đã pass.

## 3. Các sửa đổi bắt buộc

| Priority | File/symbol | Lỗi cần xử lý |
|---|---|---|
| P1 | models/regression_loader.py: load_regression_folder, LoadedRegression.predict, _run_smoke_test | Kiểm tra schema/version, object feature names, count, fitted state, output shape, cấu hình scaler mâu thuẫn. |
| P1 | models/vision_models.py: get_yolo_model | device='cpu' làm Colab bỏ GPU. |
| P1 | api/routes.py: predict | Response đổi keys mà React/phone còn đọc keys cũ. |
| P1 | api/routes.py, api/application.py | Cancellation thả gate trước khi worker kết thúc. |
| P2 | contracts.py: PredictionInput.validate; routes.py | Reject cân âm/non-finite trước khi normalize; giữ zero khác missing. |
| P2 | pipeline/grains.py, runner.py, API timing | Đang gộp cleaning/CNN vào segmentation_ms; classification_ms=0 dù CNN chạy. |
| P2 | providers, routes.py:get_system_status | Theo dõi load errors, capabilities trung thực; không đặt exception/absolute path vào file. |
| P2 | API_Server.ipynb, requirements, README | Notebook chưa theo Settings, dependency chưa rõ, preflight hardcode model legacy. |
| P2 | app.py, modules/, legacy regression/registry/scripts/tests | Code trùng và tests còn kiểm implementation cũ. |
| P2 | reports/ai_services_restructure/verification.md | Claim hoàn thành/compatibility/accuracy quá mức; mô tả schema sai. |

## 4. Tổ chức notebook và đường dẫn đích

```text
AI_SERVICES/
  app.py                         # launcher Windows, không chứa thuật toán lặp
  .env.example
  requirements.txt               # dependencies local hiện có + khai báo trực tiếp cần thiết
  requirements-colab.txt         # dependencies phục vụ notebook/Colab, xem Phase 3
  src/
    rice_ai/                     # source package inference hiện có
      settings.py
      api/
      models/
      pipeline/
      vision/
      estimation/
    notebooks/
      README.md                  # cách chạy và thứ tự cell
      API_Server.ipynb           # notebook runtime active duy nhất
  artifacts/
    yolo/
    cnn/
    regression/
  tests/
  scripts/
```

Di chuyển `AI_SERVICES/API_Server.ipynb` sang `AI_SERVICES/src/notebooks/API_Server.ipynb`, giữ tên để người dùng dễ nhận biết. Không để bản copy launcher active ở root. Chỉ move sau khi đối chiếu các cell với notebook mới và kiểm tra reference. Không gom training notebooks từ phần khác của project, không di chuyển archive_legacy notebooks: chúng không phải entrypoint runtime hiện hành.

Notebook chạy được dù mở từ Drive/Colab; không dùng __file__ hoặc cwd để suy ra project root. Một cell khai báo PROJECT_ROOT tuyệt đối tới MAIN_SOURCES; validate `PROJECT_ROOT/AI_SERVICES/src/rice_ai` trước import. Ví dụ đường dẫn có MEMBERS/GROUP_MEMBERS chỉ là ví dụ, không tự đoán hoặc ghi cứng vào core.

AI_SERVICES_DIR = PROJECT_ROOT / 'AI_SERVICES'; SRC_DIR = AI_SERVICES_DIR / 'src'; ENV_FILE = AI_SERVICES_DIR / '.env'. Thêm SRC_DIR vào sys.path một lần. Dùng create_app(Settings(...)) cho runtime mới; không phụ thuộc import `app` đã cached giữa nhiều lần chạy cell. Relative model paths vẫn tính từ AI_SERVICES, không từ src/notebooks.

## 5. Invariants giữ nguyên

- REGRESSION_MODEL_DIR trỏ bundle, YOLO_MODEL_PATH/CNN_MODEL_PATH trỏ weights; không bắt hash/manifest trung tâm và không hardcode hệ số regression.
- Exactly 31 inputs, order/version hiện tại; target Actual_Count không ở vector.
- Hybrid feature 0.62/mean mm³, geometry 0.82/median px³, ddof=0; không thay công thức hoặc fusion policy.
- Không train/refit khi inference. Scaler.transform đúng một lần; không fallback sang pipeline.joblib.
- Raw regression prediction kiểm tra hữu hạn trước clamp/round; giữ round/clamp hiện hành sau validation.
- Bảo toàn weights/dataset/source edits có sẵn. Không tự đổi active model, không benchmark nhiều model, không TASK-02+.

## Phase 0 — Recover và bổ sung regression tests

Files: tests hiện có, thêm test_vision_models.py và test_api_lifecycle.py khi cần; reports/ai_services_restructure.

1. git status --short, git diff --stat, git diff. Đọc plan này và source liên quan; không reset/stash/restore.
2. Xác định venv đúng. Local dùng AI_SERVICES/.venv/Scripts/python.exe và UTF-8. Ghi Python, sklearn, numpy, torch, CUDA; không in .env/token.
3. Chạy fast suite baseline. Reuse M001A evidence nếu code chưa thay, không chạy E2E trước/sau lặp không cần thiết.
4. Chuyển các probe review thành tests thật trước fix: wrong version, reversed object names, multi-output, no-scaler conflict, negative inputs, cancel while worker active, API keys, GPU selection.

Pass: các negative tests tái hiện fail hiện tại; có baseline rõ. Không đổi expected để pass.

## Phase 1 — Loader correctness và GPU configuration

Files: src/rice_ai/models/regression_loader.py, vision_models.py, settings.py, estimation/regression.py; tests tương ứng; .env.example; scripts/verify_artifacts.py.

### Regression

- Canonical schema_version bắt buộc đúng FEATURE_SCHEMA_VERSION; thiếu/sai -> error. List schema features đúng thứ tự ALL_31_FEATURES. Legacy lấy features từ scaler_params như adapter hiện tại, báo legacy rõ.
- Validate fitted state bằng sklearn check_is_fitted phù hợp; yêu cầu estimator/scaler n_features_in_=31 trong phạm vi supported sklearn estimators. Unsupported/custom object không có metadata đủ -> error hữu ích, không tự gán 31.
- Nếu có feature_names_in_ trên model hoặc scaler, list phải khớp ALL_31_FEATURES. Kiểm tra cả hai, không chỉ JSON.
- Input builder: object có feature_names_in_ nhận DataFrame đúng cột; object fit array nhận ndarray. Nếu dùng pandas thì khai báo dependency trực tiếp. Scaler output có thể chuyển array rồi dựng DataFrame cho model khi model cần names; không đổi thứ tự bằng phỏng đoán.
- Dùng chung routine transform/predict giữa smoke và runtime để không lệch checks: vector hữu hạn (1,31), scaler output hữu hạn (1,31), model output chỉ (1,) hoặc (1,1); reject (1,2), empty, NaN/Inf. Không lấy [0][0] để che multi-output.
- preprocessing=none chỉ hợp lệ khi scaler file không tồn tại; tồn tại cùng lúc -> reject ambiguous bundle. Không có none và scaler thiếu -> reject. Config JSON tồn tại nhưng parse lỗi -> reject, không chỉ warning.
- Publish provider cache sau toàn bộ check pass. Lỗi load không tạo partial bundle. Đọc tên estimator thực tế; alias ExtraTrees chỉ cho ExtraTreesRegressor.
- Unit reference so raw model.predict(scaler.transform(X)) với routine mới TRƯỚC rounding; kiểm tra 1 transform, 0 fit calls. Giữ các tests legacy/ARD/synthetic Ridge/DecisionTree.

### GPU

- Settings thêm YOLO_DEVICE=auto|cpu|cuda|cuda:N; default auto. auto chọn cuda:0 nếu torch.cuda.is_available(), không thì cpu; explicit CUDA không khả dụng -> lỗi rõ, không fallback âm thầm.
- vision_models dùng resolved device thay device='cpu'. Status/log cho biết thiết bị đã chọn; không nạp weights chỉ để đọc /health.
- Mock CUDA availability để test không cần GPU local; Colab acceptance phải xác nhận device thực của model. Không tự thay CNN backend/preprocessing trong fix này; ghi backend/device thực tế nếu API backend cung cấp được, không suy luận CNN GPU chỉ từ YOLO GPU.
- Model path defaults giữ thứ tự candidates cũ; không tự ưu tiên file artifacts mới khi user chưa chọn. Explicit path validate is_file, không chỉ exists. MAX_CONCURRENT_INFERENCES chỉ chấp nhận 1 trong đợt này để không quảng bá parallel model inference chưa kiểm chứng; reject giá trị khác với message rõ.

Pass: negative bundle tests reject đúng; hai họ model/scaler hợp lệ pass; GPU policy đúng dưới mocks và auto vẫn chạy CPU local.

## Phase 2 — API compatibility, lifecycle, diagnostics

Files: contracts.py, api/routes.py, api/application.py, api/schemas.py, pipeline/grains.py, runner.py, models/vision_models.py; tests API/pipeline.

### API và validation

- PredictionInput.validate kiểm finite/domain toàn bộ số đo, weight_total/sample_weight/sample_count: âm/non-finite reject 422; sample_count nguyên >=0. Route truyền raw parsed values vào validate trước normalize. Preserve legacy empty<=height tại input boundary; downstream xử lý empty sample rõ, không lén đổi thành empty<height.
- Zero hợp lệ giữ zero, không chuyển thành None; missing weight tiếp tục policy optional hiện có, warning minh bạch. Không mở rộng imputation sang feature khác.
- Khôi phục metrics_summary: total_grains_detected, whole_grains_surface, uniformity_rate_pct, pixels_per_mm, bulk_rice_volume_mm3, avg_length_mm, avg_width_mm, avg_thickness_mm, regression_model, estimator_mode. Alias mới có thể giữ additive. Tính averages từ whole_grains; regression_model phản ánh estimator chạy thực tế.
- model_bundle lấy định danh bundle/provider, không gán method_used. features_used chỉ trả khi debug=True như trước; debug giữ null và feature_validation metadata tương thích.
- Phân lỗi load 503 MODEL_UNAVAILABLE và predict failure 500 INFERENCE_FAILED; auto fallback có warning đúng loại. Không gom mọi exception regression thành 503.

### Lifecycle

- Admission token thuộc về công việc worker, không chỉ coroutine HTTP. Trước submit nếu lỗi đọc input/validation thì release; sau submit chỉ release khi worker THỰC SỰ kết thúc.
- Giữ asyncio future/task trong app.state pending set. Await bằng shield để cancellation caller không hủy wrapper future và làm callback chạy sớm; release ở completion callback trên event loop; cleanup pending đúng một lần. Không release lần nữa trong finally của request sau submit.
- Shutdown chờ pending worker hoàn tất trước đóng loop/model resources. Test cancellation sau worker-start: health trả ngay, request thứ hai vẫn 503, worker kết thúc mới được nhận request tiếp. Test success/error trước submit/error worker không leak token.
- RequestWorkspace vẫn nằm trong worker và được dọn sau stage; không để coroutine canceled dọn folder worker còn dùng. Giữ inference serial, không thêm queue server.

### Timing và readiness

- process_grains ghi timing thực cho segmentation, cleaning, classification, measurement/uniformity. Dùng field timings của GrainAnalysis hoặc result stage, không import API TimingInfo vào core.
- runner merge stage durations; geometry_ms giữ estimator time; thêm measurement_ms/uniformity_ms additive nếu cần, không trộn vào CNN. API truyền cleaning_ms/classification_ms thực. Với mocked clock test tổng/nhãn đúng; E2E CNN đã chạy phải có classification_ms>0.
- VisionModelProvider lưu state configured/loaded/error, sanitized error và actual device; nếu weights tồn tại nhưng load fail thì status error, không configured/ready.
- Status capabilities phải phản ánh component khả dụng; regression phải loaded+compatibility_checked mới ready. Không load thêm model chỉ để status. Trường file chỉ basename, error riêng không lộ absolute path; full diagnostics vào server log.

Pass: old client keys đầy đủ, invalid input không vào pipeline, cancellation gate đúng, timing đúng stage, error/readiness tests pass. Không sửa React/phone để thích nghi với breaking response không cần thiết.

## Phase 3 — Colab-first notebook trong src/notebooks

Files: move API_Server.ipynb -> src/notebooks/API_Server.ipynb; src/notebooks/README.md; requirements-colab.txt; requirements.txt; settings.py; tests/test_colab_notebook.py; README references.

### Cấu trúc cell bắt buộc

1. **Hướng dẫn**: bật GPU runtime, mount Drive, điền PROJECT_ROOT, nơi sửa model paths, cách restart khi đổi code/model; notebook chỉ vận hành, không chứa CV/regression formulas.
2. **Mount và project root**: mount Drive; khai báo duy nhất root và optional model/device overrides; validate source path; dùng pathlib, không __file__/cwd assumptions.
3. **Dependencies/environment**: cài từ requirements-colab.txt bằng interpreter notebook, kiểm tra exit code. Không rải pip install tên package khác giữa các cell. In Python/sklearn/torch/CUDA/device; nếu cần restart runtime sau package change, dừng import/server và hướng dẫn chạy lại từ đầu.
4. **Settings**: add SRC_DIR; đọc ENV_FILE qua shared Settings; optional override model/device qua process env trước tạo Settings. Không print token hoặc dump toàn CONFIG. Cho biết resolved model paths trong notebook người vận hành, không public API.
5. **Preflight**: dùng regression loader và VisionModelProvider cùng Settings; regression canonical hoặc legacy từ path đã chọn, không check filenames ExtraTrees cố định. Nạp vision để lỗi weights/device xuất hiện trước mở tunnel; tái sử dụng providers cho app, tránh load đôi. Missing artifacts -> dừng start cell với lỗi rõ.
6. **Start server và ngrok**: create_app mới với providers; dùng asyncio loop đang có trong notebook, lưu server/server_task/tunnel handles. Chỉ mở tunnel và công bố URL sau server startup + local /health và /api/status thành công; timeout bounded với cleanup nếu startup lỗi. Không asyncio.run trong notebook loop; không tạo nhiều server khi rerun.
7. **Smoke/E2E tùy chọn**: GET health/status từ localhost và public URL; dùng fixture/form M001A đã định nghĩa, strict regression; in selected device/model, timing, result; không default gọi expensive E2E mỗi lần run all mà không có switch RUN_E2E rõ.
8. **Stop/restart**: đóng tunnel do notebook này tạo, server.should_exit rồi await server_task và pending jobs hoàn tất; xóa handles. Chạy hai lần stop an toàn. Start khi đang chạy thì báo dùng stop/restart cell, không tự giành cổng. Đổi model path: stop -> tạo Settings/providers/app mới -> start. Đổi source đã import: hướng dẫn restart runtime, không dùng importlib.reload hàng loạt.

Giữ hỗ trợ NGROK_AUTH_TOKEN/NGROK_DOMAIN và cơ chế cập nhật AI_SERVER_URL vào .env mà gateway đang dùng. Update chỉ key URL, giữ keys/comment khác, không log secret; chỉ cập nhật sau health thành công. Khi restart đọc token/domain bằng cùng cơ chế dotenv, không copy parser tự viết cũ. Không ngrok.kill toàn bộ session vì có thể có tunnel khác; đóng handle của notebook.

### Dependencies và reread cấu hình

- requirements.txt khai báo trực tiếp python-dotenv; thêm pandas nếu input builder cần. Không dựa vào transitive dependency vô tình có trên máy.
- requirements-colab.txt liệt kê dependencies notebook (pyngrok, httpx và inference deps cần thiết), dùng opencv-python-headless thay bản GUI. Không cài song song nhiều OpenCV distributions, không chạy upgrade torch/CUDA tùy tiện; ưu tiên runtime GPU stack có sẵn nếu phù hợp và báo versions thực tế. Không chạy pip -r local requirements một cách mù quáng do bounds NumPy/OpenCV/local có thể khác Colab.
- Không tự chọn pin version mới bằng phỏng đoán. Ghi tương thích với artifact metadata, chạy load tests. Version mismatch warnings phải được báo; không suppress toàn cục để gọi pass. Không train lại để che incompatibility.
- Settings nên dùng dotenv_values + merge os.environ thay load_dotenv làm ô nhiễm process globals: giữ precedence process env > file > default, nhưng mỗi Settings mới đọc được .env vừa sửa. Tests: tạo Settings hai lần sau sửa file, khi không có process override phải thấy giá trị mới; process override vẫn thắng.
- Notebook overrides chỉ áp vào keys vận hành đã liệt kê; khi bỏ override ở lần rerun, khôi phục env ban đầu của các keys notebook từng đặt. Không xóa hàng loạt os.environ.

### Acceptance Colab

- Chỉ một notebook runtime active ở src/notebooks. Không còn root notebook copy. Update tất cả references active trong AI_SERVICES README/scripts; giữ report lịch sử và plan gốc như history, ghi link plan follow-up thay vì rewrite history.
- Static: nbformat hợp lệ, source code cells compile với top-level await support; không chứa saved token/output cá nhân; execution_count/outputs của notebook giao lại sạch.
- Mock lifecycle/config tests chạy local; không giả vờ mock là GPU evidence.
- Manual Colab: start sạch -> status device CUDA -> strict fixture -> stop -> chọn folder regression khác hợp lệ -> start -> xác nhận model mới và health -> stop. Ghi device thực, versions, commands/cells và outputs đã lọc secret.
- Nếu không có Colab runtime để thao tác, hoàn thành code/tests local rồi ghi COLAB_RUNTIME=NOT VERIFIED cùng checklist manual. Không claim hoàn thành nghiệm thu Colab.

## Phase 4 — Cleanup nguồn cũ có kiểm soát

Files: modules/, app.py, regression_engine.py, model_registry.py, scripts/export_artifact_bundle.py, schemas.py, tests, README.

- Sáu file modules CV cũ trùng byte với vision mới: search imports/references active cả .py và source cells .ipynb trong repo (không search outputs); move consumer sang rice_ai. Sau đó xóa modules/ cũ nếu không còn active consumer. Archive không cần sửa nội dung lịch sử, nhưng ghi chúng không phải entrypoint supported.
- test_regression_engine chuyển sang estimation/regression và pipeline/features; không test implementation cũ thay cho runtime mới. Chuyển assertions cần thiết của test_model_registry sang loader tests, bỏ hash-only checks vốn không còn là policy.
- Chỉ bỏ regression_engine.py và model_registry.py sau khi chuyển consumers. Bỏ hardcoded OLS code khỏi active repo path; Git history giữ lịch sử.
- scripts/export_artifact_bundle.py cũ không còn phù hợp: thay bằng stub deprecated không ghi file, in hướng dẫn copy nguyên folder và return nonzero; không giữ importer registry chỉ để tool cũ sống. Không thêm exporter framework mới trong fix này.
- app.py chỉ bootstrap, app export và CLI; xóa implementation lặp execute_*/compute_final_estimates sau cập nhật tests. MODEL_PATHS chỉ giữ nếu còn consumer cần, hoặc chuyển test sang Settings và xóa shim thừa.
- feature_schema.py root giữ shim vì LINEAR_REGRESSION_MODEL/tests/test_notebook_consolidation.py còn dùng. schemas.py root chỉ giữ nếu có consumer thực; ghi lý do, không xóa mù quáng.
- Giữ artifacts/manifest.json và reports lịch sử; đánh dấu historical/not runtime configuration. Không xóa model/scaler/dataset/venv/archive notebooks theo suy đoán.

Pass: import search không còn consumer active trỏ file bị xóa; tests migrated pass; chỉ một implementation runtime cho mỗi thuật toán; notebook đã move không duplicate.

## Phase 5 — Verification cuối và tài liệu trung thực

Chạy từ project root; cập nhật tên module test trong commands nếu hợp nhất nhưng không bỏ coverage. Test fixtures tổng hợp chỉ fit model cực nhỏ để test loader, không training nghiên cứu.

```powershell
$env:PYTHONIOENCODING='utf-8'
& AI_SERVICES/.venv/Scripts/python.exe -m compileall -q AI_SERVICES/src/rice_ai AI_SERVICES/app.py AI_SERVICES/scripts AI_SERVICES/tests
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_settings AI_SERVICES.tests.test_regression_loader AI_SERVICES.tests.test_feature_schema AI_SERVICES.tests.test_regression_engine AI_SERVICES.tests.test_estimators AI_SERVICES.tests.test_pipeline AI_SERVICES.tests.test_api_endpoints AI_SERVICES.tests.test_vision_models AI_SERVICES.tests.test_api_lifecycle AI_SERVICES.tests.test_colab_notebook -v
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py
& AI_SERVICES/.venv/Scripts/python.exe AI_SERVICES/scripts/verify_artifacts.py --model-dir ../LINEAR_REGRESSION_MODEL/models/ard
& AI_SERVICES/.venv/Scripts/python.exe -m unittest AI_SERVICES.tests.test_real_pipeline -v
git diff --check
git status --short
git diff --stat
```

Sửa test_real_pipeline sang TestClient lifespan context, strict regression, assert finite regression và method đúng, legacy API keys, timing stage thực. M001A final/geometry/hybrid so inherited values để phát hiện lệch cùng model; sai khác phải điều tra chứ không force 85. Không tự gọi nhiều E2E để tiết kiệm CPU. Colab GPU vs CPU có thể khác số học: ghi measurement, giải thích tolerance theo stage, không tùy tiện nới ngưỡng cho pass.

Rà external training consolidation tests sau cleanup shim nếu môi trường training có sẵn; nếu không chạy được ghi NOT VERIFIED. Không install full training environment chỉ để làm cleanup.

README cập nhật kiến trúc duy nhất, đường dẫn notebook mới và flow Colab là hướng dẫn chính; Windows là tùy chọn. Folder schema dùng key features, no-scaler nằm trong config.json, không gọi dataclass là Pydantic nếu không đúng.

Tạo reports/ai_services_restructure/remediation_verification.md với local vs Colab evidence tách biệt; thêm correction note vào verification.md cũ và bỏ claim hiện tại hoàn thành/accuracy 100% gây hiểu nhầm, giữ dữ liệu lịch sử có nguồn. Không sửa docs/PROJECT_STATUS.md nếu user chưa yêu cầu. Không commit/push tự động.

Gateway timeout 120s là giới hạn đã biết; E2E CPU 127s có thể timeout qua UI dù Python API pass. Đợt này không sửa MERN: đo luồng Colab qua gateway nếu có, nếu vẫn vượt timeout thì ghi blocker giao diện và đề xuất cấu hình timeout riêng, không báo UI pass từ direct API test.

## 6. Definition of Done

- [ ] Loader reject wrong version/order/object names/count/output shape/preprocessing conflict; legitimate model/scaler pass.
- [ ] Đổi folder training/deploy chỉ bằng path; raw parity và scaler đúng một lần.
- [ ] YOLO auto dùng GPU khi có, explicit device lỗi báo rõ; không bỏ GPU Colab.
- [ ] Old response keys, model metadata, debug policy và input errors tương thích.
- [ ] Cancellation không giải phóng gate trước worker; health responsive; cleanup/shutdown tests pass.
- [ ] Timing stage có nghĩa thật; provider error/status/capabilities trung thực.
- [ ] src/notebooks/API_Server.ipynb là launcher active duy nhất; root/config/preflight/start/stop/restart đúng.
- [ ] Dependencies được khai báo; Settings reread .env đúng trong notebook session.
- [ ] Code trùng đã dọn sau migration consumers; weights/history được giữ.
- [ ] Local suite và strict E2E pass; evidence Colab manual ghi PASS hoặc NOT VERIFIED rõ ràng.
- [ ] README/report không overclaim; git diff được review, không secret/generated junk.

Phân biệt LOCAL_VERIFIED với COLAB_VERIFIED. Chỉ gọi plan FULLY VERIFIED khi cả hai có evidence thực; thiếu Colab không cản giao code local nhưng vẫn là acceptance pending.

## 7. Stop conditions

Dừng phần phụ thuộc và báo chính xác khi model contract không xác định, weights thiếu/load fail, same-model numerical parity thay đổi không giải thích được, hoặc notebook root không có source. Không fake provenance/accuracy, không xóa uncommitted source để làm test sạch, không force expected, không retrain. Nếu schema thay đổi thật thì là công việc khác, không tự mở rộng plan.

## 8. Checklist imperative cho implementation agent

- [ ] Đọc plan này toàn bộ; giữ working tree; không thực thi lại đại refactor cũ.
- [ ] Thêm negative tests dựa các probe review trước sửa code.
- [ ] Sửa loader version/names/count/fitted/shape/finite/scaler-conflict; generic model, không hash gate.
- [ ] Sửa named input và test raw prediction parity/one transform/no fit.
- [ ] Thêm YOLO_DEVICE auto/cpu/cuda:N, restore default candidates, validate inference concurrency=1.
- [ ] Restore API metrics keys/model metadata/debug policy, validate raw negative/nonfinite inputs.
- [ ] Bind admission token với worker completion; shield và track pending; cancellation/shutdown tests.
- [ ] Tách timing stages; lưu model load errors/device; sanitize status.
- [ ] Move notebook vào src/notebooks/API_Server.ipynb; một launcher active, không duplicate source inference.
- [ ] Viết cells root/dependencies/Settings/preflight/start/smoke/stop; handles/restart dùng app/providers mới.
- [ ] Thêm dependencies trực tiếp/Colab; Settings reread dotenv không stale process pollution.
- [ ] Migrate legacy tests/consumers, dọn modules và code lặp, giữ feature_schema shim cần thiết.
- [ ] Chạy fast tests, preflight hai bundle, strict M001A một lần cuối và notebook static/lifecycle tests.
- [ ] Manual GPU Colab nếu khả dụng; thiếu runtime ghi NOT VERIFIED, không giả pass.
- [ ] Update README/reports chính xác; báo files, commands/results, blockers và trạng thái local/Colab riêng.
- [ ] Không commit/push hoặc bắt đầu task tiếp theo; dừng sau handoff.
