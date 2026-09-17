# Plan hợp nhất notebook huấn luyện hồi quy

Status: READY FOR IMPLEMENTATION — chưa triển khai, chưa training
Created: 2026-09-17
Planner: Codex
Implementation agent: Gemini 3.8 Flash
Scope: LINEAR_REGRESSION_MODEL; một notebook chính dễ đọc, cấu hình model riêng, preprocessing đúng và lưu bundle độc lập.
Updated requirement: mọi logic đọc, phân tích, kiểm tra, chuẩn bị dữ liệu, schema, scaler, train, evaluate và export nằm ngay trong notebook; không phụ thuộc mã nguồn AI_SERVICES/module nội bộ khác.

## 1. Kết luận và evidence đã đọc

Không cần duy trì ba trainer độc lập cho cùng nguồn dữ liệu/31 features. Tạo một notebook chính `RICE_SEED_REGRESSION_TRAINER.ipynb`; giữ ba bản gốc làm lịch sử trong giai đoạn chuyển đổi. Tên mới dùng Regression vì có cả tuyến tính, kernel, cây và ensemble, không chỉ linear regression.

| Notebook | Evidence theo cell index, bắt đầu từ 0 | Kết luận |
| --- | --- | --- |
| RICE_SEED_LINEAR_REGRESSION_TRAINER.ipynb | Cell 2 nạp dữ liệu/schema; 3 split/scaler; 4 sáu cấu hình; 5 xuất winner | Sáu cấu hình đều được lặp lại trong v2. Lasso chỉ được import/nhắc trong giới thiệu, không thực sự train. |
| RICE_SEED_LINEAR_REGRESSION_TRAINER_v2.ipynb | Cell 2–3 lặp data/scaler; 4 thêm PLS, ARD, KernelRidge, Stacking, RF, GB, XGBoost tùy thư viện; 5 xuất winner | 13 cấu hình khi có XGBoost, 12 nếu không có. RF/GB khác tham số trainer cây, phải giữ riêng variant. |
| RICE_SEED_DECISION_TREE_TRAINER.ipynb | Cell 1–2 lặp data/scaler; 3 sáu cấu hình cây; 4 xuất winner | Bổ sung DecisionTree, ExtraTrees, AdaBoost, HistGradientBoosting và hai variant RF/GB. |

V2 + trainer cây có 19 cấu hình ứng viên nếu giữ các variant khác tham số và có XGBoost. Đây không phải 19 họ thuật toán, cũng không phải bộ 14 model chính đã đăng ký trong TASK-04.

Các vấn đề cần xử lý trong chính luồng hợp nhất:

- Cả ba fit `StandardScaler` trên outer train trước khi chạy CV bằng `X_train_scaled`: outer test không tham gia fit scaler, nhưng validation fold trong CV có tham gia tính mean/std. CV cần pipeline fit scaler trong từng fold.
- Cả ba ghi vào `models/scaler.joblib` và `scaler_params.json`. Nhiều biểu đồ và `test_set_predictions.csv` cũng cùng tên; rerun trainer khác có thể ghép nhầm scaler/model và ghi đè kết quả.
- Hai bản gốc chọn winner bằng R2_Test, v2 bằng RMSE_Test. Test không được dùng chọn/tune model nếu muốn báo cáo chất lượng trên holdout độc lập.
- V2 chọn số components PLS trên toàn outer train rồi CV lại cấu hình đã chọn trên cùng train; metric đó không phải unbiased nested-CV estimate.
- Các bản chỉ export winner, không có bundle cho mỗi model. Data paths và import google.colab còn bắt buộc theo Colab; pip cài latest và blanket warnings suppression làm khó tái lập.
- MAPE chia trực tiếp cho y; adjusted R2 fallback thành R2 khi không đủ bậc tự do; dải sai số ±5% bị gọi confidence interval; R2 bị gọi độ tin cậy dự đoán. Phải sửa nhãn/logic, không thêm claim thống kê không có evidence.
- `models/tuned_hyperparameters.json` có các giá trị khác source notebook và cả Lasso; chưa thấy ba notebook nạp file này. Không tự áp nó làm mặc định hoặc gọi là cấu hình đã chứng minh tối ưu.

## 2. Source of truth và nguyên tắc

- Schema: FEATURE_CONTRACT tự khai báo ALL_31_FEATURES, FEATURE_DEFS, FEATURE_SCHEMA_VERSION, TARGET_COLUMN và validators trong notebook. Lấy thứ tự từ ba trainer nguồn; không import schema/helper/registry từ AI_SERVICES hoặc module nội bộ. Không tạo feature_schema.py riêng. Training xuất contract cùng bundle; deployment tuân theo bundle, không điều khiển ngược trainer.
- Dataset mặc định: `DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv`; XLSX chỉ dùng khi người chạy chọn rõ hoặc CSV không tồn tại và log source được chọn. Không đọc/ghi sheet để thay đổi nhãn.
- Tham số model: lấy từ cell cấu hình của ba notebook trong bảng mục 4. Export `get_params(deep=True)` và phiên bản thư viện vì defaults có thể đổi.
- Hybrid đã trích xuất là input, không suy từ Actual_Count. Ghi nguồn/semantics từ dataset/extractor trong notebook; không đọc manifest triển khai để quyết định công thức training. Evidence trước đó cho dataset hiện tại ghi packing fraction0.62; thiếu grain volumes gốc thì không recompute bằng mean/median guessed hoặc ép công thức cho dataset khác. Thiếu evidence ghi audit pending, không fake parity.
- TASK-04 spec: `PROJECT_TASKS/TASK_04_REGRESSION_BENCHMARK_PAPER.md`. Kết quả notebook mới vẫn PROVISIONAL trên pilot, không đánh dấu TASK-04 complete hoặc official/scientifically validated.
- Giữ nguyên `AI_SERVICES/artifacts/manifest.json`, model/scaler/metadata deployment ở root models, các script train/inference hiện có và tất cả thay đổi người dùng. Đặc biệt không restore file Untitled0.ipynb đang bị xóa.
- Không hardcode dự đoán, không missing -> 0, không đổi expected để test pass, không fake provenance. Hash chứng minh bytes; provenance run mới phải ghi từ chính execution tạo bundle, không suy từ timestamps.

Tài liệu kỹ thuật đối chiếu: [scikit-learn preprocessing/leakage](https://scikit-learn.org/stable/common_pitfalls.html), [model selection](https://scikit-learn.org/stable/model_selection.html), [model persistence](https://scikit-learn.org/stable/model_persistence.html). Không coi joblib từ nguồn không tin cậy là dữ liệu an toàn để load; dùng cùng environment phiên bản đã ghi cho bundle.

## 3. Bố cục notebook chính

Markdown tiếng Việt, có mục lục và heading đánh số. Các code cell có comment ID ổn định để test/headless runner định vị; không dựa vào execution_count. Không tách thành framework/package lớn: helpers nhỏ đặt trong notebook, một file trainer chính.

Notebook chỉ cần CSV/XLSX được chọn và thư viện bên thứ ba; REPO_ROOT chỉ là tiện ích đường dẫn tùy chọn. Không thêm sys.path/import mã dự án, không đọc schema/manifest/artifact deployment khi Run All. Mọi hàm đọc/validate/EDA/preparation/split/scaling/model/train/metrics/plot/export/reload có code trong notebook. Tests/README/requirements chỉ hỗ trợ kiểm chứng/sử dụng, không chứa logic thay notebook. Không nhúng dataset vào ipynb. Scope là bảng đặc trưng đã trích xuất, không thêm YOLO/CNN/segment ảnh.

FEATURE_CONTRACT gồm list31 đúng thứ tự trong trainer nguồn, version, TARGET_COLUMN='Actual_Count', FEATURE_DEFS(name/group/unit/required/zero policy) và validators. Assert len(list)==len(set(list))==31 và target không trong input. Units: bulk/volume mm3, area mm2, kích thước mm, weight g, scale px/mm, detected diameter px, uniformity %, counts hạt. Std=0 hợp lệ; missing khác zero; bounds phải có căn cứ, không threshold theo metric. Thay semantics/schema phải version mới. Export đầy đủ list/units/version/rules vào feature_schema.json, không import backend contract.

| Khu | Cell ID / nội dung | Invariant |
| --- | --- | --- |
| 00 — Mục tiêu và giới hạn | intro, TOC, provisional protocol | Nêu input, output, cách chạy local/Colab, không hứa loại bỏ overfit chỉ vì có CV. |
| 01 — Môi trường | ENV_SETUP, IMPORTS | Setup tùy chọn, không pip install latest mỗi Run All; Colab import/mount được guard. In Python/package versions. |
| 02 — Cấu hình dữ liệu/run | RUN_CONFIG | DATA_PATH/OUTPUT_ROOT độc lập; REPO_ROOT tùy chọn tạo defaults; seed=42, test_size=.20, cv_folds=5, split/group source, enabled models, jobs, export. |
| 03 — Đọc, phân tích và chuẩn bị dữ liệu | FEATURE_CONTRACT, DATA_LOAD, DATA_OVERVIEW, DATA_PREPARE | Schema/validators tự khai báo; đọc CSV/XLSX; shape/dtypes/missing/nonfinite/duplicates/status/QC; numeric conversion/cleaning report, tạo X31/y/row_ids/groups ngay trong notebook. Không impute hoặc fit preprocessing trước split. |
| 04 — Holdout và CV | SPLIT_CONFIG, SPLIT_BUILD | Split một lần, dùng chung mọi model; lưu row IDs/group/fold membership. |
| 04b — Phân tích dữ liệu train | TRAIN_EDA | Summary/phân bố feature/target và tương quan trên outer train; outlier chỉ flag, không tự loại theo target/test. Không dùng target test chọn feature/model/tune. |
| 05 — Scaler/preprocessing | PREPROCESSING, HELPERS | Giải thích z=(x-mean)/scale, fit fold train, raw X vào Pipeline; helper không fit khi khai báo. |
| 06 — Cài đặt từng model | MODEL_<model_id>, một cell riêng cho mỗi dòng mục 4 | Chỉ tạo spec/factory/params; không train, không plot, không chọn winner. Có enabled và lý do. |
| 07 — Huấn luyện | TRAIN_ALL | Train tất cả enabled models; chỉ log tiến độ/thời gian/lỗi, chưa display leaderboard/plot. |
| 08 — Đánh giá và so sánh | EVALUATE_ALL, COMPARE_ALL | Chọn theo CV trên train trước khi đọc test; sau train mới tính/báo cáo kết quả, sắp CV_MAE. |
| 09 — Chẩn đoán từng model | DIAGNOSTICS | Chọn model_id để xem parity/residual/coefficient/importance đúng model đó. |
| 10 — Export bundle | EXPORT_ALL | Export mọi model thành công, không chỉ winner; protected-path guard. |
| 11 — Reload và predict | RELOAD_SMOKE | Model/scaler/schema cùng bundle; raw pipeline prediction bằng model(scaler(raw)). |

Cho phép người dùng tắt model bằng `enabled` trong cell đó. RUN_CONFIG có `ENABLED_MODEL_IDS=None` nghĩa dùng enabled mặc định; override phải là subset ID hợp lệ. Không có hai bảng enable mâu thuẫn. Khi đổi cấu hình phải Restart & Run All; helper kiểm tra run signature để không export state cũ với config mới.

DATA_OVERVIEW có thể hiển thị trước train vì là QC bảng, không phải leaderboard model. TRAIN_EDA là descriptive analysis; mọi biến đổi học từ dữ liệu vẫn fit trong train/fold train. Model metrics/comparison/diagnostics chỉ hiển thị sau TRAIN_ALL.

## 4. Danh mục model và tham số phải giữ rõ ràng

Một cell riêng mỗi model_id; mỗi cell có display_name, family, source_notebook, enabled, estimator factory, params và ghi chú scaler. Không nhập Lasso/KNN/SVR/MLP mới chỉ vì TASK-04 liệt kê. Các tham số không ghi trong bảng là default thư viện, nhưng effective params phải lưu đầy đủ khi chạy.

| model_id | Class | Tham số source cần hiện rõ trong cell |
| --- | --- | --- |
| ridge_alpha_0_1 | Ridge | alpha=.1, random_state=42 |
| ols | LinearRegression | fit_intercept=True (default nguồn) |
| ridge_alpha_1 | Ridge | alpha=1, random_state=42 |
| bayesian_ridge | BayesianRidge | defaults, ghi effective params |
| huber | HuberRegressor | max_iter=1000 |
| elastic_net | ElasticNet | alpha=.01, l1_ratio=.5, random_state=42 |
| pls | PLSRegression | source tìm components 2..14 bằng CV; baseline mới cố định n_components=2, scale=True, ghi rõ không phải tuned optimum |
| ard | ARDRegression | defaults, ghi effective params |
| kernel_ridge_rbf | KernelRidge | kernel='rbf', alpha=1, gamma=None |
| stacking_linear | StackingRegressor | base OLS/Ridge(.1)/BayesianRidge, final RidgeCV(alphas=logspace(-3,3,20)), legacy inner KFold5 seed42 |
| random_forest_v2 | RandomForestRegressor | n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42 |
| gradient_boosting_v2 | GradientBoostingRegressor | n_estimators=500, learning_rate=.03, max_depth=3, subsample=.8, random_state=42 |
| xgboost_v2 | XGBRegressor | n_estimators=600, learning_rate=.03, max_depth=4, subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1, random_state=42 |
| decision_tree | DecisionTreeRegressor | random_state=42 |
| random_forest_tree | RandomForestRegressor | n_estimators=100, random_state=42 |
| extra_trees | ExtraTreesRegressor | n_estimators=100, random_state=42 |
| gradient_boosting_tree | GradientBoostingRegressor | n_estimators=100, random_state=42 |
| ada_boost | AdaBoostRegressor | n_estimators=50, random_state=42 |
| hist_gradient_boosting | HistGradientBoostingRegressor | random_state=42 |

Cho sklearn/xgboost dùng jobs config: tránh outer CV n_jobs=-1 và inner estimator n_jobs=-1 đồng thời; mặc định CV_JOBS=1, estimator jobs cấu hình được. Không coi thay số jobs là tuning chất lượng model.

Ngoại lệ phải minh bạch:

- PLS baseline fixed2 tránh selection bias. Không chuyển loop legacy chọn components trên toàn train sang metric gọi unbiased CV. Tuning PLS là tùy chọn sau này bằng nested search fit preprocess trong inner folds; ngoài baseline tối thiểu này. `scale=True` nghĩa PLS còn chuẩn hóa nội bộ; ghi trong bundle, không áp công thức coef generic của Ridge cho PLS.
- Stacking vẫn có cell riêng và toàn bộ legacy spec nhưng `enabled=False` mặc định, reason='inner CV preprocessing/group isolation not verified'. Không copy scaler fit ngoài Stacking rồi giả vờ inner CV không leakage. Nếu người dùng bật: fail explicit trước training cho đến khi có group-aware inner splitting và scaler trong từng base-estimator pipeline được kiểm chứng. Không bỏ tên stacking khỏi inventory hoặc báo nó đã train. Đây là giới hạn được chấp nhận của plan tối thiểu, không âm thầm đổi final estimator.
- XGBoost optional: nếu disabled không import package; nếu enabled mà thiếu dependency báo rõ trước train, không silently chuyển sang estimator khác. Báo skipped nếu người chạy chủ động disable.
- Giữ StandardScaler cho các model baseline thông thường kể cả cây để nhất quán với source và rõ model–scaler pair. Cây thường không cần scaling; không thêm ablation scaler-none trong lần hợp nhất. Mỗi final pipeline có scaler instance riêng, không fit lại scaler chung khi export.

## 5. Quy tắc dữ liệu, split, training và đánh giá

`load_and_validate_data()` yêu cầu đủ 31 feature + Actual_Count + Sample_ID + Image_Status. Lọc FOUND như source; numeric conversion, phát hiện NaN/Inf/domain invalid và ghi reason theo row. Không impute. Legitimate zero được giữ theo FEATURE_DEFS, đặc biệt Std=0; Actual_Count=0 có thể giữ nếu dữ liệu hợp lệ, không chia zero khi metric. Dòng thiếu target không train. Không loại outlier theo target/test performance.

Dataset fingerprint SHA-256 trên bytes source; giữ `row_id` từ index source trước lọc và Sample_ID; không yêu cầu Sample_ID unique nếu thực tế lặp nhưng row_id phải unique. Không đưa Sample_ID/QC/Actual_Count/group vào X.

`build_splits()`:

1. Mặc định GROUPED. Dùng physical sample/group column hoặc mapping do người chạy cấu hình và xác nhận; không coi mỗi ảnh flat M0001 là mẫu vật độc lập nếu cùng một ly/mẻ được chụp nhiều lần.
2. Với ID legacy đã xác nhận M001A/M001B là cùng mẫu vật, cho phép explicit legacy parser `^(M\d+)[A-Za-z]+$`; ghi mapping và kiểm tra trước chạy. Không tự tách/bỏ số hoặc suy nhóm từ ID chưa rõ quy ước.
3. GroupShuffleSplit(test_size=.2, random_state=42) cho outer split, GroupKFold5 trên outer train. Assert group intersection rỗng cho holdout và mọi CV fold. Tỉ lệ rows thực tế có thể khác20%, report actual rows/groups. Ưu tiên metadata ngày theo official protocol chỉ khi được chốt riêng, không redesign trong plan này.
4. Không có group đáng tin: fail cho GROUPED. Người chạy có thể chọn ROW_PROVISIONAL rõ ràng (train_test_split seed42 + KFold5); output phải ghi risk repeated-sample leakage, không official. Không silent fallback.
5. Thiếu số groups/rows hoặc fold validation <2 rows cho R2 thì fail với cách sửa config, không âm thầm hạ folds. Lưu split/fold membership dùng chung; mọi model dùng raw X và cùng folds.

`build_pipeline(spec)` tạo Pipeline([('scaler', StandardScaler()), ('model', estimator)]) mới. `train_one_model()` dùng `cross_validate` với raw outer-train DataFrame, fold list chung, multi-metric R2/neg MAE/neg MSE, `error_score='raise'`; RMSE tính sqrt(MSE từng fold) rồi aggregate. Sau CV, clone pipeline fit toàn outer train. Không fit full dataset, không nhìn test để tune; không dùng scaler từ fold bất kỳ làm final scaler.

`TRAIN_ALL` chỉ tập hợp fitted pipelines, CV outputs, train time và status; errors không thành PASS. Sau vòng train, `select_by_cv()` chọn min CV_MAE_mean; ties CV_RMSE_mean rồi model_id trên số chưa round. Freeze selected ID trước `EVALUATE_ALL` dùng test.

`compute_metrics()` báo R2, MAE, MSE, RMSE, max error, train/test counts; raw predictions float và shape(n,) (`np.asarray(...).reshape(-1)` + assert đúng n) thống nhất PLS. MAPE chỉ trên y!=0, ghi count excluded; không thay mẫu zero bằng epsilon để metric đẹp. Adjusted R2 chỉ trình bày tham khảo cho OLS phù hợp và n>p+1; các model khác/không đủ rows ghi null + reason, không fallback R2. JSON null thay NaN/Infinity (`allow_nan=False`). Không round/clamp prediction trước đánh giá; rounded display riêng.

Sau train, có thể hiển thị test metrics tất cả model để exploration nhưng selection/ranking chỉ dùng CV. Ghi rõ xem lại test nhiều lần hoặc đổi config theo test sẽ khiến holdout thành exploration; chưa có locked independent official test. Không tuyên bố CV mean là unbiased score của winner được chọn trong nhiều candidates.

Diagnostics đúng model_id: parity, residual, error table; ±5% gọi dải sai số tham khảo, không CI. Hệ số chỉ whitelist OLS/Ridge/BayesianRidge/Huber/ElasticNet/ARD có coef shape31 và equation parity test pass. Công thức raw w=coef/scale, b=intercept-sum(coef*mean/scale). `feature_importances_` chỉ model có hỗ trợ; không lấy RF rồi gán cho winner khác. Không thêm XAI/permutation computation lớn trong scope.

## 6. Cấu trúc lưu trữ bắt buộc

```text
LINEAR_REGRESSION_MODEL/
  RICE_SEED_REGRESSION_TRAINER.ipynb       # trainer chính
  NOTEBOOK_CONSOLIDATION_PLAN.md
  README.md                                 # hướng dẫn run/config/lưu trữ
  requirements_training.txt                 # version chính xác đã smoke-test
  models/
    # Artifact deployment của AI Services — giữ nguyên, không ghi đè
    best_tree_ensemble_model.joblib
    scaler.joblib
    best_tree_model_info.json
    scaler_params.json

    # Bundle hiện hành: một thư mục trực tiếp cho mỗi model
    ols/
      pipeline.joblib
      model.joblib
      scaler.joblib
      scaler_params.json
      config.json
      feature_schema.json
      metrics.json
      cv_results.csv
      predictions_train.csv
      predictions_test.csv
      manifest.json
      equation.json                         # chỉ nếu hợp lệ + parity PASS
      plots/parity_residual.png
    extra_trees/                            # cùng cấu trúc
    random_forest_v2/                       # cùng cấu trúc
    ... mỗi model_id một thư mục riêng ...
  results/
    comparison.csv
    comparison.png
    run_metadata.json
    split_membership.csv
    data_quality_report.json
    execution_report.json
```

Mỗi lần chạy **ghi đè** bundle hiện hành của từng `model_id` và các báo cáo toàn cục trong `results/`; người dùng chỉ cần mở đúng thư mục tên model. `run_id` timestamp + suffix ngẫu nhiên vẫn được lưu trong `manifest.json` và `run_metadata.json` để truy vết artifact hiện tại, nhưng không tạo thư mục con. Chỉ chấp nhận model_id trong registry; không input path tùy ý. Không ghi đè root deployment artifacts. Global results dùng để so sánh; metric/prediction/plot riêng vẫn nằm cạnh từng model như yêu cầu.

`export_bundle()` lấy `named_steps['model']` và `named_steps['scaler']` từ chính final fitted pipeline; không refit. `feature_schema.json` gồm version, ordered_features31, units, target và hash deterministic (ghi serialization convention; không gọi cùng hash deployment nếu dùng convention khác). Manifest gồm run_id, model_id, relative paths + SHA256 cho model/scaler/pipeline/schema/config, dataset hash, split hash, source notebook hash, package versions, estimator class, params, preprocessing contract, fit_scope='outer_train', benchmark_type='PROVISIONAL'. Run metadata lưu cleaning counts, enabled/disabled/failed IDs, seed/folds/selection rule, warnings. Run execution source hash tính trước chạy; output notebook không phải training-source hash.

Reload bằng joblib chỉ file vừa tạo tin cậy; verify hashes/schema và so sánh pipeline.predict(X_raw) với model.predict(scaler.transform(X_raw)) và in-memory predictions bằng np.testing.assert_allclose(rtol=1e-8, atol=1e-8). Scaler mean phải bằng outer train mean, không full-data mean. Không dùng artifact hợp nhất tự deploy vào AI_SERVICES; đổi model deployment là việc riêng.

## 7. Phase thực thi

### Phase 0 — Baseline và bảo vệ runtime

Goal: tránh mất thay đổi hoặc ghi đè pair deployment.

- Chạy `git status --short`, `git diff --stat`; giữ nguyên mọi thay đổi hiện có, không stash/reset/restore.
- Đọc plan và ba source notebook; ghi inventory19 specs với source cell và tham số vào README.
- Reviewer snapshot SHA256 legacy model/scaler/metadata hiện có tại root models để bảo vệ no-overwrite. Đây là check ngoài notebook, không dùng manifest/schema AI_SERVICES làm input training. Artifact cũ thiếu ghi unavailable, không tự tạo thay và không chặn notebook chạy độc lập.
- Xác định kernel Python/package versions. Local Python3.12 là lựa chọn đề xuất, không ép người dùng dùng Python hệ thống3.14 hoặc tự upgrade AI_SERVICES/.venv. Nếu cần environment riêng đặt `.venv` trong LINEAR_REGRESSION_MODEL; không commit venv.
- Pin training dependencies theo environment kiểm chứng; gồm numpy,pandas,scikit-learn,scipy,joblib,matplotlib,seaborn,openpyxl,jupyter/ipykernel,nbformat/nbconvert; XGBoost là extension optional ghi version riêng. Không tuyên bố tương thích cross-version joblib nếu chưa kiểm chứng.

Acceptance: baseline/status/hash report có thật; chưa train/overwrite. Stop nếu dirty changes overlap target hoặc thiếu source hợp lệ.

### Phase 1 — Tạo notebook có cấu trúc và model cells

Goal: một entrypoint, dễ đọc và chỉnh tham số.

Files: notebook mới, README.md, requirements_training.txt.

- Tạo Markdown/cell IDs đúng mục3; guard Colab setup, REPO_ROOT/DATA_PATH rõ ràng; optional install cell không tự cài khi Run All.
- Khai báo schema/validators ngay trong FEATURE_CONTRACT theo mục2-3, không import mã nội bộ; register19 spec cells đúng mục4, giữ RF/GB variants và ghi rõ Lasso chưa thực thi ở source.
- PLS fixed2 mới phải label baseline; Stacking disabled + explicit guard; không nạp tuned_hyperparameters.json ngầm.
- Helpers tạo spec/pipeline không train; enabled-ID validation, state signature; không tạo script/framework không cần thiết.

Tests: nbformat validate, compile toàn bộ Python code cells (IPython setup được xử lý bằng transformer nếu có magic), chỉ execute setup/helpers/model-spec cells không TRAIN_ALL.

Acceptance: đủ19specs, class/params/inventory khớp, notebook mở được và local setup không import google.colab bắt buộc; chưa export model.

### Phase 2 — Dữ liệu, scaler và train đúng

Goal: cùng raw feature contract/split, fold-local preprocessing.

- Implement load_and_validate_data/build_splits/build_pipeline/train_one_model/select_by_cv theo mục5.
- Implement DATA_OVERVIEW/DATA_PREPARE/TRAIN_EDA trong notebook: schema/quality report, cleaning counts, X/y/groups; sau split mới phân tích phân bố/target/correlation outer train. Không chuyển logic vào file .py.
- TRAIN_ALL hoàn tất CV + final outer-train fit mọi enabled model trước display metrics/plots. Một lỗi model ghi FAILED, không return prediction giả; không làm run thành SUCCESS nếu model enabled bắt buộc thất bại.
- Dùng cùng split/folds; không fit scaler trước CV; không tune/chọn theo test. Jobs không oversubscribe.

Tests: synthetic grouped31features, assert group isolation; scaler fold mean bằng train-fold mean qua cross_validate(return_estimator=True) trong test; missing/Inf bị reject/loại có report, zero Std giữ hợp lệ; target không trong X; deterministic split/order; stacking enable bị reject trước fit khi chưa hỗ trợ.

Acceptance: các invariant có assert/test thật; không có silently degraded CV. Nếu group không đủ thì báo config cần đổi, không đánh giá official.

### Phase 3 — Evaluate và export riêng mỗi model

Goal: bảng/plot sau train và bundle có thể truy xuất không nhầm.

- Implement compute_metrics/diagnostics/export_bundle/reload_and_verify; data shapes và capability guards như mục5–6.
- Chọn CV winner trước test, leaderboard sắp CV_MAE, metrics raw floats; null+reason cho metric không áp dụng.
- Export tất cả successful enabled models trực tiếp tại model_id; không export failed/disabled hoặc gọi winner là deployed model.
- Prevent overwrite/protected paths; export đủ config/schema/split/dataset/version/hash/provenance thực tế. Không giữ artifact fold estimator thay final estimator.

Tests: yzero MAPE, small-n adjustedR2, PLS output shape, metric recompute từ saved predictions; hai run không overwrite; tamperhash bị detect; equation chỉ supported + predict parity; plot model attribution đúng.

Acceptance: reload pipeline và separate model/scaler predict bằng nhau; protected legacy hashes không đổi; mỗi model đã train có bundle/kết quả riêng.

### Phase 4 — Verification, tài liệu và chuyển đổi entrypoint

Goal: chạy được từ kernel sạch, không phải notebook có stale output.

- Tạo lightweight `tests/test_notebook_consolidation.py` chỉ trong LINEAR_REGRESSION_MODEL nếu cần; dùng nbformat/IPython transformer để lấy định nghĩa/specs theo cell ID, không execute real TRAIN_ALL khi import test. Không phụ thuộc dataset private cho synthetic tests.
- Smoke synthetic dùng OLS + ExtraTrees (estimator params nhỏ chỉ cho SMOKE profile ghi rõ), groups đủ5fold, export vào temporary test directory không rootmodels. Smoke không là benchmark.
- Thêm isolated-copy smoke: copy notebook và synthetic CSV vào temp directory không có AI_SERVICES/module dự án; không repo sys.path. Setup/schema/read/validate/prepare/train/export/reload phải chạy được chỉ với dependencies cài sẵn. Không chỉ test import schema riêng lẻ.
- Chạy notebook thật trên dataset hiện có với explicit GROUPED source hoặc ROW_PROVISIONAL opt-in; restart kernel/run all, no setup install/network. Không đủ environment/data thì ghi NOT VERIFIED, không fake PASS. Không chạy tuning dài; Stacking vẫn disabled; XGBoost chỉ khi environment đủ và enabled rõ ràng.
- Nếu full pilot run chưa thực hiện, ghi chính xác smoke verified, full training pending; chưa được tuyên bố migration verification hoàn tất.
- README giải thích cell chỉnh data/model/scaler, run all, output tree, load pipeline vs separate pair, provisional limits, optional XGBoost và Stacking/PLS exceptions.
- README xác nhận toàn bộ logic nằm trong notebook, không cần AI_SERVICES. Nếu agent đã triển khai plan cũ: chỉnh đúng cells import/schema/helpers/path discovery và README liên quan; không xóa notebook/output/generated bundles hiện có. Evidence cũ không tự áp dụng cho code mới; rerun lightweight isolation/smoke trước, full training ghi trạng thái riêng.
- Giữ nguyên ba notebook cũ trong vòng này, README đánh dấu legacy/reference only. Không move trainer cây vì deployment manifest đang tham chiếu source path/hash. Sau acceptance người dùng mới quyết định archive/xóa; không tạo hai active entrypoint được khuyên dùng.
- Xóa execution output của notebook mới trước giao file nếu chứa private data/bulky plots; lưu textual execution report cần thiết không chứa raw records. Không sửa PROJECT_TASKS/docs trạng thái hoặc AI_SERVICES trong task hợp nhất.

Commands đề xuất chạy tại repository root (PowerShell; lựa chọn environment trước, không tự cài package vào runtime backend):

```powershell
# Chỉ dùng sau khi training environment riêng đã được chuẩn bị/kiểm chứng.
& '.\LINEAR_REGRESSION_MODEL\.venv\Scripts\python.exe' -m unittest discover -s 'LINEAR_REGRESSION_MODEL/tests' -p 'test_notebook_consolidation.py' -v
# Kernel ricai-training phải đăng ký từ chính training environment và config notebook hợp lệ.
& '.\LINEAR_REGRESSION_MODEL\.venv\Scripts\python.exe' -m nbconvert --to notebook --execute 'LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb' --ExecutePreprocessor.kernel_name=ricai-training --ExecutePreprocessor.timeout=1800 --output='trainer_executed.ipynb' --output-dir='LINEAR_REGRESSION_MODEL/results/verification'
git diff --stat
git status --short
```

Không commit executed notebook, private dataset, venv/cache hoặc joblib mới. Pin/môi trường và command setup phải được README ghi sau khi actual smoke pass, không dự đoán phiên bản cài thành công. Tổng thời gian real run có thể vượt timeout; báo timeout không suy thành PASS.

## 8. Definition of Done và test matrix

| Tầng | Pass condition |
| --- | --- |
| Static | nbformat hợp lệ, source cells compile, không bắt buộc Colab/local Drive path |
| Inventory | 19specs có source/params rõ; variants không mất, skipped/disabled explicit |
| Data | exactly31 ordered features tự khai báo theo trainer nguồn; overview/EDA/preparation trong notebook; target riêng; missing/Inf/QC report; zero hợp lệ không bị impute |
| Independence | không import helper/schema/module nội bộ; isolated-copy notebook + CSV không repo/AI_SERVICES chạy raw data đến train/export/reload smoke pass |
| Split | holdout/folds chung, deterministic seed, group disjoint hoặc explicit ROW_PROVISIONAL limitation |
| Scaling | fit trong từng CV fold; final scaler fit outertrain; no test fit/no double scaling |
| Selection | chỉ CV chọn winner; PLS không reused tuning CV claim; Stacking disabled nếu chưa verified |
| Metrics | train/CV/test rõ, floats chưa round, guards zero/small-n/model capabilities, no false confidence labels |
| Export | mỗi successful model có model/scaler/pipeline/schema/config/metrics/predictions/plots/manifest; nooverwrite |
| Integrity | reload parity + hashes/schema pass; protected deployment bytes unchanged |
| Execution | fresh-kernel synthetic pass + real pilot run có evidence; missing data/dependency ghi NOT VERIFIED |
| Docs | README entrypoint duy nhất, provisional status/known exceptions và exact tested environment |

Không có yêu cầu prediction mới bằng85 hoặc metric giống báo cáo cũ: split/scaler-CV/selection đã sửa sẽ thay số liệu. Điều tra khác biệt bằng dataset hash/split/params, không force expected hoặc overwrite model đang deploy.

## 9. Risks và stop conditions

- Không biết physical group hoặc ngày chụp: dừng GROUPED, hỏi/nhận mapping hoặc dùng ROW_PROVISIONAL được chọn rõ. Không claim tổng quát hóa giữa mẫu vật/ngày chỉ từ row split.
- Label nguồn không chắc/Hybrid dùng target: dừng claim scientific benchmark/deployment, ghi audit pending; không sửa nhãn để khớp model.
- Group/sample count quá ít: dừng với counts, không tự giảm folds hoặc đổi metric.
- Dependency/kernal khác version: ghi mismatch và sửa training environment riêng; không upgrade runtime AI_SERVICES tự động.
- Stacking enabled chưa đúng inner isolation: reject; không tự thiết kế custom stack framework trong task này.
- Run đổi config hoặc cells out-of-order: reject export state stale, yêu cầu Restart & Run All.
- Legacy artifact hash thay đổi: dừng ngay, báo exact paths; không git restore/reset vì có thể là thay đổi người dùng. Không tự publish/deploy bundle.
- Không đủ thời gian chạy cả enabled models: ghi partial/FAILED/NOT VERIFIED, không gọi train-all verified.
- Cần move/xóa notebook hoặc cập nhật deployment contract: xin authority riêng; không mở rộng task.
- Nếu notebook cần AI_SERVICES/schema.py/helper.py/manifest deployment để đọc/chuẩn bị dữ liệu: independence FAIL; sửa dependency trong notebook, không thêm module .py trung gian hoặc fallback import.

## 10. Execution checklist cho Gemini 3.8 Flash

- [ ] Làm đúng scope plan: notebook mới + README + requirements_training + targeted tests; không training benchmark chính thức, không đổi AI_SERVICES/TASK-02+.
- [ ] Snapshot git status/diff và hashes legacy artifact hiện có ngoài notebook; preserve dirty deletion Untitled0.ipynb. Không dùng manifest deployment làm cấu hình training.
- [ ] Tạo RICE_SEED_REGRESSION_TRAINER.ipynb với Markdown/ID cells đúng mục3, setup local/Colab được guard.
- [ ] RUN_CONFIG chứa DATA_PATH/OUTPUT_ROOT/seed/split/groups/folds/jobs/enable/export; repo root tùy chọn, không personal Drive path bắt buộc.
- [ ] FEATURE_CONTRACT tự khai báo list31/definitions/units/version/target/validators theo trainer nguồn trong notebook; không import AI_SERVICES hay feature_schema.py riêng; assert31/order/target absent.
- [ ] DATA_LOAD/DATA_OVERVIEW/DATA_PREPARE đọc CSV/XLSX, bảng/types/missing/duplicates/status/QC, filter/validate/clean report, giữ zero hợp lệ, tạo X/y/row_ids/groups và SHA256 source ngay trong notebook.
- [ ] SPLIT_BUILD freeze holdout/CV membership; grouped mặc định; explicit row-provisional nếu thiếu group và người chạy opt-in.
- [ ] TRAIN_EDA hiển thị summary/distributions/correlations outer train; không tự feature-select hoặc chỉnh data/model theo test.
- [ ] PREPROCESSING chỉ khai báo StandardScaler factory; raw X vào Pipeline, không prefit scaler cho CV.
- [ ] Tạo đủ19 model config cells đúng mục4; RF/GB variants ID khác; export effective params, không nạp tuned JSON tự động.
- [ ] PLS fixed2 baseline label rõ; Stacking cell disabled có reason/enable rejection; optional XGBoost dependency explicit.
- [ ] TRAIN_ALL multi-metric fold-local cross_validate + final outer-train pipeline fit; logs progress, chưa display comparison/plots.
- [ ] Freeze selected model theo CV_MAE/CV_RMSE/model_id trước EVALUATE_ALL đọc test.
- [ ] Evaluate all sau train; shape(n,), no rounding/clamping metrics, MAPE yzero/AdjR2 capability guards.
- [ ] COMPARE_ALL rank theo CV, ghi PROVISIONAL; DIAGNOSTICS chỉ đúng selected model_id, không false CI/confidence.
- [ ] EXPORT_ALL tạo models/model_id bundles cho tất cả successful models và results/ global comparison.
- [ ] Extract model/scaler từ same final pipeline, hash/config/schema/dataset/split/source/version provenance thực tế, no refit/no overwrite.
- [ ] RELOAD_SMOKE verify hashes và pipeline/raw vs separate/scaled prediction parity; equation whitelist có parity test.
- [ ] Thêm lightweight synthetic tests theo mục8; test no group leakage/fold mean/no missing->zero/order/zero/metrics/tamper/two-run.
- [ ] Thêm isolated-copy notebook+CSV smoke trong temp directory không AI_SERVICES/repo modules/sys.path; verify read/prepare/train/export/reload độc lập.
- [ ] Chạy tests bằng verified training environment; fresh-kernel smoke OLS+ExtraTrees temp output, profile ghi rõ.
- [ ] Chạy real pilot notebook với metadata split rõ nếu đủ data/env; ghi command/result/time/disabled/failed thật.
- [ ] So protected hashes; nếu khác dừng, không restore/overwrite. Không deploy bundle mới.
- [ ] README hướng dẫn entrypoint/config/scaler/bundle/local-Colab/dependencies/knownlimits; pin versions đã kiểm chứng.
- [ ] Giữ ba notebook cũ reference-only, không move/delete; không sửa docs task status, không commit/push nếu chưa được yêu cầu.
- [ ] Trả changed files, test commands/results, real-run verification status và limitations; không overclaim completion.

## 11. Suggested commit message (chỉ khi được yêu cầu sau)

`feat: consolidate regression training notebook and per-model bundles`

Stage source/docs/tests chính xác; không `git add .` mù quáng, không stage dataset/binaries/executed notebook hoặc deletion người dùng không thuộc task.
