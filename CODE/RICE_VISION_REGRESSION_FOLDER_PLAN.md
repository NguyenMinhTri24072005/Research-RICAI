# Plan: nạp regression theo folder trong RICE_VISION_MAIN_PIPELINE

Status: READY FOR IMPLEMENTATION
Reviewed: 2026-09-20 — bổ sung giới hạn chỉnh cell, thứ tự schema, môi trường và chống dùng lại kết quả cũ.
Created: 2026-09-20
Target agent: Gemini 3.1 Pro High
Scope: CODE/RICE_VISION_MAIN_PIPELINE.ipynb; không sửa AI_SERVICES hoặc train lại.
Các đường dẫn trong tài liệu tính từ repository root.

## 1. Diagnosis và bằng chứng đã đọc

Notebook hiện có 11 cells: markdown index 0; Bước 1–10 là code index 1–10 (index 10 là cell thứ 11 nếu đếm cả markdown). Dùng tiêu đề cell để định vị, không dựa riêng số thứ tự.

- Bước 10 đọc LINEAR_REGRESSION_MODEL/models/regression_equation.json; file này hiện không tồn tại.
- Khi thiếu file, notebook âm thầm dùng intercept=-63.0267 và 31 hệ số hardcode; dự đoán là tổng intercept + coefficient * feature. Không dùng joblib/scaler.transform/model.predict.
- Hệ số unscaled về nguyên tắc có thể dự đoán trực tiếp trên dữ liệu thô, nhưng fallback hiện tại không xác định model đã chọn và không hỗ trợ cây/ensemble.
- R²=0.9988 và MAE=2.14 được gán mặc định/ghi cố định; không phải metric của mọi model hay độ tin cậy của ảnh đang chạy.
- `PACKING_FRACTION=0.82` ở Bước 8 là cấu hình vật lý có chủ đích của người dùng; khoảng hợp lý được chấp nhận là `0.80–0.85`. Feature hybrid hiện lấy nhầm estimated_seed_count vật lý này (mean_clean sau IQR), trong khi contract regression đã train dùng hệ số 0.62 và mean toàn bộ hạt đo hợp lệ trước IQR.
- Bước 10 tính statistics trên whole_records đã round 2 chữ số; dataset extractor tính trên measurements đầy đủ rồi round summary 3 chữ số.
- Nếu thiếu hạt, Bước 8 fallback cleaned_grains; Bước 10 tạo array [0.0], default feature 0/100 và có thể tự ước lượng Weight_g. Cần ngăn các nhánh này giả làm input đúng contract train.
- Working tree lúc review có sửa chưa commit ở cả notebook mục tiêu và AI_SERVICES/src/notebooks/API_Server.ipynb. Giữ nguyên thay đổi người dùng, đặc biệt IMAGE_PATH và kích thước ly.

## 2. Source of truth

- LINEAR_REGRESSION_MODEL/RICE_SEED_REGRESSION_TRAINER.ipynb:
  - PREPROCESSING.build_pipeline: Pipeline([("scaler", StandardScaler()), ("model", estimator)]).
  - EXPORT_ALL: pipeline.joblib là toàn bộ final_pipeline; model.joblib và scaler.joblib lấy từ named_steps của cùng pipeline, không refit.
  - Xuất trực tiếp models/<model_id>/; không cần mã run trong đường dẫn.
- models/ard/{config.json,feature_schema.json,metrics.json}: schema_version=31v1, target_column=Actual_Count, 31 tên feature có thứ tự; sklearn 1.6.1, Python 3.13.15 được ghi trong bundle.
- CODE/modules/dataset_extractor.py:
  - DatasetExtractor mặc định packing_fraction=0.62.
  - Nhánh FOUND: hybrid=int(round(bulk_volume*packing_fraction/summary["volume_mean"])).
  - Whole_Grains_Count=summary["count"]; Pixels_Per_mm và Bulk_Rice_Volume_mm3 round 2; 20 statistics round 3.
- CODE/modules/ellipsoid_geometry.py::compute_folder_grains_summary: statistics từ raw metrics, np.std mặc định ddof=0, không lọc IQR trước statistics.
- CODE/modules/uniformity_evaluator.py::evaluate_batch_uniformity: tỷ lệ phần trăm sau IQR round 2.
- docs/TASK_01_AUDIT.md: evidence lịch sử 254/254 hàng hybrid khớp hệ số 0.62. Không trình bày evidence này như test mới.

### Quy ước bắt buộc về hệ số chèn lấp

- Giữ nguyên `PACKING_FRACTION = 0.82` cho ước lượng vật lý `estimated_seed_count`; không hạ về 0.62. Khi cấu hình, validate giá trị nằm trong khoảng `0.80 <= PACKING_FRACTION <= 0.85` và hiển thị giá trị thực tế trong báo cáo.
- Khoảng 0.80–0.85 là ràng buộc cấu hình của nghiên cứu theo yêu cầu người dùng, chưa phải kết luận thực nghiệm được plan này xác minh. Đặt validation ngay sau khai báo ở Bước 3 (reject cả NaN/Inf); sửa comment cũ ~0.60–0.64 cho nhất quán. Không sửa công thức/IQR ở Bước 8.
- `0.62` chỉ là hằng số của feature `Estimated_Total_Seeds_Hybrid` trong bundle regression hiện có. Đây là hai đại lượng có mục đích khác nhau và không được dùng thay thế cho nhau.
- Nếu nghiên cứu muốn regression dùng 0.82, đó là một thay đổi training contract: phải train/export bundle mới với feature semantics 0.82 và không được âm thầm đổi con số trong notebook inference.

## 3. Thiết kế chọn

Chỉ cấu hình REGRESSION_MODEL_DIR trỏ folder có model.joblib, scaler.joblib, feature_schema.json; config.json và metrics.json bổ trợ. Hỗ trợ folder tuyệt đối hoặc tương đối BASE_PATH.

Chọn nạp model.joblib + scaler.joblib, rồi scaler.transform đúng một lần và model.predict. pipeline.joblib chỉ là reference khi kiểm thử parity; không chạy nó trên input đã scale.

Không import AI_SERVICES, không registry hoặc hash bắt buộc; không tìm folder mới nhất, không tự chọn model khác khi folder sai. Không dùng equation.json làm fallback. Chỉ nạp artifact do nhóm tin cậy xuất ra (joblib có thể thực thi Python khi deserialize).

Ví dụ người dùng chỉ sửa:
```python
REGRESSION_MODEL_DIR = "LINEAR_REGRESSION_MODEL/models/ard"
# hoặc "LINEAR_REGRESSION_MODEL/models/decision_tree"
# hoặc "/content/drive/MyDrive/.../AI_SERVICES/artifacts/regression/production"
```

Không mặc định root models/ kiểu legacy. Nếu cần model legacy hãy export thành canonical folder riêng; không thêm adapter ngoài scope.

## 4. Files dự kiến

- CODE/RICE_VISION_MAIN_PIPELINE.ipynb: chỉnh markdown, dependencies và khu regression; giữ pipeline vision.
- CODE/tests/test_main_pipeline_regression.py: test các hàm/cell regression thật bằng AST extraction hoặc helper thực thi cell có marker, không copy lại thuật toán trong test.
- Không sửa training notebook, bundle artifacts, CODE/modules hoặc AI_SERVICES.
- Toàn bộ định nghĩa loader/feature/predict nằm trong notebook; test không import notebook bằng Run All.

## Phase 0 — Bảo toàn baseline

1. git status --short; git diff --stat; đọc diff notebook hiện tại.
2. Đọc lại các source-of-truth trên. Ghi nhận model folder được chọn và phiên bản sklearn.
3. Lưu snapshot source cell trong test memory để kiểm tra các cell vision không bị đổi ngoài phần có chỉ định.
4. Không commit/push, không clear output toàn notebook đang có kết quả nghiên cứu nếu người dùng chưa yêu cầu. Output regression cũ sau thay đổi phải được xóa hoặc ghi rõ stale; bảo toàn output cell khác.
5. Định vị cell theo "Bước 10", "Bước 8", "Bước 3", không đoán index nếu notebook vừa thay đổi.

Pass: có baseline rõ ràng, chỉ scope mục tiêu được sửa.

## Phase 1 — Cấu hình và loader folder

Giữ thuật toán và thông số người dùng trong Bước 1–9. Ngoại lệ chỉnh nhỏ được phép: dependencies ở Bước 1 và comment/validation PACKING_FRACTION ở Bước 3. Không sửa thuật toán vision hoặc physical/IQR ở Bước 8. Thay Bước 10 bằng nhóm cell có marker:
REGRESSION_CONFIG, REGRESSION_BUNDLE_LOADER, REGRESSION_FEATURES, REGRESSION_PREDICT, REGRESSION_REPORT.
Đặt config/loader ngay trước feature cell để đổi model rồi chạy lại khu regression mà không chạy YOLO/CNN.

REGRESSION_CONFIG:
- Khai báo REGRESSION_MODEL_DIR và INPUT_WEIGHT_G ở một nơi duy nhất. Giữ giá trị khối lượng đang có khi di chuyển; comment yêu cầu cân đúng ảnh.
- Path tương đối resolve từ BASE_PATH; tuyệt đối giữ nguyên. In folder đã resolve.
- Model mặc định đề xuất ard (đã có canonical export), ghi rõ đây là thay đổi khỏi fallback Ridge cũ, không kết luận ARD tốt nhất cho ảnh mới.
- Dependency Bước 1 thêm joblib và scikit-learn==1.6.1 cho bundle hiện tại; dùng sys.executable/pip trong Colab. Nếu đổi bundle phiên bản khác, báo mismatch kèm hướng dẫn cài đúng version/restart; không tự pip trong loader. xgboost chỉ cần khi chọn bundle xgboost, báo thiếu dependency rõ ràng.

REGRESSION_BUNDLE_LOADER định nghĩa load_regression_bundle(model_dir):
- Xóa/trả về state mới; khi load fail không để dùng model thành công từ lần trước.
- Bắt buộc directory và model.joblib + feature_schema.json.
- Schema: 31 tên duy nhất, đúng tập 31 features mà extractor của notebook cung cấp; reject Actual_Count; schema_version 31v1.
- Khai báo EXPECTED_31_FEATURES theo đúng danh sách/thứ tự trong FEATURE_CONTRACT của training notebook và bundle đã kiểm tra; yêu cầu schema.features bằng chính xác danh sách này, không chỉ cùng tập tên. Đây là contract độc lập trong notebook, không import AI_SERVICES. Khi model/scaler chỉ lưu n_features_in_ mà không lưu feature_names_in_, vẫn reject schema bị đảo thứ tự. Không tự sort để chữa schema sai.
- Lấy ordered_features từ schema; không alphabet-sort. Nếu artifact.feature_names_in_ có thì phải khớp thứ tự schema; kiểm tra n_features_in_=31 nếu có.
- scaler.joblib bắt buộc, trừ config.json khai báo chính xác preprocessing="none". none kèm scaler file là lỗi mâu thuẫn; thiếu config không đồng nghĩa không scaler.
- config hiện tại preprocessing="StandardScaler in Pipeline" vẫn yêu cầu scaler riêng.
- joblib.load cả hai từ cùng folder; check_is_fitted; predict/transform callable. Reject Pipeline trong model.joblib để tránh double preprocessing.
- Bắt cảnh báo phiên bản, trình bày rõ; không suppress rồi tuyên bố verified.
- Chính sách version: mismatch phiên bản thư viện được báo rõ và không coi là bundle/schema hỏng. Cho phép smoke/parity chẩn đoán trong local environment hiện có; không claim tương thích deployment từ kết quả đó. Acceptance Colab cần môi trường tương thích bundle. Thiếu dependency hoặc load/predict lỗi thì dừng, không fallback. Loader không tự cài package hay refit.
- config/metrics thiếu được phép với nhãn unavailable; JSON tồn tại nhưng hỏng phải báo file/path.
- Return dict chứa model, scaler, ordered_features, schema_version, folder, config, metrics. Không giả định fields estimator_mode.
- Không fit/fit_transform, không tự thay scaler từ folder khác, không sửa artifacts.

Pass: ARD và Decision Tree load theo một biến folder; folder thiếu/sai fail rõ trước prediction.

## Phase 2 — Feature semantics

REGRESSION_FEATURES định nghĩa assemble_regression_features(...) dùng measurements gốc:
- Lấy whole_grain_metrics_list đã có từ Bước 8 (raw length_mm, width_mm, thickness_mm, area_mm2, volume_mm3), không lấy whole_records round 2.
- Require len(whole_grains)>0 và measurements hợp lệ; nếu Bước 8 đã fallback sang cleaned_grains vì không có whole grains thì dừng regression với giải thích. Phần vật lý/plots đã chạy vẫn giữ.
- Kiểm tra raw metrics hữu hạn và kích thước/thể tích >0; không dùng [0.0] cho missing; không âm thầm nhận hạt lỗi.
- Whole_Grains_Count là số measurements thành công, không len(whole_grains) khi có phép đo thất bại.
- Bulk_Rice_Volume_mm3=round(bulk_volume_mm3,2); Pixels_Per_mm=round(pixels_per_mm,2).
- Các length/width/thickness/area/volume Mean/Min/Max/Std: lấy raw arrays, std ddof=0, round 3 sau thống kê.
- Uniformity_Rate_Pct tính evaluate_batch_uniformity(raw volumes)["uniformity_rate_pct"].
- Hybrid=int(round(raw_bulk_volume * 0.62 / np.mean(raw_volumes))). Không lấy mean_clean; không dùng `PACKING_FRACTION=0.82` của physical estimate để thay thế hybrid; không lấy mean đã round 3.
- Giữ physical `estimated_seed_count` dùng `PACKING_FRACTION=0.82` để so sánh, ghi rõ đây là công thức vật lý độc lập và validate hệ số trong khoảng 0.80–0.85.
- Các chiều cao/đường kính nhập là mm; area mm²; volume mm³; Weight_g là gam cân thực.
- INPUT_WEIGHT_G=None phải báo thiếu cân; bỏ ước lượng quang học tự động trong regression. Không thay bằng 0. Trường hợp mẫu không rỗng có cân <=0 phải báo dữ liệu không nhất quán.
- Require container_info["rice_height_mm"], ["inner_w_px"], valid scale; không .get(...,0) hoặc default uniformity 100.
- Exact 31 feature keys, no Actual_Count. Missing/None/NaN/Inf fail với tên feature; legitimate zero (Std=0, Empty_Height_mm=0) hợp lệ.
- Dựng DataFrame một hàng theo ordered_features; không ép mọi feature phải >0.

Reference test: bulk=1000, volumes=[10,20] -> hybrid=41; physical với `PACKING_FRACTION=0.82` -> 55; volumes có outlier chứng minh hybrid không dùng mean IQR; test cấu hình reject 0.79 và 0.86, accept 0.80/0.82/0.85.

Pass: thống kê/rounding khớp extractor cho cùng raw measurements. Nêu rõ matching feature semantics chưa chứng minh toàn bộ segmentation/crop giống dữ liệu train.

## Phase 3 — Predict và báo cáo

REGRESSION_PREDICT định nghĩa predict_regression_bundle(bundle, features):
- Validate input trước transform. DataFrame nếu fitted scaler yêu cầu names; ndarray nếu không.
- scaler.transform một lần, hoặc raw vector khi explicit preprocessing=none.
- Validate transformed shape (1,31), finite. Khi model có feature_names_in_, cấp DataFrame đúng names; ngược lại ndarray.
- model.predict; chỉ nhận scalar single-target shape (1,) hoặc (1,1), reject nhiều target/nonfinite.
- raw_predicted_count lưu nguyên float; final_predicted_seeds_regression=max(0,int(round(raw_predicted_count))).
- Reset kết quả ở đầu cell để rerun thất bại không xuất lại prediction cũ.
- Cụ thể: đầu REGRESSION_CONFIG và trước mỗi lần load/assemble/predict, vô hiệu hóa state downstream (bundle/features/result tương ứng). REGRESSION_REPORT bắt buộc result hợp lệ, folder đã resolve trùng REGRESSION_MODEL_DIR hiện tại và vector feature khớp lần predict; nếu không, raise trước mọi thao tác ghi file. File CSV cũ trên đĩa được giữ nhưng thông báo đó là kết quả lần trước, không công bố như kết quả mới. Người dùng đổi folder phải chạy tuần tự CONFIG → LOADER → FEATURES → PREDICT → REPORT.
- Không bắt buộc predict giữ số cũ: hệ số/model/input mới có thể thay đổi output.

REGRESSION_REPORT:
- Đổi title "hồi quy tuyến tính" thành "hồi quy" và in model class/name thật + folder/scaler.
- Bỏ claims R² 99.88%, MAE 2.14 cố định ở markdown/title/CSV.
- metrics nếu có đọc metrics_test.r2/mae và cv_summary.cv_r2_mean/cv_mae_mean, phân biệt CV/holdout; label saved training/pilot metrics, không phải accuracy ảnh hiện tại; thiếu ghi N/A.
- Giữ final_prediction_summary.csv; cột mới Regression_Predicted_Seeds, Regression_Raw_Prediction, Regression_Model, Regression_Bundle_Dir, Regression_Scaler, Feature_Schema_Version.
- Cột Linear_Regression_Predicted_Seeds cũ nếu giữ vì consumer thì ghi deprecated alias cùng giá trị; search consumer trước khi bỏ.
- Export ordered regression_input_features.csv của ảnh để kiểm tra.
- Không vẽ coef*x trên raw features cho model scale hoặc cây. Phương án tối thiểu: bỏ biểu đồ contributions cũ, in "chưa tính đóng góp từng feature"; giữ plots vision/physical. Không triển khai SHAP/XAI ở task này.
- Không xóa kết quả người dùng hàng loạt; nếu ảnh contributions cũ tồn tại, thông báo nó không được cập nhật và không link như kết quả mới.

Pass: đổi folder không cần sửa coefficients/scaler; báo cáo ghi model thật, không metric bịa.

## Phase 4 — Verification

Không chạy lại train, không bắt buộc GPU để test loader/predict.
Dùng interpreter có sẵn AI_SERVICES/.venv/Scripts/python.exe chỉ như môi trường test, không import AI_SERVICES. Local sklearn khác bundle phải được ghi rõ; parity trong cùng process không chứng minh cross-version reproducibility.

Commands từ root, theo thứ tự:
1. git diff --check
2. AI_SERVICES/.venv/Scripts/python.exe -m unittest discover -s CODE/tests -p test_main_pipeline_regression.py -v
3. Kiểm tra JSON notebook và syntax code qua IPython input transformer (Bước 1 có !pip; không ast.parse trực tiếp magic).
4. Manual Colab: cài version phù hợp, chạy Bước 1–9 trên ảnh thật; chạy nhóm regression với ard rồi decision_tree, giữ nguyên measurements.

Test matrix:
- Load canonical ARD, Decision Tree; lấy tên model từ class của object và metadata thật, không đoán.
- Fixed finite synthetic 31-vector: kết quả manual scaler+model khớp pipeline.joblib.predict(raw DataFrame) cùng folder với rtol=1e-7, atol=1e-7. Không so expected=85.
- Không scaler explicit none: synthetic fitted estimator test nhỏ, không training nghiên cứu.
- Missing folder/model/scaler/schema; duplicate/missing feature, mismatch order artifact/schema; unfitted, NaN/Inf, multioutput fail.
- Một lần transform; feature zero hợp lệ; None khác zero.
- Hybrid=41 reference và outlier case; no-whole-grains/empty measurements phải reject; summary rounding/Std ddof=0/count đúng.
- Thay folder A -> B rồi folder sai: không predict/export bằng state A/B cũ.
- Missing metrics không hiện 0.9988/2.14; tree không chạy contribution giả.
- So source diff: chỉ đổi vùng đã chỉ định, giữ INPUT image/geometry của người dùng.
- Colab thực tế chưa chạy phải ghi NOT VERIFIED; không biến test synthetic thành E2E.

## Definition of Done

- [ ] Một biến folder để chọn model đã train.
- [ ] Model và scaler lấy từ cùng folder; không refit/double scale.
- [ ] Đúng 31 inputs/order/units và hybrid training semantics.
- [ ] Missing inputs và invalid bundle fail explicit; không fallback coefficients.
- [ ] Hai family linear/tree pass reload parity.
- [ ] Reports phản ánh model/metrics thật, no fake contributions.
- [ ] User input và các thay đổi sẵn có được bảo toàn.
- [ ] Kết quả test có command/output thật; Colab acceptance ghi trạng thái thật.
- [ ] Không sửa AI_SERVICES, train code, weights, hoặc push khi chưa yêu cầu.

## Risks / Stop conditions

- Nếu metadata artifact mâu thuẫn schema/scaler, dừng và báo folder cụ thể; không đoán hoặc sửa model để test pass. Version mismatch xử lý theo chính sách Phase 1: cảnh báo/chẩn đoán, chưa đạt acceptance Colab cho đến khi kiểm chứng trong môi trường tương thích.
- Nếu không đủ measurements/khối lượng, dừng prediction và hướng dẫn nhập/đo lại.
- Nếu Google Drive đồng bộ sửa notebook trong lúc đang edit, đọc lại bản mới, merge theo cell source; không ghi đè các thay đổi bên ngoài.
- Chỉ bằng cùng tên 31 features chưa đủ chứng minh ảnh train/inference tương đương; báo khác biệt crop/segmentation nếu phát hiện, không mở rộng refactor vision.
- Không yêu cầu hash mới hay sửa manifest để đổi model. Không claim provenance/history mới.
- Nếu Colab thiếu phiên bản tương thích, báo environment blocker; không fake PASS.

## Prompt cho Gemini 3.1 Pro High

Bạn đang làm việc trong Research-RICAI. Hãy thực thi CODE/RICE_VISION_REGRESSION_FOLDER_PLAN.md đầy đủ.
Đọc plan và source-of-truth trước khi sửa. Đây là implementation notebook, không chỉ tư vấn.
Giữ mọi thay đổi chưa commit, đặc biệt ảnh/thông số người dùng và AI_SERVICES notebook.
Chỉ sửa CODE/RICE_VISION_MAIN_PIPELINE.ipynb và thêm test targeted trong CODE/tests.
Giữ thuật toán Bước 1–9; chỉ cập nhật dependencies Bước 1 và comment/validation PACKING_FRACTION Bước 3. Thay khu Bước 10 bằng các cell config, load bundle, feature, predict, report.
Chỉ cần REGRESSION_MODEL_DIR để nạp model.joblib + scaler.joblib + feature_schema.json cùng folder.
Độc lập AI_SERVICES; không hardcode coefficients; không bắt buộc hash/registry; không fallback ngầm.
Khớp feature semantics dataset_extractor: raw measurements, round summary đúng, hybrid=round(raw_bulk*0.62/raw_mean_volume).
Giữ `PACKING_FRACTION=0.82` cho physical estimate, validate trong 0.80–0.85, và không dùng physical estimate 0.82/IQR làm hybrid. Không missing->0, không giả cân nặng.
Transform đúng một lần; model.predict; hỗ trợ ít nhất ARD và Decision Tree.
Bỏ metric cố định và contribution giả; xuất model/scaler/path/features/raw prediction minh bạch.
Chạy test/parity và ghi rõ Colab E2E chưa chạy nếu chưa có runtime thật.
Không train lại, sửa artifact, sửa AI_SERVICES, triển khai XAI, commit hoặc push.
Cuối cùng báo files changed, cơ chế trước/sau, commands/results, blockers và chỉ dẫn đổi folder/chạy lại khu regression.
