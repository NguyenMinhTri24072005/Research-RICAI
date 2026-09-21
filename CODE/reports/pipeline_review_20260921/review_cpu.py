"""Read-only research pipeline review; outputs stay in this report directory.

Run --mode tabular first, then --mode vision. No fitting or source modification.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import argparse
import ast
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'CODE'))
import numpy as np
import pandas as pd
import joblib

DATA = ROOT / 'DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv'
BUNDLE = ROOT / 'LINEAR_REGRESSION_MODEL/models/extra_trees'
NB = json.loads((ROOT / 'CODE/RICE_VISION_MAIN_PIPELINE.ipynb').read_text(encoding='utf-8'))
SOURCES = [''.join(c['source']) for c in NB['cells']]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def metrics(y, pred):
    e = np.asarray(pred) - np.asarray(y)
    return dict(n=len(e), mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(e.mean()), mape_pct=float(np.mean(abs(e) / y) * 100),
                median_ratio=float(np.median(np.asarray(pred) / y)))


def source(marker):
    return next(s for s in SOURCES if marker in s)


def assignment(name):
    for s in SOURCES:
        try:
            nodes = ast.parse(s).body
        except SyntaxError:
            continue
        for n in nodes:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                try:
                    return ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    pass
    raise KeyError(name)


def key(s):
    return str(s).strip().upper()


def tabular():
    print('Tabular: reading dataset', flush=True)
    df = pd.read_csv(DATA)
    names = json.loads((BUNDLE / 'feature_schema.json').read_text())['features']
    valid = np.isfinite(df[names + ['Actual_Count']].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[valid].copy()
    print('Tabular: loading regression bundle', flush=True)
    model, scaler = joblib.load(BUNDLE / 'model.joblib'), joblib.load(BUNDLE / 'scaler.joblib')
    print('Tabular: predicting stored features', flush=True)
    scaled = scaler.transform(df[names])
    pred = model.predict(scaled)
    reference = joblib.load(BUNDLE / 'pipeline.joblib').predict(df[names])
    train = pd.read_csv(BUNDLE / 'predictions_train.csv')
    test = pd.read_csv(BUNDLE / 'predictions_test.csv')
    df['key'] = df.Sample_ID.map(key)
    train_keys, test_keys = set(train.Sample_ID.map(key)), set(test.Sample_ID.map(key))
    group = lambda s: re.match(r'M\d+', key(s)).group()
    train_groups, test_groups = set(train.Sample_ID.map(group)), set(test.Sample_ID.map(group))
    df['regression_now'] = pred
    df['physics_stored_mean_082'] = np.rint(df.Bulk_Rice_Volume_mm3 * 0.82 / df.Grain_Volume_mm3_Mean)
    df['physics_stored_mean_062'] = np.rint(df.Bulk_Rice_Volume_mm3 * 0.62 / df.Grain_Volume_mm3_Mean)
    df['effective_phi_to_match_label'] = df.Actual_Count * df.Grain_Volume_mm3_Mean / df.Bulk_Rice_Volume_mm3
    df['split'] = np.where(df.key.isin(test_keys), 'saved_holdout', np.where(df.key.isin(train_keys), 'saved_train', 'unknown'))
    held = df[df.split == 'saved_holdout'].copy()
    merged = held.merge(test.assign(key=test.Sample_ID.map(key)), on='key', suffixes=('', '_saved'))
    importance = sorted(zip(names, model.feature_importances_.tolist()), key=lambda v: -v[1])
    desc_columns = ['Actual_Count', 'Weight_g', 'Inner_Diameter_mm', 'Container_Height_mm',
                    'Whole_Grains_Count', 'Grain_Length_mm_Mean', 'Grain_Width_mm_Mean',
                    'Grain_Thickness_mm_Mean', 'Grain_Volume_mm3_Mean']
    report = dict(dataset=str(DATA.relative_to(ROOT)), n_total=len(valid), n_valid=len(df),
                  n_invalid=int((~valid).sum()), n_features=len(names),
                  versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
                  inference_ms=None, pipeline_max_diff=float(np.max(abs(pred - reference))),
                  holdout_prediction_max_diff=float(np.max(abs(merged.regression_now - merged.Predicted_Count))),
                  saved_train_rows=len(train), saved_test_rows=len(test),
                  train_groups=len(train_groups), test_groups=len(test_groups),
                  overlap_groups=sorted(train_groups & test_groups),
                  dataset_summary=json.loads(df[desc_columns].describe().to_json()),
                  feature_importances=importance,
                  effective_phi_quantiles=df.effective_phi_to_match_label.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
                  weight_count_corr=float(df.Weight_g.corr(df.Actual_Count)),
                  weight_per_seed_quantiles=(df.Weight_g / df.Actual_Count).quantile([0, .5, 1]).to_dict(),
                  thickness_length_ratio_median=float((df.Grain_Thickness_mm_Mean / df.Grain_Length_mm_Mean).median()),
                  thickness_greater_width_fraction=float((df.Grain_Thickness_mm_Mean > df.Grain_Width_mm_Mean).mean()),
                  scores={})
    for split, sub in [('all_mixed_not_validation', df), ('saved_holdout', held)]:
        report['scores'][split] = {col: metrics(sub.Actual_Count.to_numpy(), sub[col].to_numpy())
                                  for col in ['regression_now', 'physics_stored_mean_082', 'physics_stored_mean_062']}
    # Time after artifact load, a warmed batch, with repeated prediction only (no fitting).
    ts = time.perf_counter()
    for _ in range(10):
        model.predict(scaler.transform(df[names]))
    report['inference_ms'] = (time.perf_counter() - ts) * 1000 / 10
    selected = held.sort_values(['Actual_Count', 'Sample_ID']).groupby(held.Sample_ID.map(group)).first().sort_values('Actual_Count')
    selected = selected.iloc[[0, len(selected) // 2, len(selected) - 1]]
    report['selected_smoke_ids'] = selected.Sample_ID.tolist()
    df[['Sample_ID', 'Actual_Count', 'split', 'regression_now', 'physics_stored_mean_082',
        'physics_stored_mean_062', 'effective_phi_to_match_label']].to_csv(OUT / 'tabular_predictions.csv', index=False)
    save('tabular_summary.json', report)
    print(json.dumps({k: report[k] for k in ['n_valid', 'saved_test_rows', 'overlap_groups', 'holdout_prediction_max_diff',
                                           'scores', 'effective_phi_quantiles', 'feature_importances', 'selected_smoke_ids']}, indent=2), flush=True)


def vision():
    print('Vision: loading CPU libraries', flush=True)
    import cv2
    import torch
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sahi import AutoDetectionModel
    from modules.container_detector import detect_container_and_scale
    from modules.grain_segmenter import segment_grains_sahi
    from modules.grain_crop_cleaner import clean_single_grain_crop
    from modules.grain_classifier import GrainClassifier
    from modules.ellipsoid_geometry import compute_single_grain_metrics, draw_grain_ellipse_overlay
    torch.set_num_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    env_base = dict(os=os, np=np, pd=pd, math=math, cv2=cv2, plt=plt, display=lambda *args: None,
                    BASE_PATH=str(ROOT), detect_container_and_scale=detect_container_and_scale,
                    segment_grains_sahi=segment_grains_sahi, clean_single_grain_crop=clean_single_grain_crop,
                    compute_single_grain_metrics=compute_single_grain_metrics,
                    draw_grain_ellipse_overlay=draw_grain_ellipse_overlay)
    t = time.perf_counter()
    env_base['sahi_yolo_model'] = AutoDetectionModel.from_pretrained(model_type='yolov8',
        model_path='RESULTS/all-new-data-v1.yolov8_yolov8s-seg_trained/weights/best.pt',
        confidence_threshold=0.5, device='cpu')
    env_base['cnn_classifier'] = GrainClassifier('RESULTS/CNN_DenseNet121_Trained/best_v3_step2.keras',
        class_names=['hat_khuyet_tat', 'hat_nguyen'], target_size=(224, 224))
    load_seconds = time.perf_counter() - t
    df = pd.read_csv(DATA)
    selected = json.loads((OUT / 'tabular_summary.json').read_text())['selected_smoke_ids']
    images = {p.name.upper(): p for p in (ROOT / 'DATASET_BUILDER/1_Raw_Images').rglob('*')
              if p.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    results = []
    names = json.loads((BUNDLE / 'feature_schema.json').read_text())['features']
    for sample_id in selected:
        row = df[df.Sample_ID.map(key) == key(sample_id)].iloc[0]
        env = dict(env_base)
        timings = {}
        result = {'sample_id': sample_id, 'actual_count': int(row.Actual_Count), 'timings': timings}
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix='rice_pipeline_cpu_') as temporary:
                env.update(IMAGE_PATH=str(images[key(row.Image_Filename)].relative_to(ROOT)),
                    INPUT_WEIGHT_G=float(row.Weight_g), INNER_DIAM_MM=float(row.Inner_Diameter_mm),
                    CONTAINER_HEIGHT_MM=float(row.Container_Height_mm), EMPTY_HEIGHT_MM=float(row.Empty_Height_mm),
                    USE_CROPPED_CONTAINER=False, PACKING_FRACTION=assignment('PACKING_FRACTION'),
                    CONF_THRESHOLD_NGUYEN=assignment('CONF_THRESHOLD_NGUYEN'),
                    CLEAN_PARAMS_STEP1=assignment('CLEAN_PARAMS_STEP1'), CLEAN_PARAMS_STEP2=assignment('CLEAN_PARAMS_STEP2'),
                    ENABLE_SIZE_FILTER=assignment('ENABLE_SIZE_FILTER'), SIZE_FILTER_K=assignment('SIZE_FILTER_K'),
                    SIZE_FILTER_MIN_SAMPLES=assignment('SIZE_FILTER_MIN_SAMPLES'), OUTPUT_DIR=temporary)
                for name, suffix in [('OUTPUT_RAW_CROPS_DIR', 'raw'), ('OUTPUT_CLEANED_CROPS_DIR', 'clean'),
                                     ('OUTPUT_NGUYEN_DIR', 'whole'), ('OUTPUT_KHUYET_DIR', 'defective'), ('OUTPUT_REPORT_DIR', 'reports')]:
                    env[name] = str(Path(temporary) / suffix)
                    Path(env[name]).mkdir()
                markers = [('container', '# Bước 4:'), ('sahi', '# Bước 5:'), ('clean', '# Bước 6:'),
                           ('cnn', '# Bước 7:'), ('measurement', 'GRAIN_MEASUREMENTS'),
                           ('filter', 'GRAIN_SIZE_FILTER (BỘ LỌC'), ('physics', 'GRAIN_PHYSICAL_STATISTICS')]
                with (OUT / f'{sample_id}_execution.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                    for stage, marker in markers:
                        ts = time.perf_counter()
                        exec(source(marker), env)
                        timings[stage] = round(time.perf_counter() - ts, 3)
                        plt.close('all')
                    env['resolved_model_dir'] = str(BUNDLE)
                    exec(source('REGRESSION_BUNDLE_LOADER:'), env)
                    exec(source('REGRESSION_FEATURES:'), env)
                    ts = time.perf_counter()
                    exec(source('REGRESSION_PREDICT:'), env)
                    timings['regression_predict'] = round(time.perf_counter() - ts, 3)
                result.update(status='pass', image_shape=list(cv2.imread(env['IMAGE_PATH']).shape),
                    raw_detected=len(env['raw_grains']), cleaned=len(env['cleaned_grains']),
                    cnn_whole=len(env['whole_grains']), measured=len(env['whole_records']),
                    filtered=len(env['filtered_whole_records']), filter_stats=env['filter_stats'],
                    pixels_per_mm=env['pixels_per_mm'], saved_pixels_per_mm=float(row.Pixels_Per_mm),
                    bulk_volume_mm3=env['bulk_volume_mm3'], saved_bulk_volume_mm3=float(row.Bulk_Rice_Volume_mm3),
                    physical_prediction=env['estimated_seed_count'],
                    physical_raw_iqr_prediction=int(round(env['bulk_volume_mm3'] * env['PACKING_FRACTION'] / env['uniformity_res_raw']['mean_clean'])),
                    physical_mean_volume=env['mean_whole_grain_vol'], saved_mean_volume=float(row.Grain_Volume_mm3_Mean),
                    regression_prediction=env['regression_result']['raw_count'])
                live = env['regression_features'].iloc[0]
                comparison = pd.DataFrame({'feature': names, 'stored': [float(row[n]) for n in names], 'live': live.to_numpy()})
                comparison['delta'] = comparison.live - comparison.stored
                comparison.to_csv(OUT / f'{sample_id}_features.csv', index=False)
                pd.DataFrame(env['whole_records']).to_csv(OUT / f'{sample_id}_measurements.csv', index=False)
                if 'filtered_regression_features' in env and env['filtered_regression_features'] is not None:
                    env['filtered_regression_features'].to_csv(OUT / f'{sample_id}_filtered_features.csv', index=False)
        except Exception as e:
            result.update(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        result['total_seconds'] = round(time.perf_counter() - started, 3)
        results.append(result)
        save('vision_summary.json', dict(model_load_seconds=load_seconds, cpu_only=True,
             versions={p: importlib.metadata.version(p) for p in ['torch', 'tensorflow', 'sahi', 'ultralytics']}, results=results))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print('CPU VISION REVIEW DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tabular', 'vision'], required=True)
    args = parser.parse_args()
    try:
        (tabular if args.mode == 'tabular' else vision)()
    except Exception as exc:
        if args.mode == 'vision':
            save('vision_failure.json', dict(status='NOT VERIFIED', phase='vision_initialization',
                 error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                 note='No image-to-count E2E result. No model or dependency was changed.'))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
