"""Independent, read-only reproduction for the 2026-09-21 root-cause audit.

This script never changes training data, production source, or model artifacts.
It writes only JSON/CSV/JPG evidence beside itself.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATASET = ROOT / "DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv"
MANIFEST_PATH = ROOT / "AI_SERVICES/artifacts/manifest.json"
AI_SRC = ROOT / "AI_SERVICES/src"
sys.path.insert(0, str(AI_SRC))


def dump_json(name: str, value: object) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def metric_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "bias_pred_minus_actual": float(np.mean(error)),
        "mape_pct": float(np.mean(np.abs(error) / y_true) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
    }


def group_id(value: str) -> str:
    match = re.match(r"(M\d+)", str(value).strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Sample_ID không khớp M###: {value!r}")
    return match.group(1).upper()


def load_complete() -> tuple[pd.DataFrame, list[str], dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features = [item["name"] for item in manifest["features"]]
    raw = pd.read_csv(DATASET)
    for name in features + ["Actual_Count"]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    complete = raw.dropna(subset=features + ["Actual_Count"]).copy()
    complete["group_id"] = complete["Sample_ID"].map(group_id)
    return complete, features, manifest


def bootstrap_group_mae_difference(frame: pd.DataFrame, seed: int = 20260921) -> dict:
    """Bootstrap physical groups; positive means full model has lower MAE."""
    group_diff = frame.groupby("group_id", sort=True).apply(
        lambda g: float(np.mean(np.abs(g.actual - g.pred_baseline)) - np.mean(np.abs(g.actual - g.pred_full))),
        include_groups=False,
    )
    values = group_diff.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "definition": "baseline absolute error minus full-model absolute error; positive favors 31 features",
        "n_physical_groups": int(len(values)),
        "mean_mae_improvement_seeds": float(values.mean()),
        "bootstrap_95pct_ci": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_resamples": 10000,
        "seed": seed,
    }


def tabular_audit() -> None:
    from rice_ai.models.regression_loader import LoadedRegressionProvider
    from rice_ai.settings import Settings

    df, features, manifest = load_complete()
    raw = pd.read_csv(DATASET)
    provider = LoadedRegressionProvider(Settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = provider.get_regression()
    load_warnings = [f"{w.category.__name__}: {w.message}" for w in captured] + loaded.warnings

    X = df[features].astype(float)
    y = df["Actual_Count"].astype(float).to_numpy()
    indices = np.arange(len(df))

    # Reconstruct legacy production split exactly as declared by its manifest.
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, shuffle=True)
    production_pred = loaded.model.predict(loaded.scaler.transform(X.iloc[test_idx]))
    production_metrics = metric_dict(y[test_idx], production_pred)
    production_groups_train = set(df.iloc[train_idx].group_id)
    production_groups_test = set(df.iloc[test_idx].group_id)

    # Feature importances come from the actually loaded production object, not metadata JSON.
    importances = np.asarray(loaded.model.feature_importances_, dtype=float)
    importance_rows = [
        {"feature": name, "importance": float(value), "percentage": float(value * 100)}
        for name, value in sorted(zip(features, importances), key=lambda item: -item[1])
    ]
    importance_sums = {
        "container_first_8_pct": float(importances[:8].sum() * 100),
        "surface_features_8_to_10_pct": float(importances[8:11].sum() * 100),
        "grain_morphology_last_20_pct": float(importances[11:].sum() * 100),
    }

    model_params = loaded.model.get_params(deep=False)
    n_estimators = int(model_params.get("n_estimators", 100))
    max_features = model_params.get("max_features", 1.0)
    min_samples_leaf = int(model_params.get("min_samples_leaf", 1))

    # Temporary diagnostic models only; nothing is persisted.
    baseline_features = features[:8]  # Weight_g + seven other container features.
    def make_model():
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=2,
        )

    def fit_predict(train_rows, test_rows, cols):
        scaler = StandardScaler().fit(df.iloc[train_rows][cols])
        model = make_model().fit(scaler.transform(df.iloc[train_rows][cols]), y[train_rows])
        return model.predict(scaler.transform(df.iloc[test_rows][cols]))

    row_full = fit_predict(train_idx, test_idx, features)
    row_base = fit_predict(train_idx, test_idx, baseline_features)
    row_weight = fit_predict(train_idx, test_idx, ["Weight_g"])

    # Independent grouped evaluation: each M### appears in exactly one fold.
    gkf = GroupKFold(n_splits=5)
    grouped_predictions = []
    fold_results = []
    groups = df.group_id.to_numpy()
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        pred_full = fit_predict(tr, te, features)
        pred_base = fit_predict(tr, te, baseline_features)
        pred_weight = fit_predict(tr, te, ["Weight_g"])
        train_groups = set(groups[tr])
        test_groups = set(groups[te])
        if train_groups & test_groups:
            raise AssertionError("GroupKFold leaked physical groups")
        fold_results.append({
            "fold": fold,
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "overlap": [],
            "full_31": metric_dict(y[te], pred_full),
            "baseline_container_8": metric_dict(y[te], pred_base),
            "weight_only": metric_dict(y[te], pred_weight),
        })
        grouped_predictions.append(pd.DataFrame({
            "row_position": te,
            "source_row_id": df.index.to_numpy()[te],
            "Sample_ID": df.iloc[te].Sample_ID.to_numpy(),
            "group_id": groups[te],
            "actual": y[te],
            "pred_full": pred_full,
            "pred_baseline": pred_base,
            "pred_weight_only": pred_weight,
        }))
    grouped = pd.concat(grouped_predictions, ignore_index=True).sort_values("row_position")
    grouped.to_csv(OUT / "grouped_baseline_predictions.csv", index=False)
    grouped_summary = {
        "full_31": metric_dict(grouped.actual, grouped.pred_full),
        "baseline_container_8": metric_dict(grouped.actual, grouped.pred_baseline),
        "weight_only": metric_dict(grouped.actual, grouped.pred_weight_only),
        "paired_group_bootstrap": bootstrap_group_mae_difference(grouped),
        "folds": fold_results,
    }

    # Saved canonical group holdout used for packing-fraction reproduction.
    saved_test = pd.read_csv(ROOT / "LINEAR_REGRESSION_MODEL/models/extra_trees/predictions_test.csv")
    # The saved holdout was produced from a 280-row dataset, while the current
    # table has 276 rows. Preserve that drift as evidence and join by the
    # stable, unique Sample_ID rather than silently trusting stale row offsets.
    source_ids = saved_test["source_row_id"].astype(int)
    raw_ids = raw.Sample_ID.astype(str).str.upper()
    if raw_ids.duplicated().any():
        raise AssertionError("Current dataset has duplicate Sample_ID values; holdout join is ambiguous")
    held = saved_test[["source_row_id", "Sample_ID", "Actual_Count"]].copy()
    held["Sample_ID_normalized"] = held.Sample_ID.astype(str).str.upper()
    current = raw.copy()
    current["Sample_ID_normalized"] = raw_ids
    held = held.merge(current, on="Sample_ID_normalized", how="left", validate="one_to_one", suffixes=("_saved", ""))
    if held["Bulk_Rice_Volume_mm3"].isna().any():
        missing_ids = held.loc[held["Bulk_Rice_Volume_mm3"].isna(), "Sample_ID_saved"].tolist()
        raise AssertionError(f"Saved holdout Sample_ID absent from current dataset: {missing_ids}")
    positional_matches = 0
    for row in saved_test.itertuples(index=False):
        source_id = int(row.source_row_id)
        if source_id in raw.index and str(raw.loc[source_id, "Sample_ID"]).upper() == str(row.Sample_ID).upper():
            positional_matches += 1
    held["physics_082"] = np.rint(held.Bulk_Rice_Volume_mm3 * 0.82 / held.Grain_Volume_mm3_Mean)
    held["physics_062"] = np.rint(held.Bulk_Rice_Volume_mm3 * 0.62 / held.Grain_Volume_mm3_Mean)
    held["physics_055"] = np.rint(held.Bulk_Rice_Volume_mm3 * 0.55 / held.Grain_Volume_mm3_Mean)
    packing_metrics = {
        "holdout_source": "LINEAR_REGRESSION_MODEL/models/extra_trees/predictions_test.csv",
        "join_method": "Sample_ID because current dataset row offsets drifted",
        "saved_source_row_id_matches_current_position": int(positional_matches),
        "saved_holdout_rows": int(len(saved_test)),
        "phi_0_82": metric_dict(held.Actual_Count, held.physics_082),
        "phi_0_62": metric_dict(held.Actual_Count, held.physics_062),
        "phi_0_55": metric_dict(held.Actual_Count, held.physics_055),
    }
    held[["Sample_ID_saved", "source_row_id", "Actual_Count", "Bulk_Rice_Volume_mm3", "Grain_Volume_mm3_Mean", "physics_082", "physics_062", "physics_055"]].to_csv(
        OUT / "packing_holdout_predictions.csv", index=False
    )

    # Data quality and independent group-split verification.
    duplicate_counts = raw.Sample_ID.astype(str).str.upper().value_counts()
    missing = raw.Image_Status.astype(str).str.upper().eq("MISSING")
    complete_under_8 = df.Whole_Grains_Count.lt(8)
    cup_groups = []
    ranges = []
    for (diam, height), part in df.groupby(["Inner_Diameter_mm", "Container_Height_mm"]):
        lo, hi = float(part.Actual_Count.min()), float(part.Actual_Count.max())
        ranges.append((lo, hi))
        cup_groups.append({
            "inner_diameter_mm": float(diam), "container_height_mm": float(height),
            "rows": int(len(part)), "actual_count_min": lo, "actual_count_max": hi,
        })
    cup_overlap = max(0.0, min(r[1] for r in ranges) - max(r[0] for r in ranges)) if len(ranges) > 1 else None

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    grouped_train_idx, grouped_test_idx = next(gss.split(X, y, groups=groups))
    gss_train = set(groups[grouped_train_idx]); gss_test = set(groups[grouped_test_idx])

    saved_train = pd.read_csv(ROOT / "LINEAR_REGRESSION_MODEL/models/extra_trees/predictions_train.csv")
    saved_group_test = pd.read_csv(ROOT / "LINEAR_REGRESSION_MODEL/models/extra_trees/predictions_test.csv")
    saved_train_groups = set(saved_train.Sample_ID.map(group_id)); saved_test_groups = set(saved_group_test.Sample_ID.map(group_id))

    thickness_gt_width = df.Grain_Thickness_mm_Mean.gt(df.Grain_Width_mm_Mean)
    effective_phi = df.Actual_Count * df.Grain_Volume_mm3_Mean / df.Bulk_Rice_Volume_mm3
    quality = {
        "raw_rows": int(len(raw)),
        "complete_31_feature_rows": int(len(df)),
        "missing_image_rows": int(missing.sum()),
        "duplicate_sample_ids": {k: int(v) for k, v in duplicate_counts[duplicate_counts > 1].items()},
        "complete_rows_whole_grains_lt_8": int(complete_under_8.sum()),
        "thickness_gt_width": {
            "rows": int(thickness_gt_width.sum()),
            "denominator": int(len(df)),
            "percentage": float(thickness_gt_width.mean() * 100),
            "median_thickness_to_width_ratio": float((df.Grain_Thickness_mm_Mean / df.Grain_Width_mm_Mean).median()),
        },
        "weight_actual_pearson": float(df.Weight_g.corr(df.Actual_Count)),
        "effective_phi_quantiles": {str(k): float(v) for k, v in effective_phi.quantile([0, .1, .25, .5, .75, .9, 1]).items()},
        "cup_types": cup_groups,
        "actual_count_range_overlap_width": cup_overlap,
        "independent_GroupShuffleSplit": {
            "train_rows": int(len(grouped_train_idx)), "test_rows": int(len(grouped_test_idx)),
            "train_groups": len(gss_train), "test_groups": len(gss_test),
            "overlap_groups": sorted(gss_train & gss_test),
        },
        "saved_grouped_artifact_split": {
            "train_rows": int(len(saved_train)), "test_rows": int(len(saved_group_test)),
            "train_groups": len(saved_train_groups), "test_groups": len(saved_test_groups),
            "overlap_groups": sorted(saved_train_groups & saved_test_groups),
        },
        "production_legacy_row_split": {
            "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
            "train_groups": len(production_groups_train), "test_groups": len(production_groups_test),
            "overlap_group_count": len(production_groups_train & production_groups_test),
            "overlap_groups": sorted(production_groups_train & production_groups_test),
        },
    }

    result = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "production_resolution": {
            "settings_directory": str(loaded.model_dir.relative_to(ROOT)),
            "model_file_layout": "legacy best_tree_ensemble_model.joblib",
            "is_legacy": loaded.is_legacy,
            "model_class": loaded.model.__class__.__name__,
            "scaler_class": loaded.scaler.__class__.__name__,
            "model_params": {k: v for k, v in model_params.items() if isinstance(v, (str, int, float, bool, type(None)))},
            "load_warnings": load_warnings,
            "manifest_reported_metrics": manifest["model"]["reported_metrics"],
            "reconstructed_row_holdout_metrics": production_metrics,
        },
        "feature_importances_actual_loaded_model": importance_rows,
        "feature_importance_sums": importance_sums,
        "row_split_temporary_models": {
            "full_31": metric_dict(y[test_idx], row_full),
            "baseline_container_8": metric_dict(y[test_idx], row_base),
            "weight_only": metric_dict(y[test_idx], row_weight),
            "baseline_features": baseline_features,
        },
        "grouped_5fold_temporary_models": grouped_summary,
        "packing_fraction_reproduction": packing_metrics,
        "data_quality": quality,
    }
    dump_json("tabular_audit.json", result)
    (OUT / "tabular_audit.log").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "production": result["production_resolution"],
        "importance_sums": importance_sums,
        "row_split_models": result["row_split_temporary_models"],
        "grouped_5fold": grouped_summary,
        "packing": packing_metrics,
        "quality": quality,
    }, ensure_ascii=False, indent=2, default=str))


def scale_audit() -> None:
    import cv2
    from rice_ai.vision.container_detector import detect_container_and_scale
    import rice_ai.vision.container_detector as detector_module

    cv2.setNumThreads(2)
    df, _, _ = load_complete()
    image_root = ROOT / "DATASET_BUILDER/1_Raw_Images"
    images = {p.name.upper(): p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    sample_ids = ["M001a", "M010a", "M014a", "M020a", "M045a", "M056a"]
    rows = []
    repeated = []
    required_ratio = (800.0 / 450.0) ** (1.0 / 3.0)
    detector_source = inspect.getsource(detector_module)
    random_evidence = {
        "contains_np_random": "np.random" in detector_source,
        "contains_python_random_import": bool(re.search(r"^\s*import\s+random", detector_source, flags=re.MULTILINE)),
        "contains_ransac": "ransac" in detector_source.lower(),
        "contains_hough": "HoughCircles" in detector_source,
        "interpretation": "No stochastic RNG/RANSAC path found; OpenCV Hough/contour path is deterministic for fixed input in this test.",
    }

    def one(sample_id: str, run: int) -> dict:
        source = df[df.Sample_ID.str.upper().eq(sample_id.upper())].iloc[0]
        image_path = images[str(source.Image_Filename).upper()]
        started = time.perf_counter()
        info = detect_container_and_scale(
            image_input=image_path,
            inner_diam_mm=float(source.Inner_Diameter_mm),
            container_height_mm=float(source.Container_Height_mm),
            empty_height_mm=float(source.Empty_Height_mm),
            detect_mode="inner",
        )
        ratio = float(info["pixels_per_mm"] / source.Pixels_Per_mm)
        multiplier = ratio ** 3
        contribution = 450.0 * (multiplier - 1.0) / (800.0 - 450.0) * 100.0
        return {
            "sample_id": sample_id,
            "run": run,
            "image_filename": source.Image_Filename,
            "actual_count": float(source.Actual_Count),
            "historical_pixels_per_mm": float(source.Pixels_Per_mm),
            "current_pixels_per_mm": float(info["pixels_per_mm"]),
            "current_inner_diameter_px": int(info["inner_w_px"]),
            "detect_type": info["detect_type"],
            "scale_ratio_current_over_historical": ratio,
            "conditional_geometry_count_multiplier_ratio_cubed": multiplier,
            "conditional_pct_of_450_to_800_excess": contribution,
            "elapsed_seconds": time.perf_counter() - started,
        }, info

    for sample_id in sample_ids:
        record, info = one(sample_id, 1)
        rows.append(record)
        overlay = info["overlay_bgr"]
        factor = min(1.0, 1000.0 / max(overlay.shape[:2]))
        if factor < 1.0:
            overlay = cv2.resize(overlay, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(OUT / f"scale_{sample_id}.jpg")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    repeat_id = sample_ids[0]
    # First measurement above + four additional runs = five total values.
    repeated.append(next(r for r in rows if r["sample_id"] == repeat_id))
    for run in range(2, 6):
        record, _ = one(repeat_id, run)
        repeated.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    scale_frame = pd.DataFrame(rows)
    scale_frame.to_csv(OUT / "scale_comparison.csv", index=False)
    repeat_values = np.asarray([r["current_pixels_per_mm"] for r in repeated], dtype=float)
    repeatability = {
        "sample_id": repeat_id,
        "runs": repeated,
        "min": float(repeat_values.min()),
        "max": float(repeat_values.max()),
        "range": float(np.ptp(repeat_values)),
        "std_population": float(repeat_values.std(ddof=0)),
        "all_exactly_equal": bool(np.all(repeat_values == repeat_values[0])),
    }
    summary = {
        "detector_source": "AI_SERVICES/src/rice_ai/vision/container_detector.py",
        "required_scale_ratio_to_turn_450_into_800_if_other_terms_fixed": required_ratio,
        "required_scale_overestimate_pct": (required_ratio - 1.0) * 100.0,
        "randomness_source_audit": random_evidence,
        "repeatability": repeatability,
        "samples": rows,
        "caveat": "Current/historical scale ratio is not current/ground-truth scale. Contribution percentages are sensitivity calculations conditional on historical scale being correct and segmentation pixel volumes remaining fixed.",
    }
    dump_json("scale_audit.json", summary)
    (OUT / "scale_audit.log").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tabular", "scale"], required=True)
    args = parser.parse_args()
    if args.mode == "tabular":
        tabular_audit()
    else:
        scale_audit()
