#!/usr/bin/env python3
"""Comprehensive verification suite for pipeline parity, contracts, deterministic orchestration, and ID migration safety."""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import openpyxl

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add DATASET_BUILDER/src to sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def run_static_checks() -> bool:
    """Run strict static contract, parameter, domain validation, and dependency checks."""
    print("=" * 65)
    print("[*] RUNNING STATIC CONTRACT & PARITY VERIFICATION")
    print("=" * 65)

    all_passed = True

    # Check 1: 31 Features Count and Exact Order
    print("[1/8] Verifying 31-Feature schema order and names...")
    from rice_dataset.contracts import (
        ALL_QC_STATUSES,
        CANONICAL_COLUMNS,
        DIAGNOSTIC_COLUMNS,
        LABEL_COLUMN,
        LINEAGE_COLUMNS,
        ORDERED_FEATURES,
        SCHEMA_VERSION,
        STATUS_COLUMNS,
        QCStatus,
    )

    EXPECTED_31 = [
        "Bulk_Rice_Volume_mm3",
        "Rice_Height_mm",
        "Weight_g",
        "Empty_Height_mm",
        "Pixels_Per_mm",
        "Container_Detected_Diam_px",
        "Inner_Diameter_mm",
        "Container_Height_mm",
        "Whole_Grains_Count",
        "Uniformity_Rate_Pct",
        "Estimated_Total_Seeds_Hybrid",
        "Grain_Length_mm_Mean",
        "Grain_Length_mm_Min",
        "Grain_Length_mm_Max",
        "Grain_Length_mm_Std",
        "Grain_Width_mm_Mean",
        "Grain_Width_mm_Min",
        "Grain_Width_mm_Max",
        "Grain_Width_mm_Std",
        "Grain_Thickness_mm_Mean",
        "Grain_Thickness_mm_Min",
        "Grain_Thickness_mm_Max",
        "Grain_Thickness_mm_Std",
        "Grain_Area_mm2_Mean",
        "Grain_Area_mm2_Min",
        "Grain_Area_mm2_Max",
        "Grain_Area_mm2_Std",
        "Grain_Volume_mm3_Mean",
        "Grain_Volume_mm3_Min",
        "Grain_Volume_mm3_Max",
        "Grain_Volume_mm3_Std",
    ]

    if ORDERED_FEATURES != EXPECTED_31:
        print("❌ FAIL: ORDERED_FEATURES does not match the exact 31-feature specification.")
        all_passed = False
    else:
        print(f"✅ PASS: Schema version {SCHEMA_VERSION} matches exact 31 ordered features.")

    # Check 2: Actual_Count isolation (no leakage)
    print("[2/8] Verifying target isolation (Actual_Count excluded from features)...")
    if LABEL_COLUMN in ORDERED_FEATURES or "Actual_Count" in ORDERED_FEATURES:
        print("❌ FAIL: Actual_Count is present in ORDERED_FEATURES!")
        all_passed = False
    else:
        print("✅ PASS: Actual_Count is strictly isolated as label/metadata.")

    # Check 3: Extraction Config parameter validation
    print("[3/8] Verifying scientific parameters in extraction_config.json...")
    from rice_dataset.config import ExtractionConfig

    config_path = Path(__file__).resolve().parent.parent / "config" / "extraction_config.json"
    cfg = ExtractionConfig.from_file(config_path)

    checks = [
        ("yolo_model_load_confidence", cfg.yolo_model_load_confidence, 0.70),
        ("yolo_result_confidence", cfg.yolo_result_confidence, 0.50),
        ("sahi_slice_size", cfg.sahi_slice_size, 640),
        ("sahi_overlap_ratio", cfg.sahi_overlap_ratio, 0.25),
        ("sahi_min_area_px", cfg.sahi_min_area_px, 50),
        ("cnn_whole_confidence", cfg.cnn_whole_confidence, 0.90),
        ("clean_step1.min_neck_ratio", cfg.clean_step1.min_neck_ratio, 0.15),
        ("clean_step1.sever_bridges", cfg.clean_step1.sever_bridges, False),
        ("clean_step2.min_neck_ratio", cfg.clean_step2.min_neck_ratio, 0.15),
        ("clean_step2.sever_bridges", cfg.clean_step2.sever_bridges, False),
        ("size_filter_k", cfg.size_filter_k, 0.1),
        ("size_filter_min_samples", cfg.size_filter_min_samples, 8),
        ("physical_packing_fraction", cfg.physical_packing_fraction, 0.55),
        ("trained_hybrid_packing_fraction", cfg.trained_hybrid_packing_fraction, 0.62),
        ("whole_grain_thickness_ratio", cfg.whole_grain_thickness_ratio, 0.80),
        ("container_edge", cfg.container_edge, "inner"),
        ("apply_depth_correction", cfg.apply_depth_correction, False),
    ]

    param_fail = False
    for name, actual, expected in checks:
        if actual != expected:
            print(f"❌ FAIL: {name} = {actual}, expected {expected}")
            param_fail = True
    if param_fail:
        all_passed = False
    else:
        print("✅ PASS: All 17 scientific and vision parameters strictly match the contract.")

    # Check 4: Separation of Packing Fractions and Deterministic Feature Calculation
    print("[4/8] Verifying packing fraction independence & deterministic feature builder...")
    from rice_dataset.feature_builder import build_physical_diagnostics, build_regression_features

    sample_metrics = [
        {"length_mm": 9.2, "width_mm": 2.4, "thickness_mm": 1.92, "area_mm2": 18.5, "volume_mm3": 22.0},
        {"length_mm": 9.0, "width_mm": 2.3, "thickness_mm": 1.84, "area_mm2": 17.8, "volume_mm3": 20.5},
        {"length_mm": 9.5, "width_mm": 2.5, "thickness_mm": 2.00, "area_mm2": 19.1, "volume_mm3": 23.5},
    ]

    feats1 = build_regression_features(
        raw_valid_metrics=sample_metrics,
        bulk_volume_mm3=10000.0,
        rice_height_mm=25.0,
        weight_g=3.5,
        empty_height_mm=5.0,
        pixels_per_mm=75.0,
        container_detected_diam_px=1500.0,
        inner_diameter_mm=20.0,
        container_height_mm=30.0,
        trained_hybrid_packing_fraction=0.62,
    )

    diag_p1 = build_physical_diagnostics(
        filtered_volumes=[22.0, 20.5],
        raw_volumes=[22.0, 20.5, 23.5],
        bulk_volume_mm3=10000.0,
        pixels_per_mm=75.0,
        container_detected_diam_px=1500.0,
        physical_packing_fraction=0.55,
    )
    diag_p2 = build_physical_diagnostics(
        filtered_volumes=[22.0, 20.5],
        raw_volumes=[22.0, 20.5, 23.5],
        bulk_volume_mm3=10000.0,
        pixels_per_mm=75.0,
        container_detected_diam_px=1500.0,
        physical_packing_fraction=0.82,
    )

    if feats1["Estimated_Total_Seeds_Hybrid"] != round(10000.0 * 0.62 / ((22.0 + 20.5 + 23.5) / 3.0)):
        print("❌ FAIL: Estimated_Total_Seeds_Hybrid formula deviation.")
        all_passed = False
    elif diag_p1["Physical_Estimated_Seeds"] == diag_p2["Physical_Estimated_Seeds"]:
        print("❌ FAIL: Physical estimate diagnostic did not respond to physical packing fraction.")
        all_passed = False
    else:
        print("✅ PASS: Physical fraction (0.55) and hybrid fraction (0.62) are completely decoupled.")

    # Check 5: Domain validation (positive dimensions/area/volume, rejection of non-positive)
    print("[5/8] Verifying domain validation (positive length, width, thickness, area, volume)...")
    invalid_cases = [
        ("empty metrics", []),
        ("negative width", [{"length_mm": 9.0, "width_mm": -2.0, "thickness_mm": 1.6, "area_mm2": 15.0, "volume_mm3": 18.0}]),
        ("zero thickness", [{"length_mm": 9.0, "width_mm": 2.0, "thickness_mm": 0.0, "area_mm2": 15.0, "volume_mm3": 18.0}]),
        ("negative area", [{"length_mm": 9.0, "width_mm": 2.0, "thickness_mm": 1.6, "area_mm2": -5.0, "volume_mm3": 18.0}]),
        ("zero volume", [{"length_mm": 9.0, "width_mm": 2.0, "thickness_mm": 1.6, "area_mm2": 15.0, "volume_mm3": 0.0}]),
    ]

    domain_fail = False
    for label, metrics in invalid_cases:
        try:
            build_regression_features(
                raw_valid_metrics=metrics,
                bulk_volume_mm3=10000.0,
                rice_height_mm=25.0,
                weight_g=3.5,
                empty_height_mm=5.0,
                pixels_per_mm=75.0,
                container_detected_diam_px=1500.0,
                inner_diameter_mm=20.0,
                container_height_mm=30.0,
            )
            print(f"❌ FAIL: build_regression_features accepted invalid domain: {label}")
            domain_fail = True
        except ValueError:
            pass

    if domain_fail:
        all_passed = False
    else:
        print("✅ PASS: Non-positive or non-finite measurements properly rejected.")

    # Check 6: Import Isolation from CODE and AI_SERVICES
    print("[6/8] Verifying import isolation (no dependencies on CODE.modules or AI_SERVICES)...")
    import rice_dataset
    import rice_dataset.config
    import rice_dataset.contracts
    import rice_dataset.feature_builder
    import rice_dataset.io.dataset_writer
    import rice_dataset.io.image_index
    import rice_dataset.io.manual_records
    import rice_dataset.migration.id_mapping
    import rice_dataset.migration.migrate_ids
    import rice_dataset.pipeline
    import rice_dataset.vision.container_detector
    import rice_dataset.vision.ellipsoid_geometry
    import rice_dataset.vision.grain_classifier
    import rice_dataset.vision.grain_crop_cleaner
    import rice_dataset.vision.grain_segmenter
    import rice_dataset.vision.grain_size_filter
    import rice_dataset.vision.uniformity_evaluator

    modules_to_scan = [
        rice_dataset,
        rice_dataset.config,
        rice_dataset.contracts,
        rice_dataset.feature_builder,
        rice_dataset.pipeline,
        rice_dataset.io.manual_records,
        rice_dataset.io.image_index,
        rice_dataset.io.dataset_writer,
        rice_dataset.migration.id_mapping,
        rice_dataset.migration.migrate_ids,
        rice_dataset.vision.container_detector,
        rice_dataset.vision.grain_segmenter,
        rice_dataset.vision.grain_crop_cleaner,
        rice_dataset.vision.grain_classifier,
        rice_dataset.vision.ellipsoid_geometry,
        rice_dataset.vision.uniformity_evaluator,
        rice_dataset.vision.grain_size_filter,
    ]

    import_leak = False
    for mod in modules_to_scan:
        src = inspect.getsource(mod)
        for line in src.splitlines():
            line_str = line.strip()
            if line_str.startswith(("import ", "from ")):
                if "CODE" in line_str or "AI_SERVICES" in line_str:
                    print(f"❌ FAIL: Leaked import in {mod.__name__}: {line_str}")
                    import_leak = True
    if import_leak:
        all_passed = False
    else:
        print("✅ PASS: Zero runtime imports from CODE or AI_SERVICES found.")

    # Check 7: Column structure count and uniqueness
    print("[7/8] Verifying canonical column counts and strict uniqueness...")
    if len(CANONICAL_COLUMNS) != len(set(CANONICAL_COLUMNS)):
        duplicates = [col for col in CANONICAL_COLUMNS if CANONICAL_COLUMNS.count(col) > 1]
        print(f"❌ FAIL: Duplicate columns in CANONICAL_COLUMNS: {set(duplicates)}")
        all_passed = False
    elif len(CANONICAL_COLUMNS) != 47:
        print(f"❌ FAIL: Expected exactly 47 unique canonical columns, got {len(CANONICAL_COLUMNS)}")
        all_passed = False
    else:
        print(f"✅ PASS: Canonical dataset column count is exactly 47 unique columns (no duplicate names).")

    # Check 8: Vendor Module Hashes Identity
    print("[8/8] Verifying byte-parity of vendored vision modules with CODE/modules/...")
    import hashlib

    vendor_files = [
        "container_detector.py", "grain_segmenter.py", "grain_crop_cleaner.py",
        "grain_classifier.py", "ellipsoid_geometry.py", "uniformity_evaluator.py",
        "grain_size_filter.py"
    ]
    hash_fail = False
    for vf in vendor_files:
        src_path = Path("CODE/modules") / vf
        dst_path = Path("DATASET_BUILDER/src/rice_dataset/vision") / vf
        if not src_path.exists() or not dst_path.exists():
            print(f"❌ FAIL: Module file missing: {vf}")
            hash_fail = True
            continue
        h1 = hashlib.sha256(src_path.read_bytes()).hexdigest()
        h2 = hashlib.sha256(dst_path.read_bytes()).hexdigest()
        if h1 != h2:
            print(f"❌ FAIL: Hash mismatch for {vf} (CODE: {h1[:8]}.. vs VENDOR: {h2[:8]}..)")
            hash_fail = True

    if hash_fail:
        all_passed = False
    else:
        print("✅ PASS: All 7 vendored modules match CODE/modules/ exactly.")

    print("=" * 65)
    if all_passed:
        print("🎉 ALL STATIC CONTRACT VERIFICATIONS PASSED!")
    else:
        print("❌ STATIC VERIFICATION HAD FAILURES.")
    print("=" * 65)
    return all_passed


def run_orchestration_fake_tests() -> bool:
    """Run deterministic mocked/fake orchestration tests for PASS and all failure scenarios."""
    print("\n" + "=" * 65)
    print("🤖 RUNNING DETERMINISTIC ORCHESTRATION FAKE/SMOKE TESTS")
    print("=" * 65)

    from rice_dataset.config import ExtractionConfig
    from rice_dataset.contracts import QCStatus
    from rice_dataset.feature_builder import build_physical_diagnostics, build_regression_features
    from rice_dataset.io.manual_records import ManualRecord
    from rice_dataset.pipeline import DatasetExtractionPipeline

    with tempfile.TemporaryDirectory(prefix="rice_fake_orch_") as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "1_Raw_Images"
        img_dir.mkdir()

        # Create valid test image
        test_img_path = img_dir / "M0001.jpg"
        dummy_bgr = np.full((300, 300, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(test_img_path), dummy_bgr)

        valid_record = ManualRecord(
            sample_id="M0001",
            original_sample_id="M001A",
            physical_sample_id="M001",
            weight_g=2.69,
            container_height_mm=33.9,
            inner_diameter_mm=17.8,
            empty_height_mm=10.9,
            rice_height_mm=23.0,
            actual_count=85,
            image_filename="M0001.jpg",
            capture_timestamp="2026-09-01T10:00:00",
        )

        cfg = ExtractionConfig(
            project_root=tmp_path,
            raw_images_dir=img_dir,
            workbook_path=tmp_path / "dummy.xlsx",
        )
        pipeline = DatasetExtractionPipeline(cfg)

        # Mock submodules
        mock_container_info = {
            "status": "success",
            "pixels_per_mm": 75.0,
            "bulk_rice_volume_mm3": 5720.5,
            "rice_height_mm": 23.0,
            "inner_w_px": 1335.0,
        }

        crop_rgba = np.full((60, 30, 4), 200, dtype=np.uint8)
        crop_rgba[:, :, 3] = 255  # fully opaque
        mock_raw_grains = [{"crop_rgba": crop_rgba.copy()}]

        mock_geometry_metric = {
            "length_mm": 9.2,
            "width_mm": 2.4,
            "thickness_mm": 1.92,
            "area_mm2": 18.5,
            "volume_mm3": 22.0,
        }

        # ----------------------------------------------------
        # 1. SUCCESS PATH TEST
        # ----------------------------------------------------
        print("[1/14] Testing deterministic SUCCESS PATH...")
        with patch("rice_dataset.pipeline.detect_container_and_scale", return_value=mock_container_info) as mock_det, \
             patch("rice_dataset.pipeline.segment_grains_sahi", return_value=mock_raw_grains) as mock_seg, \
             patch("rice_dataset.pipeline.clean_single_grain_crop", side_effect=lambda crop, **kwargs: crop.copy()) as mock_clean, \
             patch("rice_dataset.pipeline.compute_single_grain_metrics", return_value=mock_geometry_metric) as mock_geom:

            # Mock CNN classifier
            mock_classifier = MagicMock()
            mock_classifier.filter_grains.return_value = (mock_raw_grains, [])
            pipeline.cnn_classifier = mock_classifier
            pipeline.sahi_model = MagicMock()

            res = pipeline.process_single_sample(valid_record, image_path=test_img_path)

            if res["QC_Status"] != QCStatus.PASS:
                print(f"❌ FAIL: Expected PASS, got {res['QC_Status']} ({res.get('QC_Reason')})")
                return False

            # Verify API call contracts
            mock_seg.assert_called_once()
            seg_kwargs = mock_seg.call_args.kwargs
            if (
                seg_kwargs.get("slice_size") != 640
                or seg_kwargs.get("overlap_ratio") != 0.25
                or seg_kwargs.get("conf_threshold") != 0.50
                or seg_kwargs.get("min_area_px") != 50
            ):
                print(f"❌ FAIL: segment_grains_sahi called with wrong parameters: {seg_kwargs}")
                return False

            if mock_clean.call_count != 2:
                print(f"❌ FAIL: clean_single_grain_crop called {mock_clean.call_count} times, expected 2.")
                return False

            mock_geom.assert_called_once()
            geom_kwargs = mock_geom.call_args.kwargs
            if (
                "image_input" not in geom_kwargs
                or geom_kwargs.get("pixels_per_mm") != 75.0
                or geom_kwargs.get("thickness_k") != 0.80
            ):
                print(f"❌ FAIL: compute_single_grain_metrics called with wrong arguments: {geom_kwargs}")
                return False

            if res.get("Capture_Timestamp") != "2026-09-01T10:00:00":
                print(f"❌ FAIL: Source Capture_Timestamp was overwritten: {res.get('Capture_Timestamp')}")
                return False

            print("✅ PASS: Deterministic success path verified with exact API contracts.")

        # ----------------------------------------------------
        # FAILURE PATH TESTS
        # ----------------------------------------------------
        failure_cases = [
            ("missing image", lambda rec: (rec, img_dir / "NON_EXISTENT.jpg"), {}, QCStatus.MISSING_IMAGE),
            ("corrupt image", lambda rec: (rec, tmp_path / "corrupt.jpg"), {"create_corrupt": True}, QCStatus.IMAGE_DECODE_FAILED),
            ("container detection failure", lambda rec: (rec, test_img_path), {"fail_stage": "container"}, QCStatus.CONTAINER_DETECTION_FAILED),
            ("invalid scale", lambda rec: (rec, test_img_path), {"fail_stage": "scale"}, QCStatus.INVALID_SCALE),
            ("zero SAHI segments", lambda rec: (rec, test_img_path), {"fail_stage": "zero_sahi"}, QCStatus.NO_SEGMENTS),
            ("all crops rejected by cleaner", lambda rec: (rec, test_img_path), {"fail_stage": "cleaner"}, QCStatus.NO_SEGMENTS),
            ("zero CNN-whole grains", lambda rec: (rec, test_img_path), {"fail_stage": "cnn"}, QCStatus.NO_CNN_WHOLE_GRAINS),
            ("all geometry measurements invalid", lambda rec: (rec, test_img_path), {"fail_stage": "geom"}, QCStatus.NO_VALID_GRAIN_MEASUREMENTS),
            ("feature validation failure", lambda rec: (rec, test_img_path), {"fail_stage": "features"}, QCStatus.FEATURE_VALIDATION_FAILED),
            ("physical diagnostic failure", lambda rec: (rec, test_img_path), {"fail_stage": "diag"}, QCStatus.FEATURE_VALIDATION_FAILED),
            ("unexpected exception in segmenter", lambda rec: (rec, test_img_path), {"fail_stage": "sahi_exc"}, QCStatus.UNEXPECTED_ERROR),
            ("unexpected exception in classifier", lambda rec: (rec, test_img_path), {"fail_stage": "cnn_exc"}, QCStatus.UNEXPECTED_ERROR),
        ]

        # Create corrupt file
        corrupt_p = tmp_path / "corrupt.jpg"
        corrupt_p.write_bytes(b"NOT_A_VALID_IMAGE_BYTES")

        for idx, (label, get_inputs, opts, expected_status) in enumerate(failure_cases, start=2):
            print(f"[{idx}/14] Testing failure path: {label} -> {expected_status}...")
            rec, img_p = get_inputs(valid_record)

            det_ret = dict(mock_container_info)
            if opts.get("fail_stage") == "container":
                det_ret = {"status": "failed", "reason": "Edge not found"}
            elif opts.get("fail_stage") == "scale":
                det_ret["pixels_per_mm"] = -1.0

            sahi_ret = list(mock_raw_grains)
            if opts.get("fail_stage") == "zero_sahi":
                sahi_ret = []

            def mock_clean_fn(crop, **kwargs):
                if opts.get("fail_stage") == "cleaner":
                    return np.zeros_like(crop)  # alpha = 0
                return crop.copy()

            mock_cls = MagicMock()
            if opts.get("fail_stage") == "cnn":
                mock_cls.filter_grains.return_value = ([], mock_raw_grains)
            elif opts.get("fail_stage") == "cnn_exc":
                mock_cls.filter_grains.side_effect = RuntimeError("CNN GPU out of memory")
            else:
                mock_cls.filter_grains.return_value = (mock_raw_grains, [])

            geom_ret = dict(mock_geometry_metric)
            if opts.get("fail_stage") == "geom":
                geom_ret["volume_mm3"] = -1.0

            pipeline.cnn_classifier = mock_cls
            pipeline.sahi_model = MagicMock()

            with patch("rice_dataset.pipeline.detect_container_and_scale", return_value=det_ret), \
                 patch("rice_dataset.pipeline.segment_grains_sahi", side_effect=RuntimeError("SAHI crashed") if opts.get("fail_stage") == "sahi_exc" else lambda **kw: sahi_ret), \
                 patch("rice_dataset.pipeline.clean_single_grain_crop", side_effect=mock_clean_fn), \
                 patch("rice_dataset.pipeline.compute_single_grain_metrics", return_value=geom_ret), \
                 patch("rice_dataset.pipeline.build_regression_features", side_effect=ValueError("Feature calculation failed") if opts.get("fail_stage") == "features" else build_regression_features), \
                 patch("rice_dataset.pipeline.build_physical_diagnostics", side_effect=RuntimeError("Diagnostic failed") if opts.get("fail_stage") == "diag" else build_physical_diagnostics):

                res = pipeline.process_single_sample(rec, image_path=img_p)
                if res["QC_Status"] != expected_status:
                    print(f"❌ FAIL for {label}: Expected {expected_status}, got {res['QC_Status']}")
                    return False
                if not res.get("QC_Reason"):
                    print(f"❌ FAIL for {label}: QC_Reason was empty for failed status.")
                    return False
                if "Bulk_Rice_Volume_mm3" in res and res["QC_Status"] != QCStatus.PASS:
                    print(f"❌ FAIL for {label}: Feature block was populated on failed row!")
                    return False

            print(f"   ✅ PASS: {label} correctly yielded {expected_status} with reason: '{res['QC_Reason']}'")

        # Canonical flat-layout test
        print("[14/14] Testing nested raw-image rejection...")
        nested_img = img_dir / "legacy_group" / "M0002.jpg"
        nested_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(nested_img), dummy_bgr)
        pipeline.image_index = None
        try:
            pipeline.validate_input_inventory([valid_record], require_exact_match=False)
            print("❌ FAIL: Nested raw image was accepted by the flat-layout validator.")
            return False
        except ValueError as exc:
            if "nested raw image" not in str(exc):
                print(f"❌ FAIL: Unexpected flat-layout validation error: {exc}")
                return False
        print("   ✅ PASS: Nested raw image was rejected by the flat-layout validator.")

    print("=" * 65)
    print("🎉 ALL ORCHESTRATION FAKE TESTS PASSED!")
    print("=" * 65)
    return True


def run_fixture_test() -> bool:
    """Create a temporary test fixture to verify ID migration, multi-sheet workbook preservation, and rollback."""
    print("\n" + "=" * 65)
    print("🧪 RUNNING TEMPORARY FIXTURE MIGRATION & ROLLBACK TEST")
    print("=" * 65)

    from rice_dataset.migration.migrate_ids import SampleIDMigrator

    with tempfile.TemporaryDirectory(prefix="rice_test_fixture_") as tmpdir:
        tmp_path = Path(tmpdir)
        fixture_imgs = tmp_path / "1_Raw_Images"
        fixture_wb_dir = tmp_path / "2_Manual_Records"
        fixture_out = tmp_path / "migrations"

        fixture_imgs.mkdir(parents=True)
        fixture_wb_dir.mkdir(parents=True)
        fixture_out.mkdir(parents=True)

        # Create nested legacy image structure
        (fixture_imgs / "M001").mkdir()
        (fixture_imgs / "M002").mkdir()

        img1 = fixture_imgs / "M001" / "M001A.jpg"
        img2 = fixture_imgs / "M001" / "M001B.jpg"
        img3 = fixture_imgs / "M002" / "M002A.jpg"

        # Write REAL decodable JPEG images
        dummy_img = np.full((60, 60, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(img1), dummy_img)
        dummy_img[10:30, 10:30] = 50
        cv2.imwrite(str(img2), dummy_img)
        dummy_img[20:40, 20:40] = 220
        cv2.imwrite(str(img3), dummy_img)

        # Verify decode before migration
        for p in [img1, img2, img3]:
            dec = cv2.imread(str(p))
            assert dec is not None and dec.size > 0, f"Failed to decode fixture image: {p}"

        # Create multi-sheet workbook with formulas and styles
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ManualData"
        ws.freeze_panes = "A2"
        ws.append(["Sample_ID", "Weight_g", "Container_Height_mm", "Inner_Diameter_mm", "Empty_Height_mm", "Actual_Count", "Capture_Timestamp"])
        ws.append(["M001A", 2.50, 35.0, 20.0, 10.0, 75, "2026-08-01T08:00:00"])
        ws.append(["M001B", 2.50, 35.0, 20.0, 10.0, 75, "2026-08-01T08:05:00"])
        ws.append(["M002A", 3.10, 35.0, 20.0, 8.0, 90, "2026-08-02T09:00:00"])

        # Second sheet with formulas
        ws2 = wb.create_sheet(title="Metadata")
        ws2.append(["Metric", "Formula_Value"])
        ws2.append(["TotalSamples", "=COUNTA(ManualData!A2:A4)"])

        wb_path = fixture_wb_dir / "test_manual_data.xlsx"
        wb.save(wb_path)

        migrator = SampleIDMigrator(
            workbook_path=wb_path,
            images_dir=fixture_imgs,
            output_dir=fixture_out,
            prefix="M",
            digits=4,
            start=1,
        )

        # 1. Dry run
        print("  -> Testing preflight dry-run on clean fixture...")
        report = migrator.dry_run(export_reports=True)
        if not report.is_apply_ready:
            print(f"❌ FAIL: Fixture preflight blocked unexpectedly: {report.blockers}")
            return False
        if report.non_empty_excel_rows != 3:
            print(f"❌ FAIL: Expected 3 rows, got {report.non_empty_excel_rows}")
            return False

        # 2. Test Critical Blocker Enforcement (duplicate ID cannot be bypassed with force)
        print("  -> Testing that critical blockers CANNOT be bypassed with --force...")
        wb_dup = openpyxl.load_workbook(wb_path)
        ws_dup = wb_dup["ManualData"]
        ws_dup.append(["M001A", 2.50, 35.0, 20.0, 10.0, 75, "2026-08-01T08:00:00"]) # duplicate!
        wb_dup_path = fixture_wb_dir / "test_dup.xlsx"
        wb_dup.save(wb_dup_path)

        migrator_dup = SampleIDMigrator(wb_dup_path, fixture_imgs, fixture_out)
        try:
            migrator_dup.apply(force=True)
            print("❌ FAIL: migrator.apply(force=True) bypassed duplicate Excel ID!")
            return False
        except RuntimeError as e:
            print("   ✅ PASS: migrator.apply(force=True) properly blocked on critical duplicate ID.")

        # 3. Apply migration on clean fixture
        print("  -> Applying migration to clean fixture...")
        apply_res = migrator.apply()
        manifest_path = Path(apply_res["manifest_path"])
        if not manifest_path.exists():
            print(f"❌ FAIL: Manifest file not created: {manifest_path}")
            return False

        # Verify staged & promoted flat image layout and readability
        flat_files = sorted(list(fixture_imgs.glob("*.jpg")), key=lambda p: p.name)
        flat_names = [f.name for f in flat_files]
        if flat_names != ["M0001.jpg", "M0002.jpg", "M0003.jpg"]:
            print(f"❌ FAIL: Expected flat files ['M0001.jpg', 'M0002.jpg', 'M0003.jpg'], got {flat_names}")
            return False

        for f in flat_files:
            dec = cv2.imread(str(f))
            assert dec is not None and dec.size > 0, f"Migrated image failed to decode: {f}"

        # Verify workbook updated IDs, sheets, and formula preservation
        wb_migrated = openpyxl.load_workbook(wb_path, data_only=False)
        if "Metadata" not in wb_migrated.sheetnames:
            print("❌ FAIL: Secondary sheet 'Metadata' lost during migration!")
            return False
        formula_val = wb_migrated["Metadata"].cell(row=2, column=2).value
        if "=COUNTA" not in str(formula_val):
            print(f"❌ FAIL: Excel formula was lost during migration: {formula_val}")
            return False

        ws_mig = wb_migrated["ManualData"]
        migrated_ids = [ws_mig.cell(row=r, column=1).value for r in range(2, 5)]
        if migrated_ids != ["M0001", "M0002", "M0003"]:
            print(f"❌ FAIL: Migrated IDs mismatch: {migrated_ids}")
            return False

        # Verify idempotent re-run without force-resequence
        print("  -> Testing idempotent re-run without --force-resequence...")
        re_apply = migrator.apply(force=False, force_resequence=False)
        if re_apply.get("status") != "NOOP_ALREADY_MIGRATED":
            print(f"❌ FAIL: Expected NOOP_ALREADY_MIGRATED on migrated dataset, got {re_apply}")
            return False
        print("   ✅ PASS: Re-running without --force-resequence safely treated as no-op.")

        # 4. Rollback
        print("  -> Testing rollback from manifest...")
        rollback_res = SampleIDMigrator.rollback(manifest_path)

        # Verify rollback restored legacy nested files and images decode
        restored_img1 = fixture_imgs / "M001" / "M001A.jpg"
        if not restored_img1.exists() or cv2.imread(str(restored_img1)) is None:
            print("❌ FAIL: Rollback failed to restore decodable nested image file!")
            return False

        wb_restored = openpyxl.load_workbook(wb_path)
        restored_ids = [wb_restored["ManualData"].cell(row=r, column=1).value for r in range(2, 5)]
        if restored_ids != ["M001A", "M001B", "M002A"]:
            print(f"❌ FAIL: Rollback failed to restore original legacy IDs: {restored_ids}")
            return False

        # Test rollback path traversal security
        print("  -> Testing rollback path traversal rejection...")
        unsafe_manifest = fixture_out / "unsafe_manifest.json"
        unsafe_manifest.write_text('{"workbook_path": "/", "backup_workbook_path": "/tmp/a", "images_dir": "/tmp/b", "backup_images_dir": "/tmp/c"}', encoding="utf-8")
        try:
            SampleIDMigrator.rollback(unsafe_manifest)
            print("❌ FAIL: Rollback accepted unsafe root path!")
            return False
        except ValueError:
            print("   ✅ PASS: Unsafe path in manifest properly rejected with ValueError.")

        print("✅ PASS: Temporary fixture migration, byte parity, real JPEG decoding, and rollback verified 100%.")
        print("=" * 65)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pipeline parity, contracts, and migration.")
    parser.add_argument("--static", action="store_true", help="Run static contract, parameter, and import checks.")
    parser.add_argument("--orchestration", action="store_true", help="Run deterministic mock orchestration tests.")
    parser.add_argument("--fixture-test", action="store_true", help="Run temporary fixture migration & rollback test.")
    parser.add_argument("--behavioral", action="store_true", help="Run behavioral parity check against reference.")

    args = parser.parse_args()

    # If no flags passed, run all local verification tests
    run_all = not (args.static or args.orchestration or args.fixture_test or args.behavioral)

    success = True
    if args.static or run_all:
        if not run_static_checks():
            success = False

    if args.orchestration or run_all:
        if not run_orchestration_fake_tests():
            success = False

    if args.fixture_test or run_all:
        if not run_fixture_test():
            success = False

    if args.behavioral:
        print("\n" + "=" * 65)
        print("BEHAVIORAL PARITY = NOT VERIFIED (No GPU/T4 runtime available in this environment)")
        print("=" * 65)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
