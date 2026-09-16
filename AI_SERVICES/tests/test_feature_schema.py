#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for feature_schema.py
"""

import math
import sys
import unittest
from pathlib import Path

# Thêm AI_SERVICES vào sys.path
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from feature_schema import (
    ALL_31_FEATURES,
    FEATURE_DEFS,
    FEATURE_SCHEMA_VERSION,
    TRAINED_HYBRID_PACKING_FRACTION,
    validate_feature_vector,
    feature_vector_to_ordered_list,
    compute_trained_hybrid_feature,
    compute_schema_hash,
)


def _make_dummy_features() -> dict:
    """Tạo vector 31 features hợp lệ phục vụ test."""
    f = {}
    for item in FEATURE_DEFS:
        if item.name == "Whole_Grains_Count":
            f[item.name] = 15.0
        elif item.name == "Uniformity_Rate_Pct":
            f[item.name] = 85.0
        elif item.name == "Weight_g":
            f[item.name] = 2.5
        elif item.min_val is not None:
            f[item.name] = item.min_val + 5.0
        else:
            f[item.name] = 10.0
    return f


class TestFeatureSchema(unittest.TestCase):
    def test_feature_count_and_version(self):
        self.assertEqual(len(ALL_31_FEATURES), 31)
        self.assertEqual(len(FEATURE_DEFS), 31)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "31v1")

    def test_feature_order_integrity(self):
        self.assertEqual(ALL_31_FEATURES[0], "Bulk_Rice_Volume_mm3")
        self.assertEqual(ALL_31_FEATURES[1], "Rice_Height_mm")
        self.assertEqual(ALL_31_FEATURES[2], "Weight_g")
        self.assertEqual(ALL_31_FEATURES[8], "Whole_Grains_Count")
        self.assertEqual(ALL_31_FEATURES[10], "Estimated_Total_Seeds_Hybrid")
        self.assertEqual(ALL_31_FEATURES[30], "Grain_Volume_mm3_Std")

    def test_valid_vector_validation(self):
        feats = _make_dummy_features()
        res = validate_feature_vector(feats, require_grains=True)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.missing_features), 0)
        self.assertEqual(len(res.non_finite_features), 0)
        self.assertEqual(len(res.out_of_domain), 0)
        self.assertEqual(res.grain_count, 15)

    def test_missing_required_feature(self):
        feats = _make_dummy_features()
        del feats["Bulk_Rice_Volume_mm3"]
        res = validate_feature_vector(feats, require_grains=True)
        self.assertFalse(res.valid)
        self.assertIn("Bulk_Rice_Volume_mm3", res.missing_features)
        self.assertEqual(res.error_code, "MISSING_FEATURE")

    def test_non_finite_feature(self):
        feats = _make_dummy_features()
        feats["Pixels_Per_mm"] = float("nan")
        res = validate_feature_vector(feats, require_grains=True)
        self.assertFalse(res.valid)
        self.assertIn("Pixels_Per_mm", res.non_finite_features)
        self.assertEqual(res.error_code, "NON_FINITE_FEATURE")

    def test_out_of_domain_negative(self):
        feats = _make_dummy_features()
        feats["Rice_Height_mm"] = -2.0
        res = validate_feature_vector(feats, require_grains=True)
        self.assertFalse(res.valid)
        self.assertEqual(res.error_code, "INVALID_INPUT")

    def test_zero_grain_count_rejection(self):
        feats = _make_dummy_features()
        feats["Whole_Grains_Count"] = 0.0
        res = validate_feature_vector(feats, require_grains=True)
        self.assertFalse(res.valid)
        self.assertTrue(any("NO_VALID_GRAINS" in w for w in res.warnings))

    def test_single_grain_zero_std_allowed(self):
        feats = _make_dummy_features()
        feats["Whole_Grains_Count"] = 1.0
        feats["Grain_Length_mm_Std"] = 0.0
        res = validate_feature_vector(feats, require_grains=True)
        self.assertTrue(res.valid)
        self.assertTrue(any("SINGLE_GRAIN_STD" in w for w in res.warnings))

    def test_ordered_list_conversion(self):
        feats = _make_dummy_features()
        lst = feature_vector_to_ordered_list(feats)
        self.assertEqual(len(lst), 31)
        self.assertEqual(lst[0], feats["Bulk_Rice_Volume_mm3"])

    def test_ordered_list_conversion_strict_rejection(self):
        feats = _make_dummy_features()
        del feats["Bulk_Rice_Volume_mm3"]
        with self.assertRaises(ValueError):
            feature_vector_to_ordered_list(feats, allow_missing=False)

    def test_ordered_list_conversion_optional_weight_allowed(self):
        feats = _make_dummy_features()
        feats["Weight_g"] = None
        lst = feature_vector_to_ordered_list(feats, allow_missing=False)
        self.assertEqual(len(lst), 31)
        self.assertEqual(lst[2], 0.0)

    def test_compute_trained_hybrid_feature_synthetic(self):
        # 8000 * 0.62 / 20 = 248.0 -> 248
        res = compute_trained_hybrid_feature(8000.0, [10.0, 20.0, 30.0])
        self.assertEqual(res, 248)

        # Edge cases: None or <= 0
        self.assertIsNone(compute_trained_hybrid_feature(None, [10.0, 20.0]))
        self.assertIsNone(compute_trained_hybrid_feature(8000.0, []))
        self.assertIsNone(compute_trained_hybrid_feature(-100.0, [10.0]))
        self.assertIsNone(compute_trained_hybrid_feature(8000.0, [0.0, -5.0]))

    def test_compute_trained_hybrid_feature_csv_parity(self):
        import csv
        csv_path = AI_SERVICES_DIR.parent / "DATASET_BUILDER" / "4_Final_Dataset" / "final_linear_regression_dataset.csv"
        if not csv_path.exists():
            self.skipTest(f"CSV file không tồn tại: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        found_rows = [r for r in rows if r.get("Image_Status") == "FOUND"]
        self.assertGreater(len(found_rows), 0)

        match_count = 0
        total_valid = 0
        for r in found_rows:
            bulk_str = r.get("Bulk_Rice_Volume_mm3", "")
            mean_vol_str = r.get("Grain_Volume_mm3_Mean", "")
            stored_str = r.get("Estimated_Total_Seeds_Hybrid", "")

            if bulk_str and mean_vol_str and stored_str:
                try:
                    bulk = float(bulk_str)
                    mean_vol = float(mean_vol_str)
                    stored_hybrid = float(stored_str)
                except ValueError:
                    continue
                if mean_vol > 0:
                    total_valid += 1
                    computed = compute_trained_hybrid_feature(bulk, [mean_vol])
                    if computed == int(round(stored_hybrid)):
                        match_count += 1

        self.assertEqual(match_count, total_valid)
        self.assertEqual(total_valid, 254)

    def test_compute_schema_hash(self):
        h = compute_schema_hash()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)
        self.assertEqual(h, "dee4b46be6aaef6cb4f696696aaa11cbcfa1e9d718deea96ca9ec5238046f5a8")


if __name__ == "__main__":
    unittest.main()
