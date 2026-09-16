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
    validate_feature_vector,
    feature_vector_to_ordered_list,
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


if __name__ == "__main__":
    unittest.main()
