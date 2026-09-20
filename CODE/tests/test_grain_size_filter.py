import unittest
import numpy as np
from CODE.modules.grain_size_filter import filter_grains_by_size

class TestGrainSizeFilter(unittest.TestCase):
    def test_disabled(self):
        measurements = [{"area_mm2": 10}, {"area_mm2": 20}]
        res = filter_grains_by_size(measurements, enabled=False)
        self.assertEqual(len(res["kept"]), 2)
        self.assertEqual(len(res["rejected"]), 0)
        self.assertEqual(res["stats"]["status"], "disabled")
        
    def test_empty_input(self):
        res = filter_grains_by_size([], min_samples=2)
        self.assertEqual(res["kept"], [])
        self.assertEqual(res["stats"]["status"], "empty_input")
        
    def test_insufficient_samples(self):
        measurements = [{"area_mm2": 10}, {"area_mm2": 20}]
        res = filter_grains_by_size(measurements, min_samples=5)
        self.assertEqual(len(res["kept"]), 2)
        self.assertEqual(res["stats"]["status"], "insufficient_samples")
        
    def test_invalid_areas(self):
        measurements = [{"area_mm2": None}, {"area_mm2": np.nan}]
        res = filter_grains_by_size(measurements, min_samples=1)
        self.assertEqual(len(res["kept"]), 2)
        self.assertEqual(res["stats"]["status"], "invalid_areas")
        
    def test_degenerate_iqr(self):
        measurements = [{"area_mm2": 10} for _ in range(10)]
        res = filter_grains_by_size(measurements, min_samples=5)
        self.assertEqual(len(res["kept"]), 10)
        self.assertEqual(res["stats"]["status"], "degenerate_iqr")
        self.assertEqual(res["stats"]["iqr"], 0.0)
        
    def test_filtering(self):
        # 10 values, one is a clear outlier
        measurements = [{"area_mm2": val, "id": i} for i, val in enumerate([10, 11, 10, 12, 10.5, 9.5, 11.5, 1, 10.2, 11.1])]
        # Q1 ~ 10, Q3 ~ 11.1, IQR ~ 1.1, lower bound ~ 10 - 1.5*1.1 = 8.35
        # The '1' should be filtered out
        res = filter_grains_by_size(measurements, k=1.5, min_samples=8)
        self.assertEqual(res["stats"]["status"], "filtered")
        self.assertEqual(len(res["kept"]), 9)
        self.assertEqual(len(res["rejected"]), 1)
        self.assertEqual(res["rejected"][0]["id"], 7)
        
    def test_non_positive_lower_bound(self):
        measurements = [{"area_mm2": val} for val in [1, 5, 10, 1, 10, 15, 1, 12, 10, 1]]
        # Spread is large, Q1=1, Q3=10, IQR=9. Lower bound = 1 - 1.5*9 = -12.5 <= 0
        res = filter_grains_by_size(measurements, k=1.5, min_samples=8)
        self.assertEqual(res["stats"]["status"], "non_positive_lower_bound")
        self.assertEqual(len(res["kept"]), 10)
        
if __name__ == "__main__":
    unittest.main()
