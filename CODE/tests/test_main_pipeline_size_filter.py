import unittest
import json
import os
import sys
import io

# Thêm CODE vào sys.path để test chạy module import được
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestMainPipelineSizeFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb_path = os.path.join(os.path.dirname(__file__), "..", "RICE_VISION_MAIN_PIPELINE.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        cls.cells = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                if "GRAIN_MEASUREMENTS" in source:
                    cls.cells["measurements"] = source
                elif "GRAIN_SIZE_FILTER" in source:
                    cls.cells["size_filter"] = source
                elif "GRAIN_PHYSICAL_STATISTICS" in source:
                    cls.cells["physical"] = source

    def setUp(self):
        # Mock environment 
        def mock_compute(mask, pixels_per_mm, label):
            # Tạo lỗi có chủ ý cho hạt thứ 3
            if mask == "error_mask":
                raise ValueError("Simulated measurement error")
            return {
                "length_mm": mask * 1.5,
                "width_mm": mask * 0.5,
                "thickness_mm": mask * 0.4,
                "area_mm2": mask * 2.0,
                "volume_mm3": mask * 3.0
            }
            
        class MockAxes:
            def hist(self, *args, **kwargs): pass
            def axvline(self, *args, **kwargs): pass
            def set_title(self, *args, **kwargs): pass
            
        class MockAxesArray:
            def __getitem__(self, idx): return MockAxes()
            
        class MockPlt:
            def show(self, *args, **kwargs): pass
            def figure(self, *args, **kwargs): pass
            def subplot(self, *args, **kwargs): pass
            def suptitle(self, *args, **kwargs): pass
            def imshow(self, *args, **kwargs): pass
            def title(self, *args, **kwargs): pass
            def axis(self, *args, **kwargs): pass
            def subplots(self, *args, **kwargs): return None, MockAxesArray()
            def legend(self, *args, **kwargs): pass
            def tight_layout(self, *args, **kwargs): pass

        sys.modules['matplotlib.pyplot'] = MockPlt()
        import matplotlib.pyplot as plt
        
        import numpy as np
        import pandas as pd
        self.env = {
            "BASE_PATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
            "pixels_per_mm": 10.0,
            "np": np,
            "pd": pd,
            "whole_grains": [
                {"confidence": 0.9, "grain_id": 1, "crop_rgba": 10},
                {"confidence": 0.9, "grain_id": 2, "crop_rgba": 11},
                {"confidence": 0.5, "grain_id": 3, "crop_rgba": "error_mask"},
                {"confidence": 0.9, "grain_id": 4, "crop_rgba": 9},
                {"confidence": 0.9, "grain_id": 5, "crop_rgba": 10},
                {"confidence": 0.9, "grain_id": 6, "crop_rgba": 12},
                {"confidence": 0.9, "grain_id": 7, "crop_rgba": 1},  # Outlier bé
                {"confidence": 0.9, "grain_id": 8, "crop_rgba": 10.5},
                {"confidence": 0.9, "grain_id": 9, "crop_rgba": 11},
                {"confidence": 0.9, "grain_id": 10, "crop_rgba": 9.5},
                {"confidence": 0.9, "grain_id": 11, "crop_rgba": 10.2},
            ],
            "compute_single_grain_metrics": mock_compute,
            "bulk_volume_mm3": 1000.0,
            "PACKING_FRACTION": 0.82,
            "OUTPUT_REPORT_DIR": "dummy_report_dir",
            "OUTPUT_DIR": "dummy_out_dir",
            "IMAGE_PATH": "dummy.jpg",
            "display": lambda x: None,
            "draw_grain_ellipse_overlay": lambda crop, metrics: None,
            "plt": plt
        }
        os.makedirs("dummy_report_dir", exist_ok=True)
        os.makedirs("dummy_out_dir", exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree("dummy_report_dir", ignore_errors=True)
        shutil.rmtree("dummy_out_dir", ignore_errors=True)

    def _exec(self, name, env):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(self.cells[name], env)
        finally:
            sys.stdout = old_stdout

    def test_filter_integration(self):
        env = dict(self.env)
        
        # 1. ĐO ĐẠC
        self._exec("measurements", env)
        self.assertEqual(len(env["whole_records"]), 10) # 1 lỗi bị bỏ qua
        self.assertEqual(len(env["raw_valid_pairs"]), 10)
        # ID của hạt lỗi là 3, nên hạt cuối (candidate index 10) có grain_id 11
        self.assertEqual(env["whole_records"][9]["grain_id"], 11)
        
        # 2. BỘ LỌC KÍCH THƯỚC (BẬT)
        env["ENABLE_SIZE_FILTER"] = True
        env["SIZE_FILTER_K"] = 1.5
        env["SIZE_FILTER_MIN_SAMPLES"] = 8
        self._exec("size_filter", env)
        
        self.assertEqual(env["filter_stats"]["status"], "filtered")
        self.assertEqual(len(env["filtered_whole_records"]), 9) # 1 outlier nhỏ bị bỏ qua
        
        # Candidate 6 (grain_id 7, area 2.0) là outlier
        rejected_ids = [r["grain_id"] for r in env["rejected_records"]]
        self.assertIn(7, rejected_ids)
        
        # 3. THỐNG KÊ VẬT LÝ
        self._exec("physical", env)
        self.assertIsNotNone(env["estimated_seed_count"])
        self.assertGreater(env["mean_whole_grain_vol"], 0)

    def test_filter_disabled_restores_raw(self):
        env = dict(self.env)
        self._exec("measurements", env)
        
        env["ENABLE_SIZE_FILTER"] = False
        self._exec("size_filter", env)
        
        self.assertEqual(env["filter_stats"]["status"], "disabled")
        self.assertEqual(len(env["filtered_whole_records"]), 10) # Không lọc outlier
        
if __name__ == "__main__":
    unittest.main()
