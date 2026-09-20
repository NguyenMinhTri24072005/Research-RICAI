import unittest
import json
import os
import sys
import io
import numpy as np
import pandas as pd
import joblib

class TestMainPipelineRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        nb_path = os.path.join(os.path.dirname(__file__), "..", "RICE_VISION_MAIN_PIPELINE.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        cls.cells = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                if "REGRESSION_CONFIG" in source:
                    cls.cells["config"] = source
                elif "REGRESSION_BUNDLE_LOADER" in source:
                    cls.cells["loader"] = source
                elif "REGRESSION_FEATURES" in source:
                    cls.cells["features"] = source
                elif "REGRESSION_PREDICT" in source:
                    cls.cells["predict"] = source
                elif "REGRESSION_REPORT" in source:
                    cls.cells["report"] = source

    def setUp(self):
        self.env = {
            "BASE_PATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
            "INPUT_WEIGHT_G": 3.71,
            "bulk_volume_mm3": 1000.0,
            "EMPTY_HEIGHT_MM": 5.0,
            "pixels_per_mm": 10.0,
            "INNER_DIAM_MM": 50.0,
            "CONTAINER_HEIGHT_MM": 100.0,
            "container_info": {"rice_height_mm": 95.0, "inner_w_px": 500.0},
            "whole_records": [
                {"length_mm_2a": 6.0, "width_mm_2b": 2.0, "thickness_mm_2c": 1.5, "area_mm2": 12.0, "volume_3d_mm3": 10.0},
                {"length_mm_2a": 6.2, "width_mm_2b": 2.1, "thickness_mm_2c": 1.6, "area_mm2": 12.5, "volume_3d_mm3": 20.0}
            ],
            "whole_grain_metrics_list": [
                {"length_mm": 6.0, "width_mm": 2.0, "thickness_mm": 1.5, "area_mm2": 12.0, "volume_mm3": 10.0},
                {"length_mm": 6.2, "width_mm": 2.1, "thickness_mm": 1.6, "area_mm2": 12.5, "volume_mm3": 20.0}
            ],
            "whole_grains": [1, 2],
            "cleaned_grains": [1, 2, 3],
            "estimated_seed_count": 55,
            "PACKING_FRACTION": 0.82,
            "OUTPUT_REPORT_DIR": "dummy_report_dir",
            "IMAGE_PATH": "dummy.jpg",
            "mean_whole_grain_vol": 15.0
        }
        os.makedirs("dummy_report_dir", exist_ok=True)

    def tearDown(self):
        if os.path.exists("dummy_report_dir"):
            import shutil
            shutil.rmtree("dummy_report_dir", ignore_errors=True)

    def _exec(self, name, env):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(self.cells[name], env)
        finally:
            sys.stdout = old_stdout

    def test_syntax_and_extracted_cells(self):
        self.assertIn("config", self.cells)
        self.assertIn("loader", self.cells)
        self.assertIn("features", self.cells)
        self.assertIn("predict", self.cells)
        self.assertIn("report", self.cells)
        
        for name, source in self.cells.items():
            try:
                compile(source, f"<cell_{name}>", "exec")
            except SyntaxError as e:
                self.fail(f"Cell {name} có lỗi cú pháp: {e}")

    def test_load_and_predict_ard(self):
        env = dict(self.env)
        self._exec("config", env)
        env["REGRESSION_MODEL_DIR"] = "LINEAR_REGRESSION_MODEL/models/ard"
        env["resolved_model_dir"] = os.path.abspath(os.path.join(env["BASE_PATH"], env["REGRESSION_MODEL_DIR"]))
        self._exec("loader", env)
        
        self.assertIn("regression_bundle", env)
        bundle = env["regression_bundle"]
        self.assertEqual(bundle["schema_version"], "31v1")
        self.assertEqual(bundle["model"].__class__.__name__, "ARDRegression")
        
        self._exec("features", env)
        self.assertIn("regression_features", env)
        features = env["regression_features"]
        self.assertEqual(features.shape, (1, 31))
        
        self.assertEqual(features["Estimated_Total_Seeds_Hybrid"].iloc[0], 41.0)
        self.assertEqual(env["INPUT_WEIGHT_G"], 3.71)
        self.assertEqual(bundle["hybrid_packing_fraction"], 0.62)
        
        self._exec("predict", env)
        self.assertIn("regression_result", env)
        res = env["regression_result"]
        self.assertIsInstance(res["raw_count"], float)
        self.assertIsInstance(res["final_count"], int)
        
        pipeline_path = os.path.join(env["resolved_model_dir"], "pipeline.joblib")
        if os.path.exists(pipeline_path):
            original_pipeline = joblib.load(pipeline_path)
            expected_pred = original_pipeline.predict(features)[0]
            np.testing.assert_allclose(res["raw_count"], expected_pred, rtol=1e-7, atol=1e-7)

    def test_load_and_predict_decision_tree(self):
        env = dict(self.env)
        self._exec("config", env)
        env["REGRESSION_MODEL_DIR"] = "LINEAR_REGRESSION_MODEL/models/decision_tree"
        env["resolved_model_dir"] = os.path.abspath(os.path.join(env["BASE_PATH"], env["REGRESSION_MODEL_DIR"]))
        
        if os.path.exists(env["resolved_model_dir"]):
            self._exec("loader", env)
            bundle = env["regression_bundle"]
            self.assertEqual(bundle["model"].__class__.__name__, "DecisionTreeRegressor")
            self._exec("features", env)
            self._exec("predict", env)
            res = env["regression_result"]
            
            pipeline_path = os.path.join(env["resolved_model_dir"], "pipeline.joblib")
            if os.path.exists(pipeline_path):
                original_pipeline = joblib.load(pipeline_path)
                expected_pred = original_pipeline.predict(env["regression_features"])[0]
                np.testing.assert_allclose(res["raw_count"], expected_pred, rtol=1e-7, atol=1e-7)

    def test_current_extra_trees_bundle(self):
        env = dict(self.env)
        self._exec("config", env)
        self._exec("loader", env)
        self.assertEqual(env["regression_bundle"]["model"].__class__.__name__, "ExtraTreesRegressor")
        self._exec("features", env)
        self._exec("predict", env)
        pipeline_path = os.path.join(env["resolved_model_dir"], "pipeline.joblib")
        original_pipeline = joblib.load(pipeline_path)
        expected = original_pipeline.predict(env["regression_features"])[0]
        np.testing.assert_allclose(env["regression_result"]["raw_count"], expected, rtol=1e-7, atol=1e-7)

    def test_missing_weight_raises_error(self):
        env = dict(self.env)
        self._exec("config", env)
        env["INPUT_WEIGHT_G"] = None
        self._exec("loader", env)
        
        with self.assertRaises(ValueError) as context:
            self._exec("features", env)
        self.assertIn("Thiếu dữ liệu cân nặng thật", str(context.exception))

    def test_empty_whole_grains_raises_error(self):
        env = dict(self.env)
        env["whole_records"] = []
        env["whole_grains"] = []
        self._exec("config", env)
        self._exec("loader", env)
        
        with self.assertRaises(ValueError) as context:
            self._exec("features", env)
        self.assertIn("Không tìm thấy hạt lúa nguyên", str(context.exception))

    def test_raw_measurements_and_bundle_fraction(self):
        env = dict(self.env)
        env["whole_grain_metrics_list"] = [
            {"length_mm": 6.01, "width_mm": 2.01, "thickness_mm": 1.51, "area_mm2": 12.01, "volume_mm3": 10.04},
            {"length_mm": 6.21, "width_mm": 2.11, "thickness_mm": 1.61, "area_mm2": 12.51, "volume_mm3": 20.04}
        ]
        self._exec("config", env)
        self._exec("loader", env)
        self._exec("features", env)
        self.assertEqual(env["regression_features"]["Grain_Volume_mm3_Mean"].iloc[0], 15.04)
        self.assertEqual(env["regression_features"]["Estimated_Total_Seeds_Hybrid"].iloc[0], 41.0)

if __name__ == "__main__":
    unittest.main()
