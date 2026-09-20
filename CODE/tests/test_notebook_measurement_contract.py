"""Use real RGBA geometry, not a fabricated mask_binary payload."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.ellipsoid_geometry import compute_single_grain_metrics


class TestMeasurementContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads((Path(__file__).resolve().parents[1] / 'RICE_VISION_MAIN_PIPELINE.ipynb').read_text(encoding='utf-8'))
        cls.source = ''.join(next(c['source'] for c in nb['cells'] if 'GRAIN_MEASUREMENTS' in ''.join(c['source'])))

    def run_measurements(self, grains):
        env = dict(np=np, whole_grains=grains, pixels_per_mm=10.0,
                   compute_single_grain_metrics=compute_single_grain_metrics)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.source, env)
        return env

    def test_real_rgba_input_and_middle_failure_keep_records_aligned(self):
        crop = np.zeros((80, 100, 4), dtype=np.uint8)
        alpha = np.zeros((80, 100), dtype=np.uint8)
        cv2.ellipse(alpha, (50, 40), (25, 10), 0, 0, 360, 255, -1)
        crop[:, :, 3] = alpha
        grains = [{'grain_id': 10, 'crop_rgba': crop},
                  {'grain_id': 20, 'crop_rgba': np.zeros_like(crop)},
                  {'grain_id': 30, 'crop_rgba': crop.copy()}]
        env = self.run_measurements(grains)
        self.assertEqual([r['grain_id'] for r in env['whole_records']], [10, 30])
        self.assertEqual(len(env['whole_grain_metrics_list']), 2)
        self.assertEqual(len(env['whole_volumes_list']), 2)
        self.assertEqual(env['measurement_errors'][0]['grain_id'], 20)
        for pair in env['raw_valid_pairs']:
            self.assertIs(pair['grain'], grains[pair['record']['candidate_index']])
            self.assertGreater(pair['metrics']['area_mm2'], 0)
            self.assertTrue(np.isfinite(pair['metrics']['volume_mm3']))

    def test_no_valid_grains_remains_explicitly_empty(self):
        env = self.run_measurements([{'grain_id': 1, 'crop_rgba': np.zeros((10, 10, 4), dtype=np.uint8)}])
        self.assertEqual(env['whole_records'], [])
        self.assertEqual(env['whole_grain_metrics_list'], [])
        self.assertEqual(len(env['measurement_errors']), 1)


if __name__ == '__main__':
    unittest.main()
