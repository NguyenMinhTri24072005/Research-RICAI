"""Verify the actual notebook crop function and SAHI input/coordinate wiring."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / 'RICE_VISION_MAIN_PIPELINE.ipynb'


def notebook_crop_function():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    source = ''.join(next(c['source'] for c in cells if 'def prepare_square_container_crop' in ''.join(c['source'])))
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_square_container_crop')
    env = {'cv2': cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), env)
    return env['prepare_square_container_crop'], cells


class TestNotebookCrop(unittest.TestCase):
    def test_preserves_pixels_scale_and_coordinates_near_image_edges(self):
        prepare, _ = notebook_crop_function()
        for height, width in [(8, 5), (5, 8), (6, 6)]:
            with self.subTest(shape=(height, width)):
                crop = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
                square, offset = prepare({'cropped_bgr': crop, 'crop_offset': (0, 11)})
                side = max(height, width)
                top, left = (side - height) // 2, (side - width) // 2
                self.assertEqual(square.shape, (side, side, 3))
                np.testing.assert_array_equal(square[top:top + height, left:left + width], crop)
                self.assertEqual((left + offset[0], top + offset[1]), (0, 11))
                padded = square.copy()
                padded[top:top + height, left:left + width] = 0
                self.assertFalse(padded.any())

    def test_rejects_empty_crop(self):
        prepare, _ = notebook_crop_function()
        with self.assertRaises(ValueError):
            prepare({'cropped_bgr': np.empty((0, 3, 3)), 'crop_offset': (0, 0)})

    def test_sahi_gets_displayed_crop_and_global_boxes_include_padding_offset(self):
        prepare, cells = notebook_crop_function()
        image, offset = prepare({'cropped_bgr': np.ones((20, 16, 3), dtype=np.uint8), 'crop_offset': (0, 100)})
        segment = Mock(return_value=[{'bbox': [2, 3, 6, 8]}])
        env = dict(USE_CROPPED_CONTAINER=True, cropped_container_bgr=image,
                   crop_offset_x=offset[0], crop_offset_y=offset[1],
                   CROPPED_CONTAINER_PATH='preview.png', sahi_yolo_model=object(),
                   OUTPUT_RAW_CROPS_DIR=None, segment_grains_sahi=segment)
        source = ''.join(next(c['source'] for c in cells if 'raw_grains = segment_grains_sahi(' in ''.join(c['source'])))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, env)
        self.assertIs(segment.call_args.kwargs['image_path'], image)
        self.assertEqual(env['raw_grains'][0]['local_bbox'], [2, 3, 6, 8])
        self.assertEqual(env['raw_grains'][0]['global_bbox'], [0, 103, 4, 108])

    def test_default_is_crop(self):
        _, cells = notebook_crop_function()
        settings = ''.join(next(c['source'] for c in cells if 'USE_CROPPED_CONTAINER = ' in ''.join(c['source'])))
        assignments = [n for n in ast.parse(settings).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'USE_CROPPED_CONTAINER' for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertIs(ast.literal_eval(assignments[0].value), True)


if __name__ == '__main__':
    unittest.main()
