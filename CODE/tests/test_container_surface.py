"""Geometry checks use known synthetic truth, not fitted real-image outputs."""
import sys
import json
from unittest.mock import Mock
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.container_detector import detect_container_and_scale


def scene():
    hsv = np.full((1000, 1000, 3), (0, 0, 200), dtype=np.uint8)
    # A displaced, pale reflection extends the initial seed down/right.
    cv2.ellipse(hsv, (525, 535), (220, 205), 0, 0, 360, (25, 85, 160), -1)
    cv2.ellipse(hsv, (500, 490), (150, 130), 15, 0, 360, (25, 180, 170), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detect(image, **kwargs):
    return detect_container_and_scale(image, 30, 50, 20, **kwargs)


def test_reflection_does_not_set_center_scale_or_crop():
    result = detect(scene())
    assert np.linalg.norm(np.array(result['center'])-[500, 490]) < 4
    assert abs(result['pixels_per_mm']-10) < .3
    assert result['surface_mask'][490, 500] == 255
    assert result['surface_mask'][705, 525] == 0
    assert not result['surface_isolated_bgr'][705, 525].any()
    x1,y1,x2,y2 = result['crop_bbox']
    assert np.array_equal(result['cropped_bgr'], result['surface_isolated_bgr'][y1:y2,x1:x2])
    assert result['inner_confidence'] is None
    assert result['surface_diagnostics']['metric_accuracy_validated'] is False


@pytest.mark.parametrize('rotate', [False, True])
def test_resize_and_rotation_preserve_known_geometry(rotate):
    image = cv2.resize(scene(), (600,600), interpolation=cv2.INTER_AREA)
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    result = detect(image)
    assert abs(result['pixels_per_mm']-6) < .2


def test_no_color_evidence_fails_without_inventing_scale():
    result = detect(np.full((600,600,3), 150, np.uint8), on_failure='dict')
    assert result['status'] == 'failed'
    assert result.get('pixels_per_mm') is None


def test_physical_wall_thickness_does_not_change_observed_inner_diameter():
    a = detect(scene(), wall_thickness_mm=.5)
    b = detect(scene(), wall_thickness_mm=3)
    assert a['inner_w_px'] == b['inner_w_px']
    assert a['pixels_per_mm'] == b['pixels_per_mm']


@pytest.mark.parametrize('crop', [False, True])
def test_notebook_sahi_uses_surface_and_preserves_coordinates(crop):
    notebook = Path(__file__).resolve().parents[1] / 'RICE_VISION_MAIN_PIPELINE.ipynb'
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    source = next(''.join(c['source']) for c in cells
                  if 'raw_grains = segment_grains_sahi(' in ''.join(c['source']))
    segment = Mock(return_value=[{'bbox': [2,3,6,8]}])
    env = dict(container_info={}, SURFACE_FULL_PATH='surface.png',
               USE_CROPPED_CONTAINER=crop, CROPPED_CONTAINER_PATH='crop.png',
               IMAGE_PATH='raw_with_reflections.jpg', crop_offset_x=100,
               crop_offset_y=200, sahi_yolo_model=object(),
               OUTPUT_RAW_CROPS_DIR=None, segment_grains_sahi=segment)
    exec(compile(source, str(notebook), 'exec'), env)
    assert segment.call_args.kwargs['image_path'] == ('crop.png' if crop else 'surface.png')
    assert env['raw_grains'][0]['global_bbox'] == ([102,203,106,208] if crop else [2,3,6,8])
