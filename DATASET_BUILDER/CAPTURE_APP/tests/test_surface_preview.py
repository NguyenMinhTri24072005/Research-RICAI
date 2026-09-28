import io
import queue
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
sys.path.insert(0, str(APP / 'src'))
from rice_capture.server.app import create_mobile_app
from rice_capture.services.surface_preview import preview_jpeg
from rice_capture.desktop.app import DatasetCaptureApp

VALUES = {'Inner_Diameter_mm': '22.8', 'Container_Height_mm': '37.2', 'Empty_Height_mm': '9.7'}


def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (640,480), (150,150,150)).save(stream, 'JPEG')
    return stream.getvalue()


def test_vendored_detector_matches_accepted_source():
    if not (ROOT / 'CODE/modules/container_detector.py').exists():
        pytest.skip('Parity check requires the full research repository')
    accepted = (ROOT / 'CODE/modules/container_detector.py').read_bytes()
    assert (APP / 'src/rice_capture/vision/container_detector.py').read_bytes() == accepted
    assert (ROOT / 'AI_SERVICES/src/rice_ai/vision/container_detector.py').read_bytes() == accepted


def test_real_glass_preview_without_modifying_input():
    path = ROOT / 'CODE/detection_test_images/M0293.jpg'
    if not path.exists():
        pytest.skip('Local real-image fixture not shipped with standalone capture app')
    payload = path.read_bytes()
    result = preview_jpeg(payload, VALUES)
    assert result['status'] == 'ok'
    assert result['detect_type'] == 'adaptive_surface_envelope'
    assert result['pixels_per_mm'] > 0
    assert result['overlay'].startswith('data:image/jpeg;base64,')
    assert result['crop'].startswith('data:image/jpeg;base64,')
    assert path.read_bytes() == payload


@pytest.mark.parametrize('value', ['nan', 'inf', '0', '-1', ''])
def test_invalid_diameter_rejected(value):
    with pytest.raises(ValueError):
        preview_jpeg(jpeg(), {**VALUES, 'Inner_Diameter_mm': value})


def test_no_surface_has_no_invented_scale():
    result = preview_jpeg(jpeg(), VALUES)
    assert result['status'] == 'failed'
    assert 'pixels_per_mm' not in result


def test_preview_endpoint_auth_and_no_storage_side_effects():
    coordinator = Mock()
    app = create_mobile_app(coordinator, 'test-token', APP / 'src/rice_capture/web')
    client = TestClient(app)
    payload = jpeg()
    args = dict(data=VALUES, files={'image': ('image.jpg', payload, 'image/jpeg')})
    assert client.post('/api/surface-preview', **args).status_code == 401
    response = client.post('/api/surface-preview', headers={'X-Capture-Token':'test-token'}, **args)
    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert coordinator.mock_calls == []
    invalid = client.post('/api/surface-preview', headers={'X-Capture-Token':'test-token'},
                          data=VALUES, files={'image':('bad.jpg',b'not-an-image','image/jpeg')})
    assert invalid.status_code == 400


@pytest.mark.parametrize('stale', [False, True])
def test_desktop_preview_runs_off_ui_thread_and_discards_stale_results(stale):
    app = DatasetCaptureApp.__new__(DatasetCaptureApp)
    app.captured_frame = np.zeros((20,20,3), np.uint8)
    original = app.captured_frame
    app.surface_preview_busy = False
    app.surface_button = Mock()
    app.manual_vars = {k: Mock(get=Mock(return_value=v)) for k,v in VALUES.items()}
    app.mobile_events = queue.Queue()
    app.root = Mock()
    app._set_status = Mock()
    app._show_surface_preview = Mock()
    result = {'status':'ok'}
    with patch('rice_capture.services.surface_preview.preview_frame', return_value=result):
        app.check_surface()
        event = app.mobile_events.get(timeout=3)
    assert app.captured_frame is original
    if stale:
        app.captured_frame = original.copy()
    app.mobile_events.put(event)
    app._poll_mobile_events()
    assert app._show_surface_preview.call_count == (0 if stale else 1)
    assert app.surface_preview_busy is False
