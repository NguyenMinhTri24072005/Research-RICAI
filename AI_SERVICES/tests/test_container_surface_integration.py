import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rice_ai.pipeline.container import analyze_container
from rice_ai.contracts import PipelineError
from rice_ai.vision.container_detector import ContainerDetectionError


def test_container_uses_accepted_strategy_and_overlay():
    overlay = np.zeros((5,5,3), np.uint8)
    with patch('rice_ai.pipeline.container.detect_container_and_scale', return_value={
            'pixels_per_mm': 40, 'overlay_bgr': overlay}) as detect:
        result = analyze_container(overlay,22.8,37.2,9.7)
    assert detect.call_args.kwargs['surface_method'] == 'adaptive'
    assert result.visual_overlay is overlay


def test_uncertain_surface_is_422_not_internal_error():
    with patch('rice_ai.pipeline.container.detect_container_and_scale',
               side_effect=ContainerDetectionError('unstable surface')):
        with pytest.raises(PipelineError) as caught:
            analyze_container(np.zeros((5,5,3),np.uint8),22.8,37.2,9.7)
    assert caught.value.status_code == 422
    assert caught.value.error_code == 'DETECTION_FAILED'
