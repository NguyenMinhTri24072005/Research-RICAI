from __future__ import annotations

import json

import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import numpy as np

from rice_ai.contracts import ContainerResult, EstimateSet, GrainAnalysis, PipelineResult
from rice_ai.pipeline.artifacts import InferenceArtifactWriter


def _result(request_id: str = "artifact-test") -> PipelineResult:
    grains = GrainAnalysis(
        whole_grains=[],
        broken_grains=[],
        chalky_grains=[],
        foreign_objects=[],
        total_detected=0,
        classified_counts={},
        volumes_px3=[],
    )
    return PipelineResult(
        request_id=request_id,
        estimates=EstimateSet(geometry_est=1, weight_est=None, regression_est=None, final=1, method_used="geometry"),
        container=ContainerResult(10.0, 20.0, 2.0, 18.0, 1000.0, 5.0, {}),
        grain_analysis=grains,
        features_31={},
        timings_ms={},
    )


def test_writer_creates_complete_request_directory(tmp_path):
    writer = InferenceArtifactWriter(tmp_path)
    manifest = writer.write(_result(), np.zeros((12, 12, 3), dtype=np.uint8), {"diam": 1.0})
    request_dir = tmp_path / "artifact-test"

    assert manifest["status"] == "completed"
    assert (request_dir / "COMPLETED").is_file()
    saved_manifest = json.loads((request_dir / "reports" / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["download_path"].endswith("/artifact-test/download")
    assert (request_dir / "input" / "original.jpg").is_file()


def test_writer_records_explicit_failure(tmp_path):
    writer = InferenceArtifactWriter(tmp_path)
    result = writer.write_failure("failed-test", RuntimeError("boom"), {"debug": True})
    assert result["status"] == "failed"
    assert (tmp_path / "failed-test" / "error.json").is_file()
    assert not (tmp_path / "failed-test" / "COMPLETED").exists()
