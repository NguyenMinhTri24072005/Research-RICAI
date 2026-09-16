from __future__ import annotations

import io
import queue
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from PIL import Image

from rice_capture.services import CaptureCoordinator


def jpeg_bytes(width: int = 800, height: int = 600) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "#d8c597").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


class MobileCaptureServiceTests(unittest.TestCase):
    def test_saves_flat_image_database_record_and_emits_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events: queue.Queue = queue.Queue()
            coordinator = CaptureCoordinator(events)
            coordinator.configure(
                image_dir=root / "images",
                database_path=root / "records" / "capture_data.sqlite3",
                sample_number=1,
                sample_digits=3,
                manual_defaults={},
            )
            result = coordinator.save_jpeg(
                jpeg_bytes(),
                {
                    "Weight_g": "120.5",
                    "Container_Height_mm": "100",
                    "Inner_Diameter_mm": "60",
                    "Empty_Height_mm": "20",
                    "Actual_Count": "500",
                },
                source_type="mobile",
                node_id="phone-1",
                request_id="request-1",
            )
            self.assertEqual(result.sample_id, "M001")
            self.assertEqual(result.next_sample_id, "M002")
            self.assertTrue((root / "images" / "M001.jpg").exists())
            self.assertEqual(events.get_nowait()["type"], "sample_saved")
            pending = coordinator.database.pending_excel_records()
            self.assertEqual(pending[0]["sample_id"], "M001")
            self.assertEqual(pending[0]["source_type"], "mobile")

    def test_request_id_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coordinator = CaptureCoordinator()
            coordinator.configure(
                image_dir=root / "images",
                database_path=root / "capture.sqlite3",
                sample_number=1,
                sample_digits=4,
            )
            values = {
                "Weight_g": 100,
                "Container_Height_mm": 90,
                "Inner_Diameter_mm": 55,
                "Empty_Height_mm": 10,
                "Actual_Count": 400,
            }
            first = coordinator.save_jpeg(
                jpeg_bytes(), values, source_type="mobile", request_id="same-request"
            )
            second = coordinator.save_jpeg(
                jpeg_bytes(), values, source_type="mobile", request_id="same-request"
            )
            self.assertEqual(first.sample_id, "M0001")
            self.assertEqual(second.sample_id, "M0001")
            self.assertTrue(second.duplicate_request)
            self.assertEqual(len(list((root / "images").glob("*.jpg"))), 1)

    def test_skips_existing_excel_and_image_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            (images / "M002.jpg").write_bytes(jpeg_bytes())
            coordinator = CaptureCoordinator()
            coordinator.configure(
                image_dir=images,
                database_path=root / "capture.sqlite3",
                sample_number=1,
                sample_digits=3,
                existing_excel_ids={"M001"},
            )
            self.assertEqual(coordinator.next_sample_id(), "M003")


if __name__ == "__main__":
    unittest.main()
