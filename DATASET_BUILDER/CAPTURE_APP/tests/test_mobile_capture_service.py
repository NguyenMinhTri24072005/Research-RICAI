from __future__ import annotations

import io
import queue
import sqlite3
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
            self.assertNotIn("capture_batch", pending[0])
            self.assertNotIn("device_id", pending[0])
            self.assertIn("+", pending[0]["captured_at"])

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
            self.assertEqual(first.record["Capture_Timestamp"], second.record["Capture_Timestamp"])

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

    def test_opens_existing_database_with_deprecated_metadata_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL UNIQUE,
                        request_id TEXT UNIQUE, image_path TEXT NOT NULL, source_type TEXT NOT NULL,
                        node_id TEXT, weight_g REAL NOT NULL, container_height_mm REAL NOT NULL,
                        inner_diameter_mm REAL NOT NULL, empty_height_mm REAL NOT NULL,
                        rice_height_mm REAL NOT NULL, actual_count INTEGER NOT NULL,
                        capture_batch TEXT, device_id TEXT,
                        captured_at TEXT, created_at TEXT NOT NULL,
                        excel_status TEXT NOT NULL DEFAULT 'pending', extra_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO samples (sample_id, image_path, source_type, weight_g, container_height_mm, inner_diameter_mm, empty_height_mm, rice_height_mm, actual_count, capture_batch, device_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("M001", "legacy.jpg", "desktop", 100, 80, 50, 10, 70, 300, "OLD_BATCH", "OLD_PHONE", "2026-09-01T10:00:00+07:00"),
                )
            connection.close()

            coordinator = CaptureCoordinator()
            coordinator.configure(image_dir=Path(temp_dir) / "images", database_path=path, sample_number=2, sample_digits=3)
            pending = coordinator.database.pending_excel_records()
            self.assertEqual(pending[0]["sample_id"], "M001")
            self.assertEqual(pending[0]["capture_batch"], "OLD_BATCH")
            self.assertEqual(pending[0]["device_id"], "OLD_PHONE")


if __name__ == "__main__":
    unittest.main()
