from __future__ import annotations

import io
import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from PIL import Image

import rice_capture.desktop.app as desktop_module


def jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1024, 768), "#d7c690").save(output, "JPEG", quality=94)
    return output.getvalue()


class DesktopMobileIntegrationTests(unittest.TestCase):
    def test_mobile_event_updates_form_preview_and_excel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            original_values = (
                desktop_module.DEFAULT_IMAGE_DIR,
                desktop_module.DEFAULT_WORKBOOK,
                desktop_module.SETTINGS_FILE,
                desktop_module.LEGACY_SETTINGS_FILE,
            )
            desktop_module.DEFAULT_IMAGE_DIR = root_dir / "images"
            desktop_module.DEFAULT_WORKBOOK = root_dir / "records" / "manual_data.xlsx"
            desktop_module.SETTINGS_FILE = root_dir / "runtime" / "settings.json"
            desktop_module.LEGACY_SETTINGS_FILE = root_dir / "legacy.json"
            window = tk.Tk()
            window.withdraw()
            app = None
            try:
                app = desktop_module.DatasetCaptureApp(window)
                app.coordinator.save_jpeg(
                    jpeg_bytes(),
                    {
                        "Weight_g": "125.5",
                        "Container_Height_mm": "100",
                        "Inner_Diameter_mm": "60",
                        "Empty_Height_mm": "15",
                        "Actual_Count": "510",
                    },
                    source_type="mobile",
                    node_id="integration-phone",
                    request_id="integration-request",
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not app.workbook_store.contains_sample_id("M0001"):
                    window.update()
                    time.sleep(0.02)
                self.assertTrue(app.workbook_store.contains_sample_id("M0001"))
                self.assertEqual(app.manual_vars["Weight_g"].get(), "125.5")
                self.assertNotIn("Capture_Batch", app.manual_vars)
                self.assertNotIn("Device_ID", app.manual_vars)
                self.assertEqual(app.sample_number_var.get(), 2)
                self.assertTrue((root_dir / "images" / "M0001.jpg").exists())
                self.assertTrue((root_dir / "records" / "manual_data.xlsx").exists())
                workbook_record = dict(app.workbook_store.rows())[2]
                self.assertEqual(workbook_record["Sample_ID"], "M0001")
                self.assertNotIn("Capture_Batch", workbook_record)
                self.assertNotIn("Device_ID", workbook_record)
                self.assertIn("+", workbook_record["Capture_Timestamp"])
                self.assertEqual(app.coordinator.database.pending_excel_records(), [])
            finally:
                if app is not None:
                    app.mobile_server.stop()
                window.destroy()
                (
                    desktop_module.DEFAULT_IMAGE_DIR,
                    desktop_module.DEFAULT_WORKBOOK,
                    desktop_module.SETTINGS_FILE,
                    desktop_module.LEGACY_SETTINGS_FILE,
                ) = original_values


if __name__ == "__main__":
    unittest.main()
