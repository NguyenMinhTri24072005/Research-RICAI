from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_app_core import NamingConfig, WorkbookStore, make_manual_record


def timestamp() -> dict[str, str]:
    return {"Capture_Timestamp": "2026-09-16T10:30:00+07:00"}


class NamingTests(unittest.TestCase):
    def test_default_flat_sequence(self):
        config = NamingConfig()
        self.assertEqual(config.sample_code, "M001")
        self.assertEqual(config.sample_id, "M001")
        config.next_id()
        self.assertEqual(config.sample_id, "M002")

    def test_variable_width_id(self):
        config = NamingConfig(
            prefix="M",
            sample_number=12,
            sample_digits=4,
        )
        self.assertEqual(config.sample_code, "M0012")
        self.assertEqual(config.sample_id, "M0012")

    def test_rejects_number_that_exceeds_selected_width(self):
        with self.assertRaises(ValueError):
            _ = NamingConfig(sample_number=1000, sample_digits=3).sample_id


class ManualRecordTests(unittest.TestCase):
    def test_calculates_rice_height_as_number(self):
        record = make_manual_record(
            "m001",
            {
                "Weight_g": "123,5",
                "Container_Height_mm": "100",
                "Inner_Diameter_mm": "60",
                "Empty_Height_mm": "12.5",
                "Actual_Count": "450",
                **timestamp(),
            },
        )
        self.assertEqual(record["Sample_ID"], "M001")
        self.assertEqual(record["Rice_Height_mm"], 87.5)
        self.assertIsInstance(record["Rice_Height_mm"], float)

    def test_rejects_impossible_empty_height(self):
        with self.assertRaises(ValueError):
            make_manual_record(
                "M001",
                {
                    "Weight_g": 100,
                    "Container_Height_mm": 80,
                    "Inner_Diameter_mm": 50,
                    "Empty_Height_mm": 90,
                    "Actual_Count": 100,
                    **timestamp(),
                },
            )

    def test_rejects_timestamp_without_timezone(self):
        values = {
            "Weight_g": 100,
            "Container_Height_mm": 80,
            "Inner_Diameter_mm": 50,
            "Empty_Height_mm": 10,
            "Actual_Count": 100,
            **timestamp(),
        }
        values["Capture_Timestamp"] = "2026-09-16T10:30:00"
        with self.assertRaisesRegex(ValueError, "múi giờ"):
            make_manual_record("M001", values)


class WorkbookStoreTests(unittest.TestCase):
    def test_create_append_edit_save_reload_and_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual_data.xlsx"
            store = WorkbookStore(path)
            store.load_or_create()
            row = store.append(
                make_manual_record(
                    "M001",
                    {
                        "Weight_g": 120,
                        "Container_Height_mm": 100,
                        "Inner_Diameter_mm": 60,
                        "Empty_Height_mm": 20,
                        "Actual_Count": 500,
                        **timestamp(),
                    },
                )
            )
            self.assertEqual(row, 2)
            self.assertTrue(store.contains_sample_id("m001"))
            self.assertEqual(dict(store.rows())[2]["Capture_Timestamp"], "2026-09-16T10:30:00+07:00")
            store.save()

            loaded = WorkbookStore(path)
            loaded.load_or_create()
            loaded.update_manual_value(2, "Empty_Height_mm", "25")
            record = dict(loaded.rows())[2]
            self.assertEqual(record["Rice_Height_mm"], 75.0)
            with self.assertRaises(ValueError):
                loaded.update_manual_value(2, "Empty_Height_mm", "125")
            self.assertEqual(dict(loaded.rows())[2]["Empty_Height_mm"], 25.0)
            with self.assertRaises(ValueError):
                loaded.append(record)
            loaded.delete_rows([2])
            loaded.save()

            final = WorkbookStore(path)
            final.load_or_create()
            self.assertEqual(final.rows(), [])

    def test_legacy_workbook_keeps_existing_data_and_adds_metadata_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.xlsx"
            import openpyxl

            workbook = openpyxl.Workbook()
            sheet = workbook.active
            legacy_headers = [
                "Sample_ID", "Weight_g", "Container_Height_mm", "Inner_Diameter_mm",
                "Empty_Height_mm", "Rice_Height_mm", "Actual_Count",
            ]
            sheet.append(legacy_headers)
            sheet.append(["M001", 120, 100, 60, 20, 80, 500])
            workbook.save(path)

            store = WorkbookStore(path)
            store.load_or_create()
            self.assertNotIn("Capture_Batch", store.headers)
            self.assertNotIn("Device_ID", store.headers)
            self.assertIn("Capture_Timestamp", store.headers)
            self.assertEqual(dict(store.rows())[2]["Sample_ID"], "M001")
            self.assertEqual(dict(store.rows())[2]["Actual_Count"], 500)


if __name__ == "__main__":
    unittest.main()
