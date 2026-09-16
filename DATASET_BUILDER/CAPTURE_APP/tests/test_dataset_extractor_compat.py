from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "DATASET_BUILDER" / "CAPTURE_APP"))
sys.path.insert(0, str(ROOT / "CODE"))

from capture_app_core import WorkbookStore, make_manual_record
from modules.dataset_extractor import DatasetExtractor, derive_sample_group


class DatasetExtractorCompatibilityTests(unittest.TestCase):
    def test_group_derivation_supports_old_and_new_ids(self):
        self.assertEqual(derive_sample_group("M001A"), "M001")
        self.assertEqual(derive_sample_group("M0001AA"), "M0001")
        self.assertEqual(derive_sample_group("M0001_02"), "M0001")

    def test_reads_openpyxl_inline_strings_and_numeric_rice_height(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "manual_data.xlsx"
            store = WorkbookStore(workbook_path)
            store.load_or_create()
            store.append(
                make_manual_record(
                    "M0002",
                    {
                        "Weight_g": 100,
                        "Container_Height_mm": 90,
                        "Inner_Diameter_mm": 55,
                        "Empty_Height_mm": 15,
                        "Actual_Count": 400,
                    },
                )
            )
            store.update_cell(2, "Rice_Height_mm", "=C2-E2")
            store.save()

            extractor = DatasetExtractor.__new__(DatasetExtractor)
            extractor.manual_excel_path = workbook_path
            records = extractor.read_manual_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["Sample_ID"], "M0002")
            self.assertEqual(float(records[0]["Rice_Height_mm"]), 75.0)

    def test_finds_flat_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "1_Raw_Images"
            image_path = raw_dir / "M0002.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"test")

            extractor = DatasetExtractor.__new__(DatasetExtractor)
            extractor.raw_images_dir = raw_dir
            self.assertEqual(extractor.find_matching_image("m0002"), image_path)

    def test_still_finds_legacy_grouped_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "1_Raw_Images"
            image_path = raw_dir / "M001" / "M001A.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"test")

            extractor = DatasetExtractor.__new__(DatasetExtractor)
            extractor.raw_images_dir = raw_dir
            self.assertEqual(extractor.find_matching_image("m001a"), image_path)


if __name__ == "__main__":
    unittest.main()
