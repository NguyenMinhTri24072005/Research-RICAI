#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static validation and integrity tests for the Colab runtime notebook.
Verifies structure, absence of secrets/outputs, and clean single-launcher placement.
"""

import ast
import json
import re
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = TESTS_DIR.parent
NOTEBOOK_PATH = AI_SERVICES_DIR / "src" / "notebooks" / "API_Server.ipynb"
ROOT_NOTEBOOK_PATH = AI_SERVICES_DIR / "API_Server.ipynb"
REQ_COLAB_PATH = AI_SERVICES_DIR / "requirements-colab.txt"
README_NOTEBOOK_PATH = AI_SERVICES_DIR / "src" / "notebooks" / "README.md"


class TestColabNotebook(unittest.TestCase):
    def test_single_active_launcher_location(self):
        """src/notebooks/API_Server.ipynb phải tồn tại và không còn bản copy ở root AI_SERVICES."""
        self.assertTrue(NOTEBOOK_PATH.is_file(), f"Notebook đích không tồn tại: {NOTEBOOK_PATH}")
        self.assertFalse(ROOT_NOTEBOOK_PATH.exists(), f"Bản copy cũ ở root vẫn còn tồn tại: {ROOT_NOTEBOOK_PATH}")

    def test_notebook_json_and_cell_structure(self):
        """Notebook phải là JSON hợp lệ theo nbformat v4, đúng 8 cells và cấu trúc chuẩn."""
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("nbformat"), 4)
        cells = data.get("cells", [])
        self.assertEqual(len(cells), 8, f"Notebook phải có chính xác 8 cells, nhận: {len(cells)}")

        # Cell 1: Markdown instructions
        self.assertEqual(cells[0].get("cell_type"), "markdown")

        # Cells 2-8: Code
        for i in range(1, 8):
            self.assertEqual(cells[i].get("cell_type"), "code", f"Cell {i+1} phải là code cell")

    def test_notebook_is_clean_without_saved_outputs_or_counts(self):
        """Notebook giao lại phải sạch hoàn toàn: không có output, không có execution_count."""
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                outputs = cell.get("outputs", [])
                self.assertEqual(outputs, [], f"Cell {idx+1} chứa saved outputs ({len(outputs)} outputs)!")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx+1} chứa saved execution_count!")

    def test_notebook_does_not_leak_secrets(self):
        """Notebook không được chứa hardcoded token cá nhân hoặc ngrok token thật."""
        content = NOTEBOOK_PATH.read_text(encoding="utf-8")

        # Kiểm tra pattern token ngrok thật (thường bắt đầu 2... độ dài ~40-50 ký tự hex/base64)
        self.assertIsNone(re.search(r"[0-9a-zA-Z]{40,50}", content))
        self.assertNotIn("provolone-duress", content)
        self.assertNotIn("11482702653395093614", content)

    def test_code_cells_syntax_compiles(self):
        """Các code cell phải có cú pháp Python hợp lệ (compile AST thành công)."""
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                # Top-level await is valid in async notebook environments; wrap in async function to parse with ast
                wrapped_code = f"async def __test_cell_{idx}():\n" + "\n".join(
                    f"    {line}" for line in code.splitlines()
                )
                try:
                    ast.parse(wrapped_code)
                except SyntaxError as se:
                    self.fail(f"Cell {idx+1} bị lỗi cú pháp Python: {se}\nCode:\n{code}")

    def test_colab_requirements_and_readme_present(self):
        """requirements-colab.txt và README.md phải tồn tại đầy đủ."""
        self.assertTrue(REQ_COLAB_PATH.is_file(), "requirements-colab.txt thiếu!")
        req_content = REQ_COLAB_PATH.read_text(encoding="utf-8")
        self.assertIn("pyngrok", req_content)
        self.assertIn("opencv-python-headless", req_content)
        self.assertIn("httpx", req_content)

        self.assertTrue(README_NOTEBOOK_PATH.is_file(), "src/notebooks/README.md thiếu!")
        readme_content = README_NOTEBOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("API_Server.ipynb", readme_content)


if __name__ == "__main__":
    unittest.main()
