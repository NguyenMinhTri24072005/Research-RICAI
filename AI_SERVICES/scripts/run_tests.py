#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TEST RUNNER: CHẠY TOÀN BỘ BỘ KIỂM THỬ TASK 01
===============================================================================
Mục đích:
  - Tự động phát hiện và chạy toàn bộ unit test, contract test và integration test.
  - Xuất báo cáo tổng kết chi tiết vào console và file reports/task_01/test_report.json.
===============================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
AI_SERVICES_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_SERVICES_DIR.parent

if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))


def run_all_tests():
    print("=" * 80)
    print("🧪 RICE VISION AI — TASK 01 TEST SUITE RUNNER")
    print("=" * 80)
    print(f"📁 Root: {PROJECT_ROOT}")
    print(f"📂 Tests Directory: {AI_SERVICES_DIR / 'tests'}")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(AI_SERVICES_DIR / "tests"), pattern="test_*.py")

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    report_dir = PROJECT_ROOT / "reports" / "task_01"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "elapsed_seconds": round(elapsed, 2),
        "error_details": [
            {"test": str(test), "traceback": str(err)} for test, err in result.errors
        ],
        "failure_details": [
            {"test": str(test), "traceback": str(fail)} for test, fail in result.failures
        ],
        "skipped_details": [
            {"test": str(test), "reason": str(reason)} for test, reason in result.skipped
        ],
    }

    report_file = report_dir / "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ KIỂM THỬ: {result.testsRun} tests đã chạy trong {elapsed:.2f}s")
    print(f"   • Thành công : {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"   • Thất bại   : {len(result.failures)}")
    print(f"   • Lỗi        : {len(result.errors)}")
    print(f"   • Bỏ qua     : {len(result.skipped)}")
    print(f"📁 Báo cáo đã lưu: {report_file}")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
