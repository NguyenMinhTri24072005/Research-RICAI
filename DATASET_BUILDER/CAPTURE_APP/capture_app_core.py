"""Compatibility exports for existing scripts and tests."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rice_capture.core import MANUAL_HEADERS, NamingConfig, make_manual_record, parse_number
from rice_capture.storage import SettingsStore, WorkbookStore


def naming_as_dict(config: NamingConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "MANUAL_HEADERS",
    "NamingConfig",
    "SettingsStore",
    "WorkbookStore",
    "make_manual_record",
    "naming_as_dict",
    "parse_number",
]
