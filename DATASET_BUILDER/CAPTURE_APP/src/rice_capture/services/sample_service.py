from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rice_capture.core.naming import NamingConfig
from rice_capture.core.validation import make_manual_record
from rice_capture.storage.database import CaptureDatabase
from rice_capture.storage.image_store import validate_jpeg_bytes, write_bytes_atomic


@dataclass(frozen=True, slots=True)
class CaptureResult:
    sample_id: str
    next_sample_id: str
    image_path: Path
    record: dict[str, Any]
    duplicate_request: bool = False


class CaptureCoordinator:
    """Owns IDs and commits samples from local cameras or mobile nodes."""

    def __init__(self, event_queue: queue.Queue[dict[str, Any]] | None = None):
        self.events = event_queue or queue.Queue()
        self._lock = threading.RLock()
        self._image_dir = Path.cwd()
        self._database: CaptureDatabase | None = None
        self._sample_number = 1
        self._sample_digits = 3
        self._existing_excel_ids: set[str] = set()
        self._manual_defaults: dict[str, str] = {}

    def configure(
        self,
        *,
        image_dir: str | Path,
        database_path: str | Path,
        sample_number: int,
        sample_digits: int,
        existing_excel_ids: set[str] | None = None,
        manual_defaults: dict[str, Any] | None = None,
    ) -> None:
        config = NamingConfig(sample_number=int(sample_number), sample_digits=int(sample_digits))
        config.validate()
        with self._lock:
            self._image_dir = Path(image_dir)
            db_path = Path(database_path)
            if self._database is None or self._database.path != db_path:
                self._database = CaptureDatabase(db_path)
            self._sample_number = config.sample_number
            self._sample_digits = config.sample_digits
            self._existing_excel_ids = {value.strip().upper() for value in (existing_excel_ids or set()) if value}
            if manual_defaults is not None:
                self._manual_defaults = {key: str(value) for key, value in manual_defaults.items()}
            self._advance_to_available_locked()

    @property
    def database(self) -> CaptureDatabase:
        if self._database is None:
            raise RuntimeError("Chưa cấu hình nơi lưu dữ liệu.")
        return self._database

    def _candidate_locked(self) -> NamingConfig:
        return NamingConfig(sample_number=self._sample_number, sample_digits=self._sample_digits)

    def _advance_to_available_locked(self) -> None:
        for _ in range(100000):
            config = self._candidate_locked()
            sample_id = config.sample_id
            if (
                sample_id not in self._existing_excel_ids
                and not (self._image_dir / f"{sample_id}.jpg").exists()
                and not self.database.contains_sample_id(sample_id)
            ):
                return
            self._sample_number += 1
        raise RuntimeError("Không tìm được mã mẫu trống trong giới hạn tìm kiếm.")

    def next_sample_id(self) -> str:
        with self._lock:
            self._advance_to_available_locked()
            return self._candidate_locked().sample_id

    def update_manual_defaults(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._manual_defaults = {key: str(value) for key, value in values.items()}

    def session_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "next_sample_id": self.next_sample_id(),
                "sample_digits": self._sample_digits,
                "manual": dict(self._manual_defaults),
                "fields": [
                    {"key": "Weight_g", "label": "Khối lượng", "unit": "g", "type": "number"},
                    {"key": "Container_Height_mm", "label": "Chiều cao ly", "unit": "mm", "type": "number"},
                    {"key": "Inner_Diameter_mm", "label": "Đường kính trong", "unit": "mm", "type": "number"},
                    {"key": "Empty_Height_mm", "label": "Chiều cao khoảng trống", "unit": "mm", "type": "number"},
                    {"key": "Actual_Count", "label": "Số hạt thực tế", "unit": "hạt", "type": "integer"},
                ],
            }

    @staticmethod
    def _record_from_database_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "Sample_ID": row["sample_id"],
            "Weight_g": row["weight_g"],
            "Container_Height_mm": row["container_height_mm"],
            "Inner_Diameter_mm": row["inner_diameter_mm"],
            "Empty_Height_mm": row["empty_height_mm"],
            "Rice_Height_mm": row["rice_height_mm"],
            "Actual_Count": row["actual_count"],
        }

    def save_jpeg(
        self,
        payload: bytes,
        values: dict[str, Any],
        *,
        source_type: str,
        node_id: str | None = None,
        request_id: str | None = None,
        captured_at: str | None = None,
    ) -> CaptureResult:
        width, height = validate_jpeg_bytes(payload)
        with self._lock:
            if request_id:
                previous = self.database.get_by_request_id(request_id)
                if previous:
                    record = self._record_from_database_row(previous)
                    return CaptureResult(
                        sample_id=record["Sample_ID"],
                        next_sample_id=self.next_sample_id(),
                        image_path=Path(previous["image_path"]),
                        record=record,
                        duplicate_request=True,
                    )

            self._advance_to_available_locked()
            config = self._candidate_locked()
            sample_id = config.sample_id
            record = make_manual_record(sample_id, values)
            image_path = self._image_dir / f"{sample_id}.jpg"
            if image_path.exists():
                raise ValueError(f"Ảnh đã tồn tại: {image_path}")

            write_bytes_atomic(image_path, payload)
            try:
                self.database.insert_sample(
                    record,
                    image_path=image_path,
                    source_type=source_type,
                    node_id=node_id,
                    request_id=request_id,
                    captured_at=captured_at or datetime.now().isoformat(timespec="seconds"),
                    extra={"width": width, "height": height},
                )
            except Exception:
                try:
                    image_path.unlink()
                except OSError:
                    pass
                raise

            self._existing_excel_ids.add(sample_id)
            self._sample_number += 1
            self._advance_to_available_locked()
            result = CaptureResult(sample_id, self._candidate_locked().sample_id, image_path, record)
            self.events.put(
                {
                    "type": "sample_saved",
                    "sample_id": result.sample_id,
                    "next_sample_id": result.next_sample_id,
                    "image_path": str(result.image_path),
                    "record": dict(result.record),
                    "source_type": source_type,
                    "node_id": node_id,
                }
            )
            return result

    def notify_node(self, event_type: str, node_id: str, label: str = "Điện thoại") -> None:
        self.events.put({"type": event_type, "node_id": node_id, "label": label})

