from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class CaptureDatabase:
    """Thread-safe SQLite source of truth for captured samples."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL UNIQUE,
                    request_id TEXT UNIQUE,
                    image_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    node_id TEXT,
                    weight_g REAL NOT NULL,
                    container_height_mm REAL NOT NULL,
                    inner_diameter_mm REAL NOT NULL,
                    empty_height_mm REAL NOT NULL,
                    rice_height_mm REAL NOT NULL,
                    actual_count INTEGER NOT NULL,
                    captured_at TEXT,
                    created_at TEXT NOT NULL,
                    excel_status TEXT NOT NULL DEFAULT 'pending',
                    extra_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_samples_created_at ON samples(created_at);
                """
            )

    def contains_sample_id(self, sample_id: str) -> bool:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM samples WHERE upper(sample_id) = upper(?) LIMIT 1", (sample_id,)
            ).fetchone()
            return row is not None

    def get_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        if not request_id:
            return None
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM samples WHERE request_id = ?", (request_id,)).fetchone()
            return dict(row) if row else None

    def insert_sample(
        self,
        record: dict[str, Any],
        *,
        image_path: Path,
        source_type: str,
        node_id: str | None,
        request_id: str | None,
        captured_at: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO samples (
                    sample_id, request_id, image_path, source_type, node_id,
                    weight_g, container_height_mm, inner_diameter_mm,
                    empty_height_mm, rice_height_mm, actual_count,
                    captured_at, created_at, excel_status, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    record["Sample_ID"], request_id or None, str(image_path), source_type, node_id or None,
                    record["Weight_g"], record["Container_Height_mm"], record["Inner_Diameter_mm"],
                    record["Empty_Height_mm"], record["Rice_Height_mm"], record["Actual_Count"],
                    captured_at or None, now, json.dumps(extra or {}, ensure_ascii=False),
                ),
            )

    def mark_excel_status(self, sample_id: str, status: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE samples SET excel_status = ? WHERE upper(sample_id) = upper(?)", (status, sample_id)
            )

    def pending_excel_records(self) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM samples WHERE excel_status != 'synced' ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]
