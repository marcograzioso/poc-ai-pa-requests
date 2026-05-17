"""SQLite helpers for metadata and audit logging."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from app.utils.config import settings


def init_db() -> None:
    """Initialize local SQLite database for auditability."""
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield SQLite connection with proper cleanup."""
    conn = sqlite3.connect(settings.sqlite_path)
    try:
        yield conn
    finally:
        conn.close()


def write_audit_log(event_type: str, payload: str) -> None:
    """Persist audit event for traceability."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_logs(event_type, payload, created_at) VALUES (?, ?, ?)",
            (event_type, payload, datetime.utcnow().isoformat()),
        )
        conn.commit()


def read_latest_audit_logs(limit: int = 100) -> list[tuple[int, str, str, str]]:
    """Return latest audit logs for governance dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, event_type, payload, created_at FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()
