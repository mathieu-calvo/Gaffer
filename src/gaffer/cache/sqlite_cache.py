"""Persistent SQLite-backed cache for FPL API responses."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any


class SqliteCache:
    """Tiny key-value cache backed by SQLite; survives process restarts."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            value TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(self._DDL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0, isolation_level=None)

    def get(self, key: str) -> Any | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT expires_at, value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        expires_at, value = row
        if time.time() > expires_at:
            self.delete(key)
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        payload = json.dumps(value)
        expires_at = time.time() + ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, expires_at, value) VALUES (?, ?, ?)",
                (key, expires_at, payload),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cache")
