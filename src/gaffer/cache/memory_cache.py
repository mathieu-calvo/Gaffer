"""In-process LRU cache with TTL for FPL API responses."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class TTLMemoryCache:
    """Small thread-safe LRU cache where every entry has a fixed TTL in seconds."""

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
