"""Cache tests — TTL eviction and SQLite persistence."""

from __future__ import annotations

import time

from gaffer.cache.memory_cache import TTLMemoryCache
from gaffer.cache.sqlite_cache import SqliteCache


class TestTTLMemoryCache:
    def test_round_trip(self):
        cache = TTLMemoryCache(maxsize=4)
        cache.set("k", {"v": 1}, ttl_seconds=10)
        assert cache.get("k") == {"v": 1}

    def test_miss_returns_none(self):
        assert TTLMemoryCache().get("nope") is None

    def test_expiry(self):
        cache = TTLMemoryCache()
        cache.set("k", "v", ttl_seconds=0)
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_lru_eviction(self):
        cache = TTLMemoryCache(maxsize=2)
        cache.set("a", 1, ttl_seconds=60)
        cache.set("b", 2, ttl_seconds=60)
        cache.get("a")  # touch a → b is now LRU
        cache.set("c", 3, ttl_seconds=60)  # should evict b
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3

    def test_clear(self):
        cache = TTLMemoryCache()
        cache.set("k", "v", ttl_seconds=10)
        cache.clear()
        assert cache.get("k") is None


class TestSqliteCache:
    def test_round_trip(self, tmp_path):
        cache = SqliteCache(tmp_path / "c.sqlite")
        cache.set("k", {"v": 1, "list": [1, 2, 3]}, ttl_seconds=10)
        assert cache.get("k") == {"v": 1, "list": [1, 2, 3]}

    def test_expiry(self, tmp_path):
        cache = SqliteCache(tmp_path / "c.sqlite")
        cache.set("k", "v", ttl_seconds=0)
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_persistence(self, tmp_path):
        path = tmp_path / "c.sqlite"
        SqliteCache(path).set("k", "value", ttl_seconds=60)
        assert SqliteCache(path).get("k") == "value"

    def test_delete(self, tmp_path):
        cache = SqliteCache(tmp_path / "c.sqlite")
        cache.set("k", "v", ttl_seconds=60)
        cache.delete("k")
        assert cache.get("k") is None
