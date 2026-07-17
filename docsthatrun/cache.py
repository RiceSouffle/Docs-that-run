"""A small thread-safe LRU cache with per-entry TTL — stdlib only.

Used to memoize answers keyed on (question, version, top_k, execute): grading a
snippet spawns a subprocess, so repeat queries are worth caching. This is the
in-process seed of the semantic cache on the ROADMAP; swap the backing store for
Redis to share it across workers.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class TTLCache:
    def __init__(self, maxsize: int, ttl: float):
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: "OrderedDict[Hashable, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the cached value, or ``None`` on miss/expiry.

        Values are never ``None`` in practice (answer dicts), so ``None`` doubles
        as the miss sentinel.
        """
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expiry, value = item
            if expiry <= now:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        if not self.enabled:
            return
        expiry = time.monotonic() + self._ttl if self._ttl > 0 else float("inf")
        with self._lock:
            self._data[key] = (expiry, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }
