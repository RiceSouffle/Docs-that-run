"""Per-key token-bucket rate limiter — stdlib, thread-safe, in-process.

Guards the expensive endpoints (each ``/ask`` may spawn a grading subprocess).
In-process is right for a single instance; put a shared store (Redis) behind the
same ``allow()`` interface to limit across a fleet.

The bucket map is an LRU capped at ``max_keys``: a flood of distinct keys evicts
the least-recently-seen bucket in O(1) rather than growing without bound.
Evicting a throttled bucket is fail-open — the key simply resets to full
capacity next time — which is safe and no worse than unbounded growth.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Tuple


class RateLimiter:
    def __init__(self, rpm: int, burst: int, max_keys: int = 10_000):
        self._rate = rpm / 60.0  # tokens replenished per second
        self._capacity = float(max(burst, 1))
        self._enabled = rpm > 0
        self._max_keys = max(max_keys, 1)
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> Tuple[bool, float]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        if not self._enabled:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                allowed, retry = True, 0.0
                tokens -= 1.0
            else:
                allowed = False
                retry = round((1.0 - tokens) / self._rate, 2) if self._rate > 0 else 60.0
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)  # mark most-recently-seen
            if len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)  # evict least-recently-seen
            return allowed, retry
