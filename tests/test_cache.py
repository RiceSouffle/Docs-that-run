"""TTLCache: hit/miss, LRU eviction, TTL expiry, disabled mode."""

import time

from docsthatrun.cache import TTLCache


def test_hit_and_miss_and_stats():
    c = TTLCache(maxsize=8, ttl=100)
    assert c.get("k") is None            # miss
    c.set("k", {"v": 1})
    assert c.get("k") == {"v": 1}        # hit
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["size"] == 1


def test_lru_eviction():
    c = TTLCache(maxsize=2, ttl=100)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")            # touch a -> b is now least-recently-used
    c.set("c", 3)         # evicts b
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_ttl_expiry():
    c = TTLCache(maxsize=8, ttl=0.05)
    c.set("k", 1)
    assert c.get("k") == 1
    time.sleep(0.08)
    assert c.get("k") is None


def test_disabled_when_maxsize_zero():
    c = TTLCache(maxsize=0, ttl=100)
    assert not c.enabled
    c.set("k", 1)
    assert c.get("k") is None
