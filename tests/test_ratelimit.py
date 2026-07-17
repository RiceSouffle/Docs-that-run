"""RateLimiter: burst, denial, retry-after, per-key isolation, disabled mode."""

from docsthatrun.ratelimit import RateLimiter


def test_burst_then_deny():
    rl = RateLimiter(rpm=60, burst=3)  # capacity 3, ~1 token/sec refill
    allowed = [rl.allow("ip")[0] for _ in range(5)]
    assert allowed == [True, True, True, False, False]


def test_retry_after_positive_when_denied():
    rl = RateLimiter(rpm=60, burst=1)
    assert rl.allow("ip")[0] is True
    ok, retry = rl.allow("ip")
    assert ok is False and retry > 0


def test_disabled_allows_everything():
    rl = RateLimiter(rpm=0, burst=1)
    assert all(rl.allow("ip")[0] for _ in range(100))


def test_keys_are_independent():
    rl = RateLimiter(rpm=60, burst=1)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True      # different key, own bucket
    assert rl.allow("a")[0] is False     # a is now exhausted


def test_bucket_map_is_bounded_under_key_flood():
    # A flood of distinct keys must not grow the map without bound — the LRU cap
    # evicts the least-recently-seen bucket in O(1).
    rl = RateLimiter(rpm=6000, burst=100, max_keys=50)
    for i in range(500):
        rl.allow(f"ip-{i}")
    assert len(rl._buckets) <= 50
