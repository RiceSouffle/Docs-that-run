"""Settings: defaults, env overrides, and safe fallback on bad values."""

import pytest

from docsthatrun.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.model.startswith("claude")
    assert 1 <= s.top_k_default <= s.top_k_max
    assert s.sandbox_cpu_seconds > 0 and s.sandbox_memory_mb > 0


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCSTHATRUN_TOP_K", "9")
    monkeypatch.setenv("DOCSTHATRUN_RATE_RPM", "123")
    monkeypatch.setenv("DOCSTHATRUN_CORS_ORIGINS", "https://a.com, https://b.com")
    monkeypatch.setenv("DOCSTHATRUN_LOG_JSON", "false")
    s = Settings.from_env()
    assert s.top_k_default == 9
    assert s.rate_limit_rpm == 123
    assert s.cors_origins == ("https://a.com", "https://b.com")
    assert s.log_json is False


def test_bad_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DOCSTHATRUN_TOP_K", "not-an-int")
    assert Settings.from_env().top_k_default == Settings.top_k_default


@pytest.mark.parametrize("raw,expected", [("0", 1), ("-1", 1), ("9999", 50)])
def test_top_k_default_is_clamped_into_range(monkeypatch, raw, expected):
    # The API's Field(ge=1, le=50) does NOT validate the *default*, so an
    # out-of-range DOCSTHATRUN_TOP_K used to sail past the same bound the wire
    # contract enforces: 0 answered from zero chunks, -1 sliced off the
    # top-ranked chunk. Clamp at the source so every consumer is covered.
    monkeypatch.setenv("DOCSTHATRUN_TOP_K", raw)
    assert Settings.from_env().top_k_default == expected


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_max_question_chars_is_clamped(monkeypatch, raw):
    # This feeds Field(min_length=1, max_length=settings.max_question_chars).
    # At 0 or negative, max_length < min_length and EVERY question is rejected.
    monkeypatch.setenv("DOCSTHATRUN_MAX_QUESTION_CHARS", raw)
    assert Settings.from_env().max_question_chars >= 1


def test_top_k_max_cannot_invert_the_range(monkeypatch):
    monkeypatch.setenv("DOCSTHATRUN_TOP_K_MAX", "0")
    s = Settings.from_env()
    assert s.top_k_max >= 1 and 1 <= s.top_k_default <= s.top_k_max


# ---- validation added after the audit --------------------------------------


def test_default_version_is_rejected_not_silently_accepted(monkeypatch):
    """A typo here used to 400 every /ask that omitted `version` — a config
    problem presenting as a client error, on every request."""
    monkeypatch.setenv("DOCSTHATRUN_DEFAULT_VERSION", "v3")
    with pytest.raises(ValueError, match="DOCSTHATRUN_DEFAULT_VERSION"):
        Settings.from_env()


@pytest.mark.parametrize(
    "name,attr,bad,floor",
    [
        ("DOCSTHATRUN_SANDBOX_TIMEOUT", "sandbox_timeout_s", "0", 1),
        ("DOCSTHATRUN_SANDBOX_TIMEOUT", "sandbox_timeout_s", "-5", 1),
        ("DOCSTHATRUN_SANDBOX_CPU", "sandbox_cpu_seconds", "-1", 1),
        ("DOCSTHATRUN_SANDBOX_FILE_MB", "sandbox_file_mb", "0", 1),
        ("DOCSTHATRUN_SANDBOX_MEM_MB", "sandbox_memory_mb", "-1", 64),
    ],
)
def test_sandbox_limits_are_clamped(monkeypatch, name, attr, bad, floor):
    """These were the only knobs with no bounds, and the ones where a bad value
    fails silently: timeout=0 made proc.wait() expire instantly so *every*
    snippet reported "timed out", and a negative rlimit raised inside the
    launcher's best-effort try/except, removing the limit altogether."""
    monkeypatch.setenv(name, bad)
    assert getattr(Settings.from_env(), attr) == floor


def test_non_positive_cache_ttl_disables_rather_than_immortalises(monkeypatch):
    monkeypatch.setenv("DOCSTHATRUN_CACHE_TTL", "-30")
    assert Settings.from_env().cache_ttl_s == 0.0
