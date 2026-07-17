"""Settings: defaults, env overrides, and safe fallback on bad values."""

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
