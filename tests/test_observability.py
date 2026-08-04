"""Metrics counters + Prometheus rendering, and the JSON log formatter."""

import json
import logging

from docsthatrun.observability import JsonFormatter, Metrics, configure_logging


def test_metrics_counts_and_prometheus_render():
    m = Metrics()
    m.record_request("/ask", 200, 12.5)
    m.record_request("/ask", 200, 7.5)
    m.record_request("/ask", 400, 1.0)
    m.record_grade("pass")
    text = m.render_prometheus({"hits": 3, "misses": 1, "size": 2})
    assert 'docsthatrun_requests_total{endpoint="/ask",status="200"} 2' in text
    assert 'docsthatrun_requests_total{endpoint="/ask",status="400"} 1' in text
    assert 'docsthatrun_grades_total{result="pass"} 1' in text
    assert "docsthatrun_cache_hits_total 3" in text

    snap = m.snapshot({"hits": 3})
    assert snap["grades"]["pass"] == 1
    assert snap["avg_latency_ms"]["/ask"] == 7.0  # (12.5+7.5+1.0)/3 rounded
    assert snap["cache"]["hits"] == 3


def test_json_formatter_includes_structured_extras():
    rec = logging.LogRecord("docsthatrun", logging.INFO, "f.py", 1, "hello", (), None)
    rec.request_id = "abc123"
    rec.status = 200
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "hello"
    assert out["level"] == "INFO"
    assert out["request_id"] == "abc123"
    assert out["status"] == 200


def test_configure_logging_survives_bad_level():
    # A bad DOCSTHATRUN_LOG_LEVEL used to raise ValueError at import and kill
    # the server before boot; it must fall back to INFO like every other knob.
    root = logging.getLogger()
    original = root.level
    try:
        configure_logging("TRACE")  # not a Python level name
        assert root.level == logging.INFO
        configure_logging("")  # empty env value
        assert root.level == logging.INFO
        configure_logging(" warning ")  # sloppy .env whitespace + case
        assert root.level == logging.WARNING
    finally:
        root.setLevel(original)
