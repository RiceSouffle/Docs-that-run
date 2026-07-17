"""HTTP API tests. Skipped when fastapi/httpx aren't installed (the core runs
without them); CI installs requirements.txt so these run there."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

os.environ.setdefault("DOCSTHATRUN_LLM", "mock")  # no API key needed for tests
os.environ.setdefault("DOCSTHATRUN_RATE_RPM", "0")  # deterministic: no rate limiting

from fastapi.testclient import TestClient  # noqa: E402

from app.main import answer_cache, app  # noqa: E402

client = TestClient(app)


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200 and "DocsThatRun" in r.text


def test_health_reports_client_and_sandbox():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["sandbox"]) == {"v1", "v2"}


def test_examples_are_nonempty():
    body = client.get("/examples").json()
    assert body["answerable"] and body["unanswerable"]


def test_ask_rejects_unknown_version():
    r = client.post("/ask", json={"question": "x", "version": "v3"})
    assert r.status_code == 400


@pytest.mark.parametrize("bad", [0, -1, 999])
def test_ask_rejects_out_of_range_top_k(bad):
    # top_k <= 0 used to silently retrieve nothing (or drop the top chunks);
    # it's now bounded and rejected with a 422.
    r = client.post("/ask", json={"question": "x", "version": "v2", "top_k": bad})
    assert r.status_code == 422


def test_ask_answerable_returns_cited_answer_and_grade():
    q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
    r = client.post("/ask", json={"question": q, "version": "v2"}).json()
    assert r["answer"]["abstained"] is False
    assert r["answer"]["citations"]  # at least one citation
    assert r["retrieved"] and r["retrieved"][0]["id"]


def test_compare_shows_both_versions():
    q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
    r = client.post("/compare", json={"question": q}).json()
    assert set(r["versions"]) == {"v1", "v2"}


# ---- production surface ----------------------------------------------------

def test_response_has_meta_and_second_call_is_cached():
    answer_cache.clear()
    q = {"question": "In Pydantic v2, how do I generate a JSON schema for a model?",
         "version": "v2"}
    r1 = client.post("/ask", json=q).json()
    assert r1["meta"]["cached"] is False and "latency_ms" in r1["meta"]
    r2 = client.post("/ask", json=q).json()
    assert r2["meta"]["cached"] is True


def test_security_headers_present():
    h = client.get("/health").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "SAMEORIGIN"
    assert "content-security-policy" in h
    assert "x-request-id" in h


def test_ready_endpoint():
    body = client.get("/ready").json()
    assert body["ready"] is True and body["corpus"] is True


def test_metrics_and_stats():
    client.get("/health")  # generate some traffic
    assert "docsthatrun_requests_total" in client.get("/metrics").text
    assert "requests" in client.get("/stats").json()


def test_rejects_overlong_question():
    r = client.post("/ask", json={"question": "x" * 5000, "version": "v2"})
    assert r.status_code == 422


def test_security_headers_on_unhandled_500(monkeypatch):
    # Force an unhandled error inside a route; the middleware must still return a
    # clean 500 carrying the security headers + request id (not a bare 500).
    import app.main as m

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(m.answer_cache, "stats", _boom)
    r = client.get("/metrics")
    assert r.status_code == 500
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in r.headers
    assert "x-request-id" in r.headers
