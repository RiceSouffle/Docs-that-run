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


def test_docs_page_csp_allows_its_cdn_but_other_routes_do_not():
    # FastAPI's /docs loads Swagger UI from cdn.jsdelivr.net; the strict global
    # CSP blocked it, so the "real OpenAPI at /docs" the README advertises
    # rendered blank. The relaxation must be scoped to the docs routes only.
    docs_csp = client.get("/docs").headers["content-security-policy"]
    assert "cdn.jsdelivr.net" in docs_csp
    for path in ("/", "/health"):
        assert "cdn.jsdelivr.net" not in client.get(path).headers["content-security-policy"]


def test_cached_response_echoes_the_callers_own_question():
    # The key was stripped but the payload stored the first caller's raw string,
    # so a later caller got a response quoting a question they never sent.
    answer_cache.clear()
    q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
    client.post("/ask", json={"question": "   " + q + "   ", "version": "v2"})
    second = client.post("/ask", json={"question": q, "version": "v2"}).json()
    assert second["meta"]["cached"] is True
    assert second["question"] == q


def test_ready_returns_503_when_corpus_empty(monkeypatch):
    # Readiness consumers (k8s probes, LBs) act on the status code — "ready":
    # false inside a 200 would still get traffic routed to a broken instance.
    from types import SimpleNamespace

    import app.main as m

    monkeypatch.setattr(m, "get_retriever", lambda: SimpleNamespace(chunks=[]))
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_rate_limit_key_honors_forwarded_header_only_when_trusted(monkeypatch):
    import dataclasses

    import app.main as m

    req = m.Request({
        "type": "http",
        "headers": [(b"x-forwarded-for", b"203.0.113.9, 10.0.0.2")],
        "client": ("10.0.0.1", 1234),
    })
    # Default: the direct TCP peer. The header is client-spoofable, so it must
    # be ignored unless a trusted proxy is explicitly declared.
    assert m._client_key(req) == "10.0.0.1"
    monkeypatch.setattr(m, "settings", dataclasses.replace(m.settings, trust_proxy=True))
    # Behind a trusted proxy: the first forwarded hop, so distinct clients get
    # distinct buckets instead of all sharing the proxy IP's one.
    assert m._client_key(req) == "203.0.113.9"


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
