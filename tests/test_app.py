"""HTTP API tests. Skipped when fastapi/httpx aren't installed (the core runs
without them); CI installs requirements.txt so these run there."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# The test environment (mock client, no rate limiting, no API key) is set in
# tests/conftest.py, which pytest imports before any test module — early enough
# for docsthatrun.config's import-time settings singleton to see it.
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

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


def test_ask_answerable_returns_cited_answer_and_grade(sandbox_ready):
    q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
    r = client.post("/ask", json={"question": q, "version": "v2"}).json()
    assert r["answer"]["abstained"] is False
    assert r["answer"]["citations"]  # at least one citation
    assert r["retrieved"] and r["retrieved"][0]["id"]
    # Every cited id must be one that was actually retrieved — the grounding
    # claim, checked rather than assumed.
    retrieved_ids = {c["id"] for c in r["retrieved"]}
    assert set(r["answer"]["citations"]) <= retrieved_ids
    # The name promises a grade, so check for one. Without this the test passed
    # with execution: null on any machine that hadn't built the venvs.
    if not sandbox_ready:
        pytest.skip("sandbox venvs not set up — cannot assert on the grade")
    assert r["execution"]["available"] is True
    assert r["execution"]["passed"] is True


def test_compare_shows_both_versions():
    q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
    r = client.post("/compare", json={"question": q}).json()
    assert set(r["versions"]) == {"v1", "v2"}


# ---- production surface ----------------------------------------------------


def test_response_has_meta_and_second_call_is_cached():
    q = {"question": "In Pydantic v2, how do I generate a JSON schema for a model?", "version": "v2"}
    r1 = client.post("/ask", json=q).json()
    assert r1["meta"]["cached"] is False and "latency_ms" in r1["meta"]
    r2 = client.post("/ask", json=q).json()
    assert r2["meta"]["cached"] is True
    # latency_ms describes THIS response, so a cache hit must be much faster
    # than the original; the original is kept under its own name. The UI used
    # to print the uncached figure next to the word "cached".
    assert r2["meta"]["latency_ms"] < r1["meta"]["latency_ms"]
    assert r2["meta"]["uncached_latency_ms"] == r1["meta"]["latency_ms"]


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

    req = m.Request(
        {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"203.0.113.9, 10.0.0.2")],
            "client": ("10.0.0.1", 1234),
        }
    )
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


# ---- rate limiting, exercised through HTTP ---------------------------------


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    """The limiter is unit-tested, but its *wiring* never was: the suite
    disables rate limiting globally, so the 429 and the Retry-After header the
    README advertises were never produced by a real request."""
    import app.main as m
    from docsthatrun.ratelimit import RateLimiter

    monkeypatch.setattr(m, "limiter", RateLimiter(rpm=1, burst=1))
    q = {"question": "In Pydantic v2, how do I serialize a model instance to a dictionary?", "version": "v2"}
    assert client.post("/ask", json=q).status_code == 200
    blocked = client.post("/ask", json=q)
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_malformed_bodies_are_rate_limited_too(monkeypatch):
    """Validation runs before the handler, so a flood of 422s used to cost the
    caller nothing. The limiter is a route dependency now, which runs first."""
    import app.main as m
    from docsthatrun.ratelimit import RateLimiter

    monkeypatch.setattr(m, "limiter", RateLimiter(rpm=1, burst=1))
    assert client.post("/ask", json={"bogus": 1}).status_code == 422
    assert client.post("/ask", json={"bogus": 1}).status_code == 429


def test_compare_costs_two_tokens_because_it_does_two_answers(monkeypatch):
    import app.main as m
    from docsthatrun.ratelimit import RateLimiter

    monkeypatch.setattr(m, "limiter", RateLimiter(rpm=1, burst=2))
    q = {"question": "In Pydantic v2, how do I serialize a model instance to a dictionary?"}
    assert client.post("/compare", json=q).status_code == 200
    assert client.post("/compare", json=q).status_code == 429


@pytest.mark.parametrize("blank", ["   ", "\n\t ", ""])
def test_blank_questions_are_rejected(blank):
    """min_length ran against the raw body while the handler stripped
    afterwards, so "   " validated and then became "" — a 200 answering an
    empty question."""
    r = client.post("/ask", json={"question": blank, "version": "v2"})
    assert r.status_code == 422


def test_ready_reports_503_and_a_reason_when_the_corpus_is_broken(monkeypatch):
    """/ready exists to return 503 for exactly this; it used to 500 instead,
    telling the probe 'broken in an unknown way' when the server knew."""
    import app.main as m

    def _boom():
        raise ValueError("corpus.jsonl:7 duplicate chunk id 'c1'")

    monkeypatch.setattr(m, "get_retriever", _boom)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert "duplicate chunk id" in r.json()["detail"]


def test_health_stays_up_when_the_client_cannot_be_built(monkeypatch):
    """Liveness must not depend on anything that can fail — a bad
    DOCSTHATRUN_LLM used to 500 the endpoint an orchestrator uses to decide
    whether to restart the process."""
    import app.main as m

    def _boom():
        raise ValueError("unknown client 'mokc'")

    monkeypatch.setattr(m, "get_llm", _boom)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["client"] == "unavailable"


def test_upstream_failure_does_not_leak_exception_text(monkeypatch):
    """SDK errors carry request URLs and account identifiers, and the UI renders
    `detail` verbatim."""
    import app.main as m

    def _boom(*a, **k):
        raise RuntimeError("connection to https://api.example/v1 failed for org_SECRET123")

    monkeypatch.setattr(m, "_answer", _boom)
    r = client.post("/ask", json={"question": "anything at all", "version": "v2"})
    assert r.status_code == 502
    assert "SECRET123" not in r.text and "api.example" not in r.text
    assert "request id" in r.json()["detail"]
