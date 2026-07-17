"""HTTP API tests. Skipped when fastapi/httpx aren't installed (the core runs
without them); CI installs requirements.txt so these run there."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

os.environ.setdefault("DOCSTHATRUN_LLM", "mock")  # no API key needed for tests

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
