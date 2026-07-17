"""End-to-end eval-harness tests using the offline MockClient."""

import pytest

from types import SimpleNamespace

from docsthatrun.evals.metrics import recall_at_k, reciprocal_rank
from docsthatrun.evals.run_evals import (
    GATE,
    _classify_failure,
    _classify_outcome,
    _latency_stats,
    check_gate,
    evaluate,
)
from docsthatrun.schema import GoldenItem


def test_metrics_basic():
    assert recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], 5) == 0.0
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_retrieval_only_meets_gate():
    report = evaluate(run_answers=False)
    assert report["retrieval"]["recall_at_5"] >= GATE["recall_at_5"]
    assert report["retrieval"]["mrr"] >= GATE["mrr"]
    assert check_gate(report) == []


def test_mock_answers_abstain_correctly():
    # MockClient replays the answer key: answerable items don't abstain,
    # unanswerable items do. This exercises the abstention plumbing.
    report = evaluate(run_answers=True, client_name="mock")
    answers = report["answers"]
    assert answers["unanswerable_abstention"] == 1.0
    assert answers["answerable_over_abstention"] == 0.0


def test_report_includes_taxonomy_and_latency():
    report = evaluate(run_answers=True, client_name="mock")
    answers = report["answers"]
    # every answerable item is bucketed, buckets sum to the golden set size
    assert sum(answers["taxonomy"].values()) == report["golden_size"]
    lat = answers["latency_ms"]
    assert lat and lat["p50"] <= lat["p95"] <= lat["max"]
    # each row carries its outcome + latency
    assert all("outcome" in r and "latency_ms" in r for r in answers["rows"])


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("AttributeError: 'User' object has no attribute 'model_dump'", "wrong_version_api"),
        ("ImportError: cannot import name 'BaseSettings'", "wrong_version_api"),
        ("  File x, line 2\n    def f(\n        ^\nSyntaxError: ...", "malformed_code"),
        ("AssertionError", "wrong_assert"),
        ("ValueError: boom", "runtime_error"),
    ],
)
def test_failure_taxonomy_classifier(stderr, expected):
    assert _classify_failure(stderr) == expected


def test_latency_stats_none_on_empty():
    assert _latency_stats([]) is None


def _fake_res(retrieved_ids, abstained=False, code="x", ex=None):
    return SimpleNamespace(
        answer=SimpleNamespace(abstained=abstained, code=code),
        retrieved=[SimpleNamespace(chunk=SimpleNamespace(id=i)) for i in retrieved_ids],
        execution=ex,
    )


_ITEM = GoldenItem(id="g", question="q", version="v2", relevant_chunk_ids=["gold"], check="")


def test_passing_answer_never_labeled_retrieval_miss():
    # Executed & passed, but the gold chunk wasn't in the retrieved set: still a
    # pass (the old ordering mislabeled this as retrieval_miss).
    ex = SimpleNamespace(available=True, passed=True, stderr="")
    assert _classify_outcome(_ITEM, _fake_res(["other"], ex=ex)) == "pass"


def test_retrieval_miss_only_when_not_passing():
    ex = SimpleNamespace(available=True, passed=False, stderr="AssertionError")
    assert _classify_outcome(_ITEM, _fake_res(["other"], ex=ex)) == "retrieval_miss"
    # gold retrieved but failed on the assert -> attributed to the assert
    assert _classify_outcome(_ITEM, _fake_res(["gold"], ex=ex)) == "wrong_assert"
