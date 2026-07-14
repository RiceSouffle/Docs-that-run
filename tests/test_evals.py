"""End-to-end eval-harness tests using the offline MockClient."""

from docsthatrun.evals.metrics import recall_at_k, reciprocal_rank
from docsthatrun.evals.run_evals import GATE, check_gate, evaluate


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
