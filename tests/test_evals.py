"""End-to-end eval-harness tests using the offline MockClient."""

from types import SimpleNamespace

import pytest

from docsthatrun.evals.metrics import recall_at_k, reciprocal_rank
from docsthatrun.evals.run_evals import (
    GATE,
    _classify_failure,
    _classify_outcome,
    _latency_stats,
    check_gate,
    evaluate,
)
from docsthatrun.schema import Answer, GoldenItem


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
    # MockClient replays the answer key, so these two numbers are true by
    # construction — this is a plumbing check (the abstention flag survives the
    # whole answer path and reaches the report), not a quality claim. The real
    # abstention behaviour is pinned by test_unknown_question_abstains below,
    # which uses a client that has no fixture for the question at all.
    report = evaluate(run_answers=True, client_name="mock")
    answers = report["answers"]
    assert answers["unanswerable_abstention"] == 1.0
    assert answers["answerable_over_abstention"] == 0.0


def test_unknown_question_abstains_rather_than_inventing_an_answer():
    """MockClient({}) has no fixtures, so every question is unknown.

    This is also the regression test for `fixtures or ...`: an explicitly empty
    dict is falsy, so MockClient({}) silently loaded all 31 golden answers and
    this test would have found a fixture for everything.
    """
    from docsthatrun.llm import MockClient

    client = MockClient({})
    assert client.fixtures == {}
    out = client.generate("anything at all", "v2", [])
    assert out["abstained"] is True
    assert out["code"] == ""


def test_report_includes_taxonomy_and_latency():
    report = evaluate(run_answers=True, client_name="mock")
    answers = report["answers"]
    # every answerable item is bucketed, buckets sum to the golden set size
    assert sum(answers["taxonomy"].values()) == report["golden_size"]
    lat = answers["answer_latency_ms"]
    assert lat and lat["p50"] <= lat["p95"] <= lat["max"]
    # each row carries its outcome + answer latency
    assert all("outcome" in r and "answer_ms" in r for r in answers["rows"])


def test_answer_latency_covers_every_question_and_excludes_grading():
    """Answer latency must describe the retrieval + LLM path only.

    It used to wrap the sandbox run too, so the published p95 was really a
    statement about subprocess startup; and the 6 unanswerable questions were
    never timed, quietly narrowing the distribution to 25 of 31.
    """
    report = evaluate(run_answers=True, client_name="mock")
    answers = report["answers"]
    rows = answers["rows"]
    graded = [r for r in rows if "grade_ms" in r]
    if not graded:
        pytest.skip("sandbox venvs not set up")
    # Grading is reported on its own axis, and it dominates — which is exactly
    # why folding it into "answer latency" was misleading.
    assert answers["grade_latency_ms"]["mean"] > answers["answer_latency_ms"]["mean"]
    assert all(r["answer_ms"] < r["grade_ms"] for r in graded)


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
    # A real Answer, not a stand-in: has_runnable_code() is the shared predicate
    # the CLI, the API and this harness all gate on, so faking it past would
    # test nothing.
    return SimpleNamespace(
        answer=Answer(answer="a", code=code, citations=[], abstained=abstained),
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


def test_whitespace_only_code_is_no_code_not_runtime_error():
    # "  \n" is truthy, so the old `if not res.answer.code` let it through to
    # the sandbox, where it failed with empty stderr -> runtime_error, blaming
    # the wrong stage.
    assert _classify_outcome(_ITEM, _fake_res(["gold"], code="  \n")) == "no_code"


def test_empty_code_answers_count_against_executable_pct(monkeypatch):
    """A client that answers (doesn't abstain) but emits no runnable code used
    to be silently dropped from the executable_pct denominator — 40% no-code
    could still report 1.0 and pass the gate."""
    from docsthatrun import llm
    from docsthatrun.evals import run_evals as re_
    from docsthatrun.sandbox import sandbox_available

    if not (sandbox_available("v1") and sandbox_available("v2")):
        pytest.skip("sandbox venvs not set up")

    class SometimesNoCode(llm.MockClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, question, version, retrieved):
            out = dict(super().generate(question, version, retrieved))
            self.calls += 1
            if not out["abstained"] and self.calls % 5 == 0:
                out["code"] = "   \n"  # answered, but nothing runnable
            return out

    monkeypatch.setattr(re_, "get_client", lambda name=None: SometimesNoCode())
    answers = re_.evaluate(run_answers=True, client_name="ignored")["answers"]

    no_code = answers["taxonomy"].get("no_code", 0)
    assert no_code > 0
    expected = round((answers["answered_count"] - no_code) / answers["answered_count"], 3)
    assert answers["executable_pct"] == expected < 1.0


def test_mock_fixture_collision_fails_loud(monkeypatch):
    # Two golden items with identical normalized question text would silently
    # replay whichever fixture loaded last; the loader must refuse instead.
    from docsthatrun import llm
    from docsthatrun.evals import run_evals as re_

    dup = SimpleNamespace(id="dup", question="Same question?", relevant_chunk_ids=[], check="")
    monkeypatch.setattr(re_, "load_golden", lambda: [dup, dup])
    monkeypatch.setattr(re_, "load_unanswerable", lambda: [])
    with pytest.raises(ValueError, match="duplicate MockClient fixture"):
        llm._load_fixtures_from_golden()


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_llm_env_var_means_auto_not_error(monkeypatch, raw):
    # `DOCSTHATRUN_LLM=` is how a .env / docker-compose default / CI matrix
    # writes "leave it alone"; os.environ.get returns "" for it, not the
    # default. Rejecting that as an unknown client refused to start the very
    # setup docker-compose.yml ships.
    from docsthatrun.llm import MockClient, get_client

    monkeypatch.setenv("DOCSTHATRUN_LLM", raw)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_client(), MockClient)


def test_unknown_client_name_raises_instead_of_falling_through():
    # `--client mokc` used to silently select "auto", which with a key exported
    # makes real billed API calls when the operator asked for the offline mock
    # (and vice versa) — producing confidently mislabelled numbers.
    from docsthatrun.llm import get_client

    with pytest.raises(ValueError, match="unknown client"):
        get_client("mokc")


def test_grading_is_per_version_not_a_single_and(monkeypatch):
    """One broken venv must not stop the *other*, healthy version from being
    graded — that turned every item into `not_graded` while the gate, which
    skips the executable check when the sandbox is down, still passed."""
    from docsthatrun.evals import run_evals as re_

    monkeypatch.setattr(re_, "sandbox_available", lambda v: v == "v2")
    report = re_.evaluate(run_answers=True, client_name="mock")

    assert report["sandbox_by_version"] == {"v1": False, "v2": True}
    assert report["sandbox_available"] is False
    rows = {r["id"]: r for r in report["answers"]["rows"]}
    v2_graded = [r for r in rows.values() if "executed" in r]
    assert v2_graded, "healthy v2 sandbox should still have graded its items"


@pytest.mark.parametrize("bad", ["0", "-1", "51"])
def test_run_evals_rejects_out_of_range_top_k(bad, capsys):
    # Unvalidated --top-k silently corrupted every metric (0 retrieves nothing,
    # negative slices off the top results).
    from docsthatrun.evals.run_evals import main

    with pytest.raises(SystemExit):
        main(["--top-k", bad])
