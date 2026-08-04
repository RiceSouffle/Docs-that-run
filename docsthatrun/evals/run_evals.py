"""Eval harness + CI gate.

Three layers, from cheapest to most expensive:

1. Retrieval metrics (recall@k, MRR) — pure offline, no LLM, no sandbox.
2. Answer executable-% and version-lock — needs an LLM client and the sandbox
   venvs. With the MockClient this exercises the plumbing; with Claude it is a
   real measurement.
3. Abstention — answerable over-abstention + unanswerable correct-abstention.

Run:
    python -m docsthatrun.evals.run_evals               # retrieval only
    python -m docsthatrun.evals.run_evals --answers     # + answer/exec/abstention
    python -m docsthatrun.evals.run_evals --answers --gate   # fail CI on regression
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import List, Optional

from ..answer import AnswerResult, build_answer
from ..corpus import load_corpus
from ..llm import CLIENT_NAMES, MockClient, get_client
from ..retrieve import HybridRetriever
from ..sandbox import sandbox_available
from ..schema import GoldenItem
from .metrics import mean, recall_at_k, reciprocal_rank

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
GOLDEN_PATH = os.path.join(_DATA_DIR, "golden", "golden_set.jsonl")
UNANSWERABLE_PATH = os.path.join(_DATA_DIR, "golden", "unanswerable.jsonl")

# Gate thresholds. These are the noise floor the committed data must clear;
# tighten them as the golden set grows. Documented in DECISIONS.md.
GATE = {
    "recall_at_5": 0.80,
    "mrr": 0.60,
    "unanswerable_abstention": 0.80,
    "answerable_over_abstention_max": 0.20,
    "executable_pct_min": 0.60,
}


# Per-item failure taxonomy (ROADMAP milestone 1). A single number
# ("executable-%") hides *why* answers fail; this splits each answerable item
# into one bucket so a regression is attributable to a stage, not a mystery.
def _classify_failure(stderr: str) -> str:
    s = stderr or ""
    if "SyntaxError" in s or "IndentationError" in s:
        return "malformed_code"
    if "AssertionError" in s:
        return "wrong_assert"
    if any(
        k in s
        for k in (
            "ImportError",
            "ModuleNotFoundError",
            "cannot import name",
            "has no attribute",
            "AttributeError",
        )
    ):
        return "wrong_version_api"
    return "runtime_error"


def _classify_outcome(item: GoldenItem, res: AnswerResult) -> str:
    ex = res.execution
    # A snippet that executed and passed is a pass, full stop — even if the gold
    # chunk happened to fall outside the retrieved set. Check this FIRST so a
    # success is never mislabeled as an upstream failure.
    if ex is not None and ex.available and ex.passed:
        return "pass"
    # Not a pass: attribute the shortfall to a stage. A missing gold chunk is an
    # upstream (retrieval) cause and takes precedence over the symptom.
    retrieved_ids = {r.chunk.id for r in res.retrieved}
    if item.relevant_chunk_ids and not (set(item.relevant_chunk_ids) & retrieved_ids):
        return "retrieval_miss"
    if res.answer.abstained:
        return "over_abstention"
    if not res.answer.has_runnable_code():
        return "no_code"
    if ex is None or not ex.available:
        return "not_graded"
    return _classify_failure(ex.stderr)


def _latency_stats(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = int(round((p / 100.0) * (len(ordered) - 1)))
        return ordered[min(idx, len(ordered) - 1)]

    return {
        "mean": round(mean(values), 1),
        "p50": pct(50),
        "p95": pct(95),
        "max": ordered[-1],
    }


def _load_items(path: str, answerable: bool) -> List[GoldenItem]:
    items: List[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            data = json.loads(raw)
            items.append(
                GoldenItem(
                    id=data["id"],
                    question=data["question"],
                    version=data["version"],
                    relevant_chunk_ids=data.get("relevant_chunk_ids", []),
                    check=data.get("check", ""),
                    answerable=answerable,
                )
            )
    return items


def load_golden() -> List[GoldenItem]:
    return _load_items(GOLDEN_PATH, answerable=True)


def load_unanswerable() -> List[GoldenItem]:
    return _load_items(UNANSWERABLE_PATH, answerable=False)


def evaluate(run_answers: bool = False, top_k: int = 5, client_name: Optional[str] = None) -> dict:
    corpus = load_corpus()
    retriever = HybridRetriever(corpus)
    golden = load_golden()
    unanswerable = load_unanswerable()

    # ---- Layer 1: retrieval (offline) --------------------------------------
    recalls_5, recalls_3, rrs = [], [], []
    per_item = []
    for item in golden:
        results = retriever.retrieve(item.question, item.version, top_k=max(top_k, 5))
        ids = [r.chunk.id for r in results]
        r5 = recall_at_k(ids, item.relevant_chunk_ids, 5)
        r3 = recall_at_k(ids, item.relevant_chunk_ids, 3)
        rr = reciprocal_rank(ids, item.relevant_chunk_ids)
        recalls_5.append(r5)
        recalls_3.append(r3)
        rrs.append(rr)
        per_item.append({"id": item.id, "recall_at_5": r5, "mrr": rr, "top_ids": ids[:5]})

    report: dict = {
        "corpus_size": len(corpus),
        "golden_size": len(golden),
        "unanswerable_size": len(unanswerable),
        "retrieval": {
            "recall_at_3": round(mean(recalls_3), 3),
            "recall_at_5": round(mean(recalls_5), 3),
            "mrr": round(mean(rrs), 3),
        },
        "per_item": per_item,
    }

    if not run_answers:
        return report

    # ---- Layers 2 & 3: answers, execution grading, abstention --------------
    client = get_client(client_name)
    report["client"] = type(client).__name__
    # Per-version, not a single AND across both. A global flag would stop grading
    # the *healthy* version too, turn every item into `not_graded`, and — because
    # the executable gate used to be skipped whenever that flag was false — let
    # the run print GATE PASSED having executed nothing. Grading is per version,
    # and so is the denominator below; a version that could not be graded is
    # reported as such and fails the gate rather than vanishing.
    sandbox_by_version = {v: sandbox_available(v) for v in ("v1", "v2")}
    report["sandbox_available"] = all(sandbox_by_version.values())
    report["sandbox_by_version"] = sandbox_by_version

    executable_hits = 0
    gradable = 0  # answers actually executed (execution.available)
    should_run = 0  # non-abstained answerable items whose sandbox was up
    ungraded = 0  # non-abstained answerable items whose sandbox was NOT up
    answerable_over_abstain = 0
    answer_rows = []
    taxonomy: Counter = Counter()
    answer_latencies: List[float] = []
    grade_latencies: List[float] = []
    for item in golden:
        t0 = time.perf_counter()
        res = build_answer(item.question, item.version, retriever, client=client, top_k=top_k)
        # Timed separately: bundling a 20s sandbox run into "answer latency"
        # makes the published p95 a number about subprocess startup, not about
        # the retrieval + LLM path anyone is actually asking about.
        answer_ms = round((time.perf_counter() - t0) * 1000, 1)
        answer_latencies.append(answer_ms)

        version_up = sandbox_by_version.get(item.version, False)
        grade_ms = None
        if not res.answer.abstained and res.answer.has_runnable_code() and version_up:
            t1 = time.perf_counter()
            res.execution_grade()
            grade_ms = round((time.perf_counter() - t1) * 1000, 1)
            grade_latencies.append(grade_ms)

        outcome = _classify_outcome(item, res)
        taxonomy[outcome] += 1
        if res.answer.abstained:
            answerable_over_abstain += 1
        elif version_up:
            should_run += 1
        else:
            ungraded += 1
        if res.execution is not None and res.execution.available:
            gradable += 1
            if res.execution.passed:
                executable_hits += 1

        row = {
            "id": item.id,
            "outcome": outcome,
            "abstained": res.answer.abstained,
            "answer_ms": answer_ms,
        }
        if grade_ms is not None:
            row["grade_ms"] = grade_ms
        if res.execution is not None:
            row["executed"] = res.execution.passed
            row["reason"] = res.execution.reason
        answer_rows.append(row)

    abstained_correct = 0
    for item in unanswerable:
        t0 = time.perf_counter()
        res = build_answer(item.question, item.version, retriever, client=client, top_k=top_k)
        # Timed too: leaving these out silently narrowed the published latency
        # distribution to 25 of the 31 questions the harness actually answers.
        answer_latencies.append(round((time.perf_counter() - t0) * 1000, 1))
        if res.answer.abstained:
            abstained_correct += 1

    report["answers"] = {
        # Denominator: every answerable item that *claimed* an answer (didn't
        # abstain) and whose version's sandbox was up. An answer with no runnable
        # snippet counts as a failure here — with the old graded-only
        # denominator, a client emitting empty code for 40% of items still
        # scored executable_pct=1.0 and passed the gate. Items we *couldn't*
        # grade are excluded and counted separately in `ungraded_count`, so a
        # half-broken sandbox never silently inflates or erases this number.
        "executable_pct": (round(executable_hits / should_run, 3) if should_run else None),
        "answered_count": should_run,
        "ungraded_count": ungraded,
        "gradable_count": gradable,
        # Two related but distinct abstention numbers, spelled out because they
        # do not have to agree: this rate counts every abstention on an
        # answerable item, while taxonomy["over_abstention"] counts only those
        # where retrieval *did* surface the gold chunk (an abstention despite
        # having the evidence). An abstention that also missed its gold chunk is
        # filed under taxonomy["retrieval_miss"], since retrieval is the
        # upstream cause, but still counts here.
        "answerable_over_abstention": round(answerable_over_abstain / len(golden), 3) if golden else 0.0,
        "abstained_count": answerable_over_abstain,
        "unanswerable_abstention": round(abstained_correct / len(unanswerable), 3) if unanswerable else 0.0,
        # Failure taxonomy: which stage each answerable item landed in.
        "taxonomy": dict(taxonomy),
        # Retrieval + LLM only, across all 31 questions.
        "answer_latency_ms": _latency_stats(answer_latencies),
        # Sandbox execution only, across the items that were graded.
        "grade_latency_ms": _latency_stats(grade_latencies),
        "rows": answer_rows,
        "note": (
            "MockClient replays the answer key: executable_pct here is a PLUMBING "
            "check, not a quality claim. Run with DOCSTHATRUN_LLM=anthropic for a "
            "real measurement."
            if isinstance(client, MockClient)
            else "Measured against Claude-generated answers."
        ),
    }
    return report


def check_gate(report: dict) -> List[str]:
    failures: List[str] = []
    ret = report["retrieval"]
    if ret["recall_at_5"] < GATE["recall_at_5"]:
        failures.append(f"recall@5 {ret['recall_at_5']} < {GATE['recall_at_5']}")
    if ret["mrr"] < GATE["mrr"]:
        failures.append(f"mrr {ret['mrr']} < {GATE['mrr']}")

    answers = report.get("answers")
    if answers:
        if answers["unanswerable_abstention"] < GATE["unanswerable_abstention"]:
            failures.append(
                f"unanswerable_abstention {answers['unanswerable_abstention']} < {GATE['unanswerable_abstention']}"
            )
        if answers["answerable_over_abstention"] > GATE["answerable_over_abstention_max"]:
            failures.append(
                "answerable_over_abstention "
                f"{answers['answerable_over_abstention']} > {GATE['answerable_over_abstention_max']}"
            )
        # An execution gate that did not execute does not pass. Previously a
        # single unavailable venv set sandbox_available=False, which skipped
        # this whole block — so a run that graded half the corpus (and threw
        # those results away) printed GATE PASSED. Any version we could not
        # grade is now a stated failure, naming the version.
        down = [v for v, up in sorted(report.get("sandbox_by_version", {}).items()) if not up]
        if down:
            failures.append(
                f"sandbox unavailable for {', '.join(down)} — "
                f"{answers['ungraded_count']} answerable item(s) went ungraded; "
                "run scripts/setup_sandbox.sh"
            )
        pct = answers["executable_pct"]
        if pct is None:
            # Nothing was gradable: either every answerable item abstained, or
            # no sandbox was up at all. Both are regressions, not passes — the
            # old guard silently skipped the gate in each case.
            failures.append("executable_pct could not be measured (nothing was graded)")
        elif pct < GATE["executable_pct_min"]:
            failures.append(f"executable_pct {pct} < {GATE['executable_pct_min']}")
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DocsThatRun eval harness")
    parser.add_argument("--answers", action="store_true", help="run LLM + sandbox layers")
    parser.add_argument("--gate", action="store_true", help="exit non-zero on threshold miss")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="retrieval depth, 1-50 (default 5). recall@3 and recall@5 are fixed "
        "at those depths regardless; this changes MRR and how much context the "
        "answer layer sees, so compare runs at the same value.",
    )
    parser.add_argument("--client", default=None, choices=CLIENT_NAMES, help="anthropic | mock | auto")
    parser.add_argument("--json", default=None, help="write full report to this path")
    args = parser.parse_args(argv)
    # Mirror the CLI/API bound: 0 retrieves nothing and a negative value slices
    # off the *top* results — both silently corrupt every metric downstream.
    if not (1 <= args.top_k <= 50):
        parser.error("--top-k must be between 1 and 50")

    report = evaluate(run_answers=args.answers, top_k=args.top_k, client_name=args.client)

    ret = report["retrieval"]
    print("=" * 60)
    print("DocsThatRun eval report")
    print("=" * 60)
    print(f"corpus={report['corpus_size']} golden={report['golden_size']} unanswerable={report['unanswerable_size']}")
    print(f"retrieval: recall@3={ret['recall_at_3']}  recall@5={ret['recall_at_5']}  mrr={ret['mrr']}")
    if report.get("answers"):
        a = report["answers"]
        by_version = "  ".join(
            f"{v}={'up' if up else 'DOWN'}" for v, up in sorted(report["sandbox_by_version"].items())
        )
        print(f"client={report['client']}  sandbox: {by_version}")
        print(
            f"answers: executable%={a['executable_pct']} (n={a['answered_count']} graded)  "
            f"unanswerable_abstention={a['unanswerable_abstention']}  "
            f"answerable_over_abstention={a['answerable_over_abstention']}"
        )
        if a["ungraded_count"]:
            # Never let ungraded items disappear into a percentage.
            print(f"WARNING: {a['ungraded_count']} answerable item(s) went ungraded (sandbox down for their version)")
        if a.get("taxonomy"):
            tax = "  ".join(f"{k}={v}" for k, v in sorted(a["taxonomy"].items()))
            print(f"taxonomy: {tax}")
        for label, key in (("answer", "answer_latency_ms"), ("grade", "grade_latency_ms")):
            lt = a.get(key)
            if lt:
                print(f"{label} latency(ms): mean={lt['mean']}  p50={lt['p50']}  p95={lt['p95']}  max={lt['max']}")
        print(f"note: {a['note']}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}")

    if args.gate:
        failures = check_gate(report)
        if failures:
            print("\nGATE FAILED:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
