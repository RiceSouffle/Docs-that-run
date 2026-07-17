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
from ..llm import get_client
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
    "executable_pct_min": 0.60,  # only enforced when the sandbox is available
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
    if not res.answer.code:
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
    sandbox_up = sandbox_available("v1") and sandbox_available("v2")
    report["sandbox_available"] = sandbox_up

    executable_hits, gradable = 0, 0
    answerable_over_abstain = 0
    answer_rows = []
    taxonomy: Counter = Counter()
    latencies: List[float] = []
    for item in golden:
        t0 = time.perf_counter()
        res = build_answer(item.question, item.version, retriever, client=client, top_k=top_k)
        if not res.answer.abstained and res.answer.code and sandbox_up:
            res.execution_grade()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        latencies.append(latency_ms)

        outcome = _classify_outcome(item, res)
        taxonomy[outcome] += 1
        if res.answer.abstained:
            answerable_over_abstain += 1
        if res.execution is not None:
            gradable += 1
            if res.execution.passed:
                executable_hits += 1

        row = {
            "id": item.id,
            "outcome": outcome,
            "abstained": res.answer.abstained,
            "latency_ms": latency_ms,
        }
        if res.execution is not None:
            row["executed"] = res.execution.passed
            row["reason"] = res.execution.reason
        answer_rows.append(row)

    abstained_correct = 0
    for item in unanswerable:
        res = build_answer(item.question, item.version, retriever, client=client, top_k=top_k)
        if res.answer.abstained:
            abstained_correct += 1

    report["answers"] = {
        "executable_pct": round(executable_hits / gradable, 3) if gradable else None,
        "gradable_count": gradable,
        "answerable_over_abstention": round(
            answerable_over_abstain / len(golden), 3
        ) if golden else 0.0,
        "unanswerable_abstention": round(
            abstained_correct / len(unanswerable), 3
        ) if unanswerable else 0.0,
        # Failure taxonomy: which stage each answerable item landed in.
        "taxonomy": dict(taxonomy),
        "latency_ms": _latency_stats(latencies),
        "rows": answer_rows,
        "note": (
            "MockClient replays the answer key: executable_pct here is a PLUMBING "
            "check, not a quality claim. Run with DOCSTHATRUN_LLM=anthropic for a "
            "real measurement."
            if type(client).__name__ == "MockClient"
            else "Measured against Claude-generated answers."
        ),
    }
    return report


def check_gate(report: dict) -> List[str]:
    failures: List[str] = []
    ret = report["retrieval"]
    if ret["recall_at_5"] < GATE["recall_at_5"]:
        failures.append(
            f"recall@5 {ret['recall_at_5']} < {GATE['recall_at_5']}"
        )
    if ret["mrr"] < GATE["mrr"]:
        failures.append(f"mrr {ret['mrr']} < {GATE['mrr']}")

    answers = report.get("answers")
    if answers:
        if answers["unanswerable_abstention"] < GATE["unanswerable_abstention"]:
            failures.append(
                "unanswerable_abstention "
                f"{answers['unanswerable_abstention']} < {GATE['unanswerable_abstention']}"
            )
        if answers["answerable_over_abstention"] > GATE["answerable_over_abstention_max"]:
            failures.append(
                "answerable_over_abstention "
                f"{answers['answerable_over_abstention']} > {GATE['answerable_over_abstention_max']}"
            )
        if report.get("sandbox_available"):
            pct = answers["executable_pct"]
            if pct is None:
                # Sandbox is up but nothing was gradable (every answer abstained
                # or produced empty code). That's a regression, not a pass — the
                # old `is not None` guard silently skipped the gate here.
                failures.append(
                    "no gradable answers (gradable_count=0) while the sandbox is up"
                )
            elif pct < GATE["executable_pct_min"]:
                failures.append(
                    f"executable_pct {pct} < {GATE['executable_pct_min']}"
                )
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DocsThatRun eval harness")
    parser.add_argument("--answers", action="store_true", help="run LLM + sandbox layers")
    parser.add_argument("--gate", action="store_true", help="exit non-zero on threshold miss")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--client", default=None, help="anthropic | mock | auto")
    parser.add_argument("--json", default=None, help="write full report to this path")
    args = parser.parse_args(argv)

    report = evaluate(run_answers=args.answers, top_k=args.top_k, client_name=args.client)

    ret = report["retrieval"]
    print("=" * 60)
    print("DocsThatRun eval report")
    print("=" * 60)
    print(f"corpus={report['corpus_size']} golden={report['golden_size']} "
          f"unanswerable={report['unanswerable_size']}")
    print(f"retrieval: recall@3={ret['recall_at_3']}  recall@5={ret['recall_at_5']}  "
          f"mrr={ret['mrr']}")
    if report.get("answers"):
        a = report["answers"]
        print(f"client={report['client']}  sandbox={report['sandbox_available']}")
        print(f"answers: executable%={a['executable_pct']} (n={a['gradable_count']})  "
              f"unanswerable_abstention={a['unanswerable_abstention']}  "
              f"answerable_over_abstention={a['answerable_over_abstention']}")
        if a.get("taxonomy"):
            tax = "  ".join(f"{k}={v}" for k, v in sorted(a["taxonomy"].items()))
            print(f"taxonomy: {tax}")
        if a.get("latency_ms"):
            lt = a["latency_ms"]
            print(f"latency(ms): mean={lt['mean']}  p50={lt['p50']}  p95={lt['p95']}  max={lt['max']}")
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
