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
from typing import List, Optional

from ..answer import build_answer
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
    for item in golden:
        res = build_answer(item.question, item.version, retriever, client=client, top_k=top_k)
        row = {"id": item.id, "abstained": res.answer.abstained}
        if res.answer.abstained:
            answerable_over_abstain += 1
        elif res.answer.code and sandbox_up:
            exec_res = res.execution_grade()
            gradable += 1
            if exec_res.passed:
                executable_hits += 1
            row["executed"] = exec_res.passed
            row["reason"] = exec_res.reason
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
        if report.get("sandbox_available") and answers["executable_pct"] is not None:
            if answers["executable_pct"] < GATE["executable_pct_min"]:
                failures.append(
                    f"executable_pct {answers['executable_pct']} < {GATE['executable_pct_min']}"
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
