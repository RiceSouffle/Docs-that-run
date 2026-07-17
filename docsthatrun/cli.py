"""Command-line interface for DocsThatRun.

    python -m docsthatrun ask "In Pydantic v2, how do I serialize a model?" --version v2
    python -m docsthatrun compare "In Pydantic v2, how do I serialize a model?"

`ask` retrieves version-filtered docs, writes a cited answer, and (unless
--no-execute) runs the generated snippet in the pinned-version sandbox, printing
a pass/fail grade. `compare` answers the same question for BOTH versions side by
side — the version-lock in a single view.

Runs offline with the MockClient (no API key). Set ANTHROPIC_API_KEY (or
DOCSTHATRUN_LLM=anthropic) for real Claude answers.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .answer import AnswerResult, build_answer
from .corpus import load_corpus
from .llm import get_client
from .retrieve import HybridRetriever
from .sandbox import sandbox_available
from .schema import VERSIONS


# ---- tiny ANSI helper (no dependencies) ------------------------------------

class _C:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s):  return self._w("1", s)
    def dim(self, s):   return self._w("2", s)
    def green(self, s): return self._w("32", s)
    def red(self, s):   return self._w("31", s)
    def yellow(self, s):return self._w("33", s)
    def blue(self, s):  return self._w("34", s)
    def cyan(self, s):  return self._w("36", s)
    def mag(self, s):   return self._w("35", s)


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _vtag(c: _C, version: str) -> str:
    f = {"v1": c.yellow, "v2": c.blue, "both": c.dim}.get(version, c.dim)
    return f(f"[{version}]")


def _render(result: AnswerResult, c: _C, show_retrieval: bool = True) -> None:
    a = result.answer
    print()
    if a.abstained:
        print("  " + c.yellow("⦸ abstained") + "  " + c.dim("(the docs don't cover this)"))
        if a.answer:
            print("  " + c.dim(a.answer))
        return

    # execution badge
    ex = result.execution
    if ex is None:
        badge = c.dim("○ not executed")
    elif not ex.available:
        badge = c.yellow("○ sandbox unavailable")
    elif ex.passed:
        badge = c.green(f"✓ PASS on {result.version} sandbox")
    else:
        badge = c.red(f"✗ FAIL on {result.version} sandbox")
    print("  " + c.bold("Answer ") + _vtag(c, result.version) + "   " + badge)

    if a.answer:
        print("  " + a.answer)
    if a.citations:
        print("  " + c.dim("cited: ") + " ".join(c.cyan(x) for x in a.citations))
    if a.code:
        print()
        for line in a.code.rstrip().splitlines():
            print("    " + c.dim("│ ") + line)
    if ex is not None and ex.available and not ex.passed and ex.stderr:
        print()
        print("  " + c.dim("stderr:"))
        for line in ex.stderr.strip().splitlines()[-6:]:
            print("    " + c.red(line))

    if show_retrieval and result.retrieved:
        print()
        print("  " + c.dim("retrieved (BM25 + TF-IDF, RRF-fused, version-filtered):"))
        for r in result.retrieved:
            mark = c.cyan("●") if r.chunk.id in set(a.citations) else c.dim("○")
            rid = r.chunk.id.ljust(16)
            meta = c.dim(f"score {r.score:.4f}  bm25#{r.bm25_rank}  dense#{r.dense_rank}")
            print(f"    {mark} {_vtag(c, r.chunk.version)} {rid} {meta}")


# ---- commands --------------------------------------------------------------

def _make_answer(question: str, version: str, retriever, client, execute: bool, top_k: int):
    result = build_answer(question, version, retriever, client=client, top_k=top_k)
    if execute and result.answer.code and not result.answer.abstained and sandbox_available(version):
        result.execution_grade()
    return result


def cmd_ask(args, retriever, client, c: _C) -> int:
    if args.version not in VERSIONS:
        print(c.red(f"version must be one of {VERSIONS}"), file=sys.stderr)
        return 2
    print(c.bold("Q: ") + args.question + "  " + _vtag(c, args.version))
    result = _make_answer(args.question, args.version, retriever, client, args.execute, args.top_k)
    _render(result, c)
    print()
    ex = result.execution
    return 0 if (ex is None or not ex.available or ex.passed or result.answer.abstained) else 1


def cmd_compare(args, retriever, client, c: _C) -> int:
    print(c.bold("Q: ") + args.question + "  " + c.dim("(v1 vs v2)"))
    print(c.dim("  The execution check is the version-correctness check: a snippet that used a"))
    print(c.dim("  removed API fails the other version's sandbox."))
    for version in VERSIONS:
        result = _make_answer(args.question, version, retriever, client, args.execute, args.top_k)
        print()
        print(c.bold(f"── Pydantic {version} ".ljust(60, "─")))
        _render(result, c, show_retrieval=False)
    print()
    # compare is a demonstration, not a check: a fail on the *other* version is
    # the expected version-lock outcome, so it always exits 0.
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="docsthatrun",
        description="Version-aware Pydantic docs RAG that grades answers by running them.",
    )
    sub = p.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("question", help="the question to ask")
    common.add_argument("--top-k", type=int, default=5, help="chunks to retrieve (default 5)")
    common.add_argument(
        "--no-execute", dest="execute", action="store_false",
        help="skip running the generated snippet in the sandbox",
    )
    common.add_argument("--client", default=None, help="anthropic | mock | auto (default auto)")

    ask = sub.add_parser("ask", parents=[common], help="answer for one version")
    ask.add_argument("--version", default="v2", help="v1 or v2 (default v2)")

    sub.add_parser("compare", parents=[common], help="answer for BOTH versions, side by side")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0

    c = _C(_color_enabled())

    # Mirror the API's bound: top_k <= 0 would retrieve nothing (or, negative,
    # drop the top chunks via ordered[:-k]) and silently yield a wrong answer.
    if not (1 <= args.top_k <= 50):
        print(c.red("--top-k must be between 1 and 50"), file=sys.stderr)
        return 2

    retriever = HybridRetriever(load_corpus())
    client = get_client(args.client)

    if not (os.environ.get("ANTHROPIC_API_KEY") or args.client == "anthropic"):
        print(c.dim(f"· using {type(client).__name__} (offline; set ANTHROPIC_API_KEY for real Claude answers)"))

    if args.command == "ask":
        return cmd_ask(args, retriever, client, c)
    if args.command == "compare":
        return cmd_compare(args, retriever, client, c)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
