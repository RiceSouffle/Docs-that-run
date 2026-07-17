"""FastAPI service for DocsThatRun.

    uvicorn app.main:app --reload

Endpoints
---------
GET  /                 the interactive demo UI (single-page app, no build step)
GET  /health           liveness + which LLM client and whether the sandbox is up
GET  /examples         sample questions (answerable + unanswerable) for the UI
POST /ask              {"question": ..., "version": "v1"|"v2", "execute": true}
POST /compare          {"question": ...}  -> answers for BOTH versions side by side

`/ask` returns the cited answer plus, if requested and the sandbox is set up, the
pass/fail result of executing the generated snippet against that version.
`/compare` is the version-lock showcase: the same question answered for v1 and v2,
each graded against its own pinned sandbox.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field  # FastAPI request model (v2, app-side only)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The API server needs fastapi + uvicorn: pip install -r requirements.txt"
    ) from exc

from docsthatrun.answer import AnswerResult, build_answer
from docsthatrun.corpus import load_corpus
from docsthatrun.llm import get_client
from docsthatrun.retrieve import HybridRetriever
from docsthatrun.sandbox import sandbox_available
from docsthatrun.schema import VERSIONS

app = FastAPI(title="DocsThatRun", version="0.2.0")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_retriever: Optional[HybridRetriever] = None
_client = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(load_corpus())
    return _retriever


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def _graded(result: AnswerResult, execute: bool) -> dict:
    """Grade the snippet against the sandbox when asked, then serialize."""
    if (
        execute
        and result.answer.code
        and not result.answer.abstained
        and sandbox_available(result.version)
    ):
        result.execution_grade()
    return result.to_dict()


class AskRequest(BaseModel):
    question: str
    version: str = "v2"
    execute: bool = True
    # Bounded: top_k <= 0 would silently retrieve nothing (or, negative, drop the
    # top chunks via ordered[:-k]) and yield a wrong/abstained answer with no error.
    top_k: int = Field(5, ge=1, le=50)


class CompareRequest(BaseModel):
    question: str
    execute: bool = True
    top_k: int = Field(5, ge=1, le=50)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = os.path.join(_STATIC_DIR, "index.html")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:  # pragma: no cover - static asset missing
        raise HTTPException(status_code=500, detail="UI asset not found")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "client": type(_get_client()).__name__,
        "sandbox": {v: sandbox_available(v) for v in VERSIONS},
    }


@app.get("/examples")
def examples() -> dict:
    """A few sample questions the UI can offer as one-click demos.

    Loaded lazily from the golden/unanswerable sets so they stay in sync with
    the committed data.
    """
    from docsthatrun.evals.run_evals import load_golden, load_unanswerable

    answerable = [
        {"question": item.question, "version": item.version}
        for item in load_golden()
    ]
    unanswerable = [
        {"question": item.question, "version": item.version}
        for item in load_unanswerable()
    ]
    return {"answerable": answerable, "unanswerable": unanswerable}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.version not in VERSIONS:
        raise HTTPException(status_code=400, detail="version must be 'v1' or 'v2'")
    try:
        result = build_answer(
            req.question,
            req.version,
            _get_retriever(),
            client=_get_client(),
            top_k=req.top_k,
        )
    except Exception as exc:  # upstream LLM / parse failure -> clean 502, not a 500
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}")
    return _graded(result, req.execute)


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    """Answer the same question for BOTH versions — the version-lock showcase.

    A v2-flavoured answer run against the v1 sandbox fails, and vice-versa; this
    endpoint surfaces that contrast in a single call for the UI's compare view.
    """
    results = {}
    try:
        for version in VERSIONS:
            results[version] = build_answer(
                req.question,
                version,
                _get_retriever(),
                client=_get_client(),
                top_k=req.top_k,
            )
    except Exception as exc:  # upstream LLM / parse failure -> clean 502, not a 500
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}")
    # Grade outside the try (as /ask does), so a grader-side error isn't
    # mislabeled "answer generation failed".
    versions = {v: _graded(r, req.execute) for v, r in results.items()}
    return {"question": req.question, "versions": versions}
