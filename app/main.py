"""FastAPI service for DocsThatRun.

    uvicorn app.main:app --reload

POST /ask {"question": "...", "version": "v1"|"v2", "execute": true}
returns the cited answer plus, if requested and the sandbox is set up, the
pass/fail result of executing the generated snippet against that version.
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel  # FastAPI request model (v2, app-side only)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The API server needs fastapi + uvicorn: pip install -r requirements.txt"
    ) from exc

from docsthatrun.answer import build_answer
from docsthatrun.corpus import load_corpus
from docsthatrun.llm import get_client
from docsthatrun.retrieve import HybridRetriever

app = FastAPI(title="DocsThatRun", version="0.1.0")

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


class AskRequest(BaseModel):
    question: str
    version: str = "v2"
    execute: bool = True
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "client": type(_get_client()).__name__}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.version not in ("v1", "v2"):
        raise HTTPException(status_code=400, detail="version must be 'v1' or 'v2'")
    result = build_answer(
        req.question, req.version, _get_retriever(), client=_get_client(), top_k=req.top_k
    )
    if req.execute and result.answer.code and not result.answer.abstained:
        result.execution_grade()
    return result.to_dict()
