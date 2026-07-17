"""FastAPI service for DocsThatRun.

    uvicorn app.main:app --reload

Production concerns handled here: env-driven config, structured JSON access logs
with a request id, a thread-safe warmed retriever/client, an answer cache, per-IP
rate limiting, security headers, and Prometheus metrics.

Endpoints
---------
GET  /                the interactive demo UI (single-page app)
GET  /health          liveness + client + sandbox status
GET  /ready           readiness (corpus loaded, sandbox usable)
GET  /metrics         Prometheus text exposition
GET  /stats           JSON metrics snapshot (human-friendly)
GET  /examples        sample questions for the UI
POST /ask             {"question","version","execute","top_k"} -> graded answer
POST /compare         {"question"} -> answers for BOTH versions (the version-lock)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The API server needs fastapi + uvicorn: pip install -r requirements.txt"
    ) from exc

from docsthatrun.answer import AnswerResult, build_answer
from docsthatrun.cache import TTLCache
from docsthatrun.config import settings
from docsthatrun.corpus import load_corpus
from docsthatrun.llm import get_client
from docsthatrun.observability import Metrics, configure_logging
from docsthatrun.ratelimit import RateLimiter
from docsthatrun.retrieve import HybridRetriever
from docsthatrun.sandbox import sandbox_available
from docsthatrun.schema import VERSIONS

configure_logging(settings.log_level, settings.log_json)
log = logging.getLogger("docsthatrun.api")

APP_VERSION = "0.3.0"

metrics = Metrics()
answer_cache = TTLCache(settings.cache_max, settings.cache_ttl_s)
limiter = RateLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ---- thread-safe, warmed singletons ---------------------------------------
_lock = threading.Lock()
_retriever: Optional[HybridRetriever] = None
_client = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:  # double-checked: build exactly once even under concurrency
            if _retriever is None:
                _retriever = HybridRetriever(load_corpus())
    return _retriever


def get_llm():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = get_client()
    return _client


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    # Warm the retriever + client at startup so the first request isn't slow and
    # any config/corpus error surfaces on boot, not mid-request.
    get_retriever()
    client = get_llm()
    log.info(
        "startup",
        extra={
            "client": type(client).__name__,
            "sandbox": {v: sandbox_available(v) for v in VERSIONS},
            "cache_max": settings.cache_max,
            "rate_rpm": settings.rate_limit_rpm,
        },
    )
    yield


app = FastAPI(title="DocsThatRun", version=APP_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

if settings.cors_origins:  # opt-in; same-origin UI needs none
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

_CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; frame-ancestors 'self'"
)


def _security_headers(rid: str) -> Dict[str, str]:
    return {
        "x-request-id": rid,
        "x-content-type-options": "nosniff",
        "x-frame-options": "SAMEORIGIN",
        "referrer-policy": "no-referrer",
        "content-security-policy": _CSP,
    }


def _route_label(request: Request) -> str:
    # The matched route *template* (e.g. "/ask"), not the raw client path, so a
    # flood of distinct URLs (404s) can't blow up metric cardinality.
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def observe(request: Request, call_next):
    """Attach a request id, time the request, log it as JSON, add security headers.

    Security headers and the request id are applied to *every* response —
    including an unhandled-500 built here — not just the happy path.
    """
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # unhandled -> log, count, and return a clean 500 with headers
        latency = round((time.perf_counter() - t0) * 1000, 1)
        log.exception("request_error", extra={"request_id": rid, "path": request.url.path, "latency_ms": latency})
        metrics.record_request(_route_label(request), 500, latency)
        response = JSONResponse(status_code=500, content={"detail": "internal server error"})
        for k, v in _security_headers(rid).items():
            response.headers[k] = v
        return response
    latency = round((time.perf_counter() - t0) * 1000, 1)
    metrics.record_request(_route_label(request), response.status_code, latency)
    log.info(
        "request",
        extra={
            "request_id": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code, "latency_ms": latency,
            "client_ip": request.client.host if request.client else None,
        },
    )
    for k, v in _security_headers(rid).items():
        response.headers[k] = v
    return response


# ---- request / response models --------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)
    version: str = settings.default_version
    execute: bool = True
    top_k: int = Field(settings.top_k_default, ge=1, le=settings.top_k_max)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)
    execute: bool = True
    top_k: int = Field(settings.top_k_default, ge=1, le=settings.top_k_max)


class ExecutionOut(BaseModel):
    passed: bool
    available: bool
    returncode: Optional[int] = None
    reason: str = ""
    stderr_tail: str = ""


class RetrievedOut(BaseModel):
    id: str
    version: str
    topic: str
    title: str
    snippet: str
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    cited: bool


class AnswerOut(BaseModel):
    answer: str
    code: str
    citations: List[str]
    abstained: bool


class MetaOut(BaseModel):
    latency_ms: float
    cached: bool
    client: str


class AskResponse(BaseModel):
    question: str
    version: str
    retrieved_ids: List[str]
    retrieved: List[RetrievedOut]
    answer: AnswerOut
    execution: Optional[ExecutionOut] = None
    meta: MetaOut


class CompareResponse(BaseModel):
    question: str
    versions: Dict[str, AskResponse]


# ---- core answer path (cache + grade + metrics) ---------------------------

def _grade_outcome(graded: dict) -> str:
    if graded["answer"]["abstained"]:
        return "abstain"
    ex = graded.get("execution")
    if ex is None or not ex["available"]:
        return "no_grade"
    return "pass" if ex["passed"] else "fail"


def _answer(question: str, version: str, execute: bool, top_k: int) -> dict:
    key = (question.strip(), version, top_k, execute)
    hit = answer_cache.get(key)
    if hit is not None:
        out = dict(hit)
        out["meta"] = {**hit["meta"], "cached": True}
        return out

    t0 = time.perf_counter()
    result: AnswerResult = build_answer(
        question, version, get_retriever(), client=get_llm(), top_k=top_k
    )
    if (
        execute and result.answer.code and not result.answer.abstained
        and sandbox_available(version)
    ):
        result.execution_grade()
    graded = result.to_dict()
    graded["meta"] = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "cached": False,
        "client": type(get_llm()).__name__,
    }
    metrics.record_grade(_grade_outcome(graded))
    answer_cache.set(key, graded)
    return graded


# ---- routes ----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        with open(os.path.join(_STATIC_DIR, "index.html"), "r", encoding="utf-8") as h:
            return h.read()
    except FileNotFoundError:  # pragma: no cover
        raise HTTPException(status_code=500, detail="UI asset not found") from None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "client": type(get_llm()).__name__,
        "sandbox": {v: sandbox_available(v) for v in VERSIONS},
    }


@app.get("/ready")
def ready() -> dict:
    corpus_ok = len(get_retriever().chunks) > 0
    sandbox = {v: sandbox_available(v) for v in VERSIONS}
    return {"ready": corpus_ok, "corpus": corpus_ok, "sandbox": sandbox}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus() -> str:
    return metrics.render_prometheus(answer_cache.stats())


@app.get("/stats")
def stats() -> dict:
    return metrics.snapshot(answer_cache.stats())


@app.get("/examples")
def examples() -> dict:
    from docsthatrun.evals.run_evals import load_golden, load_unanswerable

    answerable = [
        {"question": i.question, "version": i.version, "answerable": True}
        for i in load_golden()
    ]
    unanswerable = [
        {"question": i.question, "version": i.version, "answerable": False}
        for i in load_unanswerable()
    ]
    return {"answerable": answerable, "unanswerable": unanswerable}


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "anon"
    ok, retry = limiter.allow(ip)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(retry) + 1)},
        )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> dict:
    _rate_limit(request)
    if req.version not in VERSIONS:
        raise HTTPException(status_code=400, detail="version must be 'v1' or 'v2'")
    try:
        return _answer(req.question, req.version, req.execute, req.top_k)
    except HTTPException:
        raise
    except Exception as exc:  # upstream LLM / parse failure -> clean 502
        log.exception("ask_failed", extra={"request_id": getattr(request.state, "request_id", None)})
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}") from exc


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest, request: Request) -> dict:
    """Answer the same question for BOTH versions — the version-lock showcase."""
    _rate_limit(request)
    try:
        versions = {
            v: _answer(req.question, v, req.execute, req.top_k) for v in VERSIONS
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("compare_failed", extra={"request_id": getattr(request.state, "request_id", None)})
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}") from exc
    return {"question": req.question, "versions": versions}
