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
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The API server needs fastapi + uvicorn: pip install -r requirements.txt") from exc

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

APP_VERSION = "0.3.2"

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

_CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; frame-ancestors 'self'"
)

# FastAPI's built-in /docs and /redoc pages load Swagger UI / ReDoc from
# cdn.jsdelivr.net, which the policy above blocks — leaving a blank page where
# the README promises browsable OpenAPI. Relax it for exactly those two routes
# rather than globally: the demo UI and every API response keep the strict
# policy, and the app itself still makes zero external requests.
_CDN = "https://cdn.jsdelivr.net"
# ReDoc additionally pulls a stylesheet from fonts.googleapis.com and its font
# files from fonts.gstatic.com; without both, /redoc renders with broken
# typography and console errors.
_FONTS_CSS = "https://fonts.googleapis.com"
_FONTS_FILES = "https://fonts.gstatic.com"
_CSP_DOCS = (
    f"default-src 'self'; style-src 'self' 'unsafe-inline' {_CDN} {_FONTS_CSS}; "
    f"script-src 'self' 'unsafe-inline' {_CDN}; "
    f"font-src 'self' {_CDN} {_FONTS_FILES}; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "worker-src 'self' blob:; "  # ReDoc renders in a web worker
    "connect-src 'self'; base-uri 'none'; frame-ancestors 'self'"
)
_DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})


def _security_headers(rid: str, path: str = "") -> Dict[str, str]:
    return {
        "x-request-id": rid,
        "x-content-type-options": "nosniff",
        "x-frame-options": "SAMEORIGIN",
        "referrer-policy": "no-referrer",
        "content-security-policy": _CSP_DOCS if path in _DOCS_PATHS else _CSP,
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
        for k, v in _security_headers(rid, request.url.path).items():
            response.headers[k] = v
        return response
    latency = round((time.perf_counter() - t0) * 1000, 1)
    metrics.record_request(_route_label(request), response.status_code, latency)
    log.info(
        "request",
        extra={
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency,
            "client_ip": request.client.host if request.client else None,
        },
    )
    for k, v in _security_headers(rid, request.url.path).items():
        response.headers[k] = v
    return response


# Registered AFTER `observe` so CORS is the *outermost* layer (Starlette: last
# added wraps everything). If it were inside, the unhandled-500 built in
# `observe` would leave without an Access-Control-Allow-Origin header and a
# cross-origin caller would see an opaque network error instead of a 500.
if settings.cors_origins:  # opt-in; same-origin UI needs none
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


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
    # Normalize once, then use the normalized form everywhere. Keying on the
    # stripped question while answering the raw one meant a cache hit echoed the
    # *first* caller's whitespace back to a later caller — a response quoting a
    # question that caller never sent.
    question = question.strip()
    key = (question, version, top_k, execute)
    hit = answer_cache.get(key)
    if hit is not None:
        out = dict(hit)
        out["meta"] = {**hit["meta"], "cached": True}
        return out

    t0 = time.perf_counter()
    result: AnswerResult = build_answer(question, version, get_retriever(), client=get_llm(), top_k=top_k)
    if execute and result.answer.code and not result.answer.abstained and sandbox_available(version):
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


_SANDBOX_STATUS_TTL_S = 30.0
_sandbox_status: tuple = (0.0, None)
_sandbox_status_lock = threading.Lock()


def _sandbox_status_cached() -> Dict[str, bool]:
    """Sandbox status for the *unauthenticated, un-rate-limited* probes.

    ``sandbox_available`` caches success only, on purpose (a probe racing
    `make sandbox` must not disable grading for the process lifetime). But that
    means a venv whose install never finished re-forks a probe subprocess on
    every call, and /health and /ready are public and unmetered — a plain GET
    flood becomes one process fork per request, and since these are sync routes
    each blocked probe holds an anyio threadpool worker. Memoize for 30 s under
    a lock: at most one probe per version per TTL no matter the request rate,
    and grading still self-heals within a TTL of the install finishing.
    """
    global _sandbox_status
    ts, val = _sandbox_status
    if val is not None and time.monotonic() - ts < _SANDBOX_STATUS_TTL_S:
        return val
    # Non-blocking: if another request is already probing, serve the last known
    # value instead of queueing behind it. Waiting would still park a threadpool
    # worker for the probe's 15 s timeout, which is the thing being avoided.
    if not _sandbox_status_lock.acquire(blocking=False):
        return val if val is not None else {v: False for v in VERSIONS}
    try:
        ts, val = _sandbox_status
        if val is not None and time.monotonic() - ts < _SANDBOX_STATUS_TTL_S:
            return val
        fresh = {v: sandbox_available(v) for v in VERSIONS}
        _sandbox_status = (time.monotonic(), fresh)
        return fresh
    finally:
        _sandbox_status_lock.release()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "client": type(get_llm()).__name__,
        "sandbox": _sandbox_status_cached(),
    }


@app.get("/ready")
def ready(response: Response) -> dict:
    corpus_ok = len(get_retriever().chunks) > 0
    sandbox = _sandbox_status_cached()
    if not corpus_ok:
        # Probes (k8s readinessProbe, LB health checks) act on the status code,
        # not the body — "ready": false inside a 200 would still get traffic.
        response.status_code = 503
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

    answerable = [{"question": i.question, "version": i.version, "answerable": True} for i in load_golden()]
    unanswerable = [{"question": i.question, "version": i.version, "answerable": False} for i in load_unanswerable()]
    return {"answerable": answerable, "unanswerable": unanswerable}


def _client_key(request: Request) -> str:
    """Rate-limit key: the direct TCP peer, unless DOCSTHATRUN_TRUST_PROXY is
    set — behind a reverse proxy every client arrives from the proxy's IP, so
    without the header they'd all share one bucket (one user's burst 429s
    everyone). Off by default because X-Forwarded-For is client-spoofable when
    no trusted proxy is overwriting it."""
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "anon"


def _rate_limit(request: Request) -> None:
    ok, retry = limiter.allow(_client_key(request))
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
        versions = {v: _answer(req.question, v, req.execute, req.top_k) for v in VERSIONS}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("compare_failed", extra={"request_id": getattr(request.state, "request_id", None)})
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}") from exc
    return {"question": req.question, "versions": versions}
