# DocsThatRun

**Version-aware documentation RAG that grades its answers by running them.**

[![evals](https://github.com/RiceSouffle/Docs-that-run/actions/workflows/evals.yml/badge.svg)](https://github.com/RiceSouffle/Docs-that-run/actions/workflows/evals.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![core deps](https://img.shields.io/badge/core%20dependencies-0-brightgreen)
![tests](https://img.shields.io/badge/tests-148%20passing-brightgreen)
![lint](https://img.shields.io/badge/lint-ruff-black)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Most docs assistants answer from whatever they retrieved and hope the code is
right. DocsThatRun answers questions about a *specific* version of a
fast-moving library (Pydantic **v1** vs **v2**), cites the docs it used, refuses
when the docs don't cover the question — and then **executes the generated code
against the pinned version of the library in an isolated sandbox** and scores it
pass/fail.

Because Pydantic v2 removed several v1 names outright (their imports raise), the
execution check *is* the version-correctness check: a v2-flavoured answer run
against the v1 sandbox fails, and vice-versa.

"Pinned" is literal: the sandboxes install `pydantic==1.10.26` and
`pydantic==2.13.4` (see [`scripts/setup_sandbox.sh`](scripts/setup_sandbox.sh)),
not a `>=1.10,<2` range. Every number below is reproducible against those exact
releases; with a range they would drift between builds with no code change and
no signal.

![DocsThatRun compare view — the same question answered for v1 and v2, each graded in its own pinned sandbox: v1 FAILs, v2 PASSes](docs/compare-demo.svg)

> The `/compare` view above: the same question answered for both versions, each
> snippet run in its own pinned sandbox. The v1 answer used `model_dump()` — a v2
> API that doesn't exist in v1 — so the v1 sandbox surfaces the exact
> `AttributeError`. That's the version-lock, proven by execution rather than asserted.

```
question + target version
      │
      ▼
 hybrid retrieval  (BM25 + TF-IDF, fused with RRF, filtered to the target version)
      │
      ▼
 cited answer  (Claude, structured JSON: answer + code + citations + abstained)
      │
      ▼
 execution grade  (run the snippet in the pinned-version venv → pass/fail)
      │
      ▼
 evals + CI gate  (recall@k, MRR, executable-%, abstention, version-lock, failure taxonomy)
```

## Why this is interesting

- **Execution-graded, not vibes-graded.** The snippet has to actually run
  against the version it claims to target.
- **Version drift is handled and measured.** A v2 answer never reaches a v1
  query; a v1 answer that used a removed API fails the sandbox.
- **Honest abstention.** Out-of-corpus questions are refused, not hallucinated.
- **Failure taxonomy.** Every graded answer is bucketed (wrong-version-API /
  malformed-code / wrong-assert / retrieval-miss / pass), so a regression is
  attributable to a stage instead of a mystery.
- **Runs on the standard library.** Retrieval, the sandbox grader, and the eval
  harness have **zero pip dependencies** — clone and run the evals immediately.

## Try it — the interactive demo

```bash
make sandbox                              # build the pinned v1/v2 venvs (once)
pip install fastapi uvicorn               # server only; the core needs nothing
make serve                                # → http://localhost:8000
```

The single-page UI (vanilla JS, no build step) walks the whole pipeline: type a
question, pick **v2 / v1 / Compare both**, and watch retrieval → cited answer →
syntax-highlighted code → a green **PASS** / red **FAIL** badge from the real
sandbox. It works offline with the `MockClient` (no API key); set
`ANTHROPIC_API_KEY` for real Claude answers.

Prefer the terminal?

This runs offline, with no API key — copy and paste it (output abridged only
where marked):

```console
$ python3 -m docsthatrun compare "In Pydantic v2, how do I serialize a model instance to a dictionary?"
· using MockClient (offline; set ANTHROPIC_API_KEY for real Claude answers)
Q: In Pydantic v2, how do I serialize a model instance to a dictionary?  (v1 vs v2)
  The execution check is the version-correctness check: a snippet that used a
  removed API fails the other version's sandbox.

── Pydantic v1 ─────────────────────────────────────────────

  Answer [v1]   ✗ FAIL on v1 sandbox
  See docs c_v2_dump.

    │ from pydantic import BaseModel
    │
    │ class User(BaseModel):
    │     name: str
    │     age: int
    │
    │ u = User(name='Ada', age=36)
    │ assert u.model_dump() == {'name': 'Ada', 'age': 36}

  stderr:
    ...                                          # traceback trimmed
    AttributeError: 'User' object has no attribute 'model_dump'

── Pydantic v2 ─────────────────────────────────────────────

  Answer [v2]   ✓ PASS on v2 sandbox
  See docs c_v2_dump.
  cited: c_v2_dump

    │ from pydantic import BaseModel
    │
    │ class User(BaseModel):
    │     name: str
    │     age: int
    │
    │ u = User(name='Ada', age=36)
    │ assert u.model_dump() == {'name': 'Ada', 'age': 36}
```

Same snippet, both sandboxes. `model_dump()` does not exist in Pydantic 1.10, so
the v1 run raises — and `c_v2_dump` appears under `cited:` on the v2 side only,
because a v2 chunk is never retrieved for a v1 query.

Or run it fully containerized (API + both sandboxes baked in):

```bash
docker compose up --build                 # → http://localhost:8000
```

## Quickstart (evals & API)

```bash
# 1. Retrieval metrics — no install, no network, no API key:
python3 -m docsthatrun.evals.run_evals

# 2. Build the two pinned-version sandboxes, then run the full eval incl. real
#    execution grading (offline MockClient):
make sandbox
python3 -m docsthatrun.evals.run_evals --answers --gate --client mock

# 3. Real answers from Claude — set a key, then:
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m docsthatrun.evals.run_evals --answers --client anthropic

# 4. Serve the API + demo UI:
pip install -r requirements.txt
uvicorn app.main:app --reload        # / (UI) · POST /ask,/compare · /health,/ready,/metrics,/stats
```

## Running it in production

The core stays stdlib-only, but the service layer is built to be operated, not
just demoed. All of it is env-driven (see [`docsthatrun/config.py`](docsthatrun/config.py)):

- **Sandboxed execution with resource limits.** The grader runs each snippet in
  its own process group under `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_FSIZE`
  (and `RLIMIT_CORE=0`), so an infinite loop, a memory bomb, a disk-fill, or a
  fork that outlives the timeout is contained — not just the happy path. Output
  is captured to files rather than pipes, so a snippet printing gigabytes is
  bounded by the kernel instead of growing the server's heap.
- **Structured JSON logs** with a per-request id, method, path, status, and
  latency; **Prometheus metrics** at `/metrics` (request counts, latencies,
  grade pass/fail, cache hit-rate) and a human-readable `/stats`.
- **Answer cache** (LRU + TTL) — repeat queries skip the subprocess and return
  in ~1 ms; responses carry `meta.cached`.
- **Per-IP token-bucket rate limiting** on the expensive endpoints, with
  `Retry-After` on 429. Behind a reverse proxy, set `DOCSTHATRUN_TRUST_PROXY=1`
  to key on `X-Forwarded-For` (off by default — the header is spoofable
  without a proxy overwriting it).
- **Hardened HTTP**: typed request/response models (real OpenAPI at `/docs`),
  bounded inputs, `content-security-policy` + `x-frame-options` +
  `x-content-type-options`, a warmed thread-safe retriever/client, and a
  `/ready` readiness probe that returns a real **503** when the corpus is
  missing (probes act on status codes, not JSON bodies).
- **Container**: [`Dockerfile`](Dockerfile) runs as a non-root user with a
  `HEALTHCHECK`; `ruff` lint gates CI alongside the eval gate.

```bash
docker compose up --build            # API + both sandboxes, non-root, health-checked
```

See [DECISIONS.md](DECISIONS.md) → *Production hardening* for why each piece is
in-process stdlib (and its upgrade path to Redis / OpenTelemetry).

## Current numbers (seed corpus)

Measured by `python3 -m docsthatrun.evals.run_evals --answers` on the committed
data (27 doc chunks, 25 answerable golden questions, 6 unanswerable):

| Metric | Value | Notes |
|---|---|---|
| retrieval recall@5 | **1.00** | small, clean seed corpus — see caveat below |
| retrieval MRR | **0.98** | one item's relevant chunk isn't rank-1 on the larger corpus |
| reference snippets executable on target version | **25 / 25** | proves the sandbox + drift mechanism |
| crisply version-locked checks | **17 / 25 (68%)** | fail on the *other* version; the rest are v1 APIs kept as deprecated v2 shims |
| unanswerable abstention | **100%** | out-of-corpus questions refused |
| answerable over-abstention | **0%** | in-corpus questions answered |

Every figure above is pinned by a test, so it cannot drift out from under this
table unnoticed — the version-lock rate is asserted as exactly 17/25 rather than
as a floor, which is only meaningful because the sandbox venvs install exact
pydantic releases.

> **Honest caveat:** these are seed-corpus numbers. recall@5 = 1.0 reflects a
> small, hand-curated corpus with clean version separation, *not* messy
> real-world docs. The `MockClient` used in CI replays the golden answer key, so
> its executable-% is a **plumbing** check — the real measurement comes from
> running with `--client anthropic`. Scaling to the real messy corpus and
> reporting Claude's true executable-% is the next milestone
> ([ROADMAP.md](ROADMAP.md)).

## How the pieces map to files

| Concern | File |
|---|---|
| version-tagged corpus | [`data/corpus/pydantic_corpus.jsonl`](data/corpus/pydantic_corpus.jsonl) |
| hand-labeled golden set | [`data/golden/golden_set.jsonl`](data/golden/golden_set.jsonl) |
| hybrid retrieval + version filter | [`docsthatrun/retrieve.py`](docsthatrun/retrieve.py) |
| cited/abstaining answer via Claude | [`docsthatrun/llm.py`](docsthatrun/llm.py) |
| execution grader (pinned venvs, process-group isolation) | [`docsthatrun/sandbox.py`](docsthatrun/sandbox.py) |
| eval harness + CI gate + failure taxonomy | [`docsthatrun/evals/run_evals.py`](docsthatrun/evals/run_evals.py) |
| HTTP API (lifespan, middleware, models) | [`app/main.py`](app/main.py) |
| env-driven config · cache · rate limit · logging+metrics | [`config.py`](docsthatrun/config.py) · [`cache.py`](docsthatrun/cache.py) · [`ratelimit.py`](docsthatrun/ratelimit.py) · [`observability.py`](docsthatrun/observability.py) |
| interactive demo UI (instrument aesthetic) | [`app/static/index.html`](app/static/index.html) |
| terminal CLI (`ask` / `compare`) | [`docsthatrun/cli.py`](docsthatrun/cli.py) |
| container image (non-root, healthcheck) | [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) |
| design decisions & tradeoffs | [`DECISIONS.md`](DECISIONS.md) |
| full explainer, from zero knowledge | [`GUIDE.md`](GUIDE.md) |

New to any of this? **[GUIDE.md](GUIDE.md) explains the whole project from
zero** — the problem, every concept it rests on (RAG, BM25, RRF, execution
grading, sandboxing, recall/MRR, CI gates), a step-by-step trace of one question
through the system, and a frank account of what the numbers do and do not prove.

See [DECISIONS.md](DECISIONS.md) for why each choice was made (and its honest
limitations), and [ROADMAP.md](ROADMAP.md) for the path from this slice to a
flagship portfolio piece.

## License

MIT — see [LICENSE](LICENSE). The bundled fonts (JetBrains Mono, IBM Plex Sans)
are third-party assets under the SIL Open Font License and keep their own terms;
see [`app/static/fonts/NOTICE.txt`](app/static/fonts/NOTICE.txt).
