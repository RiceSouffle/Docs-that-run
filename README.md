# DocsThatRun

**Version-aware documentation RAG that grades its answers by running them.**

Most docs assistants answer from whatever they retrieved and hope the code is
right. DocsThatRun answers questions about a *specific* version of a
fast-moving library (Pydantic **v1** vs **v2**), cites the docs it used, refuses
when the docs don't cover the question — and then **executes the generated code
against the pinned version of the library in an isolated sandbox** and scores it
pass/fail.

Because Pydantic v2 removed several v1 names outright (their imports raise), the
execution check *is* the version-correctness check: a v2-flavoured answer run
against the v1 sandbox fails, and vice-versa.

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
 evals + CI gate  (recall@k, MRR, executable-%, abstention, version-lock)
```

## Why this is interesting

- **Execution-graded, not vibes-graded.** The snippet has to actually run
  against the version it claims to target.
- **Version drift is handled and measured.** A v3 answer never reaches a v2
  query; a v1 answer that used a removed API fails the sandbox.
- **Honest abstention.** Out-of-corpus questions are refused, not hallucinated.
- **Runs on the standard library.** Retrieval, the sandbox grader, and the eval
  harness have **zero pip dependencies** — clone and run the evals immediately.

## Quickstart

```bash
# 1. Retrieval metrics — no install, no network, no API key:
python3 -m docsthatrun.evals.run_evals

# 2. Build the two pinned-version sandboxes (pydantic 1.x and 2.x), then
#    run the full eval incl. real execution grading (offline MockClient):
make sandbox
python3 -m docsthatrun.evals.run_evals --answers --gate --client mock

# 3. Real answers from Claude — set a key, then:
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m docsthatrun.evals.run_evals --answers --client anthropic

# 4. Serve it:  POST /ask {"question": "...", "version": "v2"}
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Current numbers (seed corpus)

Measured by `python3 -m docsthatrun.evals.run_evals --answers` on the committed
data (18 doc chunks, 16 answerable golden questions, 6 unanswerable):

| Metric | Value | Notes |
|---|---|---|
| retrieval recall@5 | **1.00** | small, clean seed corpus — see caveat below |
| retrieval MRR | **1.00** | |
| reference snippets executable on target version | **16 / 16** | proves the sandbox + drift mechanism |
| crisply version-locked checks | **10 / 16 (62%)** | fail on the *other* version; the rest are v1 APIs kept as deprecated v2 shims |
| unanswerable abstention | **100%** | out-of-corpus questions refused |
| answerable over-abstention | **0%** | in-corpus questions answered |

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
| execution grader (pinned venvs) | [`docsthatrun/sandbox.py`](docsthatrun/sandbox.py) |
| eval harness + CI gate | [`docsthatrun/evals/run_evals.py`](docsthatrun/evals/run_evals.py) |
| HTTP API | [`app/main.py`](app/main.py) |
| design decisions & tradeoffs | [`DECISIONS.md`](DECISIONS.md) |

See [DECISIONS.md](DECISIONS.md) for why each choice was made (and its honest
limitations), and [ROADMAP.md](ROADMAP.md) for the path from this slice to a
flagship portfolio piece.
