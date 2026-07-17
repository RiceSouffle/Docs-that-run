# Design decisions

Short record of *why* each choice was made and what it trades off. Written so a
reviewer can interrogate any layer.

## Why Pydantic v1 → v2

The differentiator lives or dies on how many "how do I X" questions reduce to a
**self-contained, runnable, assertable** snippet. Pydantic is close to ideal:
almost every answer is 5–10 lines (`model_dump`, `field_validator`, `Field`,
`Optional` defaults), and the v1→v2 migration is the canonical version-drift
story. It's also widely known, so it reads instantly in an interview.

## Why execution grading instead of an LLM judge (for code answers)

The most common way to grade a RAG answer is an LLM-as-judge — which then needs
its *own* validation against human labels or it's just noise. For **code**
answers we can do something stronger and cheaper: run the code. A snippet that
imports a removed API or asserts the wrong behavior fails deterministically. No
judge to calibrate.

The nuance, measured and reported honestly: **17/25 (68%)** of the golden checks
are *crisply* version-locked (they fail on the other version). The rest aren't,
because v2 kept some v1 methods as **deprecated shims** (`.dict()`, `.json()`,
`.schema()` still run in v2). So execution grading catches "v2 answer on v1"
crisply (removed names raise), but "v1 answer on v2" only when the API was truly
removed (`BaseSettings` moved packages; `Optional` without a default became
required; `field_validator`/`ConfigDict` don't exist in v1). This is a real
finding, not a bug — and it's exactly the kind of failure-mode nuance the
sandbox surfaces.

Answers that *don't* reduce to a runnable check (conceptual "why" questions) are
out of scope for v0 and are the natural place a **human-validated LLM judge**
gets added next (see ROADMAP).

## Why pure-Python BM25 + TF-IDF (and not a vector DB / embeddings)

The default retriever has **zero dependencies** so the whole eval loop runs on a
fresh clone with no install and no network. The honest tradeoff: BM25 and
TF-IDF are *both lexical* — this is a strong baseline, not a true dense+sparse
hybrid. The architecture is built for the upgrade: `retrieve.py` fuses two
ranked channels with RRF, so dropping in a real sentence-transformer / Voyage /
OpenAI embedder as the second channel is a one-class change and RRF then fuses
genuine lexical + semantic signal. Documented rather than hidden.

## Why a version filter *before* fusion

A query is tagged with its target version; retrieval only considers chunks whose
version is that target or `both`. This is what makes "a v2 answer never leaks
into a v1 query" true at the retrieval layer, before generation. Enforced by a
test (`test_version_filter_excludes_other_version`).

## Why the corpus and golden set are hand-authored (and small)

The critique that kills projects like this is a thin or fabricated golden set.
So the seed set is **hand-written and every reference snippet is verified to run
on its target version** (asserted in CI). It's deliberately small (27 chunks / 25
questions) and honest about it: recall@5 = 1.0 reflects clean version separation
on a curated corpus, not messy real docs. Growing the corpus from real Pydantic
docs + GitHub issues (where version tagging is genuinely hard) is the headline
next step, not a footnote.

## Why a MockClient exists

So the answer → sandbox → eval → CI pipeline runs with **no API key** (CI, or an
offline laptop). It replays the golden answer key, so its executable-% is a
*plumbing* signal, clearly labeled as such in the eval report and README — never
presented as a quality number. Real quality comes from `--client anthropic`.

## Why Claude with structured JSON output

The answer contract (`answer` / `code` / `citations` / `abstained`) is enforced
with `output_config.format` (JSON schema), so parsing never guesses. Citations
are cross-checked against the retrieved ids and hallucinated ids are dropped
(`answer._coerce`). Model defaults to `claude-opus-4-8` with adaptive thinking;
overridable via `DOCSTHATRUN_MODEL` / `DOCSTHATRUN_EFFORT`.

## Robustness & hardening

A pass over the code (adversarially verified — each candidate bug had to survive
a "try to refute this" review before it counted) surfaced a handful of real
defects, now fixed and pinned by regression tests:

- **The sandbox now bounds *total* execution, not just the direct child.**
  `subprocess.run(..., timeout=)` only SIGKILLs the immediate process on timeout,
  so a snippet that spawns a grandchild (`subprocess.Popen`, a fork) could orphan
  it and escape the timeout. The grader now runs each snippet in its own session
  (`start_new_session=True`) and `os.killpg`s the whole group on timeout. A
  regression test spawns a grandchild and asserts it's dead after the deadline.
- **The answer path degrades to an abstain instead of crashing.** The Anthropic
  client shares its `max_tokens` budget between adaptive thinking and the JSON
  answer, so a hard question can return `stop_reason == "max_tokens"` with a
  truncated (or empty) body. Feeding that to `json.loads` used to 500 the
  request. Parsing now tolerates truncated / empty / fenced / non-JSON output and
  falls back to a clean abstain, surfacing the stop reason for observability.
- **`sandbox_available` checks that pydantic actually imports**, not just that
  `bin/python` exists — an interrupted `setup_sandbox.sh` leaves a venv with no
  package, which would otherwise mislabel every answer as a quality failure.
- **The CI gate no longer silently passes when nothing is gradable.** If the
  sandbox is up but every answer abstained or produced no code, `executable_pct`
  is `None`; the old guard skipped the check, so a total collapse looked like a
  pass. It's now an explicit gate failure.
- **API inputs are bounded** (`top_k` constrained; unknown versions 400) and
  generation failures return a clean `502` rather than leaking a stack trace.

None of these are exotic — they're the ordinary edges (timeouts, truncated model
output, half-built environments, empty result sets) that a demo skips and a real
system can't. Surfacing and fixing them, with tests, is the point.

## Production hardening

The core is deliberately stdlib-only, but the *service* is built to be operated.
The theme: do the real thing in-process with the standard library, with a clear
seam to swap in heavier infra when a second instance actually exists.

- **Sandbox resource limits.** The grader now runs each snippet under
  `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_FSIZE` / `RLIMIT_CORE=0`. The limits are
  applied *inside* the child (a small launcher that self-`setrlimit`s, then
  `runpy`s the target) rather than via `preexec_fn` — `preexec_fn` runs after
  `fork` in a possibly-threaded server and can deadlock, whereas the launcher
  runs single-threaded after `exec`. `RLIMIT_AS` is skipped on macOS (where it's
  unreliable) and generous on Linux so a legitimate pydantic import never
  false-fails. This closes the ROADMAP's headline limitation: the sandbox now
  bounds CPU, memory, disk, *and* process lifetime — not just the happy path.
  It is still a venv, not a container; running genuinely untrusted input at scale
  would want gVisor/a microVM on top, and the limits are the defence-in-depth
  layer beneath that.
- **Config is one env-driven dataclass** (`config.py`), not scattered
  `os.environ` reads, so behaviour is 12-factor and every knob is discoverable.
- **Observability is stdlib.** JSON logs with a request id (`observability.py`),
  and a Prometheus text endpoint built by hand rather than pulling in
  `prometheus_client`. The interface is the same as a real exporter; the upgrade
  path is a one-file change.
- **Cache and rate-limiter are in-process** (`cache.py`, `ratelimit.py`) — an
  LRU+TTL memo and a token bucket. Correct for one instance; both hide behind an
  interface a Redis backend can implement for a fleet. The cache is the seed of
  the ROADMAP's semantic cache.
- **Thread-safe warmed singletons.** The retriever and client are built once
  under a lock (double-checked) and warmed in the FastAPI lifespan, so there's no
  init race and the first request isn't slow.
- **Non-root container + healthcheck.** Because the sandbox runs model-authored
  code, the server process must not be root; the image drops to an unprivileged
  user and ships a `HEALTHCHECK`.

## Gate thresholds

`GATE` in `run_evals.py` is the noise floor the committed data must clear
(recall@5 ≥ 0.80, MRR ≥ 0.60, unanswerable abstention ≥ 0.80, over-abstention
≤ 0.20, executable-% ≥ 0.60 when the sandbox is up). They're intentionally below
the current numbers so a real regression trips the gate. Tighten as the corpus
grows.
