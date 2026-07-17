# Roadmap — from working slice to flagship

This repo is a **working end-to-end slice**: real hybrid retrieval, real
execution grading against pinned versions, real evals with a CI gate. What
turns it into a portfolio anchor is depth and *honest real-world numbers*. The
order below is deliberate — each step closes a specific gap a skeptical
interviewer would probe.

### Shipped since v0
- **Interactive demo UI** (`app/static/index.html`, an instrument/test-bench
  design with self-hosted fonts) + **terminal CLI** (`docsthatrun ask` /
  `compare`) + **`/compare` endpoint** — the version-lock is now something you
  can see, not just read about.
- **Failure taxonomy + per-query latency** in the eval report (part of Milestone 1).
- **Robustness pass** (DECISIONS.md → Robustness & hardening): sandbox
  process-group isolation, graceful degradation on truncated/empty model output,
  a stricter CI gate, bounded API inputs — all pinned by regression tests.
- **Production pass** (DECISIONS.md → Production hardening): sandbox **resource
  limits** (CPU/memory/file/core), env-driven config, structured JSON logs with
  request ids, Prometheus `/metrics` + `/stats`, an LRU+TTL answer cache, per-IP
  rate limiting, security headers, typed request/response models, a warmed
  thread-safe retriever, a **non-root** Docker image with a healthcheck, and a
  `ruff` lint gate in CI. This delivers much of Milestone 5 in-process.

### Milestone 1 — Real Claude measurement (highest leverage)
- Run `--client anthropic` over the golden set and publish Claude's true
  executable-%, abstention, and over-abstention (not the MockClient plumbing
  numbers). Add a `results/` snapshot to the README.
- ✅ Add a per-item failure taxonomy: retrieval miss vs wrong-version API vs
  malformed code vs wrong assert. *(Done — the taxonomy is in the eval report;
  with the MockClient every item is `pass`, so the interesting split appears once
  Claude's real failures are measured.)* Still to do: show one fix that moved the
  number.

### Milestone 2 — Real, messy corpus
- Replace the 18 hand-written chunks with ingested Pydantic docs for one v1 and
  one v2 release: markdown, changelog, and a sample of version-tagged GitHub
  issues / Stack Overflow answers.
- **This is where the project is won or lost:** version-tagging real Q&A is
  genuinely hard (a v1 answer lives on a v4 thread). Document the labeling
  protocol and report inter-version leakage honestly.
- Expect recall@5 to drop from 1.0 — that's the point. Report the real number.

### Milestone 3 — Retrieval depth
- Swap the TF-IDF channel for a real embedding model (sentence-transformers or a
  hosted embedder) as the dense channel; keep BM25 sparse; RRF already fuses.
  Report the recall lift as a leaderboard row.
- Add a cross-encoder reranker; add Anthropic-style contextual retrieval
  (chunk prefixing) as another row.

### Milestone 4 — Grade the non-executable answers
- For conceptual questions that don't reduce to a runnable check, add an
  LLM-as-judge for faithfulness — and **validate the judge against a
  human-labeled subset** (report Cohen's κ). Set the CI threshold from the
  measured judge noise floor, not an arbitrary number.

### Milestone 5 — Operate it  *(mostly shipped — see "Production pass" above)*
- ✅ Deploy shape: non-root Docker image + healthcheck; ✅ structured logs +
  Prometheus metrics (per-query latency). *Still TODO:* a cloud host, and
  token/cost capture via Langfuse/OpenTelemetry (needs the real Anthropic path).
- ✅ In-process LRU+TTL cache. *Still TODO:* a true **semantic** cache
  (embedding-keyed) and the published cost-vs-quality tradeoff.
- *TODO:* drift job — on each new library release, re-ingest and emit a
  drift-regression report.

### Milestone 6 — Tell the story
- A `DECISIONS.md`-style blog post: the version-lock finding (62% crisp), the
  honest recall drop on real docs, and the execution-grading design. This is the
  artifact that gets read in an interview.

## Known limitations (today)
- Corpus is small and clean; retrieval numbers reflect that.
- Default retrieval is lexical-only (BM25 + TF-IDF), not true dense hybrid.
- MockClient executable-% is a plumbing check, not a quality metric.
- Only code answers are execution-graded; conceptual answers aren't graded yet.
- Sandbox now applies CPU/memory/file/core resource limits and runs snippets in
  an isolated process group as a non-root user — but it's still a venv, not a
  locked-down container. Running genuinely untrusted input at scale would want
  gVisor/a microVM on top; the rlimits are the defence-in-depth layer beneath.
