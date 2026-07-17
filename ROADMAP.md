# Roadmap — from working slice to flagship

This repo is a **working end-to-end slice**: real hybrid retrieval, real
execution grading against pinned versions, real evals with a CI gate. What
turns it into a portfolio anchor is depth and *honest real-world numbers*. The
order below is deliberate — each step closes a specific gap a skeptical
interviewer would probe.

### Shipped since v0
- **Interactive demo UI** (`app/static/index.html`) + **terminal CLI**
  (`docsthatrun ask` / `compare`) + **`/compare` endpoint** — the version-lock is
  now something you can see, not just read about.
- **Failure taxonomy + per-query latency** in the eval report (part of Milestone 1).
- **Containerized** (`Dockerfile` / `docker-compose.yml`): API + both sandboxes
  in one image, works offline.
- **Robustness pass** (see DECISIONS.md → Robustness & hardening): sandbox
  process-group isolation, graceful degradation on truncated/empty model output,
  a stricter CI gate, bounded API inputs — all pinned by regression tests.

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

### Milestone 5 — Operate it
- Deploy the FastAPI service (Docker + a cloud host); wire Langfuse/OpenTelemetry
  tracing with per-query token/latency/cost capture.
- Add a semantic cache in front and publish the cost-vs-quality tradeoff.
- Drift job: on each new library release, re-ingest and emit a drift-regression
  report.

### Milestone 6 — Tell the story
- A `DECISIONS.md`-style blog post: the version-lock finding (62% crisp), the
  honest recall drop on real docs, and the execution-grading design. This is the
  artifact that gets read in an interview.

## Known limitations (today)
- Corpus is small and clean; retrieval numbers reflect that.
- Default retrieval is lexical-only (BM25 + TF-IDF), not true dense hybrid.
- MockClient executable-% is a plumbing check, not a quality metric.
- Only code answers are execution-graded; conceptual answers aren't graded yet.
- Sandbox runs snippets in a venv, not a locked-down container — fine for
  trusted golden checks and self-authored answers; would need real isolation
  before running untrusted input.
