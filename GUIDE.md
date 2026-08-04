# DocsThatRun — a complete guide

DocsThatRun answers questions about a specific version of a Python library, and then
checks its own answer by running it. The generated code is executed inside a virtual
environment holding that exact library version, and the result is a pass or a fail from
the Python interpreter rather than an opinion. The test case is Pydantic v1 versus v2,
because v2 deleted several v1 names outright — so running the code *is* the
version-correctness check.

This document assumes no prior knowledge. Every term is defined before it is used, and
every mechanism is tied to the file that implements it.

---

## Contents

1. [The problem](#1-the-problem)
2. [The idea in one paragraph](#2-the-idea-in-one-paragraph)
3. [Concepts you need](#3-concepts-you-need)
   - [3a. Finding the right documentation](#3a-finding-the-right-documentation)
   - [3b. Writing an answer you can check](#3b-writing-an-answer-you-can-check)
   - [3c. Measuring and operating](#3c-measuring-and-operating)
4. [How one question flows through the system](#4-how-one-question-flows-through-the-system)
5. [The components](#5-the-components)
6. [How it is measured](#6-how-it-is-measured)
7. [Honest limitations](#7-honest-limitations)
8. [Running it yourself](#8-running-it-yourself)
9. [Repository map](#9-repository-map)
10. [Design decisions and why](#10-design-decisions-and-why)
11. [What would come next](#11-what-would-come-next)
12. [Glossary](#12-glossary)

---

## 1. The problem

A **large language model** (LLM) is a program that predicts text. You give it a stretch
of text and it produces what most plausibly comes next, based on statistical patterns
absorbed from a very large pile of documents during **training**. It has no database and
performs no lookup. When it writes `u.model_dump()`, it is not consulting Pydantic — it
is emitting the characters its training made likely.

Training ends on a date. Anything published after that date is invisible to the model.
That date is the **training cutoff**.

Software libraries move. Pydantic is a widely used Python library that validates data
against declared types. Version 2 was a breaking rewrite:

| Task | Pydantic v1 | Pydantic v2 |
|---|---|---|
| model to dictionary | `u.dict()` | `u.model_dump()` |
| field validator | `@validator` | `@field_validator` |
| settings class | `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` |

Two failure modes follow, and the second is the dangerous one.

First, a model trained mostly on v1-era text writes v1 code when you asked about v2.
Ordinary staleness.

Second, **the model does not know it is wrong**. It has no confidence signal tied to
fact. It produces fluent, correctly indented, entirely plausible Python that raises
`AttributeError` the instant it runs. Call this the **plausible-but-wrong** problem: the
output is indistinguishable from a correct answer by inspection, so a reader cannot
filter it by reading. You find out when it breaks.

Neither failure is fixed by instructing the model to be careful. It must be handed the
current facts, or checked against them. This project does both.

## 2. The idea in one paragraph

Tag every piece of documentation with the library version it describes. When a question
arrives for v2, filter the documentation down to v2 material *before* ranking anything,
so the v1 answer is not merely unlikely — it is unreachable. Hand the surviving passages
to the model and instruct it to answer only from them, to cite which passages it used,
and to refuse outright if they do not cover the question. Then take the code it wrote and
execute it inside a virtual environment containing exactly Pydantic v2. If the process
exits 0, it passed. If it used a v1-only name, the interpreter raises and it failed. The
score is measured, not asserted.

```
question + target version (v1 | v2)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL            docsthatrun/retrieve.py                │
│   filter corpus to {version, "both"}   ← the load-bearing   │
│   BM25 ranking          ─┐                       step       │
│   TF-IDF cosine ranking ─┴→ fuse by rank (RRF) → top 5      │
└─────────────────────────────────────────────────────────────┘
      │  5 version-correct chunks
      ▼
┌─────────────────────────────────────────────────────────────┐
│ ANSWER               docsthatrun/llm.py, answer.py          │
│   prompt = rules + target version + question + chunks       │
│   model returns JSON: {answer, code, citations, abstained}  │
│   invented citation ids are dropped                         │
└─────────────────────────────────────────────────────────────┘
      │  a snippet, or an explicit refusal
      ▼
┌─────────────────────────────────────────────────────────────┐
│ EXECUTION GRADE      docsthatrun/sandbox.py                 │
│   write snippet to temp file                                │
│   run in .venvs/pydantic_v{1,2}/bin/python                  │
│   own process group · CPU/memory/file-size limits           │
│   passed = (exit code == 0)                                 │
└─────────────────────────────────────────────────────────────┘
      │  PASS / FAIL + stderr tail
      ▼
┌─────────────────────────────────────────────────────────────┐
│ EVALS + CI GATE      docsthatrun/evals/run_evals.py         │
│   recall@k · MRR · executable-% · abstention · taxonomy     │
│   thresholds missed → exit 1 → red build                    │
└─────────────────────────────────────────────────────────────┘
```

## 3. Concepts you need

### 3a. Finding the right documentation

**Retrieval-augmented generation** (RAG) is the pattern that fixes the staleness half of
the problem. Before asking the model anything, search a trusted set of documents for
passages relevant to the question, paste those passages into the request, and instruct
the model to answer only from them. The model stops being the source of facts and becomes
a rewriter of supplied facts. Its memory of Pydantic's API stops mattering; what matters
is what you handed it. The instruction is literal — rule 1 of the system prompt at
`docsthatrun/llm.py:46` reads *"Answer ONLY using the provided documentation chunks. Do
not use outside knowledge."*

A **corpus** is the fixed set of documents the system may search. A **chunk** is one
document in it, small enough to paste in cheaply. This corpus is
`data/corpus/pydantic_corpus.jsonl` — **27 chunks**, 13 tagged `v2`, 12 tagged `v1`, and
2 tagged `both` for material true of either. **JSONL** means one JSON object per line.
Each record has six string fields; here is line 1, abbreviated:

```json
{"id": "c_both_basemodel", "version": "both", "topic": "model",
 "title": "Defining a model",
 "text": "A model is a class that inherits from pydantic.BaseModel. ...",
 "code": "from pydantic import BaseModel\n\nclass User(BaseModel):\n    name: str\n    age: int"}
```

Only `title`, `text`, and `code` are searchable (`schema.py:29-31`). The `id` and `topic`
are not indexed.

**Tokenization** is cutting text into the units you match on; those units are **tokens**.
The rule is one regular expression at `corpus.py:18` — lowercase the text, then take every
maximal run of letters, digits, and underscores. One extra step matters: if a token
contains an underscore, its parts are appended too.

```
tokenize("u.model_dump()")  ->  ['u', 'model_dump', 'model', 'dump']
```

That is why the question "convert a model to a dictionary" can reach a chunk whose code
says `model_dump()`. There is no stemming and no stopword list.

**Lexical search** ranks documents by the literal words they share with the query — no
meaning is involved. **BM25** is the standard formula for it, and three intuitions drive
it:

- **Term frequency** (`f`): a document saying "dictionary" twice is more about
  dictionaries than one saying it once — but not twice as much. The tenth mention adds
  almost nothing, so the contribution must saturate.
- **Inverse document frequency** (idf): matching a rare word is informative; matching a
  common one is not. **Document frequency** (`df`) counts how many documents contain the
  word at all, and low `df` earns high weight.
- **Length normalization**: long documents accumulate matches by accident, so they are
  penalized relative to the average.

The implementation is pure Python. idf, at `retrieve.py:53`, with `N = 27`:

```
idf(t) = ln( (N − df + 0.5) / (df + 0.5) + 1 )
```

Check it. The word `dictionary` appears in 2 chunks: `(27 − 2 + 0.5)/(2 + 0.5)` = `10.2`,
and `ln(11.2)` = **2.41591**. The word `model` appears in 19: idf = **0.36179**. Matching
"dictionary" is worth about seven times as much as matching "model", which is exactly the
behaviour you want.

The score, at `retrieve.py:85-86`, with `k1 = 1.5` and `b = 0.75`:

```
score(d) = Σ  idf(t) · (f · 2.5) / ( f + 1.5 · (0.25 + 0.75 · dl / avgdl) )
```

`dl` is the document's token count and `avgdl` is the corpus mean, measured at **41.63**.
For the query "how do I convert a model to a dictionary" against chunk `c_v2_dump`
(`dl = 42`), the term `dictionary` with `f = 2` contributes:

```
denom = 2 + 1.5 · (0.25 + 0.75 · 42/41.63) = 3.51001
term  = 2.41591 · (2 · 2.5) / 3.51001      = 3.44146
```

Summed over all matching terms, that chunk scores **8.52**. The next best v2 chunk scores
2.46.

**TF-IDF with cosine similarity** is the second ranker. It turns each document into a
vector — one weight per term, here `(1 + ln(f)) · idf` — and measures the angle between
the query vector and each document vector, ignoring their lengths: the dot product
divided by both vector lengths (`retrieve.py:106`). For the same query, `c_v2_dump` scores
**0.387** and the runner-up scores 0.075. Where BM25 saturates frequency with an explicit
`k1` knob and is unbounded in scale, cosine damps frequency with a logarithm and is
bounded near 1.

Be clear about what this second channel is *not*. It shares the tokenizer and the idf
table with BM25, so it carries no semantic signal. It is a second lexical opinion, not a
meaning-based one. `DECISIONS.md` says so outright, and `retrieve.py:7-11` marks the seam
where a real embedding model would be swapped in.

**Reciprocal Rank Fusion** (RRF) combines the two rankings. You cannot simply add the
scores — 8.52 and 0.387 are not on the same scale, so BM25 would dominate by units alone.
RRF discards scores and keeps only positions (`retrieve.py:130-134`):

```
fused(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_tfidf(d))
```

A worked example with invented ranks shows why this is worth doing:

| doc | BM25 rank | cosine rank | fused score |
|---|---|---|---|
| A | 1 | 8 | 1/61 + 1/68 = 0.031099 |
| B | 2 | 2 | 1/62 + 1/62 = **0.032258** |
| C | 3 | 12 | 1/63 + 1/72 = 0.029762 |

B wins without ever being either channel's top hit, because both channels agree it is
near the top. A's single first place cannot offset its eighth. Agreement between
independent rankers beats a lone strong opinion. The `+ 60` is what makes the curve
gentle — without it, rank 1 would be worth 1.0 and rank 2 only 0.5, and one first place
would decide everything.

A **metadata filter** restricts which documents are eligible before any ranking happens,
using a structured field rather than the text. Here the field is `version`, and the whole
filter is one line at `retrieve.py:69-70`:

```python
return [c for c in self.chunks if c.version in (version, "both")]
```

It runs at `retrieve.py:123`, before both rankers and before fusion. A v2 query sees 15 of
27 chunks; a v1 query sees 14.

**This is the load-bearing idea in the project.** The v1 and v2 chunks are near-identical
in wording — `c_v2_dump` and `c_v1_dict` share the title "Serialize a model to a
dictionary" and differ only in a trailing `(v1)`/`(v2)` and one method name. No amount of
tuning `k1`, `b`, or the RRF constant separates them reliably by text. Filtering first
makes the wrong-version chunk *unreachable* rather than merely *unlikely*. The same
question, asked twice:

```
version=v2  ->  c_v2_dump   "Serialize a model to a dictionary (v2)"   code: u.model_dump()
version=v1  ->  c_v1_dict   "Serialize a model to a dictionary (v1)"   code: u.dict()
```

### 3b. Writing an answer you can check

The text you send a model is the **prompt**, and it is the only control surface — there is
no flag that makes a model accurate. Most APIs split it in two: a **system prompt** of
standing instructions that apply to every request, and a **user prompt** carrying the
specific question. Here the system prompt (`llm.py:41-57`) is five numbered rules: answer
only from the supplied chunks, be correct for the target version, return a runnable
snippet ending in an `assert`, cite the chunk ids used, and abstain rather than guess. The
user prompt (`llm.py:109-125`) is three labelled blocks — target version, question, then
each retrieved chunk rendered as `[c_v1_dict] (version=v1) <title> ...`. The chunk id is
printed in brackets deliberately: the model cannot cite an identifier it was never shown.

**Structured output** means requiring the reply to be machine-readable data rather than
prose. **JSON** is a text format for structured data; a **JSON Schema** is a second
document describing the shape the first must have. If the model replied in prose, the
program would have to guess where the code starts, whether "I'm not sure" counts as a
refusal, and which words were citations — regular-expression work that breaks on the first
answer phrased differently. Instead `ANSWER_SCHEMA` (`llm.py:29-39`) declares exactly four
required keys:

| key | type | meaning |
|---|---|---|
| `answer` | string | the prose explanation |
| `code` | string | a runnable snippet |
| `citations` | array of strings | chunk ids the answer rests on |
| `abstained` | boolean | true if refusing to answer |

The schema is passed to the API as `output_config={"format": {"type": "json_schema", ...}}`
(`llm.py:152-161`), so the model is constrained to emit that object, and downstream code
reads four fields by name and never parses English.

Constrained is not guaranteed. A reply cut short by the token budget stops mid-object and
is not valid JSON. `_extract_json` (`llm.py:74-111`) handles it: strip stray code fences,
try `json.loads`, and on failure retry with a string-aware decoder from the first `{`, so
a brace inside a code field does not derail it. If parsing still fails, or the API reports
a refusal, the client returns a pre-built abstention. **A malformed reply becomes a
refusal, never a crash.**

A **citation** here is not a URL — it is the id of a chunk that was in the prompt, like
`c_v2_dump`. It answers "which of the 27 corpus entries did this come from", which a
reader can check by eye. A model can still emit an id it was never shown, so `_coerce`
(`answer.py:40-56`) filters the list against what was actually retrieved:

```python
citations = [c for c in raw.get("citations", []) if c in retrieved_ids]
```

**Abstention** is the system declining to answer — one boolean, `Answer.abstained`.
Refusing is a feature, because the alternative is worse: if the retrieved chunks do not
cover the question, any answer is invented. The repo ships six deliberately unanswerable
questions (`data/golden/unanswerable.jsonl`), such as *"In Pydantic v2, how do I make an
HTTP GET request to a REST API?"*, so that abstention is measured rather than assumed.

Now the core of the project. **Execution grading** means the score comes from running the
generated code, not from reading it. Three definitions first:

- A **virtual environment** (venv) is a directory holding its own Python interpreter and
  its own package folder. Two venvs on one machine can hold two conflicting versions of
  the same library without interfering.
- **pip** is Python's package installer. `pip install "pydantic>=1.10,<2"` means "the
  newest release at least 1.10 and below 2.0".
- The **exit code** is the integer a finished program reports. 0 means success; an
  uncaught Python exception exits 1.

**Pinning** a version means constraining which release gets installed.
`scripts/setup_sandbox.sh` builds two venvs — `.venvs/pydantic_v1` with `pydantic>=1.10,<2`
and `.venvs/pydantic_v2` with `pydantic>=2,<3` plus `pydantic-settings>=2,<3`, because v2
moved settings support into a separate package. Measured on the development machine:
**pydantic 1.10.26** and **pydantic 2.13.4**.

`grade(code, version)` (`sandbox.py:329`) writes the snippet to a temporary `.py` file and
runs it with the matching venv's interpreter. Passing is `returncode == 0`. The subjective
question "is this answer good?" becomes the objective question "did exit code 0 come
back?".

This doubles as the version check because Pydantic 2.0 *removed* names rather than
aliasing them. Measured, both directions, for a snippet asserting on `u.model_dump()`:

| run in | result | stderr |
|---|---|---|
| v2 sandbox | `passed=True, returncode=0` | empty |
| v1 sandbox | `passed=False, returncode=1` | `AttributeError: 'U' object has no attribute 'model_dump'` |

No string matching. No second model judging the first. The interpreter for the pinned
version decides.

**Sandboxing** is the last concept here. The snippet was written by a model from
documents; running it is running code nobody reviewed. It could loop forever, exhaust
memory, fill the disk, or read the server's environment variables. Four operating-system
terms: a **process** is one running program with its own memory; a **process group** is a
set of related processes the kernel can signal as a unit, inherited by anything a process
starts; a **signal** is a one-word message the kernel delivers, of which `SIGKILL`
terminates immediately and cannot be caught; **POSIX** is the Unix standard these follow
(macOS and Linux, not Windows).

**Resource limits** (rlimits) are kernel-enforced ceilings on a single process. Four are
applied:

| Limit | Default | Stops |
|---|---|---|
| `RLIMIT_CPU` | 10 s | infinite loops |
| `RLIMIT_AS` (address space) | 1024 MB | runaway allocation |
| `RLIMIT_FSIZE` | 10 MB | disk-filling writes |
| `RLIMIT_CORE` | 0 | core-dump files |

A **wall-clock timeout** (default 20 s) measures elapsed real time and catches what
rlimits cannot: code that sleeps or blocks on a network read, burning no CPU at all. Note
the ordering — the 10-second CPU limit bites before the 20-second wall-clock limit for
compute-bound code, so a hot loop dies by `SIGXCPU` and the timeout is the fallback for
code that blocks.

### 3c. Measuring and operating

You can read one answer and judge it. You cannot read three hundred, and you cannot read
the same three hundred next week and get a comparable verdict. Evaluation turns "is this
good?" into a number a machine recomputes: **reproducible**, **comparable** across
changes, and **attributable** to a stage when it drops.

A **golden set** is a fixed, hand-written list of questions with the correct answer
material attached. *Fixed* is load-bearing — change the questions and the numbers stop
being comparable. Here it is `data/golden/golden_set.jsonl`: **25 records**, each with an
`id`, a `version`, the `question`, the `relevant_chunk_ids` that should have been
retrieved, and a `check`. The `check` is not prose to string-match against — it is
**runnable Python, verified to pass on its target version**:

```json
{"id": "g_v2_dump", "version": "v2",
 "question": "In Pydantic v2, how do I serialize a model instance to a dictionary?",
 "relevant_chunk_ids": ["c_v2_dump"],
 "check": "from pydantic import BaseModel\n\nclass User(BaseModel):\n    name: str\n    age: int\n\nu = User(name='Ada', age=36)\nassert u.model_dump() == {'name': 'Ada', 'age': 36}\n"}
```

**recall@k** = (correct items appearing in the top k) ÷ (correct items). Each golden record
names exactly one correct chunk, so per question the value is 0.0 or 1.0, and the reported
figure is the mean over 25. Take item `g_v1_config` ("how do I make a model immutable?",
gold chunk `c_v1_config`). The retrieved top 5 were:

```
1 c_v1_copy   2 c_v1_config   3 c_v1_root_validator   4 c_v1_dict   5 c_v1_json
```

recall@1 = 0.0, recall@3 = 1.0, recall@5 = 1.0. What recall misses is **position** — full
marks with the right chunk sitting second. Raise k far enough and recall approaches 1.0
for any retriever that is not broken, so recall alone rewards returning more.

**Mean Reciprocal Rank** (MRR) fixes that. Reciprocal rank is 1 ÷ (position of the first
correct result), counting from 1, or 0.0 if never found. Rank 1 scores 1.0, rank 2 scores
0.5, rank 4 scores 0.25 — a severe penalty at the top, negligible further down. The item
above sits at position 2, so its reciprocal rank is 0.5. It is the only one of the 25 not
at rank 1:

```
MRR = (24 × 1.0 + 1 × 0.5) ÷ 25 = 24.5 ÷ 25 = 0.98
```

That is where the reported `mrr=0.98` comes from. recall@5 = 1.0 says nothing was lost;
MRR = 0.98 says one item was demoted. Both are reported because they answer different
questions.

A **failure taxonomy** is a fixed set of named buckets, one per way the system can fail,
with each item assigned to exactly one. "Executable-% fell from 0.9 to 0.7" tells you to
start reading logs. "`retrieval_miss` went 1 → 6 and every other bucket is flat" tells you
which file to open.

**Continuous integration** (CI) runs commands automatically on every push. A **regression
gate** compares today's metrics against fixed thresholds and fails the build if any is
missed. Placing thresholds is the craft: above current performance and the build is red
the day you write it; at exactly current performance and it reddens on harmless jitter;
far below and a real regression sails through.

Finally, one lesson worth stating on its own, because this repository contains a perfect
example of it: **a metric lies when its denominator quietly excludes the failures.** The
story is in [§6](#6-how-it-is-measured).

On the operations side, a handful of terms recur:

- A **cache** stores expensive results so a repeat request is cheap. **LRU** (least
  recently used) keeps at most N entries and discards the one untouched longest; **TTL**
  (time to live) treats an entry past its expiry as absent.
- **Rate limiting** caps how often one caller may make requests. A **token bucket** holds
  tokens up to a **capacity**, refilled at a fixed **rate**; each request spends one, and
  a request finding none is refused. Capacity sets the allowed burst; rate sets sustained
  throughput.
- **Structured logging** emits one JSON object per log line instead of prose, so lines can
  be counted and filtered without regular expressions. A **request id** is a short unique
  string tying together every line one request produced.
- **Prometheus** is a monitoring server that periodically fetches a plain-text page of
  counters. **Label cardinality** is the number of distinct label combinations, each held
  in memory as a separate series — so labelling by a value the caller controls is a memory
  leak.
- **Liveness** asks whether a process is alive or should be restarted. **Readiness** asks
  whether it should receive traffic *now*; a process can be alive but not ready.
- A **container** is a packaged filesystem plus a start command, run as an isolated
  process on a shared kernel, so what ran in CI is what runs in production.

## 4. How one question flows through the system

Trace the real question *"In Pydantic v2, how do I serialize a model instance to a
dictionary?"*, asked against **v1** — the mismatch that makes the mechanism visible.

1. **Request.** `POST /ask` with `{"question": ..., "version": "v1", "execute": true}`.
   FastAPI validates the body against `AskRequest` (`app/main.py:238-247`) before any
   handler runs: the question must be 1–2000 characters and `top_k` an integer in 1–50.
2. **Rate limit.** `_rate_limit` (`app/main.py:511`) spends one token from the caller's
   bucket, refusing with HTTP 429 and a `Retry-After` header if it is empty.
3. **Cache lookup.** The key is `(question.strip(), version, top_k, execute)`
   (`app/main.py:320`). A hit returns immediately with `meta.cached = true`.
4. **Retrieval.** `build_answer` (`answer.py:13-38`) calls `retriever.retrieve(question,
   "v1", top_k=5)`. The version filter cuts 27 chunks to the 14 that are `v1` or `both`.
   BM25 and TF-IDF each rank those, RRF fuses the two rank lists, and the top 5 come back.
   `c_v2_dump` — the chunk that actually answers this question — is **not among them**. It
   was excluded at the candidate stage.
5. **Generation.** `client.generate(...)` builds the prompt from the 5 v1 chunks and calls
   the model, which returns the four-key JSON object. `_coerce` (`answer.py:40-56`) drops
   any citation id that was not retrieved.
6. **Grading.** `AnswerResult.execution_grade()` (`answer.py:73-76`) calls
   `grade(code, "v1")`. The snippet is written to a temp file and launched as
   `.venvs/pydantic_v1/bin/python -c "<launcher>" /tmp/xxxx.py` — its own process group,
   an environment scrubbed to just `PATH` and an empty `PYTHONPATH`, and CPU, file-size,
   and core-dump limits applied.
7. **Verdict.** Pydantic 1.10.26 has no `model_dump`; the `AttributeError` propagates and
   the process exits 1. The result is `passed=False, returncode=1`, with `stderr_tail`
   ending in the exact exception.
8. **Response.** The handler records metrics, stores the result in the cache, and returns
   the answer, the ranked retrieval list with per-channel ranks, and the execution block.
   The middleware attaches a request id, five security headers, and a JSON access log
   line.

Asked against **v2**, every step is the same and the snippet exits 0. That contrast — same
question, two versions, opposite verdicts, both produced by the interpreter rather than by
a claim — is the whole demonstration, and it is what the `/compare` endpoint shows
side by side.

## 5. The components

### Retrieval — `docsthatrun/retrieve.py`

Builds its index once in `__init__`: per-chunk token lists, term-frequency counters,
document lengths, the corpus-wide document frequencies, the idf table, and each document's
TF-IDF vector length. Then per query it filters by version, runs both rankers, fuses by
RRF, and returns the top `k` as `RetrievalResult` objects carrying the chunk, the **fused**
score, and both per-channel ranks (either may be `None`).

Two honest notes. There is no explicit tie-break — order falls out of Python's stable sort,
which resolves ties toward BM25 order and then corpus file order; that is emergent, not
designed. And the version filter restricts *candidates*, while idf, average document
length, and document norms are computed once across all 27 chunks and never recomputed per
version, so a term's rarity is measured partly against documents that can never be
returned. With this corpus the effect is small, but it is real.

### Answer layer — `docsthatrun/llm.py`, `docsthatrun/answer.py`

`answer.py` is the orchestration: retrieve, generate, coerce, optionally grade. `llm.py`
holds the prompts, the schema, the JSON extraction, and two interchangeable clients.

**`MockClient`** is the offline stand-in. It replays each golden item's own `check` snippet
as the "generated" answer, keyed on normalized question text. Because two golden items with
identical wording would silently replay each other's answer, the loader **raises on a
duplicate** rather than overwriting. It needs no API key, which is what lets CI run the
full pipeline for free.

**`AnthropicClient`** is the real path. It calls Claude with the model from config
(default `claude-opus-5`), a configurable `max_tokens` (8192), adaptive thinking, and the JSON schema as
the required output format. The SDK is configured with a 60-second timeout and 2 retries,
so a hung request cannot wedge a worker. Every failure path — refusal, truncation, empty
body, unparseable JSON — degrades to an abstention carrying the model's stop reason for
observability.

### Sandbox — `docsthatrun/sandbox.py`

The mechanism is described in [§3b](#3b-writing-an-answer-you-can-check); three
implementation decisions are worth calling out.

**Why a launcher instead of `preexec_fn`.** Python's `subprocess` offers `preexec_fn` to
run code in the child between `fork` and `exec`. At that moment the child holds copies of
every lock the parent's threads held, which in a threaded server is a classic deadlock.
Instead the limits are applied by a small generated Python program passed via `-c`, which
runs *after* `exec` in a fresh single-threaded interpreter where `setrlimit` is safe.

**Why `sys.argv` is rewritten.** Launching as `python -c "<launcher>" /tmp/x.py` leaves the
child with `sys.argv == ['-c', '/tmp/x.py']`. A snippet using `argparse` would fail for
reasons having nothing to do with Pydantic. The launcher rewrites argv to `[path]` and runs
the file with `runpy.run_path(..., run_name='__main__')`, reproducing exactly what
`python file.py` gives.

**Why the process *group* is killed.** A timeout that kills only the direct child orphans
any grandchild it spawned, which keeps running unsupervised. The child is started with
`start_new_session=True` so everything it spawns inherits its group, and the timeout path
sends `SIGKILL` to the whole group. The reap afterwards is bounded to 5 seconds, because a
process that escaped the group (a double-fork daemon) could otherwise hold the parent
waiting forever. A regression test spawns a grandchild that would write a marker file five
seconds later and asserts the file never appears.

**Why output goes to files, not pipes.** This one was a real bug, found late. Reading the
child's output through a pipe means reading it into the *parent's* memory with no ceiling —
and neither resource limit covers that: `RLIMIT_AS` bounds the child, not the parent's
buffer, and `RLIMIT_FSIZE` bounds files, not pipes. A snippet doing nothing but
`sys.stdout.write` in a loop therefore grew the API server without limit. Measured at
production defaults: 512 MiB of output took the process from 13 MB to 1,809 MB resident, and
the snippet still reported PASS. Redirecting stdout and stderr to temp files puts the same
write back under `RLIMIT_FSIZE` at the kernel, and the parent reads back only a bounded tail
(64 KB). The same measurement after the fix: 13.0 MB to 13.1 MB, and the flood correctly
fails. It also deleted the pipe-drain logic entirely — with no pipe, there is nothing to
deadlock on.

`sandbox_available(version)` probes whether Pydantic actually imports, because checking
that `bin/python` exists is not enough — the setup script creates the venv *before* pip
runs, so an interrupted setup leaves an interpreter with no package. **Only success is
cached.** A probe that fails while `make sandbox` is still installing must not disable
grading for the rest of the process's life.

### Evaluation harness — `docsthatrun/evals/run_evals.py`

Three layers, cheapest first: retrieval metrics (offline, no dependencies, no API key);
answer quality with execution grading (needs a client and the venvs); and abstention from
both sides. It emits a report, optionally as JSON, and with `--gate` exits non-zero when a
threshold is missed. Details in [§6](#6-how-it-is-measured).

### HTTP service — `app/main.py`

A FastAPI application with eight routes:

| Route | Purpose |
|---|---|
| `GET /` | the single-page demo UI |
| `POST /ask` | answer one question for one version |
| `POST /compare` | answer the same question for both versions |
| `GET /health` | liveness: version, client class, per-version sandbox status |
| `GET /ready` | readiness: **503** when the corpus failed to load |
| `GET /metrics` | Prometheus exposition text |
| `GET /stats` | the same counters as readable JSON |
| `GET /examples` | golden questions, to populate the UI's example chips |

Around them: a lifespan hook that warms the retriever and client at startup; thread-safe
double-checked-locking singletons so concurrent first requests cannot build two retrievers;
and an `observe` middleware that assigns a request id, times the request, emits the JSON
access log, records metrics, and attaches five security headers — including on an unhandled
500, which it builds by hand rather than letting a bare error escape.

Middleware order matters and is commented in the file: CORS is registered *last* so it ends
up outermost, because a hand-built 500 from inside `observe` would otherwise carry no
`Access-Control-Allow-Origin` header and a browser would see an opaque network error
instead of a readable failure.

Supporting modules are stdlib-only by design: `docsthatrun/cache.py` (LRU + TTL),
`docsthatrun/ratelimit.py` (token bucket with an LRU cap on tracked keys so the table cannot
grow without bound), `docsthatrun/observability.py` (JSON log formatter, metrics, Prometheus
rendering with a 64-label cardinality cap), and `docsthatrun/config.py`.

Every tunable is an environment variable:

| Variable | Default | Effect |
|---|---|---|
| `DOCSTHATRUN_LLM` | `auto` | `mock`, `anthropic`, or auto-detect by API key |
| `DOCSTHATRUN_MODEL` | `claude-opus-5` | model id |
| `DOCSTHATRUN_EFFORT` | `medium` | reasoning effort |
| `DOCSTHATRUN_LLM_TIMEOUT` | `60.0` | per-call timeout, seconds |
| `DOCSTHATRUN_LLM_RETRIES` | `2` | SDK retries on 429/5xx |
| `DOCSTHATRUN_TOP_K` | `5` | default retrieval depth |
| `DOCSTHATRUN_TOP_K_MAX` | `50` | upper bound accepted |
| `DOCSTHATRUN_MAX_QUESTION_CHARS` | `2000` | request size bound |
| `DOCSTHATRUN_SANDBOX_TIMEOUT` | `20` | wall-clock seconds |
| `DOCSTHATRUN_SANDBOX_CPU` | `10` | `RLIMIT_CPU` seconds |
| `DOCSTHATRUN_SANDBOX_MEM_MB` | `1024` | `RLIMIT_AS` |
| `DOCSTHATRUN_SANDBOX_FILE_MB` | `10` | `RLIMIT_FSIZE` |
| `DOCSTHATRUN_CACHE_MAX` | `256` | LRU entries |
| `DOCSTHATRUN_CACHE_TTL` | `900.0` | entry lifetime, seconds |
| `DOCSTHATRUN_RATE_RPM` | `60` | sustained requests/min; `0` disables |
| `DOCSTHATRUN_RATE_BURST` | `20` | bucket capacity |
| `DOCSTHATRUN_TRUST_PROXY` | `false` | key rate limits on `X-Forwarded-For` |
| `DOCSTHATRUN_CORS_ORIGINS` | *(empty)* | comma-separated allowed origins |
| `DOCSTHATRUN_LOG_LEVEL` | `INFO` | falls back to `INFO` if unrecognized |
| `DOCSTHATRUN_LOG_JSON` | `true` | JSON logs vs human-readable |

### Demo UI — `app/static/index.html`

One file, 598 lines, no build step. Three modes — v2, v1, and diff (both side by side).
The result is a **verdict instrument**: a large glyph cell reading PASS, FAIL, ABSTAIN, or
NO GRADE, beside metadata rows for the target sandbox, the process exit code, and the
latency (marked `· cached` on a cache hit). Below it sits the answer, the cited chunk ids,
the syntax-highlighted snippet, the stderr tail on failure, and a 7-column retrieval table
showing each chunk's fused score and both per-channel ranks.

All API text reaches the DOM through `textContent` or an escaping helper, so
model-generated text containing `<` or `&` renders as characters rather than markup. A
single `inFlight` flag guards against overlapping requests, because programmatic submission
from the example chips and Cmd+Enter fires the submit event even when the button is
disabled — two concurrent fetches would race and let a stale response overwrite a newer
one.

The aesthetic is a literal instrument panel: an engineering-graph grid background, LED
status readouts, a shell prompt beside the input, monospace uppercase micro-labels, and a
blinking block cursor while running. Fonts (JetBrains Mono, IBM Plex Sans) are self-hosted,
so the page makes zero external requests and satisfies its own `font-src 'self'` policy.

### CLI — `docsthatrun/cli.py`

Two subcommands, `ask` and `compare`, with hand-rolled ANSI colour that disables itself
when `NO_COLOR` is set or output is not a terminal. `ask` exits 1 when the snippet fails in
the sandbox, which makes it usable in a script. `compare` always exits 0 deliberately — a
failure on the *other* version is the expected result, not a defect.

## 6. How it is measured

Run the full harness:

```bash
python3 -m docsthatrun.evals.run_evals --answers --gate --client mock
```

Real output:

```
============================================================
DocsThatRun eval report
============================================================
corpus=27 golden=25 unanswerable=6
retrieval: recall@3=1.0  recall@5=1.0  mrr=0.98
client=MockClient  sandbox: v1=up  v2=up
answers: executable%=1.0 (n=25 graded)  unanswerable_abstention=1.0  answerable_over_abstention=0.0
taxonomy: pass=25
answer latency(ms): mean=0.2  p50=0.2  p95=0.3  max=0.3
grade latency(ms): mean=244.3  p50=288.2  p95=295.2  max=398.0
note: MockClient replays the answer key: executable_pct here is a PLUMBING check, not a quality claim. Run with DOCSTHATRUN_LLM=anthropic for a real measurement.

GATE PASSED
```

**The metrics.**

| Metric | Definition |
|---|---|
| `recall@k` | share of questions whose gold chunk appeared in the top k |
| `mrr` | mean of 1 ÷ (rank of the first gold chunk) |
| `executable_pct` | share of **non-abstained, gradable** answers whose snippet ran and exited 0 |
| `ungraded_count` | non-abstained answers whose version's sandbox was down — excluded from the line above, never folded into it |
| `unanswerable_abstention` | share of the 6 unanswerable questions correctly refused |
| `answerable_over_abstention` | share of the 25 answerable questions wrongly refused |
| `answer_latency_ms` | retrieval + LLM, across all 31 questions |
| `grade_latency_ms` | sandbox execution only, across the graded items |

The two latency numbers are separate on purpose. They used to be one, measured around
both stages: since a sandbox run is ~250 ms and the mock answer path is ~0.2 ms, the
published "p95 latency" was really a statement about subprocess startup. The unanswerable
questions weren't timed at all, so the distribution silently covered 25 of the 31
questions the harness answers.

`answerable_over_abstention` and `taxonomy["over_abstention"]` are related but not equal,
which is worth knowing before you try to reconcile them: the rate counts every abstention
on an answerable item, while the taxonomy bucket counts only abstentions where retrieval
*did* surface the gold chunk. An abstention that also missed its gold chunk is filed under
`retrieval_miss`, because retrieval is the upstream cause.

The two abstention rates exist as a pair on purpose. Abstaining costs nothing on
`executable_pct` — an abstention is excluded from that denominator — so without the
over-abstention gate, a system could score perfectly by refusing every hard question.

**The failure taxonomy** assigns each answerable item to exactly one bucket, first match
winning, in this order:

1. `pass` — ran and exited 0. Checked **first**, deliberately: a snippet that ran and
   passed is a pass even if the gold chunk was never retrieved.
2. `retrieval_miss` — no gold chunk in the retrieved set. Upstream cause outranks
   downstream symptom.
3. `over_abstention` — declined an answerable question.
4. `no_code` — code empty after stripping whitespace.
5. `not_graded` — no execution result, or the sandbox was unavailable.
6. Otherwise, classify by stderr: `malformed_code` (SyntaxError, IndentationError),
   `wrong_assert` (AssertionError), `wrong_version_api` (ImportError, AttributeError,
   "cannot import name", …), or `runtime_error`.

That third stderr bucket is the interesting one — `wrong_version_api` is precisely
"answered with the wrong Pydantic version's API", the failure this whole project exists to
detect. Honest caveat: this is substring matching over stderr, so a chained traceback lands
in whichever bucket comes first rather than necessarily the true cause.

**The gate.** Thresholds are deliberately below current performance — a noise floor, to be
tightened as the corpus grows:

| Gate | Threshold | Protects against |
|---|---|---|
| `recall_at_5` | ≥ 0.80 | the retriever stops surfacing the right chunk at all |
| `mrr` | ≥ 0.60 | the right chunk is found but sinking down the ranking |
| `unanswerable_abstention` | ≥ 0.80 | the system starts inventing out-of-scope answers |
| `answerable_over_abstention` | ≤ 0.20 | the system games the other gates by refusing |
| `executable_pct` | ≥ 0.60 | answers that look right but do not run |

An execution gate that did not execute does not pass. A version whose venv is missing is
a *named* gate failure — `sandbox unavailable for v1 — 12 answerable item(s) went
ungraded` — and `executable_pct` is computed over the items that were actually gradable,
with the rest counted in `ungraded_count`. This is not hypothetical: the gate previously
keyed off a single `all(...)` flag across both versions, so a broken v1 venv discarded
every real v2 result and printed **GATE PASSED** having graded half the corpus.

**How a metric lied, and what fixed it.** `executable_pct` used to divide by *items that
produced an execution result* — items that had code to run. Consider a client that answers
all 25 questions, abstains on none, and emits empty code on 10 of them. Fifteen produce
code, and all fifteen pass:

```
old denominator (graded only):    15 ÷ 15 = 1.0   -> gate passes
new denominator (answered):       15 ÷ 25 = 0.6   -> scrapes the 0.60 gate
one worse (11 empty):             14 ÷ 25 = 0.56  -> gate fails
```

A perfect score for a system that silently failed to answer 40% of its questions. The name
implied "how often does this work"; the arithmetic measured "of the answers we could grade,
how many passed". The fix makes the denominator every non-abstained answerable item, and
reports both `answered_count` and `gradable_count` so the gap stays visible. The general
lesson: **name the population a rate is a rate *of*, and check that population does not
shrink when things go wrong.**

## 7. Honest limitations

This section is the point, not a disclaimer. A portfolio project that overclaims is worse
than one with a small honest result.

**The headline retrieval numbers are a property of the corpus, not evidence of a strong
retriever.** recall@5 = 1.0 over 25 questions against 27 chunks, where each query is
choosing from at most 15 version-eligible candidates and returning 5, is close to
unavoidable. The corpus is small, hand-authored, and cleanly version-separated. These
numbers say the version filter and tokenizer behave as intended on this data. They say
nothing about behaviour on a large or noisy documentation set, and nothing in the repo
currently tests that.

**The `executable_pct` of 1.0 in CI is a plumbing check, not a quality claim.** `MockClient`
replays each golden item's own `check` snippet, and every `check` is independently verified
to pass on its target version. The 1.0 proves that retrieval hands chunks to the answer
layer, that layer hands code to the sandbox, the sandbox reports an exit code, and the
scorer counts it. It proves nothing about a model. The same applies to
`unanswerable_abstention = 1.0` and `taxonomy: pass=25`, which follow mechanically from the
fixture map. **No run against a real Claude client is recorded anywhere in this
repository.** That measurement is the single most valuable thing missing.

**The version-lock signal is asymmetric — 17 of 25, or 68%.** Some v1 APIs survive in v2 as
deprecated shims, so `u.dict()` passes on *both* versions, merely printing a deprecation
warning; warnings do not change the exit code. For those 8 items, execution proves the code
runs but not that it targets the right major version.

**Exit code 0 is a weak oracle.** A snippet that imports Pydantic and does nothing else
"passes". The hand-written reference snippets end in `assert` statements to force real
behaviour, and the system prompt asks for the same, but nothing *compels* a generated
snippet to assert anything meaningful.

**The sandbox is not a security boundary.** It is a virtual environment with resource
limits, not a container or a virtual machine. There is no network isolation — nothing stops
a snippet opening a socket. `RLIMIT_AS` is skipped on macOS because it is unreliable there,
so the memory cap exists only on Linux. Windows gets no limits and no process-group kill at
all. Genuinely untrusted input at scale would want gVisor or a microVM above all of this.

**The second retrieval channel is not semantic.** It shares a tokenizer and idf table with
BM25, so the "hybrid" is two lexical opinions rather than lexical plus meaning-based. The
two channels agree far more than a true sparse+dense hybrid would.

**The corpus is hand-curated.** Every number in this guide is measured against 27
chunks with clean version separation. Real docs are messier, and recall@5 = 1.0 would not
survive contact with them.

**Smaller known rough edges.** The RRF implementation drops zero-scoring documents before
ranking, so a document only one channel scored contributes one term rather than one term
plus a large-rank term — a mild deviation from textbook RRF. The retrieval table's score
bar is normalized to the top result, so it shows relative order, not confidence. The UI
silently ignores a failed `/health` or `/examples` call at boot — deliberately, so a
degraded sandbox doesn't make the demo page look broken, but it does mean the status
chips can be stale.

## 8. Running it yourself

**Retrieval metrics, with nothing installed.** The core is standard-library only, so this
works on a bare Python 3.9+ with no `pip install` at all:

```bash
python3 -m docsthatrun.evals.run_evals
```

**Build the two pinned sandboxes** (once; takes a minute):

```bash
make sandbox
```

**The full evaluation, including real execution grading:**

```bash
python3 -m docsthatrun.evals.run_evals --answers --gate --client mock
```

**Ask from the terminal:**

```bash
python3 -m docsthatrun ask "In Pydantic v2, how do I serialize a model instance to a dictionary?" --version v2
```

**See the version-lock directly** — the same question answered for both versions, each
graded in its own sandbox:

```bash
python3 -m docsthatrun compare "In Pydantic v2, how do I serialize a model instance to a dictionary?"
```

**Run the API and demo UI** at <http://localhost:8000>:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Use real Claude answers instead of the offline mock:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m docsthatrun.evals.run_evals --answers --client anthropic
```

**Everything in a container**, both sandboxes baked in at build time so grading needs no
network at runtime:

```bash
docker compose up --build
```

**Run the tests and the linter:**

```bash
pytest -q
ruff check docsthatrun app tests
```

## 9. Repository map

| Path | What it is |
|---|---|
| `data/corpus/pydantic_corpus.jsonl` | 27 version-tagged documentation chunks |
| `data/golden/golden_set.jsonl` | 25 labelled questions with runnable reference snippets |
| `data/golden/unanswerable.jsonl` | 6 out-of-corpus questions that must be refused |
| `docsthatrun/corpus.py` | corpus loading, validation, tokenizer |
| `docsthatrun/schema.py` | `Chunk`, `RetrievalResult`, `Answer`, `GoldenItem` |
| `docsthatrun/retrieve.py` | BM25 + TF-IDF, RRF fusion, version filter |
| `docsthatrun/llm.py` | prompts, JSON schema, `MockClient`, `AnthropicClient` |
| `docsthatrun/answer.py` | RAG orchestration and citation coercion |
| `docsthatrun/sandbox.py` | execution grader: venvs, rlimits, process-group isolation |
| `docsthatrun/evals/metrics.py` | recall@k, MRR, mean |
| `docsthatrun/evals/run_evals.py` | eval harness, failure taxonomy, CI gate |
| `docsthatrun/cli.py` | `ask` / `compare` terminal interface |
| `docsthatrun/config.py` | every tunable, loaded from the environment |
| `docsthatrun/cache.py` | LRU + TTL answer cache |
| `docsthatrun/ratelimit.py` | per-client token bucket |
| `docsthatrun/observability.py` | JSON logging, metrics, Prometheus rendering |
| `app/main.py` | FastAPI service: routes, middleware, models, lifespan |
| `app/static/index.html` | the single-page demo UI |
| `scripts/setup_sandbox.sh` | builds the two pinned venvs |
| `.github/workflows/evals.yml` | CI: lint, tests, eval gate, stdlib-only proof |
| `Dockerfile`, `docker-compose.yml` | container image, non-root, health-checked |
| `DECISIONS.md` | design decisions and their tradeoffs |
| `ROADMAP.md` | what the project does not yet prove |

## 10. Design decisions and why

**Version filtering before ranking, not after.** The alternative — retrieve broadly, then
drop wrong-version results — leaves the v1 and v2 chunks competing on nearly identical text
and makes correctness a matter of ranking luck. Filtering first makes the wrong-version
chunk unreachable. The invariant is pinned by a test asserting no v1 chunk ever appears in
a v2 query.

**Execution as the grader, rather than a model judging a model.** An LLM-as-judge would
introduce a second unverified opinion and cost an API call per item. The interpreter is
free, deterministic, and cannot be persuaded. The cost is that exit code 0 is a coarse
signal, which the golden snippets mitigate by ending in assertions.

**A zero-dependency core.** Retrieval, the sandbox grader, and the eval harness import
nothing outside the standard library — no NumPy, no vector database, no LangChain. Cloning
the repo and running the evals requires no installation, which makes the project auditable
by a reviewer in under a minute. The cost is a lexical-only second retrieval channel. CI
proves this claim rather than asserting it, with a dedicated job that runs the evals on a
bare Python 3.9 with no `pip install` step; if any core import reached outside the standard
library, that job fails.

**A resource-limit launcher instead of `preexec_fn`.** Discussed in [§5](#5-the-components):
`preexec_fn` runs between `fork` and `exec` holding copies of the parent's thread locks,
which deadlocks under a threaded server. The launcher applies limits after `exec` in a
single-threaded interpreter.

**In-process cache, rate limiter, and metrics rather than Redis and OpenTelemetry.** For a
single-instance demo, external infrastructure would add operational weight without adding
capability, and would make "clone and run" false. Each module is small and behind a narrow
interface, so the upgrade path is a swap rather than a rewrite. The tradeoff is stated
plainly: all three are per-process, so two instances mean two independent caches and two
independent rate-limit buckets.

**Only successful sandbox probes are cached.** Caching a failure is tempting for symmetry
and wrong in practice: a probe that runs while `make sandbox` is still installing would
disable execution grading — the project's entire differentiator — for the lifetime of the
process, with no way to recover short of a restart.

**Gate thresholds below current performance.** Setting a gate at the current number means
harmless run-to-run jitter turns the build red and the team learns to ignore it. These sit
at a noise floor: recall@5 measures 1.0 against a 0.80 gate, so five of 25 questions can
stop surfacing their chunk before CI complains. They are meant to tighten as the corpus
grows.

## 11. What would come next

The project's own roadmap is honest about what it has not shown.

**A real measurement against Claude.** Every answer-layer number in this repository comes
from a mock that replays the answer key. Running the harness with `--client anthropic` and
publishing the true executable-%, with the failure taxonomy showing *how* real answers fail,
is the single highest-value piece of missing work. It is also the only number a skeptical
reader should care about.

**A real, messy corpus.** Replace 27 hand-written chunks with ingested Pydantic
documentation for one v1 and one v2 release — markdown, changelog, and a sample of
version-tagged issues and Stack Overflow answers. This is where the project is won or lost,
because version-tagging real question-and-answer content is genuinely hard: a v1 answer
lives on a thread whose title says v4. Expect recall to drop, and report the drop.

**Then the interesting engineering follows.** A true dense retrieval channel to replace the
lexical stand-in. A drift job that re-ingests on each new library release and emits a
regression report. Stronger isolation — gVisor or a microVM — if the grader ever runs
genuinely untrusted input at scale.

## 12. Glossary

| Term | Meaning |
|---|---|
| **abstention** | the system declining to answer because the retrieved docs do not cover the question |
| **BM25** | a lexical ranking formula combining term frequency, inverse document frequency, and length normalization |
| **chunk** | one small documentation record in the corpus |
| **CI (continuous integration)** | commands run automatically on every push |
| **container** | a packaged filesystem plus start command, run isolated on a shared kernel |
| **corpus** | the fixed set of documents the system may search |
| **cosine similarity** | the angle between two term vectors, ignoring their magnitudes |
| **document frequency (df)** | how many documents contain a term at all |
| **exit code** | the integer a finished process reports; 0 means success |
| **execution grading** | scoring an answer by running its code rather than reading it |
| **failure taxonomy** | fixed named buckets, one per way the system can fail |
| **golden set** | a fixed, hand-labelled question set used as the evaluation yardstick |
| **hallucination** | confident output unsupported by the source; here, a v2 answer to a v1 question |
| **inverse document frequency (idf)** | a weight that is high for rare terms and low for common ones |
| **JSON Schema** | a document describing the required shape of a JSON object |
| **JSONL** | a text file holding one JSON object per line |
| **label cardinality** | the number of distinct metric label combinations, each a stored series |
| **liveness** | whether a process is alive or should be restarted |
| **LLM (large language model)** | a program that predicts text from statistical patterns learned in training |
| **LRU (least recently used)** | an eviction rule discarding the entry untouched longest |
| **metadata filter** | restricting eligible documents by a structured field before ranking |
| **MRR (mean reciprocal rank)** | mean of 1 ÷ the rank of the first correct result |
| **pinning** | constraining which release of a library gets installed |
| **pip** | Python's package installer |
| **POSIX** | the Unix standard covering processes, signals, and resource limits |
| **process group** | a set of related processes the kernel can signal as a unit |
| **prompt** | the text sent to a language model; the only control surface |
| **Prometheus** | a monitoring server that scrapes a text page of counters |
| **RAG (retrieval-augmented generation)** | search trusted documents first, then have the model answer only from them |
| **rate limiting** | capping how often one caller may make requests |
| **readiness** | whether a live process should receive traffic now |
| **recall@k** | the share of correct items appearing in the top k results |
| **regression gate** | a CI check that fails the build when a metric misses a threshold |
| **request id** | a short unique string tying together all log lines from one request |
| **rlimit (resource limit)** | a kernel-enforced ceiling on one process's CPU, memory, or file size |
| **RRF (reciprocal rank fusion)** | combining rankings by summing 1/(k + rank), discarding raw scores |
| **signal** | a one-word kernel message to a process; `SIGKILL` cannot be caught |
| **structured logging** | emitting one JSON object per log line instead of prose |
| **system prompt** | standing instructions applied to every request |
| **term frequency (f)** | how often a term occurs in one document |
| **TF-IDF** | representing documents as vectors of term-frequency × inverse-document-frequency weights |
| **token** | one searchable unit of text after tokenization |
| **token bucket** | a rate-limiting algorithm: spend a token per request, refill at a fixed rate |
| **training cutoff** | the date after which a model has seen no data |
| **TTL (time to live)** | how long a cache entry stays valid |
| **venv (virtual environment)** | a directory with its own Python interpreter and packages |
| **wall-clock timeout** | a limit on elapsed real time, catching code that blocks without using CPU |
