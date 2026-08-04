"""Plain-dataclass data model. Deliberately pydantic-free so the app never
couples to a specific pydantic version — the two pydantic versions live only
inside the sandbox venvs (see docsthatrun/sandbox.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# The two library versions this project reasons about. "both" means a doc chunk
# applies regardless of version and is never filtered out.
VERSIONS = ("v1", "v2")


@dataclass(frozen=True)
class Chunk:
    """One version-tagged documentation chunk."""

    id: str
    version: str  # "v1" | "v2" | "both"
    topic: str
    title: str
    text: str
    code: str = ""

    def indexable_text(self) -> str:
        """Everything the retriever should search over for this chunk."""
        return "\n".join([self.title, self.text, self.code])


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None


@dataclass
class Answer:
    answer: str
    code: str
    citations: List[str] = field(default_factory=list)
    abstained: bool = False

    def has_runnable_code(self) -> bool:
        """True when there is something worth handing to the sandbox.

        The ``strip()`` is the whole point, and the reason this lives here rather
        than being re-spelled at each call site: whitespace-only "code" is still
        no code, and without the strip ``"  \\n"`` reaches the sandbox, fails
        with an empty stderr, and lands in the ``runtime_error`` bucket —
        blaming the answer for what is really an empty-snippet bug. The eval
        harness got this right and the CLI and API did not; one predicate keeps
        all three honest.
        """
        return bool((self.code or "").strip())

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "code": self.code,
            "citations": list(self.citations),
            "abstained": self.abstained,
        }


@dataclass
class GoldenItem:
    """A hand-labeled eval example.

    `check` is a *reference* runnable snippet known to pass on `version`. It is
    used two ways: (1) the offline MockClient replays it so the answer->sandbox
    ->eval plumbing runs without an API key; (2) tests run it against the *other*
    version to quantify how many examples are crisply version-locked.
    """

    id: str
    question: str
    version: str
    relevant_chunk_ids: List[str]
    check: str
    answerable: bool = True
