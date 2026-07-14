"""RAG orchestration: retrieve -> generate cited answer -> (optionally) grade."""

from __future__ import annotations

from typing import List, Optional

from .llm import LLMClient, get_client
from .retrieve import HybridRetriever
from .sandbox import ExecResult, grade
from .schema import Answer, RetrievalResult


def build_answer(
    question: str,
    version: str,
    retriever: HybridRetriever,
    client: Optional[LLMClient] = None,
    top_k: int = 5,
) -> "AnswerResult":
    client = client or get_client()
    retrieved = retriever.retrieve(question, version, top_k=top_k)
    raw = client.generate(question, version, retrieved)
    answer = _coerce(raw, retrieved)
    return AnswerResult(question=question, version=version, retrieved=retrieved, answer=answer)


def _coerce(raw: dict, retrieved: List[RetrievalResult]) -> Answer:
    retrieved_ids = {r.chunk.id for r in retrieved}
    # Drop hallucinated citations: only keep ids that were actually retrieved.
    citations = [c for c in raw.get("citations", []) if c in retrieved_ids]
    return Answer(
        answer=str(raw.get("answer", "")),
        code=str(raw.get("code", "")),
        citations=citations,
        abstained=bool(raw.get("abstained", False)),
    )


class AnswerResult:
    def __init__(
        self,
        question: str,
        version: str,
        retrieved: List[RetrievalResult],
        answer: Answer,
    ):
        self.question = question
        self.version = version
        self.retrieved = retrieved
        self.answer = answer
        self.execution: Optional[ExecResult] = None

    def execution_grade(self) -> ExecResult:
        """Execute the generated snippet against the target-version sandbox."""
        self.execution = grade(self.answer.code, self.version)
        return self.execution

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "version": self.version,
            "retrieved_ids": [r.chunk.id for r in self.retrieved],
            "answer": self.answer.to_dict(),
            "execution": self.execution.to_dict() if self.execution else None,
        }
