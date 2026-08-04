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
    if not retrieved:
        # Nothing was retrieved, so there is no documentation to ground an
        # answer in and abstaining is the only correct outcome. Calling the
        # model anyway would be a billed round-trip that can only hallucinate or
        # abstain — and the prompt would contain an empty DOCUMENTATION CHUNKS
        # section, which is not a question worth asking.
        answer = Answer(
            answer="I don't have documentation that covers this.",
            code="",
            citations=[],
            abstained=True,
        )
        return AnswerResult(question=question, version=version, retrieved=[], answer=answer)
    raw = client.generate(question, version, retrieved)
    answer = _coerce(raw, retrieved)
    return AnswerResult(question=question, version=version, retrieved=retrieved, answer=answer)


def _coerce(raw: dict, retrieved: List[RetrievalResult]) -> Answer:
    retrieved_ids = {r.chunk.id for r in retrieved}
    # Drop hallucinated citations: only keep ids that were actually retrieved.
    # `raw` is not schema-guaranteed — _extract_json's salvage path can return
    # any dict shape — so a `"citations": null` used to reach this line and
    # raise TypeError straight out of the request. Anything that isn't a list of
    # strings is treated as "no citations".
    raw_citations = raw.get("citations")
    if not isinstance(raw_citations, (list, tuple)):
        raw_citations = []
    citations = [c for c in raw_citations if isinstance(c, str) and c in retrieved_ids]
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
            # Richer per-chunk retrieval detail so a UI can show *why* each chunk
            # surfaced (fused score + the per-channel ranks). `retrieved_ids` is
            # kept above for backward compatibility.
            "retrieved": [
                {
                    "id": r.chunk.id,
                    "version": r.chunk.version,
                    "topic": r.chunk.topic,
                    "title": r.chunk.title,
                    "snippet": _snippet(r.chunk.text),
                    "score": round(r.score, 5),
                    "bm25_rank": r.bm25_rank,
                    "dense_rank": r.dense_rank,
                    "cited": r.chunk.id in set(self.answer.citations),
                }
                for r in self.retrieved
            ],
            "answer": self.answer.to_dict(),
            "execution": self.execution.to_dict() if self.execution else None,
        }


def _snippet(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
