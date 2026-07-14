"""Small, dependency-free metric helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant ids that appear in the top-k retrieved ids."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(relevant & top) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1/rank of the first relevant id (0 if none retrieved)."""
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
