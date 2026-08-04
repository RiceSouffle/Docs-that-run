"""Small, dependency-free metric helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> Optional[float]:
    """Fraction of relevant ids that appear in the top-k retrieved ids.

    Returns ``None`` — not ``0.0`` — when the item has no labelled relevant
    chunks. Scoring an unlabelled item as a total miss and averaging it in would
    drag the published recall toward the gate threshold for a *data* reason,
    with no error and nothing in the report to explain it. Callers should skip
    ``None`` rather than count it (see ``mean``, which ignores them).
    """
    relevant = set(relevant_ids)
    if not relevant:
        return None
    top = set(retrieved_ids[:k])
    return len(relevant & top) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1/rank of the first relevant id (0 if none retrieved)."""
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: List[Optional[float]]) -> float:
    """Arithmetic mean, ignoring ``None`` (unscored items) entirely."""
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else 0.0
