"""Version-aware hybrid retrieval, dependency-free.

Two lexical channels — Okapi BM25 (sparse) and a TF-IDF cosine "dense-ish"
channel — fused with Reciprocal Rank Fusion. Both are pure Python so the whole
retrieval + eval loop runs with zero pip installs.

The ``Embedder`` seam is deliberate: swapping the TF-IDF channel for a real
sentence-transformer / OpenAI embedder is a one-class change, and RRF then
fuses genuine lexical + semantic signal. See DECISIONS.md for the honest
tradeoff (both default channels are lexical; that is a baseline, not the
finished retriever).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .corpus import tokenize
from .schema import Chunk, RetrievalResult

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = list(chunks)
        self._doc_tokens: Dict[str, List[str]] = {}
        self._doc_tf: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._df: Counter = Counter()
        self._idf: Dict[str, float] = {}
        self._tfidf_norm: Dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        for chunk in self.chunks:
            tokens = tokenize(chunk.indexable_text())
            tf = Counter(tokens)
            self._doc_tokens[chunk.id] = tokens
            self._doc_tf[chunk.id] = tf
            self._doc_len[chunk.id] = len(tokens)
            for term in tf:
                self._df[term] += 1

        n = max(len(self.chunks), 1)
        for term, df in self._df.items():
            # BM25 idf with the standard +0.5 smoothing, floored at a small
            # positive value so ubiquitous terms never go negative.
            self._idf[term] = max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 1e-6)

        # Precompute TF-IDF vector norms per document for cosine.
        for chunk in self.chunks:
            norm_sq = 0.0
            for term, tf in self._doc_tf[chunk.id].items():
                w = (1.0 + math.log(tf)) * self._idf.get(term, 0.0)
                norm_sq += w * w
            self._tfidf_norm[chunk.id] = math.sqrt(norm_sq) or 1.0

    @property
    def avgdl(self) -> float:
        if not self._doc_len:
            return 0.0
        return sum(self._doc_len.values()) / len(self._doc_len)

    def _candidates(self, version: str) -> List[Chunk]:
        return [c for c in self.chunks if c.version in (version, "both")]

    def _bm25_scores(self, q_tokens: List[str], candidates: List[Chunk]) -> Dict[str, float]:
        avgdl = self.avgdl or 1.0
        scores: Dict[str, float] = {}
        q_terms = set(q_tokens)
        for chunk in candidates:
            tf = self._doc_tf[chunk.id]
            dl = self._doc_len[chunk.id]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
                score += idf * (f * (BM25_K1 + 1)) / denom
            if score > 0:
                scores[chunk.id] = score
        return scores

    def _tfidf_scores(self, q_tokens: List[str], candidates: List[Chunk]) -> Dict[str, float]:
        q_tf = Counter(q_tokens)
        q_weights = {
            term: (1.0 + math.log(tf)) * self._idf.get(term, 0.0)
            for term, tf in q_tf.items()
        }
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        scores: Dict[str, float] = {}
        for chunk in candidates:
            tf = self._doc_tf[chunk.id]
            dot = 0.0
            for term, qw in q_weights.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                dw = (1.0 + math.log(f)) * self._idf.get(term, 0.0)
                dot += qw * dw
            if dot > 0:
                scores[chunk.id] = dot / (q_norm * self._tfidf_norm[chunk.id])
        return scores

    @staticmethod
    def _ranks(scores: Dict[str, float]) -> Dict[str, int]:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return {cid: rank for rank, (cid, _) in enumerate(ordered, start=1)}

    def retrieve(
        self, question: str, version: str, top_k: int = 5
    ) -> List[RetrievalResult]:
        if version not in ("v1", "v2"):
            raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
        q_tokens = tokenize(question)
        candidates = self._candidates(version)

        bm25 = self._bm25_scores(q_tokens, candidates)
        dense = self._tfidf_scores(q_tokens, candidates)
        bm25_ranks = self._ranks(bm25)
        dense_ranks = self._ranks(dense)

        fused: Dict[str, float] = defaultdict(float)
        for cid, rank in bm25_ranks.items():
            fused[cid] += 1.0 / (RRF_K + rank)
        for cid, rank in dense_ranks.items():
            fused[cid] += 1.0 / (RRF_K + rank)

        by_id = {c.id: c for c in candidates}
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        results: List[RetrievalResult] = []
        for cid, score in ordered[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=by_id[cid],
                    score=score,
                    bm25_rank=bm25_ranks.get(cid),
                    dense_rank=dense_ranks.get(cid),
                )
            )
        return results
