"""Retrieval correctness + the version-filter invariant."""

from docsthatrun.corpus import load_corpus, tokenize
from docsthatrun.evals.run_evals import load_golden
from docsthatrun.retrieve import HybridRetriever


def test_tokenize_splits_identifiers():
    toks = tokenize("model_dump")
    assert "model_dump" in toks and "model" in toks and "dump" in toks


def test_version_filter_excludes_other_version():
    retriever = HybridRetriever(load_corpus())
    # A v2 query must never surface v1-only chunks (and vice versa).
    for version, forbidden in (("v2", "v1"), ("v1", "v2")):
        results = retriever.retrieve("how do I serialize a model", version, top_k=20)
        for r in results:
            assert r.chunk.version in (version, "both"), (
                f"{r.chunk.id} ({r.chunk.version}) leaked into a {version} query"
            )
        # sanity: the forbidden version's exclusive chunks are absent
        assert forbidden not in {r.chunk.version for r in results}


def test_golden_relevant_chunk_ranks_top1():
    retriever = HybridRetriever(load_corpus())
    for item in load_golden():
        results = retriever.retrieve(item.question, item.version, top_k=5)
        ids = [r.chunk.id for r in results]
        assert item.relevant_chunk_ids[0] in ids, (
            f"{item.id}: expected {item.relevant_chunk_ids[0]} in top-5, got {ids}"
        )
