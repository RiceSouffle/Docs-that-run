"""Tests for the answer layer: the Anthropic client and citation coercion.

Both were entirely untested despite being the most-advertised behaviours in the
project — "cites the docs it used", "refuses when the docs don't cover it", and
"a truncated generation degrades to an abstain rather than crashing". The
AnthropicClient needs no network here: the SDK call is stubbed, because what's
worth pinning is how *we* interpret the response, not what the API returns.
"""

from types import SimpleNamespace

import pytest

from docsthatrun.answer import _coerce, build_answer
from docsthatrun.llm import AnthropicClient
from docsthatrun.schema import Chunk, RetrievalResult

# ---- _coerce: hallucinated citations ---------------------------------------


def _retrieved(*ids):
    return [RetrievalResult(chunk=Chunk(id=i, version="v2", topic="t", title="T", text="x"), score=1.0) for i in ids]


def test_citations_not_in_the_retrieved_set_are_dropped():
    """The grounding claim, made checkable.

    MockClient only ever emits real chunk ids, so before this test you could
    delete the filter in _coerce entirely and the whole suite stayed green.
    """
    raw = {"answer": "a", "code": "c", "citations": ["c_real", "c_invented"], "abstained": False}
    assert _coerce(raw, _retrieved("c_real", "c_other")).citations == ["c_real"]


@pytest.mark.parametrize("bad", [None, "c_real", 42, {"c_real": 1}])
def test_non_list_citations_degrade_to_empty(bad):
    # `"citations": null` used to raise TypeError straight out of the request.
    # _extract_json's salvage path can return any shape, so the schema is not a
    # guarantee here.
    raw = {"answer": "a", "code": "c", "citations": bad, "abstained": False}
    assert _coerce(raw, _retrieved("c_real")).citations == []


def test_non_string_citation_entries_are_dropped():
    raw = {"answer": "a", "code": "c", "citations": ["c_real", 7, None], "abstained": False}
    assert _coerce(raw, _retrieved("c_real")).citations == ["c_real"]


# ---- build_answer: no documentation means no API call ----------------------


class _ExplodingClient:
    def generate(self, question, version, retrieved):  # pragma: no cover - must not run
        raise AssertionError("the model was called with no retrieved documentation")


class _EmptyRetriever:
    def retrieve(self, question, version, top_k=5):
        return []


def test_empty_retrieval_abstains_without_calling_the_model():
    result = build_answer("anything", "v2", _EmptyRetriever(), client=_ExplodingClient())
    assert result.answer.abstained is True
    assert result.answer.code == ""
    assert result.retrieved == []


# ---- AnthropicClient: response interpretation ------------------------------


def _client_with(response, monkeypatch):
    """An AnthropicClient whose SDK call returns `response`, without a network."""
    fake_sdk = SimpleNamespace(Anthropic=lambda **kw: SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_sdk)
    client = AnthropicClient()
    client.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    return client


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def test_refusal_becomes_an_abstain(monkeypatch):
    resp = SimpleNamespace(stop_reason="refusal", content=[])
    out = _client_with(resp, monkeypatch).generate("q", "v2", _retrieved("c"))
    assert out["abstained"] is True
    assert out["code"] == ""
    assert "declined" in out["answer"]


def test_truncated_json_becomes_an_abstain_naming_the_stop_reason(monkeypatch):
    # Adaptive thinking shares the max_tokens budget, so a truncation here is a
    # real failure mode, not a hypothetical.
    resp = SimpleNamespace(stop_reason="max_tokens", content=[_text_block('{"answer": "To serial')])
    out = _client_with(resp, monkeypatch).generate("q", "v2", _retrieved("c"))
    assert out["abstained"] is True
    assert "max_tokens" in out["answer"]


def test_null_content_becomes_an_abstain_rather_than_raising(monkeypatch):
    resp = SimpleNamespace(stop_reason="end_turn", content=None)
    out = _client_with(resp, monkeypatch).generate("q", "v2", _retrieved("c"))
    assert out["abstained"] is True


def test_thinking_blocks_are_skipped_when_finding_the_json(monkeypatch):
    """The response starts with thinking blocks; the JSON is in a later text
    block. Taking content[0] blindly would parse the wrong thing."""
    payload = '{"answer":"use model_dump","code":"x","citations":["c"],"abstained":false}'
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), _text_block(payload)],
    )
    out = _client_with(resp, monkeypatch).generate("q", "v2", _retrieved("c"))
    assert out["abstained"] is False
    assert out["citations"] == ["c"]


def test_each_abstain_gets_its_own_citations_list(monkeypatch):
    """Abstains used to share one module-level list object, so a caller
    mutating it corrupted every later abstain in the process."""
    client = _client_with(SimpleNamespace(stop_reason="refusal", content=[]), monkeypatch)
    first = client.generate("q", "v2", _retrieved("c"))
    second = client.generate("q", "v2", _retrieved("c"))
    first["citations"].append("mutated")
    assert second["citations"] == []
