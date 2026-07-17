"""LLM clients behind one interface.

- ``AnthropicClient``  — the production path. Grounded, cited, abstaining answers
  from Claude via structured JSON output.
- ``MockClient``      — offline path. Replays the golden set's *reference*
  snippets so the answer -> sandbox -> eval plumbing runs with no API key (CI,
  or a laptop with no network). Its answers are the answer key, so any quality
  number it produces is PLUMBING, not a quality claim — the eval report labels
  it as such.

``get_client()`` picks Anthropic when it's importable and a key is configured,
otherwise Mock. Override explicitly with ``DOCSTHATRUN_LLM=anthropic|mock``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .config import settings
from .corpus import tokenize  # noqa: F401  (kept for parity / future use)
from .schema import Chunk, RetrievalResult

# Central config is the source of truth (env-driven); see docsthatrun.config.
DEFAULT_MODEL = settings.model
DEFAULT_EFFORT = settings.effort

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "code": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstained": {"type": "boolean"},
    },
    "required": ["answer", "code", "citations", "abstained"],
}

SYSTEM_PROMPT = (
    "You are a documentation assistant for the Pydantic library. You answer "
    "questions about a SPECIFIC target version (v1 = 1.x, or v2 = 2.x).\n\n"
    "Rules:\n"
    "1. Answer ONLY using the provided documentation chunks. Do not use outside "
    "knowledge.\n"
    "2. The answer must be correct for the TARGET VERSION. v1 and v2 differ "
    "(e.g. .dict() vs .model_dump(), @validator vs @field_validator).\n"
    "3. Provide a short, self-contained, runnable Python code snippet that uses "
    "the target-version API and ends with an assert proving the behavior. No "
    "prose in the code field.\n"
    "4. Cite the chunk ids you used in `citations`.\n"
    "5. If the provided chunks do not support a correct answer, set "
    "`abstained` true, leave `code` empty, and say you don't have enough "
    "information — do NOT guess.\n"
    "Return only the JSON object."
)


_ABSTAIN: Dict[str, object] = {
    "answer": "I couldn't produce a grounded answer from the retrieved docs.",
    "code": "",
    "citations": [],
    "abstained": True,
}


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    """Best-effort parse of the model's JSON answer, or ``None``.

    ``output_config.format`` constrains the model to emit a bare JSON object, but
    a truncated generation (``stop_reason == "max_tokens"`` — adaptive thinking
    shares the ``max_tokens`` budget), an empty text block, or otherwise
    malformed output must degrade to an abstain, never crash the request. We also
    tolerate stray ```json fences or leading prose by extracting the first
    balanced ``{...}`` span as a fallback.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: decode the first JSON value starting at the first '{'. raw_decode
    # is string-aware, so a '{' or '}' *inside* a string value (e.g. a code field
    # like "d = {") doesn't throw off the parse the way naive brace-counting would.
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _format_chunks(chunks: List[Chunk]) -> str:
    blocks = []
    for chunk in chunks:
        block = f"[{chunk.id}] (version={chunk.version}) {chunk.title}\n{chunk.text}"
        if chunk.code:
            block += f"\nExample:\n{chunk.code}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_user_prompt(question: str, version: str, chunks: List[Chunk]) -> str:
    return (
        f"TARGET VERSION: {version}\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENTATION CHUNKS:\n{_format_chunks(chunks)}"
    )


class LLMClient:
    def generate(
        self, question: str, version: str, retrieved: List[RetrievalResult]
    ) -> Dict[str, object]:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT):
        import anthropic  # imported lazily so the core has no hard dependency

        self._anthropic = anthropic
        # Timeout + retries from config: the SDK retries 429/5xx/connection
        # errors with backoff, and bounds each call so a hung request can't wedge
        # a worker.
        self.client = anthropic.Anthropic(
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        self.model = model
        self.effort = effort

    def generate(
        self, question: str, version: str, retrieved: List[RetrievalResult]
    ) -> Dict[str, object]:
        chunks = [r.chunk for r in retrieved]
        prompt = build_user_prompt(question, version, chunks)
        resp = self.client.messages.create(
            model=self.model,
            # Headroom for adaptive thinking *and* the JSON answer — they share
            # this budget. Too small a cap truncates mid-JSON (stop_reason
            # "max_tokens"); the answers themselves are short so this is ample.
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            return dict(_ABSTAIN, answer="The request was declined.")
        text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"),
            "",
        )
        parsed = _extract_json(text)
        if parsed is None:
            # Truncated (max_tokens while thinking), empty, or malformed output:
            # abstain rather than crash the caller. `stop` is surfaced for
            # observability so a spike in truncations is visible in logs.
            return dict(
                _ABSTAIN,
                answer=(
                    "I couldn't produce a grounded answer"
                    + (f" (model stopped: {stop})." if stop else ".")
                ),
            )
        return parsed


class MockClient(LLMClient):
    """Deterministic offline client that replays golden reference answers.

    Answers are matched by normalized question text. Unknown questions abstain,
    which is the safe default for the /ask endpoint when run without a key.
    """

    def __init__(self, fixtures: Optional[Dict[str, Dict[str, object]]] = None):
        self.fixtures = fixtures or _load_fixtures_from_golden()

    def generate(
        self, question: str, version: str, retrieved: List[RetrievalResult]
    ) -> Dict[str, object]:
        fixture = self.fixtures.get(_norm(question))
        if fixture is None:
            return {
                "answer": "I don't have documentation that covers this.",
                "code": "",
                "citations": [],
                "abstained": True,
            }
        return dict(fixture)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_fixtures_from_golden() -> Dict[str, Dict[str, object]]:
    from .evals.run_evals import load_golden, load_unanswerable  # lazy import

    fixtures: Dict[str, Dict[str, object]] = {}
    for item in load_golden():
        fixtures[_norm(item.question)] = {
            "answer": f"See docs {', '.join(item.relevant_chunk_ids)}.",
            "code": item.check,
            "citations": list(item.relevant_chunk_ids),
            "abstained": False,
        }
    for item in load_unanswerable():
        fixtures[_norm(item.question)] = {
            "answer": "I don't have documentation that covers this.",
            "code": "",
            "citations": [],
            "abstained": True,
        }
    return fixtures


def get_client(name: Optional[str] = None) -> LLMClient:
    name = name or os.environ.get("DOCSTHATRUN_LLM", "auto")
    if name == "mock":
        return MockClient()
    if name == "anthropic":
        return AnthropicClient()
    # auto
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except Exception:  # pragma: no cover - fall back if SDK missing
            pass
    return MockClient()
