"""Corpus loading + tokenization.

Tokenization keeps identifier-shaped tokens (``model_dump``) whole *and* also
emits their underscore-split parts, so a query for "model dump" still matches
the ``model_dump`` API name. That matters a lot for API docs where the method
name is the whole answer.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

from .schema import Chunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CORPUS_PATH = os.path.join(_DATA_DIR, "corpus", "pydantic_corpus.jsonl")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if "_" in match:
            tokens.extend(part for part in match.split("_") if part)
    return tokens


def load_corpus(path: str = DEFAULT_CORPUS_PATH) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            data = json.loads(raw)
            try:
                chunks.append(
                    Chunk(
                        id=data["id"],
                        version=data["version"],
                        topic=data.get("topic", ""),
                        title=data.get("title", ""),
                        text=data.get("text", ""),
                        code=data.get("code", ""),
                    )
                )
            except KeyError as exc:  # pragma: no cover - corpus authoring guard
                raise ValueError(
                    f"{path}:{line_no} missing required field {exc}"
                ) from exc
    _validate(chunks, path)
    return chunks


def _validate(chunks: List[Chunk], path: str) -> None:
    seen = set()
    for chunk in chunks:
        if chunk.id in seen:
            raise ValueError(f"{path}: duplicate chunk id {chunk.id!r}")
        seen.add(chunk.id)
        if chunk.version not in ("v1", "v2", "both"):
            raise ValueError(
                f"{path}: chunk {chunk.id!r} has bad version {chunk.version!r}"
            )
