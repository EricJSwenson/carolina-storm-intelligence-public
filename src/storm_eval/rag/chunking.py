"""Document chunking for the gold layer / vector store.

Word-window chunking with overlap. Kept deliberately simple and deterministic
so chunk-size sweeps in the experiments layer isolate the variable under test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from storm_eval.config import settings

_WORD = re.compile(r"\S+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    ordinal: int


def chunk_text(doc_id: str, text: str, size: int | None = None,
               overlap: int | None = None) -> List[Chunk]:
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    words = [m.group(0) for m in _WORD.finditer(text)]
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: List[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(doc_id, f"{doc_id}::{i}", " ".join(window), i))
        if start + size >= len(words):
            break
    return chunks
