"""Embedding backends behind one interface.

``local`` is a deterministic hashing embedder (pure numpy, no downloads) so the
pipeline runs offline and reproducibly in CI. ``openai`` and
``sentence-transformers`` are drop-in production backends selected via config.
Swapping backends is exactly the kind of change the experiments layer A/B tests.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Protocol

import numpy as np

from storm_eval.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: List[str]) -> np.ndarray: ...


class LocalHashingEmbedder:
    """Deterministic bag-of-hashed-bigrams embedding, L2-normalized.

    Not as expressive as a neural model, but it is fast, dependency-free, and
    reproducible -- ideal for tests and offline demos. Same query/text overlap
    still produces meaningful cosine similarity for retrieval.
    """

    name = "local-hashing"

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embedding_dim

    def _vector(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # imported lazily

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.dim = 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(backend: str | None = None) -> Embedder:
    backend = backend or settings.embedding_backend
    if backend == "local":
        return LocalHashingEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedding backend: {backend}")
