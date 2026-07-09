"""Vector store backends behind one interface.

``local`` is an in-process numpy store (cosine similarity) persisted to disk --
no server needed. ``chroma`` and ``pinecone`` are production adapters selected
via config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

from storm_eval.config import settings


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class VectorStore(Protocol):
    def add(self, ids, doc_ids, texts, vectors) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> List[Hit]: ...


class LocalVectorStore:
    """In-memory cosine store with JSON+npy persistence."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or settings.vectorstore_dir)
        self.ids: List[str] = []
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    def add(self, ids, doc_ids, texts, vectors) -> None:
        self.ids += list(ids)
        self.doc_ids += list(doc_ids)
        self.texts += list(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def search(self, query_vec: np.ndarray, k: int) -> List[Hit]:
        if self.matrix is None or not len(self.ids):
            return []
        q = query_vec.reshape(-1)
        sims = self.matrix @ q / ((np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q)) + 1e-9)
        top = np.argsort(-sims)[:k]
        return [Hit(self.ids[i], self.doc_ids[i], self.texts[i], float(sims[i])) for i in top]

    def persist(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "matrix.npy", self.matrix)
        (self.path / "meta.json").write_text(
            json.dumps({"ids": self.ids, "doc_ids": self.doc_ids, "texts": self.texts})
        )

    def load(self) -> "LocalVectorStore":
        self.matrix = np.load(self.path / "matrix.npy")
        meta = json.loads((self.path / "meta.json").read_text())
        self.ids, self.doc_ids, self.texts = meta["ids"], meta["doc_ids"], meta["texts"]
        return self


class ChromaVectorStore:
    def __init__(self, collection: str = "storm_eval"):
        import chromadb  # lazy

        self.client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
        self.col = self.client.get_or_create_collection(collection)

    def add(self, ids, doc_ids, texts, vectors) -> None:
        self.col.add(ids=list(ids), documents=list(texts),
                     embeddings=[v.tolist() for v in vectors],
                     metadatas=[{"doc_id": d} for d in doc_ids])

    def search(self, query_vec: np.ndarray, k: int) -> List[Hit]:
        res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=k)
        out = []
        for cid, txt, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append(Hit(cid, meta["doc_id"], txt, 1.0 - dist))
        return out


def get_vectorstore(backend: str | None = None) -> VectorStore:
    backend = backend or settings.vectorstore_backend
    if backend == "local":
        return LocalVectorStore()
    if backend == "chroma":
        return ChromaVectorStore()
    raise ValueError(f"unknown vectorstore backend: {backend}")
