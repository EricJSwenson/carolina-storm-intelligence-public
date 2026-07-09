"""Central configuration for the storm_eval platform.

Values resolve in this order: explicit argument -> environment variable ->
default below. Everything the demo needs runs locally with the defaults; the
production overrides (Databricks, OpenAI, Pinecone) are read from the
environment when present. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("STORM_DATA_DIR", ROOT / "data"))
SAMPLES_DIR = DATA_DIR / "samples"
WAREHOUSE = Path(os.getenv("STORM_WAREHOUSE", DATA_DIR / "warehouse.duckdb"))
VECTORSTORE_DIR = Path(os.getenv("STORM_VECTORSTORE", DATA_DIR / "vectorstore"))


@dataclass
class Settings:
    # --- backends: "local" keeps everything offline & deterministic ---
    embedding_backend: str = os.getenv("STORM_EMBEDDING_BACKEND", "local")
    llm_backend: str = os.getenv("STORM_LLM_BACKEND", "mock")
    vectorstore_backend: str = os.getenv("STORM_VECTORSTORE_BACKEND", "local")

    # --- chunking ---
    chunk_size: int = int(os.getenv("STORM_CHUNK_SIZE", "320"))
    chunk_overlap: int = int(os.getenv("STORM_CHUNK_OVERLAP", "48"))

    # --- retrieval ---
    top_k: int = int(os.getenv("STORM_TOP_K", "4"))
    embedding_dim: int = int(os.getenv("STORM_EMBEDDING_DIM", "256"))

    # --- external services (only used when backends != local/mock) ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = os.getenv("STORM_OPENAI_MODEL", "gpt-4o-mini")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))

    # --- tracking ---
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", str(DATA_DIR / "mlruns"))
    experiment_name: str = os.getenv("STORM_EXPERIMENT", "storm-eval")

    warehouse_path: Path = WAREHOUSE
    vectorstore_dir: Path = VECTORSTORE_DIR


settings = Settings()
