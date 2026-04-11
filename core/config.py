"""
core/config.py
──────────────
Pydantic data model for a RAG pipeline configuration.

Every RAG the user creates produces one RAGConfig object.  The config is split
into two logical groups:

  • Ingestion config  — locked once the first document is ingested because
                        the embedding vectors stored in ChromaDB are tied to a
                        specific model and chunking strategy.
  • Retrieval config  — freely editable at any time; these parameters are
                        applied at query time, not at index-build time.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Allowed values for every dropdown option
# ---------------------------------------------------------------------------

CHUNKING_STRATEGIES = Literal["recursive", "fixed", "sentence", "semantic"]

EMBEDDING_MODELS = Literal[
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "bge-large-en-v1.5",
]

VECTOR_STORES = Literal["chromadb"]           # faiss is greyed-out in UI

SPARSE_SEARCH = Literal["none", "bm25", "bm25plus"]

FUSION_METHODS = Literal["rrf", "weighted", "combsum"]

QUERY_EXPANSION = Literal["none", "llm_variants", "hyde"]

RERANKERS = Literal[
    "none",
    "cross_encoder_ms_marco",
    "bge_reranker_large",
]

LLM_MODELS = [
    "meta-llama/llama-3-70b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o-mini",
    "google/gemma-2-9b-it",
]

# Options shown greyed-out in the UI so users know they exist but cannot use
# them without additional cost / infrastructure.
DISABLED_EMBEDDING_MODELS = [
    "OpenAI text-embedding-3-small (API - disabled)",
    "OpenAI text-embedding-3-large (API - disabled)",
]

DISABLED_VECTOR_STORES = [
    "FAISS (coming soon - disabled)",
    "Pinecone (API - disabled)",
    "Qdrant (server - disabled)",
]

DISABLED_SPARSE = [
    "SPLADE (GPU required - disabled)",
]

DISABLED_RERANKERS = [
    "Cohere Rerank v3 (API - disabled)",
    "Jina Reranker (API - disabled)",
]

DISABLED_QUERY_EXPANSION = [
    "T5 Paraphrase (GPU required - disabled)",
]


# ---------------------------------------------------------------------------
# Main config model
# ---------------------------------------------------------------------------

class RAGConfig(BaseModel):
    """Complete configuration for one RAG pipeline instance."""

    # ── Identity ────────────────────────────────────────────────────────────
    id: str                   # slugified name, e.g. "medrag"
    name: str                 # display name, e.g. "MedRAG"
    description: str = ""
    created_at: str           # ISO-8601 timestamp string
    doc_count: int = 0
    chunk_count: int = 0
    last_used: Optional[str] = None

    # ── Ingestion config (locked after first doc) ────────────────────────────
    chunking_strategy: CHUNKING_STRATEGIES = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: EMBEDDING_MODELS = "all-MiniLM-L6-v2"
    vector_store: VECTOR_STORES = "chromadb"

    # ── Retrieval config (editable any time) ─────────────────────────────────
    sparse_search: SPARSE_SEARCH = "bm25"
    fusion_method: FUSION_METHODS = "rrf"
    query_expansion: QUERY_EXPANSION = "llm_variants"
    top_k_retrieval: int = 20
    reranker: RERANKERS = "cross_encoder_ms_marco"
    top_k_final: int = 5
    llm_model: str = "meta-llama/llama-3-70b-instruct"
    temperature: float = 0.0

    # ── Derived flags (computed, not stored directly in form) ────────────────
    is_locked: bool = False      # True after first document is ingested
    has_eval_set: bool = False
    has_eval_results: bool = False

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_valid(cls, v: int) -> int:
        if v not in (256, 512, 768, 1024):
            raise ValueError("chunk_size must be one of 256, 512, 768, 1024")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_valid(cls, v: int) -> int:
        if v not in (0, 25, 50, 100):
            raise ValueError("chunk_overlap must be one of 0, 25, 50, 100")
        return v

    @field_validator("top_k_retrieval")
    @classmethod
    def top_k_retrieval_valid(cls, v: int) -> int:
        if v not in (10, 20, 30, 50):
            raise ValueError("top_k_retrieval must be one of 10, 20, 30, 50")
        return v

    @field_validator("top_k_final")
    @classmethod
    def top_k_final_valid(cls, v: int) -> int:
        if v not in (3, 5, 7, 10):
            raise ValueError("top_k_final must be one of 3, 5, 7, 10")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_valid(cls, v: float) -> float:
        if v not in (0.0, 0.3, 0.7, 1.0):
            raise ValueError("temperature must be one of 0.0, 0.3, 0.7, 1.0")
        return v
