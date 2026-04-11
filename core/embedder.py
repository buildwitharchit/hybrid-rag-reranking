"""
core/embedder.py
────────────────
Sentence-transformer embedding models with a module-level singleton cache.

Why a singleton cache?
  Loading a SentenceTransformer model takes 1-5 seconds and consumes
  several hundred MB of RAM.  We load each model exactly once per process
  and reuse the loaded object for every subsequent call.  This means the
  first query is slow (model load) and every subsequent query is fast.

All models run locally on CPU inside the Docker container.
Models are pre-downloaded at image build time (see Dockerfile).
"""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

# Module-level cache: model_name → loaded SentenceTransformer object
_model_cache: Dict[str, object] = {}


def get_embedder(model_name: str):
    """
    Return a cached SentenceTransformer for the given model name.

    Loads and caches on first call; returns cached object on subsequent calls.

    Args:
        model_name: HuggingFace model identifier, e.g. "all-MiniLM-L6-v2".

    Returns:
        A sentence_transformers.SentenceTransformer instance.
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
    return _model_cache[model_name]


def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Embed a list of texts and return their vectors as Python lists.

    Args:
        texts:      List of strings to embed.
        model_name: Which SentenceTransformer model to use.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    embedder = get_embedder(model_name)
    vectors = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str, model_name: str) -> List[float]:
    """
    Embed a single query string.

    Convenience wrapper around embed_texts for single-string use.

    Args:
        query:      The query string to embed.
        model_name: Which SentenceTransformer model to use.

    Returns:
        Embedding vector as a list of floats.
    """
    return embed_texts([query], model_name)[0]


def preload_default_models() -> None:
    """
    Pre-warm the singleton cache with the two most commonly used models.

    Called once at app startup so that the first user query does not
    incur a 5-10 second model-load delay.

    Models pre-loaded:
      • all-MiniLM-L6-v2      — default embedding model (~80 MB)
      • cross-encoder/ms-marco-MiniLM-L-6-v2 — default re-ranker (~85 MB)
    """
    logger.info("Pre-loading default embedding model …")
    get_embedder("all-MiniLM-L6-v2")

    # Also prime the re-ranker cache via reranker.py
    from core.reranker import get_reranker
    logger.info("Pre-loading default cross-encoder re-ranker …")
    get_reranker("cross_encoder_ms_marco")

    logger.info("Default models ready.")
