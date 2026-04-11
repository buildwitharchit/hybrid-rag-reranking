"""
core/reranker.py
────────────────
Cross-encoder re-ranking with a module-level singleton cache.

Why cross-encoders beat bi-encoders for re-ranking:
  Bi-encoders (used for dense retrieval) encode query and document
  separately, then compare their vectors.  They cannot model interactions
  between query and document tokens.

  Cross-encoders encode the concatenated [query, document] pair and output
  a single relevance score.  Full attention runs over both, so it can
  detect negation, topic drift, and subtle relevance signals that a
  bi-encoder misses.

  The cost: cross-encoders cannot be pre-computed, so they must run at
  query time.  We solve this by applying them only to the top-20 candidates
  from the fast bi-encoder stage (two-stage retrieval).

Supported local models (free, run on CPU):
  • cross_encoder_ms_marco  → cross-encoder/ms-marco-MiniLM-L-6-v2  (~85 MB)
  • bge_reranker_large      → BAAI/bge-reranker-large               (~1.3 GB)

Disabled (API / cost):
  • Cohere Rerank v3
  • Jina Reranker
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from loguru import logger

# Mapping from config value → HuggingFace model identifier
_MODEL_MAP: Dict[str, str] = {
    "cross_encoder_ms_marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge_reranker_large": "BAAI/bge-reranker-large",
}

# Module-level singleton cache
_reranker_cache: Dict[str, object] = {}


def get_reranker(model_key: str):
    """
    Return a cached CrossEncoder for the given config key.

    Loads and caches on first call; returns cached object subsequently.

    Args:
        model_key: One of "cross_encoder_ms_marco" or "bge_reranker_large".

    Returns:
        sentence_transformers.cross_encoder.CrossEncoder instance.

    Raises:
        ValueError: If model_key is not a recognised local re-ranker.
    """
    if model_key not in _MODEL_MAP:
        raise ValueError(
            f"Unknown re-ranker key '{model_key}'. "
            f"Valid keys: {list(_MODEL_MAP.keys())}"
        )
    if model_key not in _reranker_cache:
        from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
        hf_name = _MODEL_MAP[model_key]
        logger.info(f"Loading cross-encoder re-ranker: {hf_name}")
        _reranker_cache[model_key] = CrossEncoder(hf_name)
        logger.info(f"Re-ranker loaded: {hf_name}")
    return _reranker_cache[model_key]


def rerank(
    query: str,
    chunk_ids: List[str],
    chunk_texts: List[str],
    model_key: str,
) -> List[Tuple[str, float]]:
    """
    Re-rank a list of chunks using a cross-encoder.

    For each (query, chunk_text) pair the cross-encoder outputs a single
    relevance score.  Results are sorted descending by that score.

    Args:
        query:       The original user query.
        chunk_ids:   Ordered list of chunk id strings.
        chunk_texts: Parallel list of chunk text strings.
        model_key:   Which re-ranker model to use.

    Returns:
        List of (chunk_id, score) tuples sorted by relevance score descending.
    """
    if not chunk_ids:
        return []

    if model_key == "none":
        # No re-ranking — return input order with placeholder scores
        return [(cid, float(len(chunk_ids) - i)) for i, cid in enumerate(chunk_ids)]

    try:
        model = get_reranker(model_key)
        pairs = [(query, text) for text in chunk_texts]
        scores = model.predict(pairs)  # returns numpy array

        scored = list(zip(chunk_ids, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    except Exception as exc:
        logger.error(f"Re-ranking failed: {exc}. Returning original order.")
        return [(cid, float(len(chunk_ids) - i)) for i, cid in enumerate(chunk_ids)]
