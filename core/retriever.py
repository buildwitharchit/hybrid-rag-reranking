"""
core/retriever.py
─────────────────
Dense + sparse hybrid retrieval with Reciprocal Rank Fusion (RRF).

Retrieval flow for a single query:
  1. Embed query → ChromaDB cosine search → dense ranked list
  2. Tokenize query → BM25 search → sparse ranked list
  3. Merge all lists via RRF → unified ranked list

For multiple query variants (from query expansion), steps 1-2 are
repeated for each variant, producing 2×N ranked lists that are all
merged together in step 3.

Why RRF instead of score normalisation?
  BM25 scores and cosine similarities are on incompatible scales.
  Normalising them (min-max) is sensitive to outliers.
  RRF uses only rank positions, making it scale-invariant and robust.

  RRF formula: score(doc) = Σ  1 / (k + rank_in_list)
  where k=60 is a smoothing constant (Cormack et al., 2009).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from core.config import RAGConfig
from core.embedder import embed_query
from core.sparse import search_bm25
from core.vector_store import search_dense


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is [(chunk_id, score), ...] sorted
                      descending by score.  Scores are ignored; only rank
                      position matters.
        k:            Smoothing constant (default 60 per original paper).

    Returns:
        Merged list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def hybrid_retrieve(
    queries: List[str],
    collection,
    bm25_index,
    bm25_chunk_ids: List[str],
    config: RAGConfig,
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Run dense + sparse retrieval for all query variants and fuse results.

    Args:
        queries:         List of query strings (original + expanded variants).
        collection:      ChromaDB collection to search.
        bm25_index:      BM25Okapi instance (or None if sparse disabled).
        bm25_chunk_ids:  Ordered chunk ids for the BM25 corpus.
        config:          RAGConfig for this pipeline (controls which
                         retrieval methods are enabled).
        top_k:           Number of results to fetch per individual search.

    Returns:
        List of (chunk_id, rrf_score) sorted by RRF score descending.
    """
    all_lists: List[List[Tuple[str, float]]] = []

    for query in queries:
        # Dense search
        query_vec = embed_query(query, config.embedding_model)
        dense_results = search_dense(collection, query_vec, top_k)
        # search_dense returns (chunk_id, score, meta); we need (chunk_id, score)
        dense_ranked = [(cid, score) for cid, score, _ in dense_results]
        if dense_ranked:
            all_lists.append(dense_ranked)

        # Sparse search (if enabled)
        if config.sparse_search != "none" and bm25_index is not None:
            sparse_results = search_bm25(query, bm25_index, bm25_chunk_ids, top_k)
            if sparse_results:
                all_lists.append(sparse_results)

    if not all_lists:
        return []

    # Fuse all lists
    if config.fusion_method == "rrf":
        merged = reciprocal_rank_fusion(all_lists)
    elif config.fusion_method == "weighted":
        merged = _weighted_sum_fusion(all_lists)
    elif config.fusion_method == "combsum":
        merged = _combsum_fusion(all_lists)
    else:
        merged = reciprocal_rank_fusion(all_lists)

    return merged


def naive_retrieve(
    query: str,
    collection,
    embedding_model: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Naive retrieval: dense-only, no expansion, no fusion, no re-ranking.

    Used as the baseline for evaluation comparisons.

    Args:
        query:           The raw user query.
        collection:      ChromaDB collection.
        embedding_model: Which embedding model to use.
        top_k:           Number of results to return.

    Returns:
        List of (chunk_id, cosine_score) sorted descending.
    """
    query_vec = embed_query(query, embedding_model)
    dense_results = search_dense(collection, query_vec, top_k)
    return [(cid, score) for cid, score, _ in dense_results]


# ---------------------------------------------------------------------------
# Alternative fusion methods
# ---------------------------------------------------------------------------

def _weighted_sum_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Normalised weighted sum fusion.

    Normalises each list to [0,1] via min-max, then averages.
    Sensitive to outliers; use RRF for more robust results.
    """
    normalised: List[List[Tuple[str, float]]] = []
    for ranked in ranked_lists:
        if not ranked:
            continue
        scores = [s for _, s in ranked]
        lo, hi = min(scores), max(scores)
        denom = (hi - lo) if hi != lo else 1.0
        norm = [((cid, (s - lo) / denom)) for cid, s in ranked]
        normalised.append(norm)

    agg: dict[str, float] = {}
    count: dict[str, int] = {}
    for ranked in normalised:
        for cid, score in ranked:
            agg[cid] = agg.get(cid, 0.0) + score
            count[cid] = count.get(cid, 0) + 1

    averaged = {cid: agg[cid] / count[cid] for cid in agg}
    return sorted(averaged.items(), key=lambda x: x[1], reverse=True)


def _combsum_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
) -> List[Tuple[str, float]]:
    """
    CombSUM: sum raw scores across all lists (no normalisation).

    Faster than weighted sum but scale-sensitive.
    """
    agg: dict[str, float] = {}
    for ranked in ranked_lists:
        for cid, score in ranked:
            agg[cid] = agg.get(cid, 0.0) + score
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)
