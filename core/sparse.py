"""
core/sparse.py
──────────────
BM25 sparse retrieval index — build, save, load, and query.

BM25 (Best Match 25) is a probabilistic ranking function that scores
documents by term frequency weighted by inverse document frequency (IDF).
It excels at exact keyword matching where dense embeddings fall short
(e.g. product codes, medical terms, rare proper nouns).

The BM25 index is an in-memory Python object.  It is serialised to disk
with pickle after every ingestion so it survives Streamlit reruns.
Each RAG has its own index file at store/{rag_id}/bm25_index.pkl.

The pickle stores a tuple: (BM25Okapi index, List[chunk_id]).
The chunk_ids list must stay in sync with the tokenised corpus order
so we can map BM25 scores back to ChromaDB chunk ids.

SECURITY NOTE: The pickle file is written and read only by this application
running in a controlled Docker container. It is never sourced from user
uploads. BM25Okapi is a pure-Python data structure with no code execution
on deserialisation.
"""

from __future__ import annotations

import os
import pickle  # noqa: S403 — intentional; see module docstring for rationale
from typing import List, Optional, Tuple

from loguru import logger

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")


def _bm25_path(rag_id: str) -> str:
    return os.path.join(_STORE_ROOT, rag_id, "bm25_index.pkl")


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# Build / Save / Load
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: List[str], chunk_ids: List[str]):
    """
    Build a BM25Okapi index from a list of text chunks.

    Args:
        chunks:    List of chunk text strings.
        chunk_ids: Parallel list of chunk id strings.

    Returns:
        BM25Okapi index object.
    """
    from rank_bm25 import BM25Okapi  # type: ignore

    tokenized = [_tokenize(c) for c in chunks]
    index = BM25Okapi(tokenized)
    logger.debug(f"Built BM25 index over {len(chunks)} chunks")
    return index


def save_bm25_index(rag_id: str, index, chunk_ids: List[str]) -> None:
    """
    Serialise the BM25 index and chunk_ids to disk.

    Args:
        rag_id:    The RAG this index belongs to.
        index:     BM25Okapi instance.
        chunk_ids: Ordered list of chunk ids matching the index corpus.
    """
    path = _bm25_path(rag_id)
    try:
        with open(path, "wb") as f:
            pickle.dump((index, chunk_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved BM25 index to {path}")
    except Exception as exc:
        logger.error(f"Failed to save BM25 index: {exc}")
        raise


def load_bm25_index(rag_id: str) -> Optional[Tuple]:
    """
    Load the BM25 index from disk.

    Args:
        rag_id: The RAG whose index to load.

    Returns:
        Tuple of (BM25Okapi, List[chunk_id]) or None if no index exists.
    """
    path = _bm25_path(rag_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)  # noqa: S301
        logger.debug(f"Loaded BM25 index from {path}")
        return result
    except Exception as exc:
        logger.error(f"BM25 index corrupt at {path}: {exc}. Deleting.")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_bm25(
    query: str,
    index,
    chunk_ids: List[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Query the BM25 index and return ranked results.

    Args:
        query:     Raw query string (will be tokenized internally).
        index:     BM25Okapi instance.
        chunk_ids: Ordered chunk ids matching the index corpus.
        top_k:     Number of top results to return.

    Returns:
        List of (chunk_id, score) tuples sorted by score descending.
        Returns an empty list if the index is empty.
    """
    if index is None or not chunk_ids:
        logger.warning("BM25 search called with no index — returning empty results.")
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = index.get_scores(tokenized_query)

    # Pair each score with its chunk_id, then sort
    scored = list(zip(chunk_ids, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
