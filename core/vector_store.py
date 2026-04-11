"""
core/vector_store.py
────────────────────
ChromaDB wrapper for dense vector storage and similarity search.

One ChromaDB client is shared across the entire process (singleton).
Each RAG pipeline gets its own named collection inside that shared client.
Collections are completely isolated — querying one never touches another.

ChromaDB persists data to disk at store/chroma_db/ so collections survive
app restarts (within the same Docker container lifecycle).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from loguru import logger

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")

# Module-level singleton ChromaDB client
_chroma_client = None


def get_chroma_client():
    """
    Return the shared ChromaDB persistent client.

    Creates the client on first call; returns cached instance subsequently.
    The client points to store/chroma_db/ for on-disk persistence.

    Returns:
        chromadb.PersistentClient instance.
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb  # type: ignore
        db_path = os.path.join(_STORE_ROOT, "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=db_path)
        logger.info(f"ChromaDB client initialised at {db_path}")
    return _chroma_client


def get_or_create_collection(rag_id: str, embedding_model: str):
    """
    Return (or create) the ChromaDB collection for a RAG.

    The embedding_model name is stored in the collection metadata for
    informational purposes; ChromaDB itself does not use it for anything.

    Args:
        rag_id:          Unique RAG identifier (used as collection name).
        embedding_model: Name of the embedding model used to generate vectors.

    Returns:
        chromadb.Collection instance.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=rag_id,
        metadata={"embedding_model": embedding_model},
    )
    logger.debug(f"Using ChromaDB collection '{rag_id}'")
    return collection


def add_chunks(
    collection,
    chunk_ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict],
) -> None:
    """
    Add pre-embedded chunks to a ChromaDB collection.

    Chunks that already exist (same id) are upserted (overwritten).

    Args:
        collection: ChromaDB Collection object.
        chunk_ids:  Unique id for each chunk.
        texts:      Raw text of each chunk.
        embeddings: Pre-computed embedding vectors for each chunk.
        metadatas:  Metadata dicts (source_doc, page_number, etc.).
    """
    if not chunk_ids:
        return
    collection.upsert(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.debug(f"Upserted {len(chunk_ids)} chunks into '{collection.name}'")


def search_dense(
    collection,
    query_embedding: List[float],
    top_k: int,
) -> List[Tuple[str, float, dict]]:
    """
    Perform approximate nearest-neighbour search in the collection.

    Args:
        collection:      ChromaDB Collection to search.
        query_embedding: Embedded query vector.
        top_k:           Number of results to return.

    Returns:
        List of (chunk_id, cosine_score, metadata) tuples, sorted by score
        descending.  cosine_score is in [0, 1] (higher = more similar).
    """
    count = collection.count()
    if count == 0:
        return []
    k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["distances", "metadatas", "documents"],
    )

    ids = results["ids"][0]
    # ChromaDB returns L2 distance by default when using cosine metric; however
    # with cosine space the distance IS 1 - cosine_sim.  We convert back.
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    output = []
    for chunk_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
        cosine_score = max(0.0, 1.0 - dist)   # distance → similarity
        meta_with_text = {**meta, "text": doc}
        output.append((chunk_id, cosine_score, meta_with_text))

    # Already sorted by distance ascending (similarity descending) from ChromaDB
    return output


def get_chunks_by_ids(collection, chunk_ids: List[str]) -> List[dict]:
    """
    Retrieve full chunk data (text + metadata) for a list of chunk ids.

    Used by the Internals tab to display retrieved chunks.

    Args:
        collection: ChromaDB Collection.
        chunk_ids:  List of chunk id strings to look up.

    Returns:
        List of dicts with keys: chunk_id, text, and all metadata fields.
    """
    if not chunk_ids:
        return []
    results = collection.get(
        ids=chunk_ids,
        include=["documents", "metadatas"],
    )
    output = []
    for cid, doc, meta in zip(
        results["ids"], results["documents"], results["metadatas"]
    ):
        output.append({"chunk_id": cid, "text": doc, **meta})
    return output


def get_all_chunks(collection) -> List[dict]:
    """
    Return every chunk in the collection.

    Used to rebuild the BM25 index after ingestion.

    Args:
        collection: ChromaDB Collection.

    Returns:
        List of dicts with keys: chunk_id, text, and metadata fields.
    """
    count = collection.count()
    if count == 0:
        return []
    results = collection.get(include=["documents", "metadatas"])
    output = []
    for cid, doc, meta in zip(
        results["ids"], results["documents"], results["metadatas"]
    ):
        output.append({"chunk_id": cid, "text": doc, **meta})
    return output


def delete_collection(rag_id: str) -> None:
    """
    Drop the ChromaDB collection for a RAG.

    Safe to call even if the collection does not exist (no-op).

    Args:
        rag_id: The RAG whose collection should be removed.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=rag_id)
        logger.info(f"Dropped ChromaDB collection '{rag_id}'")
    except Exception as exc:
        logger.warning(f"delete_collection('{rag_id}'): {exc}")
