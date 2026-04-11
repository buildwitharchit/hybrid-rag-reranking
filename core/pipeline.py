"""
core/pipeline.py
────────────────
The Pipeline class — the central orchestrator of one RAG pipeline instance.

Each RAG the user creates corresponds to one Pipeline object.  The object
holds references to that RAG's ChromaDB collection and BM25 index, and
provides three public methods:

  • ingest_text(text, source_name, page_number) → int
      Chunk → embed → store in ChromaDB + BM25.

  • query(user_query) → dict
      Full 7-step query pipeline.  Returns the answer plus all
      intermediate internals (for the Internals tab).

  • rebuild_bm25() → None
      Re-build the BM25 index from all chunks currently in ChromaDB.
      Called automatically after ingestion.

The Pipeline object is stored in st.session_state so it is reused across
Streamlit reruns without re-loading models or re-opening connections.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from loguru import logger

from core.config import RAGConfig
from core.chunker import chunk_document
from core.embedder import embed_texts, embed_query
from core.generator import generate
from core.query_expander import expand_query
from core.registry import update_rag, rag_dir
from core.reranker import rerank
from core.retriever import hybrid_retrieve, naive_retrieve, reciprocal_rank_fusion
from core.sparse import build_bm25_index, load_bm25_index, save_bm25_index, search_bm25
from core.vector_store import (
    get_or_create_collection,
    add_chunks,
    get_chunks_by_ids,
    get_all_chunks,
    search_dense,
)


class Pipeline:
    """
    Orchestrates ingestion, retrieval, re-ranking, and generation for one RAG.

    Attributes:
        config:          The RAGConfig for this pipeline.
        api_key:         OpenRouter API key.
        collection:      ChromaDB collection for this RAG.
        bm25_index:      In-memory BM25 index (or None if not yet built).
        bm25_chunk_ids:  Ordered list of chunk ids in the BM25 corpus.
    """

    def __init__(self, config: RAGConfig, api_key: str) -> None:
        self.config = config
        self.api_key = api_key

        # ChromaDB collection — created or retrieved by name
        self.collection = get_or_create_collection(config.id, config.embedding_model)

        # BM25 index — loaded from disk if available
        result = load_bm25_index(config.id)
        if result is not None:
            self.bm25_index, self.bm25_chunk_ids = result
        else:
            self.bm25_index = None
            self.bm25_chunk_ids = []

        logger.info(
            f"Pipeline ready: '{config.name}' | "
            f"{config.chunk_count} chunks | "
            f"BM25={'loaded' if self.bm25_index else 'empty'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        source_name: str,
        page_number: Optional[int] = None,
    ) -> int:
        """
        Chunk a document, embed the chunks, and store them in ChromaDB + BM25.

        Args:
            text:        Full text of the document to ingest.
            source_name: Display name (filename or URL) for source attribution.
            page_number: Optional page number for PDF documents.

        Returns:
            Number of chunks created and stored.
        """
        text = text.strip()
        if not text:
            logger.warning("ingest_text called with empty text — skipping.")
            return 0

        # 1. Chunk the text
        chunks = chunk_document(
            text,
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return 0

        # 2. Generate globally unique chunk ids
        # We use the current ChromaDB count as the starting offset so ids
        # never collide across multiple ingestion calls.
        existing_count = self.collection.count()
        chunk_ids = [
            f"{self.config.id}_chunk_{existing_count + i}"
            for i in range(len(chunks))
        ]

        # 3. Embed all chunks in batches of 64
        logger.info(f"Embedding {len(chunks)} chunks from '{source_name}'…")
        embeddings = embed_texts(chunks, self.config.embedding_model)

        # 4. Build metadata for each chunk
        metadatas = [
            {
                "chunk_id": chunk_ids[i],
                "source_doc": source_name,
                "page_number": page_number if page_number is not None else -1,
                "chunk_index": existing_count + i,
                "char_count": len(chunks[i]),
            }
            for i in range(len(chunks))
        ]

        # 5. Store in ChromaDB
        add_chunks(self.collection, chunk_ids, chunks, embeddings, metadatas)

        # 6. Rebuild BM25 from scratch (incorporates all existing + new chunks)
        self.rebuild_bm25()

        # 7. Update registry stats
        new_chunk_count = self.collection.count()
        # Track unique source docs via metadata
        try:
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            unique_sources = len({m["source_doc"] for m in all_meta})
        except Exception:
            unique_sources = self.config.doc_count + 1

        update_rag(
            self.config.id,
            {
                "doc_count": unique_sources,
                "chunk_count": new_chunk_count,
                "is_locked": True,
            },
        )
        # Refresh local config reference
        from core.registry import get_rag
        refreshed = get_rag(self.config.id)
        if refreshed:
            self.config = refreshed

        logger.info(f"Ingested {len(chunks)} chunks from '{source_name}'.")
        return len(chunks)

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, user_query: str) -> Dict:
        """
        Run the full retrieval-augmented generation pipeline.

        Steps:
          1. Query expansion (LLM or HyDE, if enabled).
          2. Hybrid retrieval — dense + sparse for all query variants.
          3. Cross-encoder re-ranking of top candidates.
          4. Context assembly from top-K final chunks.
          5. LLM generation with citations.

        Args:
            user_query: The raw question from the user.

        Returns:
            Dict with keys:
              answer:              str
              sources:             List[dict]
              query_variants:      List[str]
              pre_rerank_results:  List[dict]
              post_rerank_results: List[dict]
              latency_breakdown:   dict of component timings in ms
        """
        # ── Step 1: Query expansion ──────────────────────────────────────────
        t0 = time.perf_counter()
        queries = expand_query(
            user_query,
            method=self.config.query_expansion,
            llm_model=self.config.llm_model,
            api_key=self.api_key,
        )
        expansion_ms = int((time.perf_counter() - t0) * 1000)

        # ── Step 2: Hybrid retrieval ─────────────────────────────────────────
        t0 = time.perf_counter()
        fused = hybrid_retrieve(
            queries=queries,
            collection=self.collection,
            bm25_index=self.bm25_index,
            bm25_chunk_ids=self.bm25_chunk_ids,
            config=self.config,
            top_k=self.config.top_k_retrieval,
        )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        if not fused:
            return self._empty_result(user_query, queries)

        # Fetch full chunk texts for the retrieved ids
        top_ids = [cid for cid, _ in fused[: self.config.top_k_retrieval]]
        chunk_data = {
            c["chunk_id"]: c
            for c in get_chunks_by_ids(self.collection, top_ids)
        }

        # Also gather BM25 ranks for display in Internals
        bm25_ranks = {}
        if self.bm25_index is not None:
            bm25_results = search_bm25(
                user_query, self.bm25_index, self.bm25_chunk_ids,
                len(self.bm25_chunk_ids)
            )
            bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_results)}

        # Dense scores for display
        query_vec = embed_query(user_query, self.config.embedding_model)
        dense_raw = search_dense(self.collection, query_vec, self.config.top_k_retrieval)
        dense_scores = {cid: score for cid, score, _ in dense_raw}

        # Build pre-rerank results list
        pre_rerank_results = []
        for cid, rrf_score in fused[: self.config.top_k_retrieval]:
            c = chunk_data.get(cid, {})
            pre_rerank_results.append(
                {
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "source": c.get("source_doc", ""),
                    "rrf_score": round(rrf_score, 6),
                    "dense_score": round(dense_scores.get(cid, 0.0), 4),
                    "bm25_rank": bm25_ranks.get(cid, None),
                }
            )

        # ── Step 3: Re-ranking ───────────────────────────────────────────────
        t0 = time.perf_counter()
        rerank_input_ids = top_ids
        rerank_input_texts = [
            chunk_data.get(cid, {}).get("text", "") for cid in rerank_input_ids
        ]

        if self.config.reranker != "none":
            reranked = rerank(
                query=user_query,
                chunk_ids=rerank_input_ids,
                chunk_texts=rerank_input_texts,
                model_key=self.config.reranker,
            )
        else:
            # No re-ranker: use fused order with placeholder scores
            reranked = [(cid, float(len(rerank_input_ids) - i))
                        for i, cid in enumerate(rerank_input_ids)]
        rerank_ms = int((time.perf_counter() - t0) * 1000)

        # Build post-rerank results list (with rank-change deltas)
        pre_rank_map = {cid: i + 1 for i, (cid, _) in enumerate(fused[: self.config.top_k_retrieval])}
        post_rerank_results = []
        for new_rank, (cid, score) in enumerate(reranked[: self.config.top_k_final], start=1):
            c = chunk_data.get(cid, {})
            post_rerank_results.append(
                {
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "source": c.get("source_doc", ""),
                    "cross_encoder_score": round(score, 4),
                    "rank_before": pre_rank_map.get(cid, new_rank),
                    "rank_after": new_rank,
                }
            )

        # ── Step 4: Assemble context ─────────────────────────────────────────
        final_ids = [cid for cid, _ in reranked[: self.config.top_k_final]]
        context_chunks = []
        for cid in final_ids:
            c = chunk_data.get(cid, {})
            context_chunks.append(
                {
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "source_doc": c.get("source_doc", ""),
                    "page_number": c.get("page_number"),
                }
            )

        # ── Step 5: LLM generation ───────────────────────────────────────────
        t0 = time.perf_counter()
        answer, sources = generate(
            query=user_query,
            context_chunks=context_chunks,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            api_key=self.api_key,
        )
        generation_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "answer": answer,
            "sources": sources,
            "query_variants": queries,
            "pre_rerank_results": pre_rerank_results,
            "post_rerank_results": post_rerank_results,
            "latency_breakdown": {
                "expansion_ms": expansion_ms,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": generation_ms,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def rebuild_bm25(self) -> None:
        """
        Rebuild the BM25 index from all chunks currently in ChromaDB.

        Called automatically after every ingestion.  Can also be triggered
        manually if the BM25 index becomes desynchronised.
        """
        all_chunks = get_all_chunks(self.collection)
        if not all_chunks:
            self.bm25_index = None
            self.bm25_chunk_ids = []
            return

        texts = [c["text"] for c in all_chunks]
        ids = [c["chunk_id"] for c in all_chunks]

        self.bm25_index = build_bm25_index(texts, ids)
        self.bm25_chunk_ids = ids
        save_bm25_index(self.config.id, self.bm25_index, self.bm25_chunk_ids)
        logger.debug(f"BM25 index rebuilt: {len(ids)} chunks.")

    def get_all_chunk_texts(self) -> List[Tuple[str, str]]:
        """
        Return (chunk_id, text) pairs for all chunks in this RAG's collection.

        Useful for building eval sets.

        Returns:
            List of (chunk_id, text) tuples.
        """
        all_chunks = get_all_chunks(self.collection)
        return [(c["chunk_id"], c["text"]) for c in all_chunks]

    def _empty_result(self, user_query: str, queries: List[str]) -> Dict:
        """Return a well-structured empty result when no chunks are retrieved."""
        return {
            "answer": (
                "No relevant documents were found in the index. "
                "Please ingest documents before querying."
            ),
            "sources": [],
            "query_variants": queries,
            "pre_rerank_results": [],
            "post_rerank_results": [],
            "latency_breakdown": {
                "expansion_ms": 0,
                "retrieval_ms": 0,
                "rerank_ms": 0,
                "generation_ms": 0,
            },
        }
