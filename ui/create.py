"""
ui/create.py
────────────
RAG creation form.

Presents all pipeline configuration options grouped by pipeline stage.
Options that require paid APIs or GPUs are shown but disabled (greyed out)
with a "(disabled)" suffix so users understand what exists without being
able to accidentally select it.

All settings are validated server-side on submit before creating the RAG.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from core.config import (
    RAGConfig,
    DISABLED_EMBEDDING_MODELS,
    DISABLED_RERANKERS,
    DISABLED_SPARSE,
    DISABLED_QUERY_EXPANSION,
    LLM_MODELS,
)
from core.registry import create_rag, make_rag_id, load_all_rags

# ── Disabled option sentinel strings ─────────────────────────────────────────
# These must contain "(disabled)" — validated on submit.

_DISABLED_VECTOR_STORES = ["FAISS (coming soon - disabled)", "Pinecone (API - disabled)"]


def render_create_form() -> None:
    """Render the RAG creation form."""
    st.title("➕ Create New RAG Pipeline")
    st.caption(
        "Choose your pipeline components. "
        "Greyed-out options require a paid API or GPU — they are shown for reference only."
    )

    with st.form("create_rag_form", clear_on_submit=False):

        # ── Identity ────────────────────────────────────────────────────────
        st.subheader("Identity")
        name = st.text_input("Name *", placeholder="e.g. MedRAG, LegalDocs")
        description = st.text_input("Description", placeholder="What documents will this index?")

        st.divider()

        # ── Chunking ────────────────────────────────────────────────────────
        st.subheader("📄 Chunking  *(locked after first document ingested)*")
        st.caption(
            "Chunking strategy determines how your documents are split. "
            "Fixed is fastest; recursive respects structure; semantic is most accurate but slow."
        )
        col1, col2 = st.columns(2)
        chunking_strategy = col1.selectbox(
            "Strategy",
            ["recursive", "fixed", "sentence", "semantic"],
            index=0,
            help="recursive: prefers paragraph/sentence breaks. fixed: hard cut every N tokens.",
        )
        chunk_size = col2.selectbox(
            "Chunk size (tokens)",
            [256, 512, 768, 1024],
            index=1,
            help="Maximum tokens per chunk. all-MiniLM max is 512 tokens.",
        )
        chunk_overlap = st.selectbox(
            "Overlap (tokens)",
            [0, 25, 50, 100],
            index=2,
            help="Tokens repeated between adjacent chunks to prevent boundary cuts.",
        )

        st.divider()

        # ── Embeddings ──────────────────────────────────────────────────────
        st.subheader("🧬 Embeddings  *(locked after first document ingested)*")
        col1, col2 = st.columns(2)
        embedding_model = col1.selectbox(
            "Embedding model (local, free)",
            [
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
                "bge-large-en-v1.5",
            ],
            index=0,
            help="all-MiniLM-L6-v2 is the fastest. bge-large-en-v1.5 is most accurate but 1.3 GB.",
        )
        col1.selectbox(
            "API embedding models (disabled)",
            DISABLED_EMBEDDING_MODELS,
            index=0,
            disabled=True,
            help="These require a paid API key.",
        )
        vector_store = col2.selectbox(
            "Vector store",
            ["chromadb"],
            index=0,
            help="ChromaDB runs locally in-process. No server required.",
        )
        col2.selectbox(
            "Other vector stores (disabled)",
            _DISABLED_VECTOR_STORES,
            index=0,
            disabled=True,
            help="These require separate infrastructure or a paid cloud service.",
        )

        st.divider()

        # ── Retrieval ────────────────────────────────────────────────────────
        st.subheader("🔍 Retrieval  *(editable at any time)*")
        col1, col2 = st.columns(2)
        sparse_search = col1.selectbox(
            "Sparse search (local, free)",
            ["none", "bm25", "bm25plus"],
            index=1,
            help="BM25 excels at exact keyword matching. Combine with dense for hybrid search.",
        )
        col1.selectbox(
            "GPU sparse models (disabled)",
            DISABLED_SPARSE,
            index=0,
            disabled=True,
            help="SPLADE requires a GPU to run at practical speeds.",
        )
        fusion_method = col2.selectbox(
            "Fusion method",
            ["rrf", "weighted", "combsum"],
            index=0,
            help="RRF is robust and requires no tuning. Recommended.",
        )

        col1, col2 = st.columns(2)
        query_expansion = col1.selectbox(
            "Query expansion",
            ["none", "llm_variants", "hyde"],
            index=1,
            help=(
                "llm_variants: LLM generates 3 rephrased queries. "
                "hyde: LLM writes a hypothetical answer for embedding."
            ),
        )
        col1.selectbox(
            "GPU expansion models (disabled)",
            DISABLED_QUERY_EXPANSION,
            index=0,
            disabled=True,
        )
        top_k_retrieval = col2.selectbox(
            "Top-K (pre-rerank)",
            [10, 20, 30, 50],
            index=1,
            help="Number of candidates fetched before re-ranking.",
        )

        st.divider()

        # ── Re-ranking ───────────────────────────────────────────────────────
        st.subheader("🎯 Re-ranking  *(editable at any time)*")
        col1, col2 = st.columns(2)
        reranker = col1.selectbox(
            "Re-ranker (local, free)",
            ["none", "cross_encoder_ms_marco", "bge_reranker_large"],
            index=1,
            format_func=lambda x: {
                "none": "None",
                "cross_encoder_ms_marco": "Cross-encoder ms-marco (85 MB, recommended)",
                "bge_reranker_large": "BGE Reranker Large (1.3 GB, most accurate)",
            }.get(x, x),
            help="Cross-encoder sees query+chunk together for more accurate relevance scoring.",
        )
        col1.selectbox(
            "API re-rankers (disabled)",
            DISABLED_RERANKERS,
            index=0,
            disabled=True,
            help="Cohere and Jina require paid API keys.",
        )
        top_k_final = col2.selectbox(
            "Final Top-K",
            [3, 5, 7, 10],
            index=1,
            help="Chunks passed to the LLM for answer generation.",
        )

        st.divider()

        # ── Generation ───────────────────────────────────────────────────────
        st.subheader("💬 Generation  *(editable at any time)*")
        st.caption("All models accessed via OpenRouter API. Only external paid service.")
        col1, col2 = st.columns(2)
        llm_model = col1.selectbox(
            "LLM (via OpenRouter)",
            LLM_MODELS,
            index=0,
        )
        temperature = col2.selectbox(
            "Temperature",
            [0.0, 0.3, 0.7, 1.0],
            index=0,
            help="0.0 = deterministic (best for RAG). Higher = more creative.",
        )

        st.divider()

        # ── Submit ───────────────────────────────────────────────────────────
        submitted = st.form_submit_button("🚀 Create RAG", type="primary", use_container_width=True)

    if submitted:
        _handle_submit(
            name=name,
            description=description,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            vector_store=vector_store,
            sparse_search=sparse_search,
            fusion_method=fusion_method,
            query_expansion=query_expansion,
            top_k_retrieval=top_k_retrieval,
            reranker=reranker,
            top_k_final=top_k_final,
            llm_model=llm_model,
            temperature=temperature,
        )


def _handle_submit(**kwargs) -> None:
    """Validate form values and create the RAG if valid."""

    name = kwargs["name"].strip()
    if not name:
        st.error("Name is required.")
        return

    # Server-side validation: reject any disabled option that slipped through
    disabled_sentinel = "disabled"
    for field, value in kwargs.items():
        if isinstance(value, str) and disabled_sentinel in value.lower():
            st.error(
                f"Option '{value}' is not available. Please choose a supported option."
            )
            return

    # Check name uniqueness
    existing = load_all_rags()
    existing_names = {r.name.lower() for r in existing}
    if name.lower() in existing_names:
        st.error(f"A RAG named '{name}' already exists. Choose a different name.")
        return

    rag_id = make_rag_id(name)

    config = RAGConfig(
        id=rag_id,
        name=name,
        description=kwargs["description"].strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
        chunking_strategy=kwargs["chunking_strategy"],
        chunk_size=kwargs["chunk_size"],
        chunk_overlap=kwargs["chunk_overlap"],
        embedding_model=kwargs["embedding_model"],
        vector_store=kwargs["vector_store"],
        sparse_search=kwargs["sparse_search"],
        fusion_method=kwargs["fusion_method"],
        query_expansion=kwargs["query_expansion"],
        top_k_retrieval=kwargs["top_k_retrieval"],
        reranker=kwargs["reranker"],
        top_k_final=kwargs["top_k_final"],
        llm_model=kwargs["llm_model"],
        temperature=float(kwargs["temperature"]),
    )

    try:
        create_rag(config)
    except Exception as exc:
        st.error(f"Failed to create RAG: {exc}")
        return

    st.session_state.active_rag_id = rag_id
    st.session_state.active_tab = "workspace"
    st.session_state.active_pipeline = None
    st.session_state.chat_session_id = None
    st.success(f"✅ RAG '{name}' created! Start by ingesting documents in the Ingest tab.")
    st.rerun()
