"""
ui/config_view.py
─────────────────
Configuration viewer and editor tab for a RAG pipeline.

Shows two sections:
  • Ingestion config — locked after first document ingestion.
  • Retrieval config — freely editable at any time.

Also shows corpus stats and a danger zone for RAG deletion.
"""

from __future__ import annotations

import streamlit as st

from core.config import RAGConfig, LLM_MODELS
from core.pipeline import Pipeline
from core.registry import delete_rag, update_rag, get_rag


def render_config(config: RAGConfig, pipeline: Pipeline) -> None:
    """Render the configuration view and editor."""

    st.subheader(f"⚙️ Configuration — {config.name}")
    if config.description:
        st.caption(config.description)

    col1, col2 = st.columns(2)

    # ── Ingestion config (read-only after lock) ───────────────────────────────
    with col1:
        st.markdown("### 🔒 Ingestion Config")
        if config.is_locked:
            st.caption("Locked — change these by deleting and recreating the RAG.")
        else:
            st.caption("Will lock after first document is ingested.")

        _kv_row("Chunking strategy", config.chunking_strategy, config.is_locked)
        _kv_row("Chunk size", f"{config.chunk_size} tokens", config.is_locked)
        _kv_row("Chunk overlap", f"{config.chunk_overlap} tokens", config.is_locked)
        _kv_row("Embedding model", config.embedding_model, config.is_locked)
        _kv_row("Vector store", config.vector_store, config.is_locked)

    # ── Retrieval config (editable) ───────────────────────────────────────────
    with col2:
        st.markdown("### ✏️ Retrieval Config")
        st.caption("Edit at any time — changes take effect immediately.")

        if st.button("Edit retrieval config", key="edit_retrieval_btn"):
            st.session_state.editing_retrieval = True

        if st.session_state.get("editing_retrieval"):
            _render_retrieval_editor(config)
        else:
            _kv_row("Sparse search", config.sparse_search)
            _kv_row("Fusion method", config.fusion_method)
            _kv_row("Query expansion", config.query_expansion)
            _kv_row("Top-K (retrieval)", str(config.top_k_retrieval))
            _kv_row("Re-ranker", config.reranker)
            _kv_row("Top-K (final)", str(config.top_k_final))
            _kv_row("LLM model", config.llm_model)
            _kv_row("Temperature", str(config.temperature))

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Stats")
    cols = st.columns(3)
    cols[0].metric("Documents", config.doc_count)
    cols[1].metric("Chunks", config.chunk_count)
    cols[2].metric("Created", config.created_at[:10])

    st.divider()

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Danger Zone")
    with st.container(border=True):
        st.warning(
            f"Deleting **{config.name}** permanently removes all documents, "
            "vector embeddings, BM25 index, chat history, and test results. "
            "This cannot be undone.",
        )
        confirm = st.text_input(
            f"Type `{config.name}` to confirm deletion",
            key="delete_confirm_input",
        )
        can_delete = confirm.strip() == config.name
        if st.button(
            "Delete RAG",
            type="primary",
            disabled=not can_delete,
            key="delete_rag_btn",
        ):
            delete_rag(config.id)
            st.session_state.active_rag_id = None
            st.session_state.active_pipeline = None
            st.session_state.active_tab = "home"
            st.session_state.pop("editing_retrieval", None)
            st.session_state.pop("chat_session_id", None)
            st.success(f"Deleted '{config.name}'.")
            st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kv_row(label: str, value: str, locked: bool = False) -> None:
    """Render a key-value configuration row."""
    c1, c2 = st.columns([1, 1])
    c1.markdown(f"**{label}**")
    suffix = " 🔒" if locked else ""
    c2.markdown(f"`{value}`{suffix}")


def _render_retrieval_editor(config: RAGConfig) -> None:
    """Render an in-place form for editing retrieval config."""

    with st.form("edit_retrieval_form"):
        sparse = st.selectbox(
            "Sparse search",
            ["none", "bm25", "bm25plus"],
            index=["none", "bm25", "bm25plus"].index(config.sparse_search),
        )
        fusion = st.selectbox(
            "Fusion method",
            ["rrf", "weighted", "combsum"],
            index=["rrf", "weighted", "combsum"].index(config.fusion_method),
        )
        expansion = st.selectbox(
            "Query expansion",
            ["none", "llm_variants", "hyde"],
            index=["none", "llm_variants", "hyde"].index(config.query_expansion),
        )
        top_k_r = st.selectbox(
            "Top-K (retrieval)",
            [10, 20, 30, 50],
            index=[10, 20, 30, 50].index(config.top_k_retrieval),
        )
        reranker = st.selectbox(
            "Re-ranker",
            ["none", "cross_encoder_ms_marco", "bge_reranker_large"],
            index=["none", "cross_encoder_ms_marco", "bge_reranker_large"].index(
                config.reranker
            ),
        )
        top_k_f = st.selectbox(
            "Top-K (final)",
            [3, 5, 7, 10],
            index=[3, 5, 7, 10].index(config.top_k_final),
        )
        llm = st.selectbox(
            "LLM model",
            LLM_MODELS,
            index=LLM_MODELS.index(config.llm_model)
            if config.llm_model in LLM_MODELS
            else 0,
        )
        temperature = st.selectbox(
            "Temperature",
            [0.0, 0.3, 0.7, 1.0],
            index=[0.0, 0.3, 0.7, 1.0].index(config.temperature),
        )

        col1, col2 = st.columns(2)
        save_clicked = col1.form_submit_button("💾 Save", type="primary", use_container_width=True)
        cancel_clicked = col2.form_submit_button("Cancel", use_container_width=True)

    if save_clicked:
        update_rag(
            config.id,
            {
                "sparse_search": sparse,
                "fusion_method": fusion,
                "query_expansion": expansion,
                "top_k_retrieval": top_k_r,
                "reranker": reranker,
                "top_k_final": top_k_f,
                "llm_model": llm,
                "temperature": float(temperature),
            },
        )
        # Reload the pipeline with updated config
        refreshed = get_rag(config.id)
        if refreshed:
            from core.pipeline import Pipeline
            st.session_state.active_pipeline = Pipeline(
                refreshed, st.session_state.get("api_key", "")
            )
        st.session_state.editing_retrieval = False
        st.success("Configuration saved.")
        st.rerun()

    if cancel_clicked:
        st.session_state.editing_retrieval = False
        st.rerun()
