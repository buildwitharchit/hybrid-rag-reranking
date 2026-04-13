"""
ui/home.py
──────────
Home screen: a card-grid dashboard showing all created RAG pipelines.

Each card summarises the RAG's configuration and stats at a glance.
A "Create new RAG" card always appears as the last item.
"""

from __future__ import annotations

import streamlit as st

from core.registry import load_all_rags


def render_home() -> None:
    """Render the RAG manager dashboard."""

    st.title("RAG Builder")
    st.caption(
        "Create and manage independent RAG pipelines. "
        "Each pipeline has its own document corpus, retrieval configuration, and eval set."
    )
    st.divider()

    rags = load_all_rags()

    if not rags:
        # Empty state — invite user to create first RAG
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### No RAG pipelines yet")
            st.write(
                "Create your first RAG pipeline to get started. "
                "You will be able to ingest documents, chat, and compare "
                "naive vs hybrid retrieval performance."
            )
            if st.button("Create your first RAG", type="primary", use_container_width=True):
                st.session_state.active_tab = "create"
                st.rerun()
        return

    # ── 2-column card grid ────────────────────────────────────────────────────
    cards_per_row = 2
    all_items = list(rags) + ["create_new"]  # sentinel for the "+" card

    for row_start in range(0, len(all_items), cards_per_row):
        cols = st.columns(cards_per_row)
        for col_idx, item in enumerate(all_items[row_start : row_start + cards_per_row]):
            with cols[col_idx]:
                if item == "create_new":
                    _render_create_card()
                else:
                    _render_rag_card(item)


def _render_rag_card(rag) -> None:
    """Render a single RAG pipeline card."""
    with st.container(border=True):
        st.subheader(rag.name)
        if rag.description:
            st.caption(rag.description)

        # Configuration badges
        badges = []
        if rag.sparse_search != "none":
            badges.append(f"Hybrid ({rag.sparse_search.upper()})")
        else:
            badges.append("Dense-only")
        if rag.reranker != "none":
            badges.append("Re-ranked")
        if rag.query_expansion != "none":
            badges.append(rag.query_expansion)

        st.markdown(" &nbsp; ".join(f"`{b}`" for b in badges))

        st.divider()

        # Stats
        c1, c2 = st.columns(2)
        c1.metric("Documents", rag.doc_count)
        c2.metric("Chunks", rag.chunk_count)

        if rag.last_used:
            st.caption(f"Last used: {rag.last_used[:10]}")
        else:
            st.caption(f"Created: {rag.created_at[:10]}")

        # Eval status indicator
        if rag.has_eval_results:
            st.success("Eval results available")
        elif rag.has_eval_set:
            st.info("Eval set ready — run evaluation")
        else:
            st.caption("No eval set yet")

        st.divider()

        if st.button("Open", key=f"open_{rag.id}", use_container_width=True, type="primary"):
            if st.session_state.get("active_rag_id") != rag.id:
                st.session_state.active_pipeline = None
                st.session_state.chat_session_id = None
            st.session_state.active_rag_id = rag.id
            st.session_state.active_tab = "workspace"
            st.rerun()


def _render_create_card() -> None:
    """Render the 'Create new RAG' call-to-action card."""
    with st.container(border=True):
        st.markdown("### Create New RAG")
        st.write(
            "Configure a new hybrid retrieval pipeline with your own "
            "document corpus, chunking strategy, embeddings, and re-ranker."
        )
        st.divider()
        if st.button(
            "Create new RAG",
            key="create_new_card_btn",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.active_tab = "create"
            st.rerun()
