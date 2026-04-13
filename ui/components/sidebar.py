"""
ui/components/sidebar.py
────────────────────────
Renders the persistent left sidebar: RAG list + navigation.

The sidebar is rendered on every Streamlit rerun.  Clicking a RAG name
switches the active workspace.  Clicking "+ New RAG" opens the creation form.
"""

from __future__ import annotations

import streamlit as st

from core.registry import load_all_rags


def render_sidebar() -> None:
    """Render the application sidebar with RAG navigation controls."""

    st.sidebar.title("RAG Builder")
    st.sidebar.caption("Hybrid Retrieval-Augmented Generation")
    st.sidebar.divider()

    # ── New RAG button ────────────────────────────────────────────────────────
    if st.sidebar.button("+ New RAG", use_container_width=True, type="primary"):
        st.session_state.active_tab = "create"
        st.session_state.active_rag_id = None
        st.session_state.active_pipeline = None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("My RAGs")

    # ── RAG list ──────────────────────────────────────────────────────────────
    rags = load_all_rags()

    if not rags:
        st.sidebar.info("No RAGs yet.\nCreate your first one above.")
        return

    for rag in rags:
        is_active = st.session_state.get("active_rag_id") == rag.id

        # Summary badge line
        badges = []
        if rag.sparse_search != "none":
            badges.append("Hybrid")
        else:
            badges.append("Dense-only")
        if rag.reranker != "none":
            badges.append("Re-ranked")
        if rag.query_expansion != "none":
            badges.append("Expanded")
        badge_str = " · ".join(badges)

        # Sidebar button — highlighted when active
        button_label = f"{'> ' if is_active else ''}{rag.name}"
        help_text = f"{badge_str} | {rag.chunk_count} chunks"

        if st.sidebar.button(
            button_label,
            key=f"rag_btn_{rag.id}",
            help=help_text,
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.get("active_rag_id") != rag.id:
                # Clear the old pipeline so the new one loads fresh
                st.session_state.active_pipeline = None
                st.session_state.chat_session_id = None
            st.session_state.active_rag_id = rag.id
            st.session_state.active_tab = "workspace"
            st.rerun()

        # Sub-caption (shown below the button)
        st.sidebar.caption(f"  {rag.chunk_count} chunks · {badge_str}")

    st.sidebar.divider()
    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.active_tab = "home"
        st.session_state.active_rag_id = None
        st.rerun()
