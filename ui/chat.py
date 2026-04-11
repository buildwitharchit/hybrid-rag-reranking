"""
ui/chat.py
──────────
Chat interface tab.

Provides a conversational interface where users query their ingested
documents.  Each response is grounded in retrieved chunks and annotated
with source citations shown as coloured chips below the answer.

Chat history is persisted to SQLite within the session.
Users can start a new session or switch between previous sessions via a
sidebar widget in this tab.
"""

from __future__ import annotations

import streamlit as st

from core.config import RAGConfig
from core.database import (
    create_session,
    load_session_messages,
    list_sessions,
    save_message,
)
from core.pipeline import Pipeline


def render_chat(pipeline: Pipeline, config: RAGConfig, api_key: str) -> None:
    """Render the chat interface."""

    if config.chunk_count == 0:
        st.warning(
            "⚠️ No documents ingested yet. Go to the **Ingest** tab to add documents first.",
            icon="📚",
        )
        return

    # ── Session management ────────────────────────────────────────────────────
    _ensure_session(config.id)

    # ── Session switcher in the sidebar area ──────────────────────────────────
    with st.expander("💬 Chat sessions", expanded=False):
        _render_session_controls(config.id)

    # ── Load and display existing messages ────────────────────────────────────
    session_id = st.session_state.chat_session_id
    messages = load_session_messages(config.id, session_id)

    # Render existing conversation
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        f"Ask a question about your {config.name} documents…",
        key="chat_input",
    )

    if user_input:
        # Save user message immediately
        save_message(config.id, session_id, "user", user_input, [])

        # Show user message in UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                result = pipeline.query(user_input)

            st.markdown(result["answer"])

            if result["sources"]:
                _render_sources(result["sources"])

            # Store internals for the Internals tab
            st.session_state.last_internals = result

        # Persist assistant message
        save_message(
            config.id,
            session_id,
            "assistant",
            result["answer"],
            result["sources"],
        )

        # Update last_used timestamp in registry
        from datetime import datetime, timezone
        from core.registry import update_rag
        update_rag(config.id, {"last_used": datetime.now(timezone.utc).isoformat()})

        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_session(rag_id: str) -> None:
    """Create a new chat session if none is active for this RAG."""
    if not st.session_state.get("chat_session_id"):
        session_id = create_session(rag_id)
        st.session_state.chat_session_id = session_id


def _render_sources(sources: list) -> None:
    """Render source citation chips below an assistant message."""
    if not sources:
        return

    st.caption("📎 Sources:")
    cols = st.columns(min(len(sources), 4))
    for i, source in enumerate(sources[:4]):
        label = source.get("source_doc", "Unknown")
        page = source.get("page_number")
        if page and page != -1:
            label += f" (p.{page})"
        cited = source.get("cited", True)
        badge_color = "#d4edda" if cited else "#f8d7da"
        with cols[i % 4]:
            st.markdown(
                f'<span style="background:{badge_color};padding:3px 8px;'
                f'border-radius:4px;font-size:12px;display:inline-block;'
                f'margin:2px">[{source.get("citation_index", i+1)}] {label}</span>',
                unsafe_allow_html=True,
            )


def _render_session_controls(rag_id: str) -> None:
    """Show session list and allow starting a new session."""
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🆕 New session", key="new_session_btn"):
            session_id = create_session(rag_id)
            st.session_state.chat_session_id = session_id
            st.rerun()

    sessions = list_sessions(rag_id)
    if not sessions:
        col1.caption("No sessions yet.")
        return

    current_id = st.session_state.get("chat_session_id")
    for sess in sessions[:10]:  # show up to 10 recent sessions
        ts = sess["created_at"][:19].replace("T", " ")
        label = f"{ts}  ({sess['message_count']} msgs)"
        is_current = sess["id"] == current_id

        if is_current:
            col1.markdown(f"▶ **{label}** *(current)*")
        else:
            if col1.button(label, key=f"sess_{sess['id']}"):
                st.session_state.chat_session_id = sess["id"]
                st.rerun()
