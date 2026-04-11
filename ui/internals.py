"""
ui/internals.py
───────────────
Retrieval internals tab.

Shows what happened inside the pipeline for the last chat query:
  • The original query and all expanded variants.
  • Retrieved chunks BEFORE re-ranking with their RRF and dense scores.
  • Retrieved chunks AFTER re-ranking with cross-encoder scores and rank deltas.
  • Latency breakdown per pipeline stage.

The internals dict is populated by Pipeline.query() and stored in
st.session_state["last_internals"] after each chat message.
"""

from __future__ import annotations

import streamlit as st

from core.config import RAGConfig


def render_internals(config: RAGConfig) -> None:
    """Render the retrieval internals viewer."""

    internals = st.session_state.get("last_internals")

    if not internals:
        st.info(
            "💡 Send a message in the **Chat** tab to see retrieval internals here.",
            icon="🔬",
        )
        return

    st.subheader("🔬 Retrieval Internals")
    st.caption("Showing internals for the last query in the Chat tab.")

    # ── Query variants ────────────────────────────────────────────────────────
    variants = internals.get("query_variants", [])
    if variants:
        st.markdown("**Query expansion**")
        for i, v in enumerate(variants):
            label = "Original" if i == 0 else f"Variant {i}"
            st.markdown(f"`{label}`: {v}")

    st.divider()

    # ── Pre / post re-rank columns ────────────────────────────────────────────
    col_pre, col_post = st.columns(2)

    with col_pre:
        st.markdown("**Before re-ranking** *(RRF top results)*")
        pre = internals.get("pre_rerank_results", [])
        if not pre:
            st.caption("No pre-rerank data.")
        for i, chunk in enumerate(pre, start=1):
            text = chunk.get("text", "")
            preview = text[:500] + "…" if len(text) > 500 else text
            label = f"#{i} — {chunk.get('source', 'unknown')} (RRF: {chunk.get('rrf_score', 0):.5f})"
            with st.expander(label, expanded=(i == 1)):
                st.markdown(preview)
                c1, c2 = st.columns(2)
                c1.caption(f"Dense score: {chunk.get('dense_score', 0):.4f}")
                bm25_rank = chunk.get("bm25_rank")
                c2.caption(f"BM25 rank: {bm25_rank if bm25_rank else '—'}")

    with col_post:
        post = internals.get("post_rerank_results", [])
        if post:
            st.markdown("**After re-ranking** *(cross-encoder scored)*")
        else:
            st.markdown("**After re-ranking** *(no re-ranker active)*")

        if not post:
            st.caption("No post-rerank data. Enable a re-ranker in Config.")
        for i, chunk in enumerate(post, start=1):
            rank_before = chunk.get("rank_before", i)
            rank_after = chunk.get("rank_after", i)
            delta = rank_before - rank_after
            if delta > 0:
                delta_str = f"↑{delta}"
            elif delta < 0:
                delta_str = f"↓{abs(delta)}"
            else:
                delta_str = "="
            score = chunk.get("cross_encoder_score", 0.0)
            label = f"#{i} — {chunk.get('source', 'unknown')} ({delta_str}) score: {score:.4f}"
            with st.expander(label, expanded=(i == 1)):
                text = chunk.get("text", "")
                preview = text[:500] + "…" if len(text) > 500 else text
                st.markdown(preview)

    st.divider()

    # ── Latency breakdown ─────────────────────────────────────────────────────
    st.markdown("**Latency breakdown**")
    breakdown = internals.get("latency_breakdown", {})
    cols = st.columns(4)
    cols[0].metric("Query expansion", f"{breakdown.get('expansion_ms', 0)} ms")
    cols[1].metric("Retrieval", f"{breakdown.get('retrieval_ms', 0)} ms")
    cols[2].metric("Re-ranking", f"{breakdown.get('rerank_ms', 0)} ms")
    cols[3].metric("Generation", f"{breakdown.get('generation_ms', 0)} ms")

    total = sum([
        breakdown.get("expansion_ms", 0),
        breakdown.get("retrieval_ms", 0),
        breakdown.get("rerank_ms", 0),
        breakdown.get("generation_ms", 0),
    ])
    st.caption(f"Total end-to-end: **{total} ms**")
