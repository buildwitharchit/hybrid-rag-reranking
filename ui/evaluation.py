"""
ui/evaluation.py
────────────────
Evaluation dashboard tab.

Two panels side by side:

  Left  — Results: metric cards (NDCG@5, MRR, Faithfulness, Latency)
           plus a per-query breakdown table.
  Right — Eval-set management: auto-generate from corpus or upload CSV,
          then trigger a full benchmark run.

Results are cached to disk so the tab loads instantly without re-running
the expensive evaluation pipeline each time.
"""

from __future__ import annotations

import csv
import json
import os

import streamlit as st

from core.config import RAGConfig
from core.pipeline import Pipeline
from ui.components.metrics import metric_card

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")


def render_evaluation(pipeline: Pipeline, config: RAGConfig, api_key: str) -> None:
    """Render the evaluation dashboard."""

    if config.chunk_count == 0:
        st.warning("Ingest documents before running evaluation.", icon="📚")
        return

    st.subheader("📊 Evaluation Dashboard")
    st.caption(
        "Compare naive RAG (dense-only, no expansion, no re-ranking) vs your full "
        "hybrid pipeline on identical queries. All metrics use your custom test set."
    )

    col_results, col_setup = st.columns([2, 1])

    with col_results:
        _render_results(config)

    with col_setup:
        _render_setup(pipeline, config, api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Results panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(config: RAGConfig) -> None:
    """Render metric cards and per-query table from cached results."""
    results_path = os.path.join(_STORE_ROOT, config.id, "eval_results.json")

    if not os.path.exists(results_path):
        st.info(
            "No results yet. Generate a test set on the right and click "
            "**Run Full Evaluation**."
        )
        return

    with open(results_path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    naive = results["naive"]
    hybrid = results["hybrid"]
    run_at = results.get("run_at", "")[:19].replace("T", " ")

    st.subheader("Naive vs Hybrid — Summary")
    st.caption(f"Last run: {run_at} UTC")

    cols = st.columns(4)
    with cols[0]:
        metric_card("NDCG@5", hybrid["ndcg_at_5"], naive["ndcg_at_5"])
    with cols[1]:
        metric_card("MRR", hybrid["mrr"], naive["mrr"])
    with cols[2]:
        metric_card("Faithfulness", hybrid["faithfulness"], naive.get("faithfulness", 0.0))
    with cols[3]:
        metric_card(
            "Avg Latency",
            hybrid["avg_latency_seconds"],
            naive["avg_latency_seconds"],
            unit="s",
            higher_is_better=False,
            fmt=".2f",
        )

    st.divider()

    per_query = results.get("per_query", [])
    if not per_query:
        return

    st.subheader("Per-Query Breakdown")
    import pandas as pd  # type: ignore

    rows = []
    for item in per_query:
        n_rank = item.get("naive_first_relevant_rank")
        h_rank = item.get("hybrid_first_relevant_rank")
        rows.append(
            {
                "Query": item["query"][:80],
                "Naive rank": n_rank if n_rank else "—",
                "Hybrid rank": h_rank if h_rank else "—",
                "NDCG (naive)": round(item["naive_ndcg"], 3),
                "NDCG (hybrid)": round(item["hybrid_ndcg"], 3),
                "Faithful": "Yes" if item.get("faithful") else "No",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set management panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_setup(pipeline: Pipeline, config: RAGConfig, api_key: str) -> None:
    """Render test-set generation controls and the run button."""
    test_set_path = os.path.join(_STORE_ROOT, config.id, "eval_set.json")

    st.subheader("Test Set")
    test_set: list = []

    if os.path.exists(test_set_path):
        with open(test_set_path, "r", encoding="utf-8") as fh:
            test_set = json.load(fh)

    if test_set:
        st.success(f"**{len(test_set)}** questions loaded")
        with st.expander("Preview"):
            import pandas as pd  # type: ignore
            preview = [{"Query": item["query"][:80]} for item in test_set]
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
    else:
        st.info("No test set yet.")

    st.divider()

    # ── Auto-generate ────────────────────────────────────────────────────────
    st.markdown("**Auto-Generate from Corpus**")
    st.caption("LLM generates Q&A pairs from random chunks via OpenRouter.")
    n = st.slider("Number of questions", 5, 100, 20, step=5, key="n_questions_slider")
    if st.button("Generate", type="primary", use_container_width=True):
        with st.spinner(f"Generating {n} questions…"):
            from core.evaluator import generate_eval_set_from_corpus
            new_set = generate_eval_set_from_corpus(pipeline, n, api_key, config.llm_model)
        if new_set:
            _write_json(test_set_path, new_set)
            from core.registry import update_rag
            update_rag(config.id, {"has_eval_set": True})
            st.success(f"Generated {len(new_set)} questions.")
            st.rerun()
        else:
            st.error("Generation failed. Check your OpenRouter API key.")

    st.divider()

    # ── Upload CSV ───────────────────────────────────────────────────────────
    st.markdown("**Upload CSV**")
    st.caption("Required columns: `query`, `ideal_answer`, `relevant_chunk_ids`, `source_doc`")
    uploaded = st.file_uploader("CSV file", type=["csv"], key="testset_csv")
    if uploaded and st.button("Import", use_container_width=True):
        try:
            content = uploaded.read().decode("utf-8")
            reader = csv.DictReader(content.splitlines())
            parsed = []
            for row in reader:
                raw_ids = row.get("relevant_chunk_ids", "")
                ids = [c.strip() for c in raw_ids.split(",") if c.strip()]
                entry = {
                    "query": row.get("query", "").strip(),
                    "ideal_answer": row.get("ideal_answer", "").strip(),
                    "relevant_chunk_ids": ids,
                    "source_doc": row.get("source_doc", "").strip(),
                }
                if entry["query"]:
                    parsed.append(entry)
            _write_json(test_set_path, parsed)
            from core.registry import update_rag
            update_rag(config.id, {"has_eval_set": True})
            st.success(f"Imported {len(parsed)} queries.")
            st.rerun()
        except Exception as exc:
            st.error(f"Parse error: {exc}")

    # ── Run evaluation ───────────────────────────────────────────────────────
    if test_set:
        st.divider()
        st.markdown("**Run Full Benchmark**")
        st.caption(
            f"Runs naive + hybrid pipeline on {len(test_set)} queries and scores "
            "NDCG@5, MRR, Faithfulness, and Latency."
        )
        st.warning(
            f"Uses ~{len(test_set) * 2} OpenRouter API calls.", icon="💰"
        )
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            prog = st.progress(0, text="Running…")
            try:
                from core.evaluator import run_evaluation
                run_evaluation(pipeline, test_set, api_key)
                prog.progress(100, text="Done!")
                from core.registry import update_rag
                update_rag(config.id, {"has_eval_results": True})
                st.success("Evaluation complete! Refresh the results panel.")
                st.rerun()
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json(path: str, data: list) -> None:
    """Atomically write a JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
