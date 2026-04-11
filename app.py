"""
app.py
──────
Streamlit entry point for the RAG Builder application.

This file is a pure router — it contains NO business logic.
All logic lives in core/ (ML pipeline) and ui/ (rendering).

Responsibilities:
  1. Load environment variables (OpenRouter API key).
  2. Configure Streamlit page.
  3. Pre-warm ML model caches on first run.
  4. Initialise session state.
  5. Render the sidebar.
  6. Route to the correct view based on st.session_state.active_tab.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

# Load .env file if present (for local development)
load_dotenv()

# ── Page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="RAG Builder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure store directory exists ────────────────────────────────────────────
from core.registry import ensure_store_exists  # noqa: E402
ensure_store_exists()

# ── API key validation ────────────────────────────────────────────────────────
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error(
        "**OPENROUTER_API_KEY** environment variable is not set.\n\n"
        "Set it in a `.env` file or pass it when running the container:\n"
        "```\nOPENROUTER_API_KEY=sk-or-... streamlit run app.py\n```"
    )
    st.stop()

# ── Session state initialisation (idempotent) ─────────────────────────────────
_DEFAULTS: dict = {
    "active_tab": "home",
    "active_rag_id": None,
    "active_pipeline": None,
    "chat_session_id": None,
    "last_internals": None,
    "workspace_tab": "Ingest",
    "editing_retrieval": False,
    "api_key": api_key,
    "_models_loaded": False,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Pre-warm ML models (once per process, not once per rerun) ─────────────────
if not st.session_state._models_loaded:
    with st.spinner("Loading ML models into memory… (first run only)"):
        try:
            from core.embedder import preload_default_models
            preload_default_models()
            st.session_state._models_loaded = True
        except Exception as exc:
            st.warning(f"Model pre-load warning: {exc}. Models will load on first use.")
            st.session_state._models_loaded = True  # don't retry on every rerun


# ── Workspace renderer (defined before routing block) ────────────────────────

def _render_workspace(key: str) -> None:
    """
    Load the active RAG pipeline and render the 5-tab workspace.

    Args:
        key: OpenRouter API key.
    """
    from core.pipeline import Pipeline
    from core.registry import get_rag

    rag_id = st.session_state.active_rag_id
    if not rag_id:
        st.session_state.active_tab = "home"
        st.rerun()
        return

    config = get_rag(rag_id)
    if config is None:
        st.error(f"RAG '{rag_id}' not found. It may have been deleted.")
        st.session_state.active_tab = "home"
        st.session_state.active_rag_id = None
        st.rerun()
        return

    # Reuse cached pipeline or load a fresh one when RAG changes
    pipeline = st.session_state.active_pipeline
    if pipeline is None or pipeline.config.id != rag_id:
        with st.spinner(f"Loading pipeline '{config.name}'…"):
            try:
                pipeline = Pipeline(config, key)
                st.session_state.active_pipeline = pipeline
            except Exception as exc:
                st.error(f"Failed to load pipeline: {exc}")
                return

    # Page header
    st.title(f"🔍 {config.name}")
    if config.description:
        st.caption(config.description)

    # 5-tab workspace
    tab_ingest, tab_chat, tab_results, tab_internals, tab_config = st.tabs(
        ["📥 Ingest", "💬 Chat", "📊 Evaluation", "🔬 Internals", "⚙️ Config"]
    )

    with tab_ingest:
        from ui.ingest import render_ingest
        render_ingest(pipeline, config)

    with tab_chat:
        from ui.chat import render_chat
        render_chat(pipeline, config, key)

    with tab_results:
        from ui.evaluation import render_evaluation
        render_evaluation(pipeline, config, key)

    with tab_internals:
        from ui.internals import render_internals
        render_internals(config)

    with tab_config:
        from ui.config_view import render_config
        render_config(config, pipeline)


# ── Sidebar (rendered on every rerun) ────────────────────────────────────────
from ui.components.sidebar import render_sidebar  # noqa: E402
render_sidebar()

# ── Main routing ──────────────────────────────────────────────────────────────
_active = st.session_state.active_tab

if _active == "home":
    from ui.home import render_home
    render_home()

elif _active == "create":
    from ui.create import render_create_form
    render_create_form()

elif _active == "workspace":
    _render_workspace(api_key)

else:
    # Unknown state — reset to home
    st.session_state.active_tab = "home"
    st.rerun()
