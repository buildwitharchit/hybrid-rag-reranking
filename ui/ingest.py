"""
ui/ingest.py
────────────
Document ingestion tab.

Supports three ingestion methods:
  • Upload PDF, TXT, or DOCX file.
  • Paste a URL to scrape.
  • Paste raw plain text.

After ingestion, shows a table of all documents currently in the corpus
grouped by source document with chunk counts.
"""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st

from core.config import RAGConfig
from core.pipeline import Pipeline
from core.vector_store import get_all_chunks


def render_ingest(pipeline: Pipeline, config: RAGConfig) -> None:
    """Render the document ingestion tab."""

    st.subheader("📥 Ingest Documents")
    st.caption(
        f"Using **{config.chunking_strategy}** chunking · "
        f"chunk size **{config.chunk_size}** tokens · "
        f"overlap **{config.chunk_overlap}** tokens · "
        f"embedding model **{config.embedding_model}**"
    )

    if config.chunk_count > 0:
        st.info(
            f"**{config.chunk_count}** chunks across **{config.doc_count}** documents "
            f"currently indexed. You can add more documents at any time.",
            icon="📚",
        )

    # ── Ingestion tabs ────────────────────────────────────────────────────────
    tab_file, tab_url, tab_text = st.tabs(["📎 Upload File", "🌐 Paste URL", "📝 Paste Text"])

    with tab_file:
        _render_file_upload(pipeline, config)

    with tab_url:
        _render_url_ingest(pipeline, config)

    with tab_text:
        _render_text_ingest(pipeline, config)

    # ── Corpus table ──────────────────────────────────────────────────────────
    st.divider()
    _render_corpus_table(config)


# ── Ingestion sub-panels ──────────────────────────────────────────────────────

def _render_file_upload(pipeline: Pipeline, config: RAGConfig) -> None:
    uploaded = st.file_uploader(
        "Upload a PDF, TXT, or DOCX file",
        type=["pdf", "txt", "docx"],
        help="The file will be chunked and embedded using your pipeline's configuration.",
    )
    if uploaded:
        st.caption(f"Selected: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")
        if st.button("⚡ Ingest File", type="primary"):
            with st.spinner(f"Reading, chunking and embedding '{uploaded.name}'…"):
                text = extract_text_from_file(uploaded)
                if not text.strip():
                    st.error("Could not extract text from this file.")
                    return
                n = pipeline.ingest_text(text, uploaded.name, None)
            if n > 0:
                st.success(f"✅ Ingested **{n}** chunks from `{uploaded.name}`")
                st.rerun()
            else:
                st.warning("No chunks were created. The document may be empty or too short.")


def _render_url_ingest(pipeline: Pipeline, config: RAGConfig) -> None:
    url = st.text_input("URL", placeholder="https://example.com/article")
    if url:
        if st.button("🌐 Scrape and Ingest", type="primary"):
            with st.spinner(f"Fetching `{url}`…"):
                text = scrape_url(url)
                if not text.strip():
                    st.error("Could not extract text from this URL.")
                    return
                n = pipeline.ingest_text(text, url, None)
            if n > 0:
                st.success(f"✅ Ingested **{n}** chunks from `{url}`")
                st.rerun()
            else:
                st.warning("No chunks were created from this URL.")


def _render_text_ingest(pipeline: Pipeline, config: RAGConfig) -> None:
    pasted = st.text_area(
        "Paste plain text",
        height=200,
        placeholder="Paste any text here — articles, notes, transcripts…",
    )
    doc_name = st.text_input(
        "Document name",
        placeholder="my_notes.txt",
        help="A name to identify this text in the source citations.",
    )
    if st.button("📝 Ingest Text", type="primary"):
        if not pasted.strip():
            st.error("Please paste some text before ingesting.")
            return
        name = doc_name.strip() or "pasted_text"
        with st.spinner("Chunking and embedding…"):
            n = pipeline.ingest_text(pasted, name, None)
        if n > 0:
            st.success(f"✅ Ingested **{n}** chunks as `{name}`")
            st.rerun()
        else:
            st.warning("No chunks were created. The text may be too short.")


# ── Corpus table ──────────────────────────────────────────────────────────────

def _render_corpus_table(config: RAGConfig) -> None:
    st.subheader("📚 Indexed Documents")

    if config.chunk_count == 0:
        st.caption("No documents ingested yet. Use the tabs above to add documents.")
        return

    # Fetch all chunk metadata from ChromaDB
    from core.vector_store import get_or_create_collection
    collection = get_or_create_collection(config.id, config.embedding_model)
    all_chunks = get_all_chunks(collection)

    if not all_chunks:
        st.caption("No chunks found in the index.")
        return

    # Group by source document
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for chunk in all_chunks:
        grouped[chunk.get("source_doc", "Unknown")].append(chunk)

    rows = []
    for source, chunks in sorted(grouped.items()):
        rows.append(
            {
                "Document": source,
                "Chunks": len(chunks),
                "Avg chunk length (chars)": int(
                    sum(c.get("char_count", len(c.get("text", ""))) for c in chunks)
                    / len(chunks)
                ),
            }
        )

    import pandas as pd  # type: ignore
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Total: **{config.chunk_count}** chunks across **{config.doc_count}** documents"
    )


# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_text_from_file(uploaded_file) -> str:
    """
    Extract plain text from a Streamlit UploadedFile object.

    Supports PDF (via pypdf), DOCX (via python-docx), and TXT.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted plain text string.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        return _extract_pdf(raw_bytes)
    elif name.endswith(".docx"):
        return _extract_docx(raw_bytes)
    else:
        # TXT or unknown — attempt UTF-8 decode
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")


def _extract_pdf(raw_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(raw_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    import docx  # type: ignore

    doc = docx.Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def scrape_url(url: str) -> str:
    """
    Fetch a URL and extract readable text from its HTML.

    Extracts text from headings (h1-h6), paragraphs, and list items.

    Args:
        url: The URL to fetch.

    Returns:
        Plain text extracted from the page.
    """
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore

    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "RAGBuilder/1.0"})
        response.raise_for_status()
    except Exception as exc:
        st.error(f"Failed to fetch URL: {exc}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract meaningful text tags
    text_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"])
    pieces = [tag.get_text(separator=" ", strip=True) for tag in text_tags]
    pieces = [p for p in pieces if len(p) > 20]  # filter very short fragments
    return "\n\n".join(pieces)
