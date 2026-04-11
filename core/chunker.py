"""
core/chunker.py
───────────────
Document chunking strategies.

All strategies return List[str] — a list of non-empty text chunks.
The chunk_size parameter is measured in *tokens* (approximated as
word count × 1.3, which is accurate enough for chunking decisions).

Strategies:
  • recursive  — Preferred: splits on \n\n → \n → ". " → " " in order.
                  Preserves document structure when possible.
  • fixed      — Split by character count with overlap. Simple, predictable.
  • sentence   — Split only at sentence boundaries (". ", "! ", "? ").
  • semantic   — Embed every sentence; cut at low-similarity boundaries.
                  Expensive — only suitable for small documents.
"""

from __future__ import annotations

import re
from typing import List

from loguru import logger


# ---------------------------------------------------------------------------
# Token count approximation
# (Exact tokenisation would require loading a tokeniser; this approximation
#  is sufficient for chunking decisions and avoids adding a dependency.)
# ---------------------------------------------------------------------------

def _approx_token_count(text: str) -> int:
    """Approximate token count as word_count × 1.3."""
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split a document into chunks using the chosen strategy.

    Args:
        text:       The full document text.
        strategy:   One of "recursive", "fixed", "sentence", "semantic".
        chunk_size: Target maximum chunk size in tokens.
        overlap:    Number of tokens of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "recursive":
        chunks = _recursive_split(text, chunk_size, overlap)
    elif strategy == "fixed":
        chunks = _fixed_split(text, chunk_size, overlap)
    elif strategy == "sentence":
        chunks = _sentence_split(text, chunk_size)
    elif strategy == "semantic":
        chunks = _semantic_split(text, chunk_size)
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}', falling back to recursive.")
        chunks = _recursive_split(text, chunk_size, overlap)

    # Remove empty strings and strip whitespace
    chunks = [c.strip() for c in chunks if c.strip()]
    logger.debug(f"Chunked document into {len(chunks)} chunks (strategy={strategy})")
    return chunks


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text recursively, preferring larger structural separators.

    Separator priority: paragraph → line → sentence → space.
    If a piece is still larger than chunk_size after splitting on the
    current separator, try the next one.
    """
    # Overlap in characters (rough approximation: 1 token ≈ 5 chars)
    overlap_chars = overlap * 5

    def _split(t: str, separators: List[str]) -> List[str]:
        if _approx_token_count(t) <= chunk_size:
            return [t]
        if not separators:
            # Last resort: hard cut by characters
            return _hard_cut(t, chunk_size * 5, overlap_chars)

        sep = separators[0]
        parts = t.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if _approx_token_count(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Sub-split the part that is too large
                if _approx_token_count(part) > chunk_size:
                    sub = _split(part, separators[1:])
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # Apply overlap: prepend the tail of the previous chunk onto the next
        if overlap_chars > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-overlap_chars:]
                overlapped.append(prev_tail + " " + chunks[i])
            return overlapped

        return chunks

    return _split(text, ["\n\n", "\n", ". ", " "])


def _fixed_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split by approximate character count derived from chunk_size in tokens.

    overlap is the number of tokens to repeat at the start of each new chunk.
    """
    char_size = chunk_size * 5        # 1 token ≈ 5 chars
    overlap_chars = overlap * 5
    return _hard_cut(text, char_size, overlap_chars)


def _hard_cut(text: str, char_size: int, overlap_chars: int) -> List[str]:
    """Cut text into fixed-size character slices with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start += char_size - overlap_chars
        if start < 0:
            start = 0
    return chunks


def _sentence_split(text: str, chunk_size: int) -> List[str]:
    """
    Split text at sentence boundaries, grouping sentences until chunk_size is reached.

    Sentence delimiters: ". ", "! ", "? "
    """
    # Split on sentence endings while keeping the delimiter attached
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _approx_token_count(sentence)
        if current_tokens + s_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_tokens = 0
        current_sentences.append(sentence)
        current_tokens += s_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _semantic_split(text: str, chunk_size: int) -> List[str]:
    """
    Split text at topic boundaries detected by embedding similarity.

    For each pair of adjacent sentences, compute cosine similarity of their
    embeddings.  Where similarity drops below 0.5, a new chunk begins.

    This is expensive (embeds every sentence) and intended only for small
    documents.  Falls back to recursive split if sentence count is very high.

    Args:
        text:       Full document text.
        chunk_size: Maximum tokens per chunk (used when grouping segments).

    Returns:
        List of semantically coherent chunk strings.
    """
    import numpy as np
    from core.embedder import embed_texts

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) > 500:
        logger.warning(
            "Document has >500 sentences — semantic chunking is too expensive. "
            "Falling back to recursive split."
        )
        return _recursive_split(text, chunk_size, overlap=50)

    if len(sentences) <= 1:
        return sentences

    # Embed all sentences with the default model
    vectors = embed_texts(sentences, "all-MiniLM-L6-v2")
    vectors_np = np.array(vectors)

    # Compute cosine similarity between consecutive sentence pairs
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalised = vectors_np / norms

    similarities = (normalised[:-1] * normalised[1:]).sum(axis=1)

    # Find split points where similarity drops below threshold
    threshold = 0.5
    split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # Group sentences into segments at split boundaries
    segments: List[List[str]] = []
    prev = 0
    for idx in split_indices:
        segments.append(sentences[prev:idx])
        prev = idx
    segments.append(sentences[prev:])

    # Merge short segments and re-split long ones to respect chunk_size
    chunks = []
    for seg in segments:
        seg_text = " ".join(seg)
        if _approx_token_count(seg_text) > chunk_size:
            chunks.extend(_recursive_split(seg_text, chunk_size, overlap=25))
        else:
            chunks.append(seg_text)

    return chunks
