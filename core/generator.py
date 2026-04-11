"""
core/generator.py
─────────────────
LLM answer generation via the OpenRouter API.

OpenRouter is a meta-API: one API key gives access to 100+ models from
Anthropic, OpenAI, Meta, Mistral, and others.  We use the openai SDK
with a custom base_url pointing at OpenRouter.

The generator's job is narrow:
  • Build a prompt from the top-K retrieved chunks.
  • Call the LLM.
  • Parse which source numbers the model cited in its answer.
  • Return the answer text and the subset of sources that were actually cited.

The LLM is instructed to cite sources as [1], [2], etc.  If it does not
cite a source, that source is still returned but marked as uncited.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from loguru import logger


def generate(
    query: str,
    context_chunks: List[Dict],
    llm_model: str,
    temperature: float,
    api_key: str,
) -> Tuple[str, List[Dict]]:
    """
    Generate an answer grounded in the provided context chunks.

    Args:
        query:          The original user question.
        context_chunks: List of chunk dicts, each with at minimum:
                          - text:        chunk text content
                          - source_doc:  source document name
                          - page_number: int or None
        llm_model:      OpenRouter model identifier string.
        temperature:    Sampling temperature (0.0 = deterministic).
        api_key:        OpenRouter API key.

    Returns:
        Tuple of:
          • answer_text (str): The LLM-generated answer with [N] citations.
          • sources_used (List[dict]): The cited source dicts with added
            "citation_index" key for display.
    """
    if not context_chunks:
        return (
            "I could not find relevant information in the indexed documents "
            "to answer this question.",
            [],
        )

    # Build the numbered context block
    context_lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_label = chunk.get("source_doc", "Unknown")
        page = chunk.get("page_number")
        page_str = f", page {page}" if page else ""
        context_lines.append(f"[{i}] (Source: {source_label}{page_str})\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)

    system_prompt = (
        "You are a precise question-answering assistant.\n"
        "Answer the question using ONLY the information in the provided context.\n"
        "For each claim you make, cite the source number in square brackets like [1] or [2].\n"
        "If the context does not contain enough information to answer the question, "
        "say so explicitly — do not invent information.\n"
        "Keep your answer concise and factual."
    )

    user_prompt = f"Context:\n\n{context_block}\n\nQuestion: {query}"

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        return (
            f"Generation error: {exc}. Please check your OpenRouter API key and model.",
            [],
        )

    # Identify which citation indices appear in the answer
    cited_indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}

    sources_used = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_entry = {
            "citation_index": i,
            "source_doc": chunk.get("source_doc", "Unknown"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id", ""),
            "cited": i in cited_indices,
        }
        sources_used.append(source_entry)

    logger.debug(
        f"Generated answer ({len(answer_text)} chars); "
        f"cited sources: {sorted(cited_indices)}"
    )
    return answer_text, sources_used
