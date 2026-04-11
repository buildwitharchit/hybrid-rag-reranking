"""
core/query_expander.py
──────────────────────
Query expansion strategies that broaden the retrieval net.

Why expand queries?
  Users write short, ambiguous queries.  A single embedding captures one
  semantic direction.  Expansion generates multiple phrasings that cover
  different angles, increasing the chance that at least one matches the
  vocabulary used in the indexed documents.

Supported methods:
  • none         — Return the original query unchanged.
  • llm_variants — Ask the LLM to generate 3 alternative phrasings.
                   Returns [original] + 3 variants = 4 queries total.
  • hyde         — HyDE (Hypothetical Document Embedding): ask the LLM
                   to write a short hypothetical answer; embed that
                   answer for dense retrieval alongside the original query.

On any LLM failure, all methods gracefully fall back to [original_query].
"""

from __future__ import annotations

import json
from typing import List

from loguru import logger


def expand_query(
    query: str,
    method: str,
    llm_model: str,
    api_key: str,
) -> List[str]:
    """
    Expand a user query into multiple search queries.

    Args:
        query:     The original user query.
        method:    "none", "llm_variants", or "hyde".
        llm_model: OpenRouter model string used for LLM-based expansion.
        api_key:   OpenRouter API key.

    Returns:
        List of query strings.  Always contains at least [query].
    """
    if method == "none":
        return [query]
    elif method == "llm_variants":
        return _expand_llm_variants(query, llm_model, api_key)
    elif method == "hyde":
        return _expand_hyde(query, llm_model, api_key)
    else:
        logger.warning(f"Unknown expansion method '{method}', using original query.")
        return [query]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, llm_model: str, api_key: str) -> str:
    """Make a minimal OpenRouter completion call and return the content string."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _expand_llm_variants(query: str, llm_model: str, api_key: str) -> List[str]:
    """
    Use the LLM to generate 3 alternative phrasings of the query.

    Returns [original] + up to 3 variants = 4 queries.
    Falls back to [query] on any failure.
    """
    prompt = (
        "You are a search query optimizer.\n"
        "Given a user's question, generate exactly 3 alternative phrasings that:\n"
        "  - Preserve the original intent\n"
        "  - Use different vocabulary or approach the topic from a different angle\n"
        "  - Are concise (no longer than the original)\n\n"
        'Return ONLY a JSON array of 3 strings, nothing else. Example: ["alt1", "alt2", "alt3"]\n\n'
        f'Original question: {query}'
    )
    try:
        raw = _call_llm(prompt, llm_model, api_key)
        # Strip markdown code fences if present
        raw = raw.strip().strip("```json").strip("```").strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            variants = [str(v) for v in variants[:3] if v]
            logger.debug(f"LLM variants: {variants}")
            return [query] + variants
    except Exception as exc:
        logger.warning(f"LLM query expansion failed: {exc}. Using original query.")
    return [query]


def _expand_hyde(query: str, llm_model: str, api_key: str) -> List[str]:
    """
    Hypothetical Document Embedding (HyDE) expansion.

    Ask the LLM to write a short hypothetical answer to the query.
    The answer text is returned as a second "query" alongside the
    original; both are embedded for dense retrieval.

    The intuition: a good answer is closer in embedding space to
    relevant documents than the question itself.
    """
    prompt = (
        "Write a short paragraph (3-5 sentences) that would be a perfect answer "
        "to the following question. Be specific and factual. "
        "Do not include any preamble — just the answer paragraph.\n\n"
        f"Question: {query}"
    )
    try:
        hypothetical_answer = _call_llm(prompt, llm_model, api_key)
        if hypothetical_answer:
            logger.debug("HyDE generated hypothetical answer.")
            return [query, hypothetical_answer]
    except Exception as exc:
        logger.warning(f"HyDE expansion failed: {exc}. Using original query.")
    return [query]
