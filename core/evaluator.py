"""
core/evaluator.py
─────────────────
Evaluation metrics and eval-set generation for a RAG pipeline.

This module measures two things independently:

  1. Retrieval quality — did the pipeline fetch the right chunks?
       • NDCG@5  (Normalised Discounted Cumulative Gain)
       • MRR     (Mean Reciprocal Rank)

  2. Generation quality — did the LLM use those chunks faithfully?
       • Faithfulness — what fraction of claims are supported by context?

Both metrics are computed for a *naive* pipeline (dense-only, no expansion,
no re-ranking) and for the *hybrid* pipeline (full stack), on the same eval
set so the comparison is apples-to-apples.

Eval set format (stored as eval_set.json):
  [
    {
      "query": str,
      "ideal_answer": str,
      "relevant_chunk_ids": [str, ...],
      "source_doc": str
    }
  ]
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

from loguru import logger

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg_at_5(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 5,
) -> float:
    """
    Compute NDCG@k for a single query result.

    NDCG discounts the value of a relevant document by its position — a
    relevant document at rank 1 is worth more than one at rank 5.

    Args:
        retrieved_ids: Ranked list of chunk ids returned by the pipeline.
        relevant_ids:  Ground-truth set of relevant chunk ids.
        k:             Cutoff depth (default 5).

    Returns:
        NDCG@k score in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant docs appear at the top positions
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    Compute the reciprocal rank for a single query result.

    MRR = 1 / rank_of_first_relevant_document.
    Returns 0.0 if no relevant document is found in the retrieved list.

    Args:
        retrieved_ids: Ranked list of chunk ids.
        relevant_ids:  Ground-truth relevant chunk ids.

    Returns:
        Reciprocal rank score in [0.0, 1.0].
    """
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness scoring (LLM judge)
# ─────────────────────────────────────────────────────────────────────────────

def _score_faithfulness_llm(
    answer: str,
    context_chunks: List[str],
    llm_model: str,
    api_key: str,
) -> float:
    """
    Use an LLM judge to score faithfulness of an answer against its sources.

    Sends a prompt asking the LLM what fraction of claims in the answer
    are directly supported by the provided context chunks.

    Returns:
        Float in [0.0, 1.0].  Returns 0.5 on any parsing failure.
    """
    from openai import OpenAI  # type: ignore

    context_text = "\n\n".join(
        f"[Source {i+1}]: {text}" for i, text in enumerate(context_chunks)
    )
    prompt = (
        "You are an evaluation assistant.\n"
        "Given an answer and a set of source passages, "
        "determine what fraction of the claims made in the answer "
        "are directly supported by the source passages.\n"
        "Return ONLY a single float between 0.0 and 1.0, nothing else.\n"
        "0.0 means no claims are supported; 1.0 means all claims are supported.\n\n"
        f"Source passages:\n{context_text}\n\n"
        f"Answer:\n{answer}\n\n"
        "Faithfulness score (0.0 to 1.0):"
    )
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw)
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning(f"Faithfulness scoring failed: {exc}. Using 0.5 as default.")
        return 0.5


def _try_ragas_faithfulness(
    query: str,
    answer: str,
    contexts: List[str],
) -> Optional[float]:
    """
    Attempt to compute faithfulness using the RAGAS library.

    Returns None if RAGAS is not available or fails, so the caller can
    fall back to the LLM-judge approach.

    Args:
        query:    The original user query.
        answer:   The generated answer.
        contexts: List of retrieved chunk texts used as context.

    Returns:
        Float score in [0, 1] or None on failure.
    """
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness  # type: ignore

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness])
        return float(result["faithfulness"])
    except Exception as exc:
        logger.debug(f"RAGAS unavailable or failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Eval set generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_eval_set_from_corpus(
    pipeline,
    n_questions: int,
    api_key: str,
    llm_model: str,
) -> List[Dict]:
    """
    Auto-generate an eval set by sampling chunks and asking the LLM to
    produce question-answer pairs from each chunk.

    Args:
        pipeline:    A Pipeline instance with ingested documents.
        n_questions: Number of questions to generate.
        api_key:     OpenRouter API key.
        llm_model:   Which model to use for question generation.

    Returns:
        List of eval set dicts ready to save as eval_set.json.
    """
    import random
    from openai import OpenAI  # type: ignore

    all_chunks = pipeline.get_all_chunk_texts()
    if not all_chunks:
        logger.warning("No chunks available to generate eval set from.")
        return []

    # Sample without replacement (or repeat if corpus is small)
    sample_size = min(n_questions, len(all_chunks))
    sampled = random.sample(all_chunks, sample_size)
    if len(sampled) < n_questions:
        # Repeat samples to reach n_questions
        sampled = (sampled * (n_questions // len(sampled) + 1))[:n_questions]

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    eval_set = []

    for chunk_id, chunk_text in sampled:
        # Use a system + user split so models don't refuse the JSON instruction.
        # Keep the passage short to stay well within max_tokens.
        passage = chunk_text[:800].strip()
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a question-generation assistant. "
                            "Given a passage, produce exactly one question the passage answers "
                            "and a short ideal answer. "
                            'Respond with valid JSON only: {"question": "...", "answer": "..."}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Passage:\n{passage}",
                    },
                ],
                temperature=0.3,
                max_tokens=512,
            )

            # Guard against None content (model refusal / content filter)
            content = response.choices[0].message.content
            if not content:
                logger.warning(f"Empty response for chunk {chunk_id} — skipping.")
                continue

            raw = content.strip()
            logger.debug(f"Raw LLM response for {chunk_id}: {raw[:120]}")

            # Strip markdown code fences if present
            raw = raw.strip("```json").strip("```").strip()

            # Attempt 1: direct JSON parse
            parsed = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt 2: extract the first {...} block via regex
                import re
                m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass

            if not parsed:
                logger.warning(f"Could not parse JSON for chunk {chunk_id}. Raw: {raw[:200]}")
                continue

            question = parsed.get("question", "").strip()
            answer = parsed.get("answer", "").strip()
            if question and answer:
                eval_set.append(
                    {
                        "query": question,
                        "ideal_answer": answer,
                        "relevant_chunk_ids": [chunk_id],
                        "source_doc": "",
                    }
                )
        except Exception as exc:
            logger.warning(f"Failed to generate question for chunk {chunk_id}: {exc}")

    logger.info(f"Generated {len(eval_set)} eval questions.")
    return eval_set


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation run
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    pipeline,
    eval_set: List[Dict],
    api_key: str,
) -> Dict:
    """
    Run the complete evaluation comparing naive vs hybrid retrieval.

    For each query in the eval set:
      1. Run naive retrieval → compute NDCG@5, MRR, latency.
      2. Run hybrid pipeline.query() → compute NDCG@5, MRR, latency.
      3. Score faithfulness of the hybrid answer.

    Saves results to store/{rag_id}/eval_results.json automatically.

    Args:
        pipeline: A fully initialised Pipeline instance.
        eval_set: List of eval dicts (query, ideal_answer, relevant_chunk_ids).
        api_key:  OpenRouter API key for generation and faithfulness scoring.

    Returns:
        Eval results dict (same structure as eval_results.json).
    """
    from datetime import datetime, timezone
    from core.retriever import naive_retrieve
    from core.vector_store import get_chunks_by_ids

    per_query = []
    naive_ndcgs, naive_mrrs, naive_latencies = [], [], []
    hybrid_ndcgs, hybrid_mrrs, hybrid_latencies = [], [], []
    faithfulness_scores = []

    for i, item in enumerate(eval_set):
        query = item["query"]
        relevant_ids = item.get("relevant_chunk_ids", [])
        logger.info(f"Evaluating query {i+1}/{len(eval_set)}: {query[:60]}…")

        # ── Naive retrieval ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        naive_results = naive_retrieve(
            query=query,
            collection=pipeline.collection,
            embedding_model=pipeline.config.embedding_model,
            top_k=5,
        )
        naive_latency = time.perf_counter() - t0
        naive_ids = [cid for cid, _ in naive_results]
        naive_ndcg = compute_ndcg_at_5(naive_ids, relevant_ids)
        naive_mrr = compute_mrr(naive_ids, relevant_ids)
        naive_first_rank = next(
            (i + 1 for i, cid in enumerate(naive_ids) if cid in set(relevant_ids)),
            None,
        )

        # ── Hybrid pipeline ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        hybrid_result = pipeline.query(query)
        hybrid_latency = time.perf_counter() - t0

        hybrid_post = hybrid_result["post_rerank_results"]
        hybrid_ids = [r["chunk_id"] for r in hybrid_post]
        hybrid_ndcg = compute_ndcg_at_5(hybrid_ids, relevant_ids)
        hybrid_mrr = compute_mrr(hybrid_ids, relevant_ids)
        hybrid_first_rank = next(
            (i + 1 for i, cid in enumerate(hybrid_ids) if cid in set(relevant_ids)),
            None,
        )

        # ── Faithfulness ─────────────────────────────────────────────────────
        context_texts = [r["text"] for r in hybrid_post]
        faith = _try_ragas_faithfulness(query, hybrid_result["answer"], context_texts)
        if faith is None:
            faith = _score_faithfulness_llm(
                answer=hybrid_result["answer"],
                context_chunks=context_texts,
                llm_model=pipeline.config.llm_model,
                api_key=api_key,
            )
        faithfulness_scores.append(faith)

        # Accumulate
        naive_ndcgs.append(naive_ndcg)
        naive_mrrs.append(naive_mrr)
        naive_latencies.append(naive_latency)
        hybrid_ndcgs.append(hybrid_ndcg)
        hybrid_mrrs.append(hybrid_mrr)
        hybrid_latencies.append(hybrid_latency)

        per_query.append(
            {
                "query": query,
                "naive_first_relevant_rank": naive_first_rank,
                "hybrid_first_relevant_rank": hybrid_first_rank,
                "naive_ndcg": round(naive_ndcg, 4),
                "hybrid_ndcg": round(hybrid_ndcg, 4),
                "faithful": faith >= 0.7,
            }
        )

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "naive": {
            "ndcg_at_5": _avg(naive_ndcgs),
            "mrr": _avg(naive_mrrs),
            "faithfulness": 0.0,  # Faithfulness only computed for hybrid
            "avg_latency_seconds": _avg(naive_latencies),
        },
        "hybrid": {
            "ndcg_at_5": _avg(hybrid_ndcgs),
            "mrr": _avg(hybrid_mrrs),
            "faithfulness": _avg(faithfulness_scores),
            "avg_latency_seconds": _avg(hybrid_latencies),
        },
        "per_query": per_query,
    }

    # Save to disk
    results_path = os.path.join(_STORE_ROOT, pipeline.config.id, "eval_results.json")
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, results_path)

    logger.info(
        f"Evaluation complete. "
        f"Hybrid NDCG@5={results['hybrid']['ndcg_at_5']}, "
        f"Naive NDCG@5={results['naive']['ndcg_at_5']}"
    )
    return results
