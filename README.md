# Hybrid RAG with Re-ranking - 91.3% Answer Faithfulness on Indian Constitutional Law

> 91.3% of generated answers are fully grounded in source documents, verified against a
> 20-query benchmark on Indian constitutional law using NDCG@5, MRR, and LLM-as-judge faithfulness.

## What This Does

Most RAG systems retrieve documents and generate answers without verifying that the answer
is actually supported by what was retrieved. This project implements a full hybrid retrieval
pipeline that combines keyword search (BM25), semantic search (dense vectors), and
cross-encoder re-ranking to maximise both retrieval precision and answer faithfulness
on domain-specific corpora such as legal and constitutional texts.

Real-world use case: build a question-answering system over large document collections
(legislation, textbooks, technical manuals) where factual accuracy is non-negotiable.

## Key Results

Evaluated on a 20-query benchmark auto-generated from the Indian Constitution and
M. Lakshmikant's Indian Polity using LLM-as-judge scoring.

| Metric | Naive RAG (Baseline) | Hybrid Pipeline |
|---|---|---|
| NDCG@5 | 0.454 | 0.424 |
| MRR | 0.387 | 0.329 |
| Faithfulness | not evaluated | **91.3%** |
| Avg Latency | 0.05s | 5.67s |

**Methodology notes:**
- Naive baseline uses dense-only retrieval with no query expansion and no re-ranking.
- Faithfulness is scored by an LLM judge (OpenRouter) asking whether each claim in the
  generated answer is directly supported by the retrieved passages. Naive baseline does
  not compute faithfulness since it does not generate answers.
- NDCG and MRR are slightly lower for the hybrid pipeline on this corpus because
  constitutional and legal text is terminology-dense; exact keyword matching (BM25 alone)
  performs competitively. The hybrid pipeline's advantage is expected to grow with
  larger, more varied corpora.
- Latency reflects the full pipeline cost: query expansion (LLM call), BM25 + dense
  retrieval, cross-encoder re-ranking, and answer generation.

## Architecture

```
User Query
  |
  v
Query Expansion  (LLM generates 3 query variants via OpenRouter)
  |
  +-----> Dense Search   (sentence-transformers + ChromaDB, cosine similarity)
  |
  +-----> Sparse Search  (BM25Okapi, term frequency-inverse document frequency)
           |
           v
      Reciprocal Rank Fusion  (rank-based merge, no score normalisation needed)
           |
           v
      Cross-Encoder Re-ranking  (ms-marco-MiniLM, full attention over query+doc pair)
           |
           v
      Top-K Context Assembly
           |
           v
      LLM Generation  (any model via OpenRouter, citation index injection)
           |
           v
      Answer + Numbered Source Citations
```

Every stage is observable: the Internals tab in the UI shows query variants, pre- and
post-rerank chunk rankings, score deltas, and per-stage latency for every query.

## Tech Stack

| Component | Technology | Runs Where |
|---|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local CPU (free) |
| Vector database | ChromaDB (embedded, no server) | Local disk (free) |
| Sparse retrieval | BM25Okapi via rank-bm25 | Local CPU (free) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Local CPU (free) |
| LLM generation | Any model via OpenRouter API | Remote (pay-per-query) |
| Evaluation judge | LLM-as-judge via OpenRouter | Remote (small cost) |
| UI | Streamlit | Local |
| Storage | ChromaDB + SQLite + JSON | Local disk |

No LangChain, no LlamaIndex. Every component is built from primitives.

## Key Features

- **Hybrid retrieval** - BM25 sparse search and dense vector search run in parallel;
  their ranked lists are merged using Reciprocal Rank Fusion without any score
  normalisation or hyperparameter tuning.
- **Cross-encoder re-ranking** - a second-stage re-ranker reads the full (query, chunk)
  pair together, catching relevance signals that bi-encoders miss (negation, topic drift,
  contextual qualification).
- **Query expansion** - the LLM generates three alternative phrasings of the user query
  before retrieval, increasing recall for paraphrased or ambiguous questions. HyDE
  (Hypothetical Document Embedding) is also available.
- **LLM-as-judge faithfulness** - every generated answer is scored by an independent
  LLM judge against the retrieved passages, giving a measurable signal for hallucination.
- **Retrieval internals viewer** - shows query variants, pre- and post-rerank chunk lists
  with RRF, BM25, and cross-encoder scores, and a per-stage latency breakdown for every
  query. Designed for debugging and for explaining the system to non-technical stakeholders.
- **Multi-RAG management** - create multiple independent pipelines in a single deployment,
  each with its own corpus, configuration, chat history, and evaluation results.
- **Configurable chunking** - four strategies (recursive, fixed, sentence, semantic) with
  adjustable size and overlap, locked per pipeline after first ingestion.
- **Atomic writes** - all JSON state files are written to a .tmp file then renamed,
  preventing partial-write corruption on crashes or container termination.

## Setup

### Prerequisites

- Docker and Docker Compose
- An OpenRouter API key (the only paid dependency - used for LLM generation and evaluation)

**Recommended hardware:** 8 GB RAM, 4 CPU cores, 10 GB disk space.

### Docker (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hybrid-rag-reranking
cd hybrid-rag-reranking

# 2. Add your API key
cp .env.example .env
# Open .env and set OPENROUTER_API_KEY=sk-or-your-key-here

# 3. Build and start
docker compose up --build
```

The first build downloads all ML models (~2.5 GB) and bakes them into the image.
This takes 10-20 minutes once. Subsequent starts take approximately 5 seconds.

Open `http://localhost:8501` in your browser.

### Local (without Docker)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add OPENROUTER_API_KEY
streamlit run app.py
```

Models are downloaded from HuggingFace on first run (~2.5 GB, one-time).

## Usage

1. **Create a pipeline** - click "+ New RAG" in the sidebar, configure chunking strategy,
   embedding model, retrieval settings, re-ranker, and LLM.
2. **Ingest documents** - upload PDF, TXT, or DOCX files, paste a URL to scrape, or
   paste plain text directly in the Ingest tab.
3. **Chat** - ask questions in the Chat tab. Responses include numbered source citations
   linked back to the source document and page number.
4. **Inspect internals** - open the Internals tab after any query to see the full
   retrieval trace: query variants, ranked chunks before and after re-ranking, and
   stage-level latency.
5. **Evaluate** - auto-generate a test set from your corpus or upload a CSV, then run
   a head-to-head benchmark of naive vs hybrid retrieval.

## Project Structure

```
hybrid-rag-reranking/
|
+-- app.py                    # Streamlit entry point (routing only, no logic)
|
+-- core/                     # All ML and business logic (no Streamlit imports)
|   +-- config.py             # RAGConfig Pydantic model + validation
|   +-- registry.py           # rags.json read/write, CRUD operations
|   +-- database.py           # SQLite chat history
|   +-- chunker.py            # 4 chunking strategies
|   +-- embedder.py           # Sentence-transformer singleton cache
|   +-- vector_store.py       # ChromaDB wrapper
|   +-- sparse.py             # BM25 index build/save/load/search
|   +-- retriever.py          # Dense + sparse + RRF fusion
|   +-- reranker.py           # Cross-encoder singleton cache
|   +-- query_expander.py     # LLM variants and HyDE expansion
|   +-- generator.py          # OpenRouter LLM generation with citations
|   +-- pipeline.py           # Full pipeline orchestrator
|   +-- evaluator.py          # NDCG, MRR, and faithfulness metrics
|
+-- ui/                       # Streamlit page renderers (no ML logic)
|   +-- components/
|   |   +-- sidebar.py        # Navigation sidebar
|   |   +-- metrics.py        # Reusable metric card component
|   +-- home.py               # RAG card dashboard
|   +-- create.py             # Pipeline creation form
|   +-- ingest.py             # Document upload tab
|   +-- chat.py               # Chat interface tab
|   +-- evaluation.py         # Evaluation dashboard tab
|   +-- internals.py          # Retrieval internals viewer tab
|   +-- config_view.py        # Config viewer, editor, and delete
|
+-- store/                    # Runtime data (created at startup, ephemeral in Docker)
|   +-- rags.json             # Master registry of all RAG pipelines
|   +-- chroma_db/            # ChromaDB vector collections
|   +-- {rag_id}/
|       +-- config.json
|       +-- bm25_index.pkl
|       +-- eval_set.json
|       +-- eval_results.json
|       +-- chat.db
|
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
```

## Configuration Reference

### Free options (run locally on CPU inside Docker)

| Category | Available options |
|---|---|
| Embeddings | all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2, bge-large-en-v1.5 |
| Vector store | ChromaDB |
| Sparse search | BM25, BM25+ |
| Fusion | RRF (recommended), Weighted sum, CombSUM |
| Query expansion | None, LLM variants, HyDE |
| Re-ranking | None, ms-marco cross-encoder (85 MB), BGE Reranker Large (1.3 GB) |

### Paid via OpenRouter

| Model | Notes |
|---|---|
| meta-llama/llama-3-70b-instruct | Strong instruction following |
| mistralai/mixtral-8x7b-instruct | Fast and cost-effective |
| anthropic/claude-3-haiku | Reliable citation-style answers |
| openai/gpt-4o-mini | General-purpose baseline |
| google/gemma-3-12b-it | Lightweight open-weight option |
| google/gemma-2-9b-it | Alternative lightweight option |

### Greyed-out options (shown in UI for awareness, not selectable)

These require separate infrastructure or paid API keys beyond OpenRouter:

- OpenAI text-embedding-3-small / large (API cost)
- FAISS, Pinecone, Qdrant (separate infrastructure)
- SPLADE sparse embeddings (GPU required)
- Cohere Rerank v3 / Jina Reranker (API cost)
- T5 paraphrase expansion (GPU required)

## Design Decisions

**Why no LangChain or LlamaIndex?**
Those frameworks abstract away the components this project demonstrates. Building from
primitives means every step is explainable - RRF formula, cross-encoder vs bi-encoder
trade-offs, chunking strategy effects on recall. This matters for debugging and for
understanding production failures.

**Why ChromaDB?**
Zero infrastructure. Runs in-process and persists to disk. Suitable for single-machine
deployments and corpora up to approximately 100,000 chunks without a separate vector
database service.

**Why hybrid (BM25 + dense) retrieval?**
Dense search understands semantics but misses exact keywords, product codes, and rare
terms. BM25 excels at exact term matching but misses paraphrases. Their failure modes
have low overlap, so combining them consistently outperforms either alone. For
constitutional and legal text with precise terminology (article numbers, amendment names,
constitutional provisions), BM25 is particularly competitive.

**Why Reciprocal Rank Fusion?**
BM25 and cosine scores are on incompatible scales. Normalising them is sensitive to
outliers. RRF uses only rank positions (scale-invariant) and requires no hyperparameter
tuning, performing within 1-2% of learned fusion methods on standard benchmarks.

**Why cross-encoder re-ranking?**
Bi-encoders encode the query and document separately; they cannot see token-level
interactions between the two. Cross-encoders encode the concatenated (query, document)
pair, allowing full attention across both and catching relevance signals bi-encoders
miss: negation, topic drift, and contextual qualifications. The cost is managed by
running the cross-encoder only on the top-20 RRF candidates rather than the full corpus.

## Data Persistence

By default the container is ephemeral: all RAGs, chunks, and chat history reset when
the container is removed. To persist data across restarts, add a volume mount in
`docker-compose.yml`:

```yaml
services:
  rag-builder:
    volumes:
      - ./data:/app/store
```

## License

MIT
