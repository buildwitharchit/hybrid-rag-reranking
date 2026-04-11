# RAG Builder — Hybrid Retrieval-Augmented Generation Platform

A self-hosted, production-grade RAG platform that lets you create and benchmark
multiple independent RAG pipelines — each with its own document corpus, retrieval
configuration, and evaluation results.

Built from primitives (no LangChain, no LlamaIndex) so every component is
explainable and debuggable.

---

## What it does

Each RAG pipeline you create runs a full hybrid retrieval stack:

```
User Query
  │
  ▼
Query Expansion (LLM generates 3 variants)
  │
  ├──► Dense Search   (sentence-transformers + ChromaDB)
  │
  └──► Sparse Search  (BM25)
           │
           ▼
      Reciprocal Rank Fusion  (merge all ranked lists)
           │
           ▼
      Cross-Encoder Re-ranking  (ms-marco-MiniLM)
           │
           ▼
      Top-K Context Assembly
           │
           ▼
      LLM Generation  (via OpenRouter)
           │
           ▼
      Answer + Source Citations
```

The built-in evaluation system benchmarks your hybrid pipeline against naive RAG
(dense-only, no expansion, no re-ranking) on the same query set and reports:

| Metric | Description |
|---|---|
| **NDCG@5** | Normalised Discounted Cumulative Gain — did relevant docs appear at the top? |
| **MRR** | Mean Reciprocal Rank — how often is the first hit the right one? |
| **Faithfulness** | What fraction of the LLM's claims are grounded in retrieved context? |
| **Latency** | End-to-end wall-clock time with per-stage breakdown |

---

## Tech stack

| Component | Technology | Cost |
|---|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free — runs locally |
| Vector DB | ChromaDB (local, no server) | Free |
| Sparse search | BM25 via rank_bm25 | Free |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free — runs locally |
| LLM | Any model via OpenRouter API | Pay-per-query (~$0.001) |
| UI | Streamlit | Free |
| Evaluation | NDCG/MRR (local) + LLM judge (OpenRouter) | Small cost per run |

All ML models run **locally on CPU** inside the Docker container. The only
external service is OpenRouter for LLM generation.

---

## Prerequisites

- Docker and Docker Compose
- An [OpenRouter API key](https://openrouter.ai/keys) (the only paid dependency)

**Server requirements:**
- 8 GB RAM recommended (4 GB minimum)
- 4+ CPU cores recommended
- ~10 GB disk space (Docker image with models)

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid-rag-reranking
cd hybrid-rag-reranking
```

### 2. Set your API key

```bash
cp .env.example .env
# Open .env and set OPENROUTER_API_KEY=sk-or-your-key-here
```

### 3. Build and run

```bash
docker compose up --build
```

The first build downloads all ML models (~2.5 GB). This takes 10-20 minutes
once and is cached in the Docker image. Subsequent starts take ~5 seconds.

### 4. Open the app

Navigate to `http://localhost:8501` in your browser.

---

## Running locally without Docker

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# Run the app
streamlit run app.py
```

On first run, models are downloaded from HuggingFace (~2.5 GB total, one-time).

---

## How to use

### 1. Create a RAG pipeline

Click **+ New RAG** in the sidebar. Configure:

- **Chunking**: strategy (recursive / fixed / sentence / semantic), size, overlap
- **Embeddings**: which local sentence-transformer model to use
- **Retrieval**: BM25 sparse search, query expansion method, fusion algorithm
- **Re-ranking**: cross-encoder model and final Top-K
- **Generation**: which OpenRouter LLM and temperature

Greyed-out options require a paid API or GPU — they are shown for awareness
only and cannot be selected.

### 2. Ingest documents

Open your RAG → **Ingest** tab. Upload PDFs, TXT, or DOCX files, paste a URL,
or type plain text directly.

### 3. Chat

Open the **Chat** tab. Ask questions about your documents. Each response shows
the generated text with numbered source citations and coloured chips linking to
source documents.

### 4. View retrieval internals

Open the **Internals** tab after a chat message to see:
- What query variants the LLM generated
- Which chunks were retrieved before re-ranking (with RRF and BM25 scores)
- How the cross-encoder re-ordered them (with rank deltas like up 3 or down 1)
- Latency breakdown per pipeline stage

### 5. Benchmark your pipeline

Open the **Evaluation** tab:
1. Click **Generate** to auto-create a test set from your corpus, or upload a CSV.
2. Click **Run Evaluation** to benchmark naive vs hybrid retrieval.
3. View metric cards and a per-query breakdown table.

---

## Project structure

```
hybrid-rag-reranking/
│
├── app.py                    # Streamlit entry point (routing only, no logic)
│
├── core/                     # All ML and business logic (no Streamlit imports)
│   ├── config.py             # RAGConfig Pydantic model + validation
│   ├── registry.py           # rags.json read/write, CRUD operations
│   ├── database.py           # SQLite chat history
│   ├── chunker.py            # 4 chunking strategies
│   ├── embedder.py           # Sentence-transformer singleton cache
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── sparse.py             # BM25 index build/save/load/search
│   ├── retriever.py          # Dense + sparse + RRF fusion
│   ├── reranker.py           # Cross-encoder singleton cache
│   ├── query_expander.py     # LLM variants and HyDE expansion
│   ├── generator.py          # OpenRouter LLM generation with citations
│   ├── pipeline.py           # Full pipeline orchestrator
│   └── evaluator.py          # NDCG, MRR, and faithfulness metrics
│
├── ui/                       # Streamlit page renderers (no ML logic)
│   ├── components/
│   │   ├── sidebar.py        # Navigation sidebar
│   │   └── metrics.py        # Reusable metric card component
│   ├── home.py               # RAG card dashboard
│   ├── create.py             # Pipeline creation form
│   ├── ingest.py             # Document upload tab
│   ├── chat.py               # Chat interface tab
│   ├── evaluation.py         # Evaluation dashboard tab
│   ├── internals.py          # Retrieval internals viewer tab
│   └── config_view.py        # Config viewer, editor, and delete
│
├── store/                    # Runtime data (created at startup, ephemeral)
│   ├── rags.json             # Master registry of all RAG pipelines
│   ├── chroma_db/            # ChromaDB vector collections
│   └── {rag_id}/
│       ├── config.json
│       ├── bm25_index.pkl
│       ├── eval_set.json
│       ├── eval_results.json
│       └── chat.db
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Configuration reference

### Free options (run locally on CPU, inside Docker)

| Category | Available options |
|---|---|
| Embeddings | all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2, bge-large-en-v1.5 |
| Vector store | ChromaDB |
| Sparse search | BM25, BM25+ |
| Fusion | RRF (recommended), Weighted sum, CombSUM |
| Query expansion | None, LLM variants, HyDE |
| Re-ranking | None, ms-marco cross-encoder (85 MB), BGE Reranker Large (1.3 GB) |

### Paid via OpenRouter

| Model ID | Notes |
|---|---|
| meta-llama/llama-3-70b-instruct | Excellent instruction following |
| mistralai/mixtral-8x7b-instruct | Fast and cost-effective |
| anthropic/claude-3-haiku | Best at citation-style answers |
| openai/gpt-4o-mini | Reliable general-purpose option |
| google/gemma-2-9b-it | Lightweight option |

### Disabled options (shown greyed-out in UI)

These are displayed so you understand what exists, but cannot be selected
without additional infrastructure or API keys:

- OpenAI text-embedding-3-small / large (API cost)
- FAISS, Pinecone, Qdrant (separate infrastructure)
- SPLADE sparse embeddings (GPU required)
- Cohere Rerank v3 / Jina Reranker (API cost)
- T5 paraphrase expansion (GPU required)

---

## Data persistence

By default the container is **ephemeral** — all RAGs, chunks, and chat history
reset when the container is removed. To persist data across restarts, add a
volume mount in docker-compose.yml:

```yaml
services:
  rag-builder:
    volumes:
      - ./data:/app/store
```

---

## Architecture decisions

**Why no LangChain or LlamaIndex?**
Those frameworks abstract away the components this project demonstrates.
Building from primitives means every step is explainable — RRF formula,
cross-encoder vs bi-encoder trade-offs, chunking strategy effects.

**Why ChromaDB?**
Zero infrastructure. Runs in-process, persists to disk. Perfect for
single-machine deployments and corpora up to ~100k chunks.

**Why hybrid (BM25 + dense) retrieval?**
Dense search understands semantics but misses exact keywords, product codes,
and rare terms. BM25 excels at exact term matching but misses paraphrases.
Their failure modes have low overlap — hybrid consistently outperforms either
alone by 5-15% NDCG on standard benchmarks.

**Why Reciprocal Rank Fusion?**
BM25 and cosine scores are on incompatible scales. Normalising them is
sensitive to outliers. RRF uses only rank positions (scale-invariant) and
requires no hyperparameter tuning, performing within 1-2% of learned fusion
on standard benchmarks.

**Why cross-encoder re-ranking?**
Bi-encoders encode query and document separately — they cannot see interactions
between query and document tokens. Cross-encoders encode the concatenated pair,
allowing full attention across both and catching relevance signals bi-encoders
miss: negation, topic drift, contextual qualifications. The cost (one forward
pass per candidate) is managed by running the cross-encoder only on the top-20
RRF candidates.

---

## License

MIT
