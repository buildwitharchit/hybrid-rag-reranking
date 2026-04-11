# ─────────────────────────────────────────────────────────────────────────────
# RAG Builder — Dockerfile
#
# Build strategy:
#   1. Install all Python dependencies.
#   2. Pre-download all local ML models at BUILD time so the container
#      starts instantly with no HuggingFace download delay at runtime.
#   3. Copy application code.
#   4. Run Streamlit on port 8501.
#
# Image size estimate: ~3-4 GB (mostly model weights).
# RAM requirement at runtime: 8 GB recommended, 4 GB minimum.
#
# Build:
#   docker build -t rag-builder .
#
# Run:
#   docker run -p 8501:8501 -e OPENROUTER_API_KEY=sk-or-... rag-builder
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download ML models at build time ──────────────────────────────────────
# Models are baked into the image so runtime startup is instant.
# bge-large-en-v1.5 is intentionally excluded (1.3 GB) — it downloads
# on first use if the user selects it in the UI.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from sentence_transformers.cross_encoder import CrossEncoder; \
print('Downloading all-MiniLM-L6-v2 ...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Downloading all-MiniLM-L12-v2 ...'); \
SentenceTransformer('all-MiniLM-L12-v2'); \
print('Downloading all-mpnet-base-v2 ...'); \
SentenceTransformer('all-mpnet-base-v2'); \
print('Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 ...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('All models downloaded successfully.'); \
"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Create the runtime data directory
RUN mkdir -p store

# ── Port and startup ──────────────────────────────────────────────────────────
EXPOSE 8501

# Streamlit configuration via CLI flags (no config.toml needed)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=200"]
