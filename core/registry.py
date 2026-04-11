"""
core/registry.py
────────────────
Manages the master registry of all RAG pipeline instances.

All RAGs are stored in store/rags.json.  Each RAG also gets its own
subdirectory under store/{rag_id}/ for BM25 indexes, eval sets, etc.

All JSON writes are atomic (write to .tmp, then os.rename) so a crash
cannot leave rags.json in a half-written state.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from core.config import RAGConfig

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")


def _rags_json_path() -> str:
    return os.path.join(_STORE_ROOT, "rags.json")


def rag_dir(rag_id: str) -> str:
    """Return the directory path for a single RAG's persistent data."""
    return os.path.join(_STORE_ROOT, rag_id)


def chroma_path() -> str:
    """Return the shared ChromaDB persistence directory."""
    return os.path.join(_STORE_ROOT, "chroma_db")


def ensure_store_exists() -> None:
    """Create the store/ directory tree on first run."""
    os.makedirs(_STORE_ROOT, exist_ok=True)
    os.makedirs(chroma_path(), exist_ok=True)


# ---------------------------------------------------------------------------
# Registry read / write
# ---------------------------------------------------------------------------

def load_all_rags() -> List[RAGConfig]:
    """
    Load the list of all RAG configs from rags.json.

    Returns an empty list if the file does not yet exist (first run).
    """
    path = _rags_json_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [RAGConfig(**item) for item in raw]
    except Exception as exc:
        logger.error(f"Failed to load rags.json: {exc}")
        return []


def save_all_rags(rags: List[RAGConfig]) -> None:
    """
    Atomically write the list of RAG configs to rags.json.

    Uses a .tmp file + os.rename to prevent corruption on crash.
    """
    ensure_store_exists()
    path = _rags_json_path()
    tmp_path = path + ".tmp"
    data = [r.model_dump() for r in rags]
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.error(f"Failed to save rags.json: {exc}")
        raise


def get_rag(rag_id: str) -> Optional[RAGConfig]:
    """Return the RAGConfig for a given id, or None if not found."""
    for rag in load_all_rags():
        if rag.id == rag_id:
            return rag
    return None


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Convert a display name to a safe, lowercase slug for use as a directory / id."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "rag"


def create_rag(config: RAGConfig) -> RAGConfig:
    """
    Persist a new RAG configuration.

    Steps:
      1. Validate that the name is unique (case-insensitive slug comparison).
      2. Create store/{rag_id}/ directory.
      3. Write an empty eval_set.json.
      4. Initialise the SQLite chat database.
      5. Append to rags.json.

    Args:
        config: A fully populated RAGConfig (id + created_at must be set by caller).

    Returns:
        The saved RAGConfig.

    Raises:
        ValueError: If a RAG with the same name already exists.
    """
    from core.database import init_db

    existing = load_all_rags()
    existing_ids = {r.id for r in existing}
    if config.id in existing_ids:
        raise ValueError(f"A RAG with id '{config.id}' already exists.")

    # Create directory
    d = rag_dir(config.id)
    os.makedirs(d, exist_ok=True)

    # Empty eval set
    _write_json_atomic(os.path.join(d, "eval_set.json"), [])

    # SQLite DB
    init_db(config.id)

    # Persist
    existing.append(config)
    save_all_rags(existing)
    logger.info(f"Created RAG '{config.name}' (id={config.id})")
    return config


def delete_rag(rag_id: str) -> None:
    """
    Permanently remove a RAG:
      1. Drop its ChromaDB collection.
      2. Delete store/{rag_id}/ recursively.
      3. Remove from rags.json.

    Args:
        rag_id: The id of the RAG to delete.
    """
    # Drop ChromaDB collection
    try:
        from core.vector_store import delete_collection
        delete_collection(rag_id)
    except Exception as exc:
        logger.warning(f"Could not drop ChromaDB collection '{rag_id}': {exc}")

    # Delete the RAG's data directory
    d = rag_dir(rag_id)
    if os.path.exists(d):
        shutil.rmtree(d)
        logger.info(f"Deleted data directory: {d}")

    # Remove from registry
    rags = [r for r in load_all_rags() if r.id != rag_id]
    save_all_rags(rags)
    logger.info(f"Deleted RAG id='{rag_id}'")


def update_rag(rag_id: str, updates: dict) -> RAGConfig:
    """
    Apply a partial update to a RAG's config and persist.

    Args:
        rag_id:  The id of the RAG to update.
        updates: Dict of field names → new values.

    Returns:
        The updated RAGConfig.

    Raises:
        ValueError: If no RAG with rag_id is found.
    """
    rags = load_all_rags()
    for i, r in enumerate(rags):
        if r.id == rag_id:
            updated = r.model_copy(update=updates)
            rags[i] = updated
            save_all_rags(rags)
            logger.info(f"Updated RAG '{rag_id}': {list(updates.keys())}")
            return updated
    raise ValueError(f"RAG '{rag_id}' not found.")


def make_rag_id(name: str) -> str:
    """Generate a unique slug id from a display name."""
    base = _slugify(name)
    existing_ids = {r.id for r in load_all_rags()}
    if base not in existing_ids:
        return base
    # Append a counter to ensure uniqueness
    for i in range(2, 100):
        candidate = f"{base}_{i}"
        if candidate not in existing_ids:
            return candidate
    raise ValueError("Could not generate a unique id for this RAG name.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json_atomic(path: str, data: object) -> None:
    """Write JSON to path atomically via a .tmp file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
