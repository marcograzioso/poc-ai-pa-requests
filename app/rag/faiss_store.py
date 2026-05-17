"""FAISS index management for semantic retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from app.utils.config import settings


def save_index(vectors: np.ndarray, metadata: List[Dict[str, str]]) -> None:
    """Persist FAISS index and corresponding metadata."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    vectors32 = vectors.astype("float32")
    index = faiss.IndexFlatIP(vectors32.shape[1])
    index.add(vectors32)

    faiss.write_index(index, str(settings.vector_index_path))
    with Path(settings.vector_meta_path).open("wb") as f:
        pickle.dump(metadata, f)


def load_index() -> tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Load FAISS index and metadata from disk."""
    index = faiss.read_index(str(settings.vector_index_path))
    with Path(settings.vector_meta_path).open("rb") as f:
        metadata = pickle.load(f)
    return index, metadata
