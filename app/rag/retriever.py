"""Semantic retrieval logic over FAISS vectors."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from app.ml.embeddings import EmbeddingService
from app.rag.faiss_store import load_index
from app.utils.config import settings


class SemanticRetriever:
    """Retrieve similar historical requests by vector similarity."""

    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.index, self.metadata = load_index()

    def search(self, text: str, top_k: int | None = None) -> List[Dict[str, str | float]]:
        """Return top-k nearest historical cases."""
        k = top_k or settings.top_k_retrieval
        query_vector = self.embedder.encode([text]).astype("float32")
        scores, indices = self.index.search(query_vector, k)

        results: List[Dict[str, str | float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results
