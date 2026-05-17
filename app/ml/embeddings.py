"""Sentence embedding wrapper for ML and RAG pipelines."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.config import settings


class EmbeddingService:
    """Thin wrapper around sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Compute normalized embeddings for input texts."""
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
