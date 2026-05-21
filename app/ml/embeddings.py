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
        self.uses_query_passage_prefixes = "e5" in self.model_name.lower()

    def _apply_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        """Apply E5-style prefixes only when the selected model requires them."""
        if not self.uses_query_passage_prefixes:
            return texts

        prefix = "query: " if is_query else "passage: "
        return [f"{prefix}{text}" for text in texts]

    def encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        """Compute normalized embeddings for input texts."""
        prepared_texts = self._apply_prefix(texts, is_query=is_query)
        return self.model.encode(prepared_texts, show_progress_bar=False, normalize_embeddings=True)
