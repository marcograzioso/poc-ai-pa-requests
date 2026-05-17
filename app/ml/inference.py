"""Runtime inference service for request classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from app.ml.classifier import load_model
from app.ml.embeddings import EmbeddingService
from app.utils.config import settings
from app.utils.constants import CATEGORY_TO_OFFICE


@dataclass
class PredictionResult:
    """Container for label predictions and confidence."""

    label: str
    confidence: float


class ClassificationService:
    """Loads trained artifacts and predicts category, office, and priority."""

    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.category_model = load_model(settings.category_model_path)
        self.priority_model = load_model(settings.priority_model_path)

    @staticmethod
    def _predict_label(model: object, vector: np.ndarray) -> PredictionResult:
        probs = model.predict_proba(vector)[0]
        max_idx = int(np.argmax(probs))
        return PredictionResult(label=str(model.classes_[max_idx]), confidence=float(probs[max_idx]))

    def predict(self, text: str) -> Dict[str, PredictionResult]:
        """Predict all target labels from citizen request text."""
        vector = self.embedder.encode([text])
        category = self._predict_label(self.category_model, vector)
        priority = self._predict_label(self.priority_model, vector)
        office = PredictionResult(label=CATEGORY_TO_OFFICE.get(category.label, "Ufficio URP"), confidence=category.confidence)

        return {
            "category": category,
            "priority": priority,
            "office": office,
        }
