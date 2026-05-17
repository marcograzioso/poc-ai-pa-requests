"""Classification training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.utils.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """Bundle common classification metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    report: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]


def _optional_model_candidates(random_state: int = 42) -> Dict[str, object]:
    """Return classifier candidates, adding optional boosters if available."""
    models: Dict[str, object] = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        )
    except Exception:
        logger.info("xgboost not available: skipping")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(n_estimators=350, learning_rate=0.05, random_state=random_state)
    except Exception:
        logger.info("lightgbm not available: skipping")

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    """Compute classification metrics for evaluation and explainability."""
    return MetricsResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        report=classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def train_best_model(X: np.ndarray, y: np.ndarray) -> Tuple[str, object, MetricsResult]:
    """Train multiple models and keep the one with best macro F1."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = _optional_model_candidates()

    best_name = ""
    best_model = None
    best_metrics = None
    best_f1 = -1.0

    for name, base_model in candidates.items():
        logger.info("Training model: %s", name)
        needs_encoding = name in ("xgboost", "lightgbm")
        le = LabelEncoder()

        if needs_encoding:
            y_train_fit = le.fit_transform(y_train)
            y_test_eval = le.transform(y_test)
        else:
            y_train_fit = y_train
            y_test_eval = y_test

        base_model.fit(X_train, y_train_fit)
        raw_preds = base_model.predict(X_test)

        if needs_encoding:
            preds = le.inverse_transform(raw_preds)
        else:
            preds = raw_preds

        metrics = compute_metrics(y_test, preds)
        if metrics.f1_macro > best_f1:
            best_f1 = metrics.f1_macro
            best_name = name
            # Wrap boosters together with their encoder for transparent inference.
            if needs_encoding:
                best_model = _EncodedModel(base_model, le)
            else:
                best_model = base_model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return best_name, best_model, best_metrics


class _EncodedModel:
    """Thin wrapper that encodes labels for boosters and exposes a sklearn-like API."""

    def __init__(self, estimator: object, le: LabelEncoder) -> None:
        self.estimator = estimator
        self.le = le
        self.classes_ = le.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.le.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)  # type: ignore[attr-defined]


def save_model(model: object, path: Path) -> None:
    """Serialize trained model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> object:
    """Load serialized model artifact."""
    return joblib.load(path)
