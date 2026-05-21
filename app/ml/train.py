"""Training script for category and priority classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.ml.classifier import save_model, train_best_model
from app.ml.embeddings import EmbeddingService
from app.utils.config import settings
from app.utils.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Train classifiers and store artifacts locally."""
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Train category and priority models")
    parser.add_argument("--dataset", type=str, default=str(settings.dataset_path))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df.dropna(subset=["citizen_request_text", "category", "priority"])

    embedder = EmbeddingService()
    X = embedder.encode(df["citizen_request_text"].astype(str).tolist(), is_query=False)

    category_name, category_model, category_metrics = train_best_model(X, df["category"].to_numpy())
    priority_name, priority_model, priority_metrics = train_best_model(X, df["priority"].to_numpy())

    save_model(category_model, settings.category_model_path)
    save_model(priority_model, settings.priority_model_path)

    metadata = {
        "category_model": category_name,
        "priority_model": priority_name,
        "embedding_model": settings.embedding_model_name,
        "category_metrics": category_metrics.__dict__,
        "priority_metrics": priority_metrics.__dict__,
    }

    metadata_path = Path(settings.artifacts_dir) / "training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Training completed. Artifacts saved in %s", settings.artifacts_dir)


if __name__ == "__main__":
    main()
