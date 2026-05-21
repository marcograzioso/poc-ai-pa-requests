"""Evaluation helper script for trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.ml.classifier import compute_metrics, load_model
from app.ml.embeddings import EmbeddingService
from app.utils.config import settings


def main() -> None:
    """Evaluate persisted models on a dataset split."""
    parser = argparse.ArgumentParser(description="Evaluate category and priority models")
    parser.add_argument("--dataset", type=str, default=str(settings.dataset_path))
    parser.add_argument("--output", type=str, default=str(Path(settings.artifacts_dir) / "evaluation_report.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).dropna(subset=["citizen_request_text", "category", "priority"])
    embedder = EmbeddingService()
    X = embedder.encode(df["citizen_request_text"].astype(str).tolist(), is_query=False)

    category_model = load_model(settings.category_model_path)
    priority_model = load_model(settings.priority_model_path)

    category_preds = category_model.predict(X)
    priority_preds = priority_model.predict(X)

    report = {
        "category": compute_metrics(df["category"].to_numpy(), category_preds).__dict__,
        "priority": compute_metrics(df["priority"].to_numpy(), priority_preds).__dict__,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report written to {output_path}")


if __name__ == "__main__":
    main()
