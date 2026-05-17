"""Build semantic index from historical resolved requests."""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from app.ml.embeddings import EmbeddingService
from app.rag.faiss_store import save_index
from app.utils.config import settings


def main() -> None:
    """Create FAISS index over historical requests and responses."""
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG retrieval")
    parser.add_argument("--dataset", type=str, default=str(settings.dataset_path))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[df["status"].astype(str).str.lower().eq("resolved")].copy()

    text_for_embedding = (
        df["citizen_request_text"].fillna("").astype(str)
        + "\nRISPOSTA STORICA:\n"
        + df["operator_response"].fillna("").astype(str)
    ).tolist()

    embedder = EmbeddingService()
    vectors = embedder.encode(text_for_embedding)

    metadata: List[Dict[str, str]] = []
    for row in df.to_dict(orient="records"):
        metadata.append(
            {
                "request_id": str(row["request_id"]),
                "citizen_request_text": str(row["citizen_request_text"]),
                "operator_response": str(row["operator_response"]),
                "category": str(row["category"]),
                "priority": str(row["priority"]),
                "office": str(row["office"]),
            }
        )

    save_index(vectors, metadata)
    print(f"Index built with {len(metadata)} records")


if __name__ == "__main__":
    main()
