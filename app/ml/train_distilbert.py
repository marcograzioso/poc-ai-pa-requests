"""Fine-tune DistilBERT models for category and priority classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.utils.config import settings
from app.utils.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute key metrics for trainer evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def fine_tune_task(
    df: pd.DataFrame,
    label_column: str,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    """Fine-tune DistilBERT for one task and save artifacts."""
    labels = sorted(df[label_column].astype(str).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_df, eval_df = train_test_split(
        df[["citizen_request_text", label_column]].copy(),
        test_size=0.2,
        random_state=42,
        stratify=df[label_column].astype(str),
    )

    train_df["label"] = train_df[label_column].map(label2id)
    eval_df["label"] = eval_df[label_column].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(batch["citizen_request_text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df[["citizen_request_text", "label"]], preserve_index=False).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(eval_df[["citizen_request_text", "label"]], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting DistilBERT fine-tuning for task: %s", label_column)
    trainer.train()
    metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_map_path = output_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "precision_macro": float(metrics.get("eval_precision_macro", 0.0)),
        "recall_macro": float(metrics.get("eval_recall_macro", 0.0)),
        "f1_macro": float(metrics.get("eval_f1_macro", 0.0)),
    }


def main() -> None:
    """Train DistilBERT models for both category and priority tasks."""
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT classifiers")
    parser.add_argument("--dataset", type=str, default=str(settings.dataset_path))
    parser.add_argument("--model-name", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).dropna(subset=["citizen_request_text", "category", "priority"]).copy()
    df["citizen_request_text"] = df["citizen_request_text"].astype(str)

    category_dir = Path(settings.artifacts_dir) / "distilbert_category"
    priority_dir = Path(settings.artifacts_dir) / "distilbert_priority"

    category_metrics = fine_tune_task(
        df=df,
        label_column="category",
        output_dir=category_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    priority_metrics = fine_tune_task(
        df=df,
        label_column="priority",
        output_dir=priority_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    metadata = {
        "base_model": args.model_name,
        "category_model_dir": str(category_dir),
        "priority_model_dir": str(priority_dir),
        "category_metrics": category_metrics,
        "priority_metrics": priority_metrics,
    }
    (Path(settings.artifacts_dir) / "distilbert_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("DistilBERT fine-tuning completed. Artifacts in %s", settings.artifacts_dir)


if __name__ == "__main__":
    main()
