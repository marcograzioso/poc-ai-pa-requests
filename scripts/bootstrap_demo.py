"""Bootstrap all local artifacts for demo execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run(module_path: str, *args: str) -> None:
    """Run module as subprocess and stop on first failure."""
    cmd = [sys.executable, "-m", module_path, *args]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap local PA AI demo artifacts")
    parser.add_argument("--skip-distilbert", action="store_true", help="Skip DistilBERT fine-tuning step")
    # add argument to skip all trainings
    parser.add_argument("--skip-training", action="store_true", help="Skip all model training steps")
    args = parser.parse_args()

    # run("app.data.generate_synthetic_dataset")
    if args.skip_training:
        print("Skipping training steps as per argument. Make sure to have trained models for the demo to work.")
    if not args.skip_training:
        run("app.data.generate_synthetic_dataset")
        run("app.ml.train")
        if not args.skip_distilbert:
            run("app.ml.train_distilbert")
    run("app.ml.evaluate")
    run("app.rag.indexing")
    print("Bootstrap completato: dataset, modelli e indice FAISS pronti.")
