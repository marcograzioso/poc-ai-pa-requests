"""Bootstrap all local artifacts for demo execution."""

from __future__ import annotations

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
    run("app.data.generate_synthetic_dataset")
    run("app.ml.train")
    run("app.ml.evaluate")
    run("app.rag.indexing")
    print("Bootstrap completato: dataset, modelli e indice FAISS pronti.")
