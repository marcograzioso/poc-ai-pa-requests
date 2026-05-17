"""Convenience script to launch FastAPI app."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for uvicorn reload subprocesses.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[PROJECT_ROOT],
        app_dir=PROJECT_ROOT,
    )
