"""Convenience script to launch Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "dashboard" / "streamlit_app.py")],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
