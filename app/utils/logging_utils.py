"""Logging configuration for the PoC."""

from __future__ import annotations

import logging


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logger with a concise structured format."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return module logger."""
    return logging.getLogger(name)
