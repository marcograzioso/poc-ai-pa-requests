"""Wrapper for local Ollama API interactions."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

import requests

from app.utils.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OllamaService:
    """Simple synchronous client for Ollama generation endpoint."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def generate_stream(self, prompt: str, temperature: float = 0.1, model: str | None = None) -> Iterator[str]:
        """Stream text chunks from local LLM with defensive error handling."""
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = str(data.get("response", ""))
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            logger.exception("Ollama request failed")
            raise RuntimeError(f"Errore chiamata Ollama: {exc}") from exc
        except json.JSONDecodeError as exc:
            logger.exception("Ollama stream parse failed")
            raise RuntimeError(f"Errore parsing stream Ollama: {exc}") from exc

    def generate(self, prompt: str, temperature: float = 0.1, model: str | None = None) -> str:
        """Generate full text by joining streamed chunks."""
        return "".join(self.generate_stream(prompt=prompt, temperature=temperature, model=model)).strip()
