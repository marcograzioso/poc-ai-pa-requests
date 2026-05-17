"""Orchestration of retrieval-augmented response drafting."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterator, List, Tuple

from app.rag.prompts import RESPONSE_DRAFT_PROMPT
from app.rag.retriever import SemanticRetriever
from app.services.ollama_service import OllamaService


class RagAssistant:
    """Generate operator draft replies using retrieved historical context."""

    def __init__(self) -> None:
        self.retriever = SemanticRetriever()
        self.llm = OllamaService()

    @staticmethod
    def _format_retrieved_context(cases: List[Dict[str, str | float]]) -> str:
        chunks = []
        for i, case in enumerate(cases, start=1):
            chunks.append(
                f"Caso {i} | ID {case['request_id']} | categoria {case['category']} | priorita {case['priority']}\n"
                f"Richiesta: {case['citizen_request_text']}\n"
                f"Risposta storica: {case['operator_response']}"
            )
        return "\n\n".join(chunks)

    def _build_prompt_and_cases(self, new_request: str, top_k: int) -> Tuple[List[Dict[str, str | float]], str]:
        """Retrieve similar cases and build generation prompt."""
        similar_cases = self.retriever.search(new_request, top_k=top_k)
        prompt = RESPONSE_DRAFT_PROMPT.format(
            new_request=new_request,
            retrieved_context=self._format_retrieved_context(similar_cases),
        )
        return similar_cases, prompt

    def draft_response_stream(
        self,
        new_request: str,
        top_k: int = 5,
        model: str | None = None,
    ) -> Tuple[List[Dict[str, str | float]], Iterator[str]]:
        """Return retrieved cases and streamed draft chunks."""
        similar_cases, prompt = self._build_prompt_and_cases(new_request, top_k)
        stream = self.llm.generate_stream(prompt=prompt, temperature=0.1, model=model)
        return similar_cases, stream

    def draft_response(self, new_request: str, top_k: int = 5, model: str | None = None) -> Dict[str, object]:
        """Retrieve context and ask local LLM for a formal draft."""
        similar_cases, prompt = self._build_prompt_and_cases(new_request, top_k)
        selected_model = model or self.llm.model
        draft = self.llm.generate(prompt=prompt, temperature=0.1, model=selected_model)

        return {
            "draft_response": draft,
            "retrieved_cases": similar_cases,
            "generated_at": datetime.utcnow(),
            "model_used": selected_model,
        }
