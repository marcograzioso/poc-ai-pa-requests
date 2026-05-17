"""Business service orchestrating classification and RAG workflows."""

from __future__ import annotations

import json
from typing import Dict, Iterator

from app.ml.inference import ClassificationService
from app.rag.prompts import EXPLANATION_PROMPT
from app.rag.rag_pipeline import RagAssistant
from app.services.ollama_service import OllamaService
from app.utils.db import write_audit_log


class RequestOrchestrationService:
    """Unified service used by API and dashboard layers."""

    def __init__(self) -> None:
        self.classifier = None
        self.rag_assistant = None
        self.llm = None
        self.classifier_error = ""
        self.rag_error = ""
        self.llm_error = ""

        try:
            self.classifier = ClassificationService()
        except Exception as exc:
            self.classifier_error = str(exc)

        try:
            self.rag_assistant = RagAssistant()
        except Exception as exc:
            self.rag_error = str(exc)

        try:
            self.llm = OllamaService()
        except Exception as exc:
            self.llm_error = str(exc)

    def analyze_request(self, text: str) -> Dict[str, object]:
        """Predict category, office, and priority with confidence."""
        if self.classifier is None:
            raise RuntimeError(
                "Modelli classificazione non disponibili. Eseguire: python scripts/bootstrap_demo.py "
                f"(dettaglio: {self.classifier_error})"
            )

        predictions = self.classifier.predict(text)
        response = {
            "category": {
                "label": predictions["category"].label,
                "confidence": predictions["category"].confidence,
            },
            "office": {
                "label": predictions["office"].label,
                "confidence": predictions["office"].confidence,
            },
            "priority": {
                "label": predictions["priority"].label,
                "confidence": predictions["priority"].confidence,
            },
        }
        write_audit_log(
            "analysis",
            json.dumps(
                {
                    "text": text,
                    "result": response,
                    "model_used": "classification_pipeline",
                },
                ensure_ascii=False,
            ),
        )
        return response

    def generate_draft(self, text: str, top_k: int = 5, model: str | None = None) -> Dict[str, object]:
        """Create AI-assisted operator draft response from retrieved cases."""
        if self.rag_assistant is None:
            raise RuntimeError(
                "Pipeline RAG non disponibile. Eseguire: python scripts/bootstrap_demo.py "
                f"(dettaglio: {self.rag_error})"
            )

        selected_model = model or (self.llm.model if self.llm is not None else "unknown")
        result = self.rag_assistant.draft_response(new_request=text, top_k=top_k, model=selected_model)
        write_audit_log(
            "rag_draft",
            json.dumps({"text": text, "top_k": top_k, "model_used": selected_model}, ensure_ascii=False),
        )
        return result

    def generate_draft_stream(self, text: str, top_k: int = 5, model: str | None = None) -> Dict[str, object]:
        """Create streamed AI-assisted draft response from retrieved cases."""
        if self.rag_assistant is None:
            raise RuntimeError(
                "Pipeline RAG non disponibile. Eseguire: python scripts/bootstrap_demo.py "
                f"(dettaglio: {self.rag_error})"
            )

        selected_model = model or (self.llm.model if self.llm is not None else "unknown")
        retrieved_cases, stream = self.rag_assistant.draft_response_stream(
            new_request=text,
            top_k=top_k,
            model=selected_model,
        )

        def tracked_stream() -> Iterator[str]:
            chunks: list[str] = []
            for chunk in stream:
                chunks.append(chunk)
                yield chunk

            write_audit_log(
                "rag_draft_stream",
                json.dumps(
                    {
                        "text": text,
                        "top_k": top_k,
                        "model_used": selected_model,
                        "draft_len": len("".join(chunks)),
                    },
                    ensure_ascii=False,
                ),
            )

        return {
            "retrieved_cases": retrieved_cases,
            "stream": tracked_stream(),
            "model_used": selected_model,
        }

    def explain_prediction(self, text: str, model: str | None = None) -> str:
        """Generate human-readable explanation for model output."""
        if self.classifier is None:
            raise RuntimeError(
                "Modelli classificazione non disponibili. Eseguire: python scripts/bootstrap_demo.py "
                f"(dettaglio: {self.classifier_error})"
            )
        if self.llm is None:
            raise RuntimeError(f"Servizio LLM non disponibile: {self.llm_error}")

        prediction = self.classifier.predict(text)
        prompt = EXPLANATION_PROMPT.format(
            category=prediction["category"].label,
            priority=prediction["priority"].label,
            category_confidence=prediction["category"].confidence,
            priority_confidence=prediction["priority"].confidence,
        )
        selected_model = model or self.llm.model
        explanation = self.llm.generate(prompt, temperature=0.1, model=selected_model)
        write_audit_log(
            "explain",
            json.dumps(
                {
                    "text": text,
                    "explanation": explanation,
                    "model_used": selected_model,
                },
                ensure_ascii=False,
            ),
        )
        return explanation
