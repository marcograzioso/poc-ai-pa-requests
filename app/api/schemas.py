"""Pydantic schemas for API and service contracts."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Input payload for classification analysis."""

    citizen_request_text: str = Field(..., min_length=3)


class LabelScore(BaseModel):
    """Predicted label with confidence."""

    label: str
    confidence: float


class AnalysisResponse(BaseModel):
    """Classification output."""

    category: LabelScore
    office: LabelScore
    priority: LabelScore


class SimilarCase(BaseModel):
    """Retrieved case info returned by retriever."""

    request_id: str
    citizen_request_text: str
    operator_response: str
    category: str
    priority: str
    score: float


class DraftRequest(BaseModel):
    """Input payload for RAG draft generation."""

    citizen_request_text: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)
    model: Literal["llama3", "qwen3.5:4b", "gemma3n:e2b"] = "llama3"


class DraftResponse(BaseModel):
    """RAG generated draft with provenance."""

    draft_response: str
    retrieved_cases: List[SimilarCase]
    generated_at: datetime


class ExplainRequest(BaseModel):
    """Input for natural-language explanation generation."""

    citizen_request_text: str = Field(..., min_length=3)


class ExplainResponse(BaseModel):
    """Explanation output for classification results."""

    explanation: str
