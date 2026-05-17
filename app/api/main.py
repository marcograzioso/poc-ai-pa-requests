"""FastAPI backend for the Public Administration AI PoC."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.api.schemas import AnalysisResponse, AnalyzeRequest, DraftRequest, DraftResponse, ExplainRequest, ExplainResponse, LabelScore, SimilarCase
from app.services.request_service import RequestOrchestrationService
from app.utils.config import settings
from app.utils.db import init_db
from app.utils.logging_utils import configure_logging

configure_logging(settings.log_level)
init_db()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PoC AI platform for intelligent citizen request management",
)

orchestrator = RequestOrchestrationService()


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "environment": settings.app_env}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_request(payload: AnalyzeRequest) -> AnalysisResponse:
    """Classify citizen request and predict responsible office and priority."""
    try:
        result = orchestrator.analyze_request(payload.citizen_request_text)
        return AnalysisResponse(
            category=LabelScore(**result["category"]),
            office=LabelScore(**result["office"]),
            priority=LabelScore(**result["priority"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore analisi richiesta: {exc}") from exc


@app.post("/rag/draft", response_model=DraftResponse)
def generate_draft(payload: DraftRequest) -> DraftResponse:
    """Generate draft response using semantic retrieval and local LLM."""
    try:
        result = orchestrator.generate_draft(payload.citizen_request_text, top_k=payload.top_k)
        cases = [SimilarCase(**item) for item in result["retrieved_cases"]]
        return DraftResponse(
            draft_response=str(result["draft_response"]),
            retrieved_cases=cases,
            generated_at=result["generated_at"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore generazione bozza: {exc}") from exc


@app.post("/explain", response_model=ExplainResponse)
def explain_result(payload: ExplainRequest) -> ExplainResponse:
    """Generate explanation for classification output using local LLM."""
    try:
        explanation = orchestrator.explain_prediction(payload.citizen_request_text)
        return ExplainResponse(explanation=explanation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore generazione spiegazione: {exc}") from exc
