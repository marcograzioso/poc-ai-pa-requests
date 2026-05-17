# Public Administration AI Platform - PoC

Consulting-grade Proof of Concept for intelligent citizen request management in a municipality / PA office.

## Objectives

The PoC demonstrates end-to-end automation support for:

- request classification (category, office, priority)
- urgency prediction
- routing recommendation
- AI-assisted response drafting with human validation

The solution is designed for local execution with explainability, auditability, governance messaging, and GDPR-aware demo posture.

## Tech Stack

- Python 3.10+
- FastAPI backend
- Streamlit dashboard
- pandas, scikit-learn
- sentence-transformers + transformers
- FAISS vector search
- Ollama local LLM integration
- SQLite audit metadata

## Project Structure

```text
app/
  api/
  ml/
  rag/
  services/
  data/
  dashboard/
  models/
  utils/
data/
scripts/
requirements.txt
.env.example
README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment file:

```bash
cp .env.example .env
```

4. Ensure Ollama is running locally (optional for classification-only demo):

```bash
ollama serve
ollama pull llama3
```

## Bootstrap Demo Artifacts

Generate dataset, train models, evaluate, and build FAISS index:

```bash
python scripts/bootstrap_demo.py
```

Artifacts are saved under `app/models/artifacts/`.

## Run Backend API

```bash
python scripts/run_api.py
```

Swagger docs: `http://localhost:8000/docs`

### Main Endpoints

- `GET /health`
- `POST /analyze`
- `POST /rag/draft`
- `POST /explain`

Example `/analyze` payload:

```json
{
  "citizen_request_text": "Buongiorno, devo rinnovare la carta d'identita e vorrei sapere come prenotare appuntamento."
}
```

## Run Streamlit Dashboard

```bash
python scripts/run_dashboard.py
```

Dashboard pages:

1. Overview (KPI and distributions)
2. Live Request Analysis
3. RAG Response Assistant
4. Explainability
5. Governance & Privacy

## Synthetic Dataset

Generator script:

```bash
python -m app.data.generate_synthetic_dataset --rows 1500
```

Output fields:

- request_id
- citizen_request_text
- category
- office
- priority
- status
- operator_response
- created_at
- resolved_at

The generator injects:

- class imbalance
- noisy requests
- short and long texts
- duplicated requests
- ambiguous requests

## Explainability and Governance Notes

- Confidence scores are returned in prediction payloads.
- Training metrics and confusion matrix are saved in `training_metadata.json` and `evaluation_report.json`.
- All analysis/drafting actions are logged in SQLite (`audit_logs` table).
- Generated responses are explicitly drafts to be validated by operators.

## Prompt Templates

Prompt templates are stored in `app/rag/prompts.py`:

- response drafting
- summarization
- classification explanation

Prompts enforce:

- formal Italian PA tone
- context-grounded output
- hallucination minimization constraints

## Demo Positioning for Client Panel

This PoC is intentionally modular and local-first, suitable as an AI accelerator baseline to evolve into enterprise-grade PA deployment with:

- stronger security controls
- identity management
- richer monitoring and MLOps lifecycle
- policy and legal validation workflows
