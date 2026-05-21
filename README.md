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
- sentence-transformers + transformers + DistilBERT fine-tuning
- FAISS vector search
- Ollama local LLM integration
- SQLite audit metadata
- PyTorch with CUDA support for local training

For embedding-based retrieval the project supports E5-style models such as `intfloat/multilingual-e5-base`; when one of these is selected in `.env`, the code automatically applies the `query:` / `passage:` prefixes.

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

Optional:

```bash
python scripts/bootstrap_demo.py --skip-distilbert
```

Artifacts are saved under `app/models/artifacts/`.

The bootstrap now also trains the DistilBERT fine-tuning pipeline when available.

## Model Selection

The RAG assistant supports selecting the generation model from the dashboard or API.

Available options:

- `llama3`
- `qwen3.5:4b`
- `gemma3n:e2b`

The selected model is tracked in SQLite audit logs together with the request payload.

## Embeddings

Recommended default embedding model for Italian PA retrieval:

```bash
EMBEDDING_MODEL_NAME="intfloat/multilingual-e5-base"
```

When the selected model name contains `e5`, the code automatically prefixes queries with `query:` and corpus passages with `passage:`.

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
python -m app.data.generate_synthetic_dataset --rows 5000
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
- non-pertinent / spam requests

Each category includes multiple realistic Italian request variants, while responses are written as solution-oriented PA operator replies rather than generic confirmations.

## Classification Models

The project includes two classification approaches:

- sentence-transformers embeddings + scikit-learn classifiers
- DistilBERT fine-tuning for category and priority classification

DistilBERT training artifacts are stored under `app/models/artifacts/distilbert_category/` and `app/models/artifacts/distilbert_priority/`.

If you want to train only the lightweight pipeline, run:

```bash
python scripts/bootstrap_demo.py --skip-distilbert
```

## Explainability and Governance Notes

- Confidence scores are returned in prediction payloads.
- Training metrics and confusion matrix are saved in `training_metadata.json` and `evaluation_report.json`.
- Explainability page shows readable summaries for model metadata, confidence scores, and nearest similar requests retrieved semantically from FAISS.
- All analysis/drafting actions are logged in SQLite (`audit_logs` table).
- Generated responses are explicitly drafts to be validated by operators.

## Local GPU Setup

For the development machine with a GTX 1060, CUDA 12.6 is supported in this PoC.

Recommended PyTorch stack inside `demo-ntt`:

```bash
conda run -n demo-ntt python -m pip install --force-reinstall torch==2.11.0+cu126 torchvision==0.26.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Verify the installation with:

```bash
conda run -n demo-ntt python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If you are only running the demo inference paths, CPU execution is still supported.

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
