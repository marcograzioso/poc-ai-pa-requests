"""Streamlit dashboard for Public Administration AI platform demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load as joblib_load

from app.services.request_service import RequestOrchestrationService
from app.utils.config import settings
from app.utils.db import init_db, read_latest_audit_logs


st.set_page_config(page_title="PA AI Platform", page_icon="🏛️", layout="wide")
init_db()


@st.cache_resource
def get_orchestrator() -> RequestOrchestrationService:
    """Initialize orchestration services once per session."""
    return RequestOrchestrationService()


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load synthetic dataset for dashboard analytics."""
    if not Path(settings.dataset_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.dataset_path)
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def page_overview(df: pd.DataFrame) -> None:
    """Overview page with operational KPIs and distributions."""
    st.header("Overview")
    if df.empty:
        st.warning("Dataset non disponibile. Generare prima il dataset sintetico.")
        return

    resolved_df = df[df["status"].eq("resolved")].copy()
    resolution_hours = (resolved_df["resolved_at"] - resolved_df["created_at"]).dt.total_seconds() / 3600

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Richieste totali", f"{len(df):,}")
    c2.metric("Richieste risolte", f"{len(resolved_df):,}")
    c3.metric("Categorie", f"{df['category'].nunique()}")
    c4.metric("Tempo medio risoluzione (h)", f"{resolution_hours.mean():.1f}" if not resolution_hours.empty else "n/a")

    left, right = st.columns(2)
    with left:
        st.subheader("Distribuzione categorie")
        category_counts = df["category"].value_counts().rename_axis("category").reset_index(name="count")
        fig_cat = px.bar(category_counts, x="category", y="count", labels={"category": "Categoria", "count": "Volume"})
        st.plotly_chart(fig_cat, use_container_width=True)

    with right:
        st.subheader("Distribuzione priorita")
        fig_pr = px.pie(df, names="priority", hole=0.4)
        st.plotly_chart(fig_pr, use_container_width=True)


def page_live_analysis(service: RequestOrchestrationService) -> None:
    """Real-time prediction page for new requests."""
    st.header("Live Request Analysis")
    text = st.text_area("Inserisci richiesta cittadino", height=180, placeholder="Es. Vorrei rinnovare la carta d'identita...")
    if st.button("Analizza richiesta", type="primary"):
        if not text.strip():
            st.error("Inserire una richiesta valida.")
            return
        result = service.analyze_request(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Categoria", result["category"]["label"], f"conf: {result['category']['confidence']:.2f}")
        c2.metric("Ufficio", result["office"]["label"], f"conf: {result['office']['confidence']:.2f}")
        c3.metric("Priorita", result["priority"]["label"], f"conf: {result['priority']['confidence']:.2f}")


def page_rag_assistant(service: RequestOrchestrationService) -> None:
    """RAG page to inspect retrieved cases and generated draft response."""
    st.header("RAG Response Assistant")
    text = st.text_area("Nuova richiesta", height=180)
    top_k = st.slider("Numero casi simili", min_value=1, max_value=10, value=5)

    if "draft_text" not in st.session_state:
        st.session_state["draft_text"] = ""

    if st.button("Genera bozza"):
        if not text.strip():
            st.error("Inserire una richiesta valida.")
            return
        result = service.generate_draft_stream(text, top_k=top_k)

        st.subheader("Casi simili recuperati")
        if result["retrieved_cases"]:
            retrieved_df = pd.DataFrame(result["retrieved_cases"])
            st.dataframe(retrieved_df[["request_id", "category", "priority", "score", "citizen_request_text", "operator_response"]], use_container_width=True)

        st.subheader("Bozza generata (stream markdown)")
        stream_placeholder = st.empty()
        streamed_text = ""
        for chunk in result["stream"]:
            streamed_text += chunk
            stream_placeholder.markdown(streamed_text)

        st.session_state["draft_text"] = streamed_text

    st.subheader("Bozza di risposta (editable)")
    edited = st.text_area("Bozza", value=st.session_state.get("draft_text", ""), height=260)
    st.session_state["draft_text"] = edited
    st.caption("Bozza AI: richiede validazione umana prima dell'invio al cittadino.")

    if st.button("Rigenera bozza"):
        if text.strip():
            regen = service.generate_draft_stream(text, top_k=top_k)
            stream_placeholder = st.empty()
            streamed_text = ""
            for chunk in regen["stream"]:
                streamed_text += chunk
                stream_placeholder.markdown(streamed_text)
            st.session_state["draft_text"] = streamed_text
            st.rerun()


def page_explainability(df: pd.DataFrame) -> None:
    """Explainability page with confidence examples and model metadata."""
    st.header("Explainability")
    metadata_path = Path(settings.artifacts_dir) / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.subheader("Model metadata")
        st.json(metadata)
    else:
        st.info("Metadata modelli non trovati. Eseguire pipeline di training.")

    st.subheader("Confidence scores (testo di esempio)")
    sample_text = st.text_area(
        "Testo per spiegazione classificazione",
        value="Richiedo chiarimenti su avviso TARI ricevuto nonostante immobile non occupato.",
        height=120,
    )
    if st.button("Calcola confidenza"):
        try:
            service = get_orchestrator()
            result = service.analyze_request(sample_text)
            st.json(result)
        except Exception as exc:
            st.warning(f"Impossibile calcolare confidenza: {exc}")

    st.subheader("Feature importance (proxy su embedding dimensions)")
    try:
        model = joblib_load(settings.category_model_path)
        if hasattr(model, "coef_"):
            coef_importance = pd.Series(abs(model.coef_).mean(axis=0)).sort_values(ascending=False).head(15)
            fig_imp = px.bar(
                coef_importance.reset_index(name="importance").rename(columns={"index": "embedding_dim"}),
                x="embedding_dim",
                y="importance",
                labels={"embedding_dim": "Dimensione embedding", "importance": "Importanza media"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance disponibile solo per modelli lineari con coefficienti.")
    except Exception as exc:
        st.info(f"Feature importance non disponibile: {exc}")

    if not df.empty:
        st.subheader("Nearest similar requests (campione)")
        st.dataframe(df[["request_id", "category", "priority", "citizen_request_text"]].head(10), use_container_width=True)


def page_governance() -> None:
    """Governance and privacy controls page."""
    st.header("Governance & Privacy")
    st.markdown(
        """
### GDPR & Compliance
- Minimizzazione dati: dataset sintetico e anonimizzato per la demo.
- Conservazione controllata: tracciamento eventi in SQLite locale.
- Trasparenza: output AI etichettati come bozza e soggetti a validazione umana.

### Audit & Human-in-the-Loop
- Log eventi principali: classificazione, retrieval, generazione bozza.
- Operatore responsabile della validazione finale prima di qualsiasi invio.
- Possibilita di modifica manuale integrale della risposta.

### Limiti AI
- Classificazione probabilistica, non deterministica.
- Possibili errori su richieste ambigue o incomplete.
- Necessaria verifica operatore per conformita normativa.
        """
    )

    st.subheader("Audit logs (ultimi eventi)")
    try:
        logs = read_latest_audit_logs(limit=50)
        if logs:
            logs_df = pd.DataFrame(logs, columns=["id", "event_type", "payload", "created_at"])
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("Nessun audit log disponibile al momento.")
    except Exception as exc:
        st.warning(f"Impossibile leggere audit log: {exc}")


def main() -> None:
    """Main dashboard router."""
    service = get_orchestrator()
    df = load_data()

    st.sidebar.title("PA AI Platform")
    page = st.sidebar.radio(
        "Navigazione",
        [
            "Overview",
            "Live Request Analysis",
            "RAG Response Assistant",
            "Explainability",
            "Governance & Privacy",
        ],
    )

    if page == "Overview":
        page_overview(df)
    elif page == "Live Request Analysis":
        page_live_analysis(service)
    elif page == "RAG Response Assistant":
        page_rag_assistant(service)
    elif page == "Explainability":
        page_explainability(df)
    else:
        page_governance()


if __name__ == "__main__":
    main()
