"""Application configuration utilities."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings for local PoC execution."""

    app_name: str = "PA AI Citizen Request Platform"
    app_env: str = "local"
    log_level: str = "INFO"

    dataset_path: Path = BASE_DIR / "data" / "synthetic_requests.csv"
    sqlite_path: Path = BASE_DIR / "data" / "pa_ai_metadata.db"

    artifacts_dir: Path = BASE_DIR / "app" / "models" / "artifacts"
    category_model_path: Path = artifacts_dir / "category_model.joblib"
    priority_model_path: Path = artifacts_dir / "priority_model.joblib"
    vector_index_path: Path = artifacts_dir / "requests.index"
    vector_meta_path: Path = artifacts_dir / "requests_meta.pkl"

    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    top_k_retrieval: int = 5

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8")


settings = Settings()
