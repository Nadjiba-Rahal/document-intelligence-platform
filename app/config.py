"""Application configuration loaded from environment variables or Streamlit secrets."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _get_groq_api_key() -> str | None:
    """Safely retrieve the Groq API key from Streamlit secrets or OS environment variables."""
    # 1. Try reading from Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # 2. Fallback to OS environment variables / .env
    return os.getenv("GROQ_API_KEY")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the RAG system."""

    groq_api_key: str | None = None
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    groq_model_name: str = "llama-3.1-8b-instant"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings(
        groq_api_key=_get_groq_api_key(),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        chunk_size=_get_int("CHUNK_SIZE", 500),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 50),
        groq_model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
    )