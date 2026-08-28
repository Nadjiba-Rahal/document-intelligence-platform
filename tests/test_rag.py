"""Tests for API and RAG engine behavior."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_rag_engine
from app.evaluation import evaluate_retrieval
from app.rag_engine import DocumentRAGEngine


class FakeRAGEngine:
    def ingest_pdf(self, file_path: str) -> int:
        return 2

    def query(self, question: str, language: str = "English", answer_style: str = "Executive") -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        return {
            "answer": "This is a grounded test answer.",
            "source_documents": [
                {
                    "content": "Grounded source text.",
                    "metadata": {"source": "test.pdf", "page": 0},
                }
            ],
            "retrieval_metrics": {"source_count": 1, "unique_pages": 1},
        }


@pytest.fixture
def client():
    app.dependency_overrides[get_rag_engine] = lambda: FakeRAGEngine()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint_does_not_expose_api_key(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert "groq_configured" in response.json()
    assert "GROQ_API_KEY" not in response.text


def test_query_endpoint_returns_answer_and_sources(client):
    response = client.post("/query", json={"question": "What is tested?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "This is a grounded test answer."
    assert payload["source_documents"][0]["metadata"]["source"] == "test.pdf"
    assert payload["retrieval_metrics"]["source_count"] == 1


def test_query_endpoint_accepts_french_language(client):
    response = client.post(
        "/query",
        json={
            "question": "Quel est le sujet du document?",
            "language": "French",
            "answer_style": "Study Notes",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a grounded test answer."


def test_query_endpoint_rejects_empty_question(client):
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


def test_upload_rejects_non_pdf(client):
    response = client.post("/upload", content=b"not a multipart request")

    assert response.status_code == 400
    assert "multipart/form-data" in response.json()["detail"]


def test_engine_query_rejects_empty_question_without_external_services():
    engine = object.__new__(DocumentRAGEngine)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        engine.query(" ")


def test_retrieval_evaluation_metrics():
    metrics = evaluate_retrieval({"source-a"}, ["source-b", "source-a"], k=2)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.5
    assert metrics["mrr"] == 0.5


def test_reciprocal_rank_fusion_prefers_documents_seen_by_both_rankers():
    from langchain_core.documents import Document

    shared = Document(page_content="shared evidence")
    vector_only = Document(page_content="vector evidence")
    lexical_only = Document(page_content="lexical evidence")
    merged = DocumentRAGEngine._rrf_merge([shared, vector_only], [lexical_only, shared])

    assert merged[0].page_content == "shared evidence"


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/upload",
        files={"file": ("notes.exe", b"not supported", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
