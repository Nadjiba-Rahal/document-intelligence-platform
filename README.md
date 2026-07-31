
# Enterprise RAG Engine
### Production-Grade Multilingual Intelligence System for Documents
A lightweight, production-ready Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex PDF documents with zero-hallucination constraints and full multi-language support.

**Live Demo:** [https://nadjiba-rag-engine.streamlit.app](https://nadjiba-rag-engine.streamlit.app)

---

## Overview

This repository provides an enterprise-ready pipeline for document intelligence. It processes raw PDF documents, chunks and embeds text locally, performs hybrid vector retrieval via ChromaDB, and generates context-grounded answers using Llama-3 (Groq API).

### Core Capabilities
* **Hybrid Retrieval:** Combines vector similarity search with page-diversity filtering to prevent Table of Contents and reference leakage.
* **Fast LLM Inference:** Powered by Groq Llama-3-8B for sub-second response times.
* **Multilingual Support:** Handles document parsing and response generation in English, French, and Arabic.
* **RAG Evaluation Suite:** Includes a CLI tool to measure answer latency, groundedness, and context redundancy.
* **Containerized Deployment:** Fully configured with Docker and `docker-compose` for local and cloud environments.
* **Zero-Cost Stack:** Uses free local embeddings (`all-MiniLM-L6-v2`) and free inference endpoints.

---

## System Architecture

```text
[ User PDF Upload ]
       │
       ▼
[ PyPDFLoader & Text Splitter ]      (Chunk Size: 500 | Overlap: 50)
       │
       ▼
[ Sentence-Transformers Embeddings ] (all-MiniLM-L6-v2)
       │
       ▼
[ Persistent Chroma Vector DB ]      (Local Vector Storage)
       │
       ▼
[ Hybrid Search & Context Filter ]   (Top-3 Ranked Similarity)
       │
       ▼
[ Strict Prompt + Groq Llama 3 ]     (Zero-Hallucination Guardrails)
       │
       ▼
[ Grounded Answer + Citations ]      (English / French / Arabic)

```

---

## Repository Structure

```text
rag-enterprise-engine/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend & REST endpoints
│   ├── rag_engine.py    # Core RAG pipeline logic
│   ├── evaluate.py      # Automated RAG evaluation CLI
│   ├── ui.py            # Streamlit interactive dashboard
│   └── config.py        # Centralized settings & environment loader
├── data/                # Local ChromaDB vector storage
├── tests/
│   └── test_rag.py      # Pytest integration & unit tests
├── .env.example         # Environment template
├── Dockerfile           # Production Docker container setup
├── docker-compose.yml   # Multi-container orchestrator (API + UI)
├── requirements.txt     # Python dependencies
└── README.md            # System documentation

```

---

## Quickstart

### 1. Clone Repository & Configure Environment

```bash
git clone [https://github.com/Nadjiba04/rag-enterprise-engine.git](https://github.com/Nadjiba04/rag-enterprise-engine.git)
cd rag-enterprise-engine
cp .env.example .env

```

Add your Groq API Key to `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here

```

### 2. Local Python Setup

```bash
# Initialize virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and start FastAPI backend
pip install -r requirements.txt
uvicorn app.main:app --reload

```

* **Interactive API Docs:** `http://localhost:8000/docs`

Launch the Streamlit frontend in a separate terminal:

```bash
streamlit run app/ui.py

```

* **Frontend Web App:** `http://localhost:8501`

### 3. Docker Compose Setup (Recommended)

To launch the full backend and UI stack inside isolated containers:

```bash
docker compose up --build

```

---

## API Documentation

### Health Check

```bash
curl http://localhost:8000/health

```

### Upload & Index Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"

```

### Query Document Pipeline

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the core technical findings in this document?",
    "language": "English",
    "answer_style": "Executive"
  }'

```

---

## Evaluation & Benchmarking

The project includes an automated evaluation tool to measure performance metrics locally:

```bash
python -m app.evaluate --language English

```

To export evaluation results to JSON:

```bash
python -m app.evaluate --language French --output evaluation-results.json

```

### Evaluated Metrics

* **E2E Latency:** Total response latency (seconds).
* **Groundedness Score:** Adherence of LLM output to retrieved document context.
* **Duplicate Source Ratio:** Measure of chunk redundancy.
* **Context Leakage:** Rate of irrelevant sections (e.g., Table of Contents) filtered out.

---

## Configuration Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Yes | None | API Key for Groq Llama-3 inference |
| `GROQ_MODEL_NAME` | No | `llama-3.1-8b-instant` | Target LLM model |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | ChromaDB vector storage path |
| `CHUNK_SIZE` | No | `500` | Chunk size (characters) |
| `CHUNK_OVERLAP` | No | `50` | Chunk overlap (characters) |

---

## Testing

Run unit and integration tests (LLM calls are mocked):

```bash
pytest

```

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.

```

