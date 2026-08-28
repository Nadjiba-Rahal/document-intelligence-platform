
<div align="center">

# Enterprise RAG Engine
### Production-Grade Multilingual Intelligence System for Documents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1E90FF?style=for-the-badge&logo=google-cloud&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq_Llama_3-1E1E1E?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

<p align="center">
  <a href="https://nadjiba-rag-engine.streamlit.app"><img src="https://img.shields.io/badge/🚀_Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" /></a>
  <img src="https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  A lightweight, production-ready Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex PDF documents with <b>zero-hallucination constraints</b> and <b>full multi-language support</b>.
</p>

</div>

---

## <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="22" height="22" /> Overview

This repository provides an enterprise-ready pipeline for document intelligence. It processes PDF, DOCX, text, Markdown, HTML, and OCR-capable image documents, chunks and embeds text locally, fuses lexical and vector retrieval, and generates context-grounded answers using Groq.

### Core Capabilities

<table>
  <thead>
    <tr>
      <th width="30%">Capability</th>
      <th width="70%">Technical Realization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Hybrid Retrieval</b></td>
      <td>Combines Chroma vector retrieval with lexical BM25-style ranking using reciprocal-rank fusion, then applies page-diversity and reference filtering.</td>
    </tr>
    <tr>
      <td><b>Fast LLM Inference</b></td>
      <td>Powered by a configurable Groq model, currently <code>openai/gpt-oss-20b</code>.</td>
    </tr>
    <tr>
      <td><b>Multilingual Support</b></td>
      <td>Handles document parsing and response generation in <b>English</b>, <b>French</b>, and <b>Arabic</b>.</td>
    </tr>
    <tr>
      <td><b>RAG Evaluation Suite</b></td>
      <td>Provides Recall@K, Precision@K, MRR, Hit Rate, and nDCG helpers for retrieval regression datasets.</td>
    </tr>
    <tr>
      <td><b>Containerized Deployment</b></td>
      <td>Fully configured with Docker and <code>docker-compose</code> for local and cloud environments.</td>
    </tr>
    <tr>
      <td><b>Zero-Cost Stack</b></td>
      <td>Uses free local embeddings (<code>all-MiniLM-L6-v2</code>) and free inference endpoints.</td>
    </tr>
  </tbody>
</table>

### Production additions

- Supported ingestion: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, and OCR-capable image files.
- Evidence IDs such as `[E1]` are assigned by the application and passed to the model; the model is instructed never to invent citations or follow document instructions.
- `GET /ready` reports whether Groq configuration is present without exposing the API key.
- Uploads are extension-validated and bounded by `MAX_UPLOAD_MB` (default: 50).
- The API remains backward-compatible: `POST /upload` and `POST /query` retain their existing routes and defaults.

---

## <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" width="22" height="22" /> System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DOCUMENT INGESTION STAGE                                  │
│   [ User PDF ] ──> [ PyPDFLoader ] ──> [ Recursive Character Split (Chunk: 500, Overlap: 50) ] │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                VECTOR EMBEDDING & INDEXING                                  │
│   [ sentence-transformers/all-MiniLM-L6-v2 ] ──> [ Local Persistent Chroma Vector DB ]        │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               RETRIEVAL & GENERATION PIPELINE                               │
│   [ Query ] ──> [ Hybrid Search & Page Diversity Filter ] ──> [ Top-3 Context Ranking ]     │
│                                                                          │                  │
│   [ Grounded Response + Citations ] <── [ Groq Llama-3 API ] <── [ Zero-Hallucination Prompt ] │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

```

---

##  Repository Structure

```text
rag-enterprise-engine/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend & REST endpoints
│   ├── rag_engine.py    # Core RAG pipeline logic
│   ├── evaluate.py      # Automated RAG evaluation CLI
│   ├── ui.py            # Streamlit interactive dashboard
│   └── config.py        # Centralized settings & environment loader
├── data/                # Local ChromaDB vector storage persistence
├── tests/
│   └── test_rag.py      # Pytest integration & unit tests
├── .env.example         # Environment variables template
├── Dockerfile           # Production Docker container definition
├── docker-compose.yml   # Multi-container service setup (API + UI)
├── requirements.txt     # Python dependencies
└── README.md            # Technical documentation

```

---

##  Quickstart Guide

### 1. Clone Repository & Environment Setup

```bash
git clone [https://github.com/Nadjiba04/rag-enterprise-engine.git](https://github.com/Nadjiba04/rag-enterprise-engine.git)
cd rag-enterprise-engine
cp .env.example .env

```

Add your Groq API Key to `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here

```

---

### 2. Local Execution

```bash
# Initialize virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and start FastAPI backend
pip install -r requirements.txt
uvicorn app.main:app --reload

```

Launch the Streamlit interface in a separate terminal:

```bash
streamlit run app/ui.py

```

---

### 3. Docker Containerization 

Spin up both the FastAPI backend and Streamlit dashboard instantly:

```bash
docker compose up --build

```

---

##  API Endpoint Reference

### Health Verification

```bash
curl http://localhost:8000/health

```

### Document Upload & Vector Indexing

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"

```

### Query Context Pipeline

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

##  Benchmarking & Evaluation

An automated test harness evaluates performance across latency, context quality, and adherence:

```bash
# Run CLI benchmarking tool
python -m app.evaluate --language English

# Export metric outputs to JSON
python -m app.evaluate --language French --output evaluation-results.json

```

### Metrics Tracked

---

##  System Configuration Variables

---

## Testing

Execute unit and integration test suites with mocked LLM calls:

```bash
pytest

```

---

## License

Distributed under the **MIT License**. See `LICENSE` for further details.
