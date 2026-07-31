
# Enterprise RAG Engine
### Production-Grade Multilingual Intelligence System for Documents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1E90FF?style=for-the-badge&logo=chromadb&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-1E1E1E?style=for-the-badge&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

<p align="center">
  <a href="https://nadjiba-rag-engine.streamlit.app"><img src="https://img.shields.io/badge/Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" /></a>
  <img src="https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge" />
</p>

A lightweight, production-ready Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex PDF documents with zero-hallucination constraints and full multi-language support.

---

## <img src="https://cdn.simpleicons.org/stackshare/2563EB" height="22" /> Overview

This repository provides an enterprise-ready pipeline for document intelligence. It processes raw PDF documents, chunks and embeds text locally, performs hybrid vector retrieval via ChromaDB, and generates context-grounded answers using Llama-3 (Groq API).

### <img src="https://cdn.simpleicons.org/features/2563EB" height="20" /> Core Capabilities

<table>
  <tr>
    <td><b>Hybrid Retrieval</b></td>
    <td>Combines vector similarity search with page-diversity filtering to prevent Table of Contents and reference leakage</td>
  </tr>
  <tr>
    <td><b>Fast LLM Inference</b></td>
    <td>Powered by Groq Llama-3-8B for sub-second response times</td>
  </tr>
  <tr>
    <td><b>Multilingual Support</b></td>
    <td>Handles document parsing and response generation in English, French, and Arabic</td>
  </tr>
  <tr>
    <td><b>RAG Evaluation Suite</b></td>
    <td>Includes a CLI tool to measure answer latency, groundedness, and context redundancy</td>
  </tr>
  <tr>
    <td><b>Containerized Deployment</b></td>
    <td>Fully configured with Docker and <code>docker-compose</code> for local and cloud environments</td>
  </tr>
  <tr>
    <td><b>Zero-Cost Stack</b></td>
    <td>Uses free local embeddings (<code>all-MiniLM-L6-v2</code>) and free inference endpoints</td>
  </tr>
</table>

---

## <img src="https://cdn.simpleicons.org/architecture/2563EB" height="22" /> System Architecture

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

## <img src="https://cdn.simpleicons.org/folder/2563EB" height="22" /> Repository Structure

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

## <img src="https://cdn.simpleicons.org/rocket/2563EB" height="22" /> Quickstart

### 1. Clone Repository & Configure Environment

```bash
git clone https://github.com/Nadjiba04/rag-enterprise-engine.git
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

<table>
  <tr>
    <td><b>Interactive API Docs</b></td>
    <td><code>http://localhost:8000/docs</code></td>
  </tr>
</table>

Launch the Streamlit frontend in a separate terminal:

```bash
streamlit run app/ui.py
```

<table>
  <tr>
    <td><b>Frontend Web App</b></td>
    <td><code>http://localhost:8501</code></td>
  </tr>
</table>

### 3. Docker Compose Setup <img src="https://cdn.simpleicons.org/docker/2496ED" height="16" />

To launch the full backend and UI stack inside isolated containers:

```bash
docker compose up --build
```

---

## <img src="https://cdn.simpleicons.org/swagger/2563EB" height="22" /> API Documentation

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

## <img src="https://cdn.simpleicons.org/benchmark/2563EB" height="22" /> Evaluation & Benchmarking

The project includes an automated evaluation tool to measure performance metrics locally:

```bash
python -m app.evaluate --language English
```

To export evaluation results to JSON:

```bash
python -m app.evaluate --language French --output evaluation-results.json
```

### Evaluated Metrics

<table>
  <tr>
    <th>Metric</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>E2E Latency</b></td>
    <td>Total response latency (seconds)</td>
  </tr>
  <tr>
    <td><b>Groundedness Score</b></td>
    <td>Adherence of LLM output to retrieved document context</td>
  </tr>
  <tr>
    <td><b>Duplicate Source Ratio</b></td>
    <td>Measure of chunk redundancy</td>
  </tr>
  <tr>
    <td><b>Context Leakage</b></td>
    <td>Rate of irrelevant sections (e.g., Table of Contents) filtered out</td>
  </tr>
</table>

---

## <img src="https://cdn.simpleicons.org/settings/2563EB" height="22" /> Configuration Variables

<table>
  <tr>
    <th>Variable</th>
    <th>Required</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>GROQ_API_KEY</code></td>
    <td><b>Yes</b></td>
    <td>None</td>
    <td>API Key for Groq Llama-3 inference</td>
  </tr>
  <tr>
    <td><code>GROQ_MODEL_NAME</code></td>
    <td>No</td>
    <td><code>llama-3.1-8b-instant</code></td>
    <td>Target LLM model</td>
  </tr>
  <tr>
    <td><code>EMBEDDING_MODEL_NAME</code></td>
    <td>No</td>
    <td><code>sentence-transformers/all-MiniLM-L6-v2</code></td>
    <td>Embedding model</td>
  </tr>
  <tr>
    <td><code>CHROMA_PERSIST_DIR</code></td>
    <td>No</td>
    <td><code>./data/chroma_db</code></td>
    <td>ChromaDB vector storage path</td>
  </tr>
  <tr>
    <td><code>CHUNK_SIZE</code></td>
    <td>No</td>
    <td><code>500</code></td>
    <td>Chunk size (characters)</td>
  </tr>
  <tr>
    <td><code>CHUNK_OVERLAP</code></td>
    <td>No</td>
    <td><code>50</code></td>
    <td>Chunk overlap (characters)</td>
  </tr>
</table>

---

## <img src="https://cdn.simpleicons.org/pytest/2563EB" height="22" /> Testing

Run unit and integration tests (LLM calls are mocked):

```bash
pytest
```

---

## <img src="https://cdn.simpleicons.org/license/2563EB" height="22" /> License

Distributed under the **MIT License**. See `LICENSE` for details.

---

<p align="center">
  <img src="https://img.shields.io/badge/Made_with-❤️-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Built_for-Production-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Open_Source-Yes-brightgreen?style=flat-square" />
</p>

<p align="center">
  <a href="https://github.com/Nadjiba04/rag-enterprise-engine">
    <img src="https://img.shields.io/badge/View_on-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://nadjiba-rag-engine.streamlit.app">
    <img src="https://img.shields.io/badge/Try_Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>
