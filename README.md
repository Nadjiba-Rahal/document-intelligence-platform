```markdown
<div align="center">

# 🚀 Enterprise RAG Engine
### Production-Grade Multilingual Intelligence System for Documents

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Llama-3](https://img.shields.io/badge/LLM-Llama--3--8B-orange?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A lightweight, production-ready Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex PDF documents with zero-hallucination constraints and full multi-language support.*

[Live Demo](#-free-deployment) • [API Specs](#-api-usage) • [Evaluation Suite](#-evaluation--benchmarking)

---

</div>

## 💡 System Architecture

```text
       [ User Upload: PDF File ]
                   │
                   ▼
       [ PyPDFLoader & Text Splitter ]  ◄── (Chunk Size: 500 | Overlap: 50)
                   │
                   ▼
 [ Sentence-Transformers Embeddings ]   ◄── (all-MiniLM-L6-v2)
                   │
                   ▼
    [ Persistent Chroma Vector DB ]     ◄── (Local Persistent Storage)
                   │
                   ▼
  [ Hybrid Search & Context Filtering ] ◄── (Ranked Top-3 Similarity)
                   │
                   ▼
   [ Strict Prompt + Groq Llama 3 ]    ◄── (Zero-Hallucination Constraints)
                   │
                   ▼
 [ Grounded Answer + Source Citations ] ◄── (English / French / Arabic)

```

---

## ✨ Key Features & Highlights

* 🔍 **Hybrid Retrieval Engine:** Combines vector similarity search, keyword filtering, and page-diversity metrics to prevent table-of-contents leakage.
* ⚡ **Ultra-Fast LLM Inference:** Powered by **Groq Llama-3-8B**, delivering answers in sub-seconds.
* 🌐 **Native Multilingual Support:** Formulates contextual answers fluently in **English, French, and Arabic** based on user preference.
* 📊 **Built-In RAG Evaluation Suite:** Includes an evaluation CLI tool measuring latency, groundedness, reference leaks, and source redundancy.
* 🐳 **Cloud & Container Ready:** Completely Dockerized with `docker-compose` support for seamless local and production deployments.
* 🛡️ **Zero-Cost Stack:** Uses free local embeddings (`all-MiniLM-L6-v2`) and free inference endpoints, avoiding expensive API dependencies.

---

## 📁 Repository Structure

```text
rag-enterprise-engine/
│-- app/
│   ├── __init__.py
│   ├── main.py            # FastAPI REST backend & endpoints
│   ├── rag_engine.py      # Core RAG Pipeline logic
│   ├── evaluate.py        # Automated RAG evaluation suite
│   ├── ui.py              # Streamlit Interactive Dashboard
│   └── config.py          # Centralized Configuration & Envs
│-- data/                  # Persistent Chroma Vector storage
│-- tests/
│   └── test_rag.py        # Unit & Integration tests with pytest
│-- .env.example           # Environment template
│-- Dockerfile             # Multi-stage optimized Dockerfile
│-- docker-compose.yml     # Multi-container orchestrator (API + UI)
│-- render.yaml            # Render Blueprint deployment config
│-- requirements.txt       # Dependencies
└── README.md              # Project Documentation

```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment

```bash
git clone [https://github.com/YOUR_USERNAME/rag-enterprise-engine.git](https://github.com/YOUR_USERNAME/rag-enterprise-engine.git)
cd rag-enterprise-engine
cp .env.example .env

```

Set your **Groq API Key** in `.env`:

```env
GROQ_API_KEY=gsk_your_key_here

```

### 2. Run Locally (Python)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

# Install dependencies & launch FastAPI backend
pip install -r requirements.txt
uvicorn app.main:app --reload

```

* 🔗 **Interactive Swagger API Docs:** `http://localhost:8000/docs`

Launch the **Streamlit UI** in a second terminal:

```bash
streamlit run app/ui.py

```

* 🔗 **Frontend Web App:** `http://localhost:8501`

---

### 3. Run with Docker Compose (Recommended)

Run the entire stack with a single command:

```bash
docker compose up --build

```

---

## 🔌 API Usage

### Health Check

```bash
curl http://localhost:8000/health

```

### Upload & Index Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"

```

### Query Contextual Pipeline

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the core engineering takeaways in this document?",
    "language": "English",
    "answer_style": "Executive"
  }'

```

---

## 🌍 Multilingual Answers

The application native context extraction supports multi-language responses:

* **English**
* **French**
* **Arabic**

Select the response language directly in the Streamlit UI sidebar, or send the `language` key via the API payload:

```json
{
  "question": "Résume ce document en trois points.",
  "language": "French",
  "answer_style": "Study Notes"
}

```

---

## 📈 Evaluation & Benchmarking

Evaluate retrieval precision, answer latency, and source quality locally:

```bash
python -m app.evaluate --language English

```

Optional export to JSON:

```bash
python -m app.evaluate --language French --output evaluation-results.json

```

**Metrics Measured:**

* ⏱️ **E2E Latency:** Measures end-to-end processing time in seconds.
* 🎯 **Groundedness Proxy Score:** Evaluates model adherence to retrieved context.
* 📑 **Duplicate Source Ratio:** Identifies chunk redundancy.
* 🛡️ **Leakage Detection:** Filters out Table-of-Contents and References pages.

---

## 🌐 Free Deployment

### Render Deployment

1. Push repository to GitHub.
2. Create a new **Web Service** on Render from the repository.
3. Select **Docker** as the Runtime.
4. Add `GROQ_API_KEY` under Environment Variables.
5. Deploy using the included `render.yaml` blueprint.

### Hugging Face Spaces

1. Create a new Space on Hugging Face Spaces selecting **Docker**.
2. Push your repo and add `GROQ_API_KEY` under Space Secrets.
3. Launch Streamlit UI via the Space configuration:

```bash
streamlit run app/ui.py --server.address 0.0.0.0 --server.port 7860

```

---

## ⚙️ Environment Variables

| Variable | Required | Default Value | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | **Yes** | *None* | Groq API Key for Llama-3 inference |
| `GROQ_MODEL_NAME` | No | `llama3-8b-8192` | Target LLM model |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Persistent Vector Store path |
| `CHUNK_SIZE` | No | `500` | Document chunk character limit |
| `CHUNK_OVERLAP` | No | `50` | Text split overlap size |

---

## 🧪 Testing

Execute automated unit tests (which mock LLM endpoints to run without an API key):

```bash
pytest

```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

```

```