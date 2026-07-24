# Enterprise RAG Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

A lightweight, production-ready Retrieval-Augmented Generation system for PDF question answering. It combines FastAPI, Streamlit, ChromaDB, local HuggingFace embeddings, and Llama 3 through the Groq API.

## Architecture

```text
PDF Upload
    |
    v
PyPDFLoader
    |
    v
Recursive Text Chunks
    |
    v
HuggingFace all-MiniLM-L6-v2 Embeddings
    |
    v
Persistent Chroma Vector DB
    |
    v
Top-3 Similarity Retrieval
    |
    v
Strict Context Prompt + Groq Llama 3
    |
    v
Grounded Answer + Source Snippets
```

## Features

- PDF ingestion with chunking and persistent vector storage
- Local free embeddings through Sentence Transformers
- Groq-hosted Llama 3 answer generation
- Strict prompt grounding to reduce hallucinations
- FastAPI backend with upload, query, and health endpoints
- Streamlit dashboard for demos and portfolio screenshots
- Docker and Docker Compose support
- Pytest coverage for API behavior and validation

## Repository Structure

```text
rag-enterprise-engine/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- rag_engine.py
|   |-- ui.py
|   `-- config.py
|-- data/
|-- tests/
|   `-- test_rag.py
|-- .env.example
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- render.yaml
|-- LICENSE
`-- README.md
```

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/rag-enterprise-engine.git
cd rag-enterprise-engine
cp .env.example .env
```

Add your Groq key to `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 2. Run locally with Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the API docs at:

```text
http://localhost:8000/docs
```

Run the Streamlit UI in another terminal:

```bash
streamlit run app/ui.py
```

Open:

```text
http://localhost:8501
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Upload a PDF:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the key points in this document?\"}"
```

## Free Deployment

### Render

1. Push this repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Choose Docker as the runtime.
4. Add `GROQ_API_KEY` as an environment variable.
5. Deploy. The included `render.yaml` can be used as a blueprint.

### Hugging Face Spaces

1. Create a new Space and choose Docker.
2. Push this repository to the Space.
3. Add `GROQ_API_KEY` under Space secrets.
4. Use the Streamlit command for a UI-first demo:

```bash
streamlit run app/ui.py --server.address 0.0.0.0 --server.port 7860
```

For Hugging Face Spaces, update the Docker `CMD` to the command above or set the Space startup command accordingly.

## Environment Variables

| Variable | Required | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | None |
| `GROQ_MODEL_NAME` | No | `llama3-8b-8192` |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` |
| `CHUNK_SIZE` | No | `500` |
| `CHUNK_OVERLAP` | No | `50` |

## Testing

```bash
pytest
```

The tests mock the RAG engine at the API boundary, so they run without a Groq key.

## License

MIT
