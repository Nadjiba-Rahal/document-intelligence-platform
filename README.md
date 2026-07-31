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
