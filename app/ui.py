"""Streamlit dashboard for the RAG system."""

import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.config import get_settings
from app.rag_engine import DocumentRAGEngine, MissingAPIKeyError, RAGEngineError

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def reset_engine_cache() -> None:
    get_settings.cache_clear()
    get_engine.clear()


@st.cache_resource(show_spinner=False)
def get_engine() -> DocumentRAGEngine:
    return DocumentRAGEngine()


st.set_page_config(page_title="Enterprise RAG Engine", page_icon="📄", layout="wide")
st.title("Enterprise RAG Engine")
st.caption("Upload PDFs, index them locally with Chroma, and ask grounded questions with Llama 3 via Groq.")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if api_key_input and api_key_input != os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = api_key_input
        reset_engine_cache()
        st.success("API key loaded for this session.")

    st.divider()
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file and st.button("Index document", use_container_width=True):
        destination = DATA_DIR / f"{uuid4().hex}_{uploaded_file.name}"
        destination.write_bytes(uploaded_file.getbuffer())
        try:
            with st.spinner("Indexing document..."):
                chunks = get_engine().ingest_pdf(str(destination))
            st.success(f"Indexed {chunks} chunks from {uploaded_file.name}.")
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except (ValueError, RAGEngineError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Unexpected ingestion error. Check the PDF and try again.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for index, source in enumerate(message.get("sources", []), start=1):
            metadata = source.get("metadata", {})
            label = f"Source {index}: {metadata.get('source', 'unknown')} page {metadata.get('page', 'unknown')}"
            with st.expander(label):
                st.write(source.get("content", ""))

question = st.chat_input("Ask a question about your indexed PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching sources and asking Llama 3..."):
                result = get_engine().query(question)
            st.markdown(result["answer"])
            for index, source in enumerate(result["source_documents"], start=1):
                metadata = source.get("metadata", {})
                label = f"Source {index}: {metadata.get('source', 'unknown')} page {metadata.get('page', 'unknown')}"
                with st.expander(label):
                    st.write(source.get("content", ""))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"],
                }
            )
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Unexpected query error. Confirm a PDF has been indexed and try again.")

