"""Streamlit dashboard for the RAG system."""

import hashlib
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def uploaded_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


st.set_page_config(page_title="Enterprise RAG Engine", page_icon="📄", layout="wide")
st.title("Enterprise RAG Engine")
st.caption("Multilingual, evaluation-ready PDF intelligence with grounded answers and inspectable evidence.")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if api_key_input and api_key_input != os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = api_key_input
        reset_engine_cache()
        st.success("API key loaded for this session.")

    language = st.selectbox("Answer language", ["English", "French", "Arabic"])
    answer_style = st.segmented_control(
        "Answer style",
        ["Executive", "Detailed", "Study Notes"],
        default="Executive",
    )
    show_metrics = st.toggle("Show retrieval metrics", value=True)

    st.divider()
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Clear indexed documents", use_container_width=True):
        try:
            with st.spinner("Clearing local vector database..."):
                get_engine().clear_index()
            st.session_state.messages = []
            st.success("Index cleared. Upload and index one PDF again.")
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not clear index: {exc}")

    if uploaded_file and st.button("Index document", use_container_width=True):
        content = uploaded_file.getbuffer().tobytes()
        file_id = uploaded_file_hash(content)[:16]
        destination = DATA_DIR / f"{file_id}_{Path(uploaded_file.name).name}"
        destination.write_bytes(content)
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
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"Inspect evidence ({len(sources)} sources)"):
                for index, source in enumerate(sources, start=1):
                    metadata = source.get("metadata", {})
                    st.markdown(
                        f"**Source {index}:** {metadata.get('source', 'unknown')} "
                        f"page {metadata.get('page', 'unknown')}"
                    )
                    st.write(source.get("content", ""))
        metrics = message.get("metrics")
        if metrics and show_metrics:
            st.caption(
                "Retrieval quality: "
                f"{metrics.get('source_count', 0)} chunks, "
                f"{metrics.get('unique_pages', 0)} unique pages, "
                f"{metrics.get('reference_chunk_ratio', 0)} reference ratio, "
                f"{metrics.get('toc_chunk_ratio', 0)} TOC ratio"
            )

question = st.chat_input("Ask a question about your indexed PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching sources and asking Llama 3..."):
                result = get_engine().query(
                    question,
                    language=language,
                    answer_style=answer_style,
                )
            st.markdown(result["answer"])
            sources = result["source_documents"]
            with st.expander(f"Inspect evidence ({len(sources)} sources)"):
                for index, source in enumerate(sources, start=1):
                    metadata = source.get("metadata", {})
                    st.markdown(
                        f"**Source {index}:** {metadata.get('source', 'unknown')} "
                        f"page {metadata.get('page', 'unknown')}"
                    )
                    st.write(source.get("content", ""))
            metrics = result.get("retrieval_metrics", {})
            if show_metrics and metrics:
                st.caption(
                    "Retrieval quality: "
                    f"{metrics.get('source_count', 0)} chunks, "
                    f"{metrics.get('unique_pages', 0)} unique pages, "
                    f"{metrics.get('reference_chunk_ratio', 0)} reference ratio, "
                    f"{metrics.get('toc_chunk_ratio', 0)} TOC ratio"
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "metrics": metrics,
                }
            )
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except (ValueError, RAGEngineError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected query error: {exc}")
