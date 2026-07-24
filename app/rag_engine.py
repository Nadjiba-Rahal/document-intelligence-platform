"""Core Retrieval-Augmented Generation engine."""

from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


STRICT_SYSTEM_PROMPT = (
    "Answer the user query ONLY using the provided context. "
    "If the answer is not contained in the context, politely state that you do not know."
)


class RAGEngineError(RuntimeError):
    """Raised when document ingestion or retrieval fails."""


class MissingAPIKeyError(RAGEngineError):
    """Raised when the Groq API key is required but missing."""


class DocumentRAGEngine:
    """PDF ingestion and question-answering engine backed by Chroma and Groq."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is required. Add it to .env or provide it in the Streamlit sidebar."
            )

        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        from langchain_chroma import Chroma
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model_name)
        self.vector_store = Chroma(
            persist_directory=str(self.settings.chroma_path),
            embedding_function=self.embeddings,
        )
        self.llm = ChatGroq(
            api_key=self.settings.groq_api_key,
            model=self.settings.groq_model_name,
            temperature=0,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def ingest_pdf(self, file_path: str) -> int:
        """Load a PDF, split it into chunks, and persist chunks to Chroma."""

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        try:
            from langchain_community.document_loaders import PyPDFLoader

            documents = PyPDFLoader(str(path)).load()
        except Exception as exc:
            raise RAGEngineError(f"Could not read PDF '{path.name}'. The file may be corrupt.") from exc

        if not documents:
            raise RAGEngineError("The PDF did not contain readable pages.")

        chunks = self.text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise RAGEngineError("The PDF did not contain extractable text.")

        for chunk in chunks:
            chunk.metadata["source"] = path.name

        self.vector_store.add_documents(chunks)
        return len(chunks)

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question using the top three retrieved source chunks."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        from langchain_core.messages import HumanMessage, SystemMessage

        source_documents = self.vector_store.similarity_search(clean_question, k=3)
        context = self._format_context(source_documents)
        messages = [
            SystemMessage(content=STRICT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"User query: {clean_question}\n\n"
                    "Answer with a concise, factual response."
                )
            ),
        ]
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "source_documents": [self._serialize_document(doc) for doc in source_documents],
        }

    @staticmethod
    def _format_context(documents: list[Any]) -> str:
        if not documents:
            return "No relevant context was retrieved."
        return "\n\n".join(
            f"Source {index} ({doc.metadata.get('source', 'unknown')}, "
            f"page {doc.metadata.get('page', 'unknown')}):\n{doc.page_content}"
            for index, doc in enumerate(documents, start=1)
        )

    @staticmethod
    def _serialize_document(document: Any) -> dict[str, Any]:
        return {
            "content": document.page_content,
            "metadata": document.metadata,
        }
