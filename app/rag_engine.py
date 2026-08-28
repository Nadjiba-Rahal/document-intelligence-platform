"""Core Retrieval-Augmented Generation engine."""

import hashlib
import shutil
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


SUPPORTED_LANGUAGES = {
    "English": "English",
    "French": "French",
    "Arabic": "Arabic",
}

ANALYSIS_MODES = {
    "Custom": "Answer the user's question directly.",
    "Executive Summary": "Prioritize the document's purpose, strongest findings, implications, and bottom line.",
    "Key Findings": "Extract the most important findings and support each one with evidence from the context.",
    "Methodology Audit": "Examine the data, methods, evaluation setup, and whether the evidence supports the claims.",
    "Risks and Limitations": "Identify explicit limitations, threats to validity, risks, and unanswered questions.",
    "Study Guide": "Create structured study notes with definitions, concepts, questions, and concise answers.",
}

STRICT_SYSTEM_PROMPT = (
    "Answer the user query ONLY using the provided context. "
    "If the answer is not contained in the context, politely state that you do not know. "
    "When the context contains enough evidence, give a useful, specific answer instead of a vague one."
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
        self._chroma_cls = Chroma
        self.vector_store = Chroma(
            collection_name="rag_documents",
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

    def clear_index(self) -> None:
        """Clear all indexed documents and rebuild an empty Chroma collection."""

        try:
            existing = self.vector_store.get()
            ids = existing.get("ids", [])
            if ids:
                self.vector_store.delete(ids=ids)
        except Exception:
            if self.settings.chroma_path.exists():
                shutil.rmtree(self.settings.chroma_path)
            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self.vector_store = self._chroma_cls(
                collection_name="rag_documents",
                persist_directory=str(self.settings.chroma_path),
                embedding_function=self.embeddings,
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
            chunk.metadata["document_id"] = self._file_hash(path)

        self.vector_store.add_documents(chunks)
        return len(chunks)

    def query(
        self,
        question: str,
        language: str = "English",
        answer_style: str = "Executive",
        analysis_mode: str = "Custom",
    ) -> dict[str, Any]:
        """Answer a question using the top three retrieved source chunks."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")
        response_language = SUPPORTED_LANGUAGES.get(language, "English")
        response_style = answer_style if answer_style in {"Executive", "Detailed", "Study Notes"} else "Executive"
        selected_mode = analysis_mode if analysis_mode in ANALYSIS_MODES else "Custom"

        from langchain_core.messages import HumanMessage, SystemMessage

        source_documents = self._retrieve_documents(clean_question)
        if not source_documents:
            return {
                "answer": (
                    "I do not know yet because no relevant document context was found. "
                    "Upload and index a PDF first, then ask a question about that document."
                ),
                "source_documents": [],
            }

        context = self._format_context(source_documents)
        messages = [
            SystemMessage(content=STRICT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"User query: {clean_question}\n\n"
                    f"Answer language: {response_language}\n"
                    f"Answer style: {response_style}\n\n"
                    f"Analysis mode: {selected_mode}\n"
                    f"Analysis instruction: {ANALYSIS_MODES[selected_mode]}\n\n"
                    "Answer with clear reasoning in simple language. "
                    "For summary, problem, contribution, limitation, or hypothesis questions, prefer bullet points. "
                    "Do not summarize the references section unless the user explicitly asks for references. "
                    "If Answer style is Executive, be concise and decision-oriented. "
                    "If Answer style is Detailed, include evidence and nuance. "
                    "If Answer style is Study Notes, explain as if preparing for an exam."
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            raise RAGEngineError(
                "Groq could not generate an answer. Check that GROQ_API_KEY is valid and that the model is available."
            ) from exc

        return {
            "answer": response.content,
            "source_documents": [self._serialize_document(doc) for doc in source_documents],
            "retrieval_metrics": self._retrieval_metrics(source_documents),
        }

    def _retrieve_documents(self, question: str) -> list[Any]:
        """Retrieve diverse, non-duplicate chunks using vector and keyword signals."""

        candidates: list[Any] = []
        expanded_query = self._expand_query(question)
        try:
            candidates.extend(self.vector_store.max_marginal_relevance_search(
                expanded_query,
                k=10,
                fetch_k=24,
                lambda_mult=0.35,
            ))
        except Exception:
            candidates.extend(self.vector_store.similarity_search(expanded_query, k=12))

        candidates.extend(self._keyword_candidates(question))

        candidates = self._deduplicate_documents(candidates)
        if not self._asks_for_references(question):
            non_reference_candidates = [doc for doc in candidates if not self._looks_like_references(doc)]
            if non_reference_candidates:
                candidates = non_reference_candidates

        if self._is_broad_question(question):
            useful_candidates = [doc for doc in candidates if not self._looks_like_table_of_contents(doc)]
            if useful_candidates:
                candidates = useful_candidates

        ranked = sorted(candidates, key=lambda doc: self._rank_document(question, doc), reverse=True)
        return self._diverse_pages(ranked, limit=6)

    def _keyword_candidates(self, question: str) -> list[Any]:
        """Find chunks that contain important task words missed by vector search."""

        try:
            collection = self.vector_store.get(include=["documents", "metadatas"])
        except Exception:
            return []

        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        keywords = self._query_keywords(question)
        if not keywords:
            return []

        from langchain_core.documents import Document

        matches = []
        for content, metadata in zip(documents, metadatas, strict=False):
            text = content.lower()
            score = sum(1 for keyword in keywords if keyword in text)
            if score:
                matches.append((score, Document(page_content=content, metadata=metadata or {})))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in matches[:12]]

    @staticmethod
    def _expand_query(question: str) -> str:
        text = question.lower()
        expansions = {
            "contribution": "contribution contributions novelty proposed approach findings paper contributes",
            "main idea": "abstract introduction conclusion key finding overall objective",
            "summarize": "abstract introduction conclusion key finding overall objective",
            "summary": "abstract introduction conclusion key finding overall objective",
            "problem": "problem challenge motivation issue limitation domain shift language shift objective",
            "hypothesis": "hypothesis hypotheses h1 h2 h3 h4 h5 h6 test expected",
            "dataset": "dataset datasets experimental setting evaluation data protocol",
            "limitation": "limitation limitations risks negative results honest limits",
            "risk": "limitation limitations risks negative results honest limits",
        }
        extra_terms = [terms for trigger, terms in expansions.items() if trigger in text]
        return f"{question} {' '.join(extra_terms)}".strip()

    @staticmethod
    def _query_keywords(question: str) -> list[str]:
        text = question.lower()
        keyword_groups = {
            "contribution": ["contribution", "contributions", "contribute", "proposed", "novel", "finding"],
            "main idea": ["abstract", "introduction", "conclusion", "objective", "study"],
            "summarize": ["abstract", "introduction", "conclusion", "objective", "study"],
            "summary": ["abstract", "introduction", "conclusion", "objective", "study"],
            "problem": ["problem", "challenge", "motivation", "domain shift", "language shift", "objective"],
            "hypothesis": ["hypothesis", "hypotheses", "h1", "h2", "h3", "h4", "h5", "h6"],
            "dataset": ["dataset", "nih", "vindr", "shenzhen", "montgomery", "experimental"],
            "limitation": ["limitation", "limitations", "risk", "risks", "negative results", "honest limits"],
            "risk": ["limitation", "limitations", "risk", "risks", "negative results", "honest limits"],
            "jepa": ["jepa"],
            "alb": ["alb"],
        }
        keywords: list[str] = []
        for trigger, group in keyword_groups.items():
            if trigger in text:
                keywords.extend(group)
        return keywords

    @staticmethod
    def _deduplicate_documents(documents: list[Any]) -> list[Any]:
        seen: set[str] = set()
        unique_documents = []
        for document in documents:
            key = " ".join(document.page_content.lower().split())[:500]
            if key in seen:
                continue
            seen.add(key)
            unique_documents.append(document)
        return unique_documents

    @staticmethod
    def _looks_like_table_of_contents(document: Any) -> bool:
        text = " ".join(document.page_content.lower().split())
        if "table of contents" in text:
            return True
        section_words = ["conclusion", "appendix", "risks", "contributions", "deliverable"]
        return sum(word in text for word in section_words) >= 3 and len(text) < 900

    @staticmethod
    def _looks_like_references(document: Any) -> bool:
        text = " ".join(document.page_content.lower().split())
        citation_words = ["arxiv", "proceedings", "conference", "journal", "preprint", "doi", "references"]
        return sum(word in text for word in citation_words) >= 3

    @staticmethod
    def _is_broad_question(question: str) -> bool:
        text = question.lower()
        broad_terms = ["main idea", "summary", "summarize", "problem", "contribution", "about", "whole"]
        return any(term in text for term in broad_terms)

    @staticmethod
    def _asks_for_references(question: str) -> bool:
        text = question.lower()
        return "reference" in text or "citation" in text or "related work" in text

    def _rank_document(self, question: str, document: Any) -> int:
        text = document.page_content.lower()
        score = 0

        for keyword in self._query_keywords(question):
            if keyword in text:
                score += 4

        page = document.metadata.get("page")
        if isinstance(page, int) and page <= 1 and self._is_broad_question(question):
            score += 2

        if self._looks_like_table_of_contents(document):
            score -= 8
        if self._looks_like_references(document) and not self._asks_for_references(question):
            score -= 10
        if len(text) > 350:
            score += 1
        return score

    @staticmethod
    def _diverse_pages(documents: list[Any], limit: int) -> list[Any]:
        selected = []
        page_counts: dict[Any, int] = {}
        for document in documents:
            page = document.metadata.get("page", "unknown")
            if page_counts.get(page, 0) >= 2:
                continue
            selected.append(document)
            page_counts[page] = page_counts.get(page, 0) + 1
            if len(selected) == limit:
                break
        return selected

    def _retrieval_metrics(self, documents: list[Any]) -> dict[str, Any]:
        if not documents:
            return {
                "source_count": 0,
                "unique_pages": 0,
                "duplicate_page_ratio": 0.0,
                "reference_chunk_ratio": 0.0,
                "toc_chunk_ratio": 0.0,
            }

        pages = [doc.metadata.get("page", "unknown") for doc in documents]
        unique_pages = len(set(pages))
        total = len(documents)
        return {
            "source_count": total,
            "unique_pages": unique_pages,
            "duplicate_page_ratio": round(1 - (unique_pages / total), 3),
            "reference_chunk_ratio": round(
                sum(1 for doc in documents if self._looks_like_references(doc)) / total,
                3,
            ),
            "toc_chunk_ratio": round(
                sum(1 for doc in documents if self._looks_like_table_of_contents(doc)) / total,
                3,
            ),
        }

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

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
