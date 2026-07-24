"""FastAPI entrypoint for the RAG service."""

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_engine import DocumentRAGEngine, MissingAPIKeyError, RAGEngineError

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Enterprise RAG Engine",
    description="Production-ready PDF Retrieval-Augmented Generation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is this document about?"])


class QueryResponse(BaseModel):
    answer: str
    source_documents: list[dict]


@lru_cache
def get_rag_engine() -> DocumentRAGEngine:
    return DocumentRAGEngine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(
    request: Request,
    engine: DocumentRAGEngine = Depends(get_rag_engine),
) -> dict[str, str | int]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload must be sent as multipart/form-data with a PDF file field.",
        )

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse upload form. Ensure python-multipart is installed.",
        ) from exc

    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing PDF file field named 'file'.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported.",
        )

    safe_name = f"{uuid4().hex}_{Path(file.filename).name}"
    destination = DATA_DIR / safe_name

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )
        destination.write_bytes(content)
        chunks = engine.ingest_pdf(str(destination))
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except HTTPException:
        raise
    except (ValueError, RAGEngineError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected upload processing error.",
        ) from exc

    return {
        "status": "success",
        "filename": file.filename,
        "stored_as": safe_name,
        "chunks_indexed": chunks,
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(
    payload: QueryRequest,
    engine: DocumentRAGEngine = Depends(get_rag_engine),
) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        return engine.query(question)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected query processing error.",
        ) from exc
