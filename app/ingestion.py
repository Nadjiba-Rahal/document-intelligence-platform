"""Safe, metadata-preserving document loading for supported file types."""

from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_documents(path: Path) -> list[Any]:
    """Load a document with a format-specific parser and stable source metadata."""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'.")

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        documents = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        documents = Docx2txtLoader(str(path)).load()
    elif suffix in {".txt", ".md", ".markdown"}:
        from langchain_community.document_loaders import TextLoader

        documents = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    elif suffix in {".html", ".htm"}:
        from langchain_community.document_loaders import BSHTMLLoader

        documents = BSHTMLLoader(str(path)).load()
    else:
        documents = _load_image(path)

    for document in documents:
        document.metadata.update({"source": path.name, "file_type": suffix.lstrip(".")})
    return documents


def _load_image(path: Path) -> list[Any]:
    """Extract image text when optional OCR dependencies and Tesseract are available."""

    try:
        import pytesseract
        from langchain_core.documents import Document
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as exc:
        raise RuntimeError(
            "Image OCR requires Pillow, pytesseract, and a Tesseract installation."
        ) from exc
    if not text:
        return []
    return [Document(page_content=text, metadata={"page": 0, "ocr": True})]