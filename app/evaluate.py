"""Terminal evaluation for the RAG system.

Run after indexing documents:
    python -m app.evaluate
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from app.rag_engine import DocumentRAGEngine


DEFAULT_QUESTIONS = [
    "What is the main research problem studied in this document?",
    "What are the main contributions of the paper?",
    "What datasets or experimental settings are mentioned?",
    "What are the hypotheses tested in this paper?",
    "What risks or limitations are mentioned?",
]


@dataclass
class EvaluationResult:
    question: str
    latency_seconds: float
    answer_length: int
    groundedness_score: float
    source_count: int
    unique_pages: int
    duplicate_page_ratio: float
    reference_chunk_ratio: float
    toc_chunk_ratio: float


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        if token
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "are",
            "was",
            "were",
            "into",
            "using",
            "paper",
            "document",
        }
    }


def groundedness(answer: str, sources: list[dict[str, Any]]) -> float:
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    source_tokens = tokenize(" ".join(source.get("content", "") for source in sources))
    return round(len(answer_tokens & source_tokens) / len(answer_tokens), 3)


def evaluate_question(engine: DocumentRAGEngine, question: str, language: str) -> EvaluationResult:
    started = time.perf_counter()
    result = engine.query(question, language=language, answer_style="Detailed")
    latency = time.perf_counter() - started
    metrics = result.get("retrieval_metrics", {})
    answer = result.get("answer", "")
    sources = result.get("source_documents", [])
    return EvaluationResult(
        question=question,
        latency_seconds=round(latency, 2),
        answer_length=len(answer),
        groundedness_score=groundedness(answer, sources),
        source_count=metrics.get("source_count", len(sources)),
        unique_pages=metrics.get("unique_pages", 0),
        duplicate_page_ratio=metrics.get("duplicate_page_ratio", 0.0),
        reference_chunk_ratio=metrics.get("reference_chunk_ratio", 0.0),
        toc_chunk_ratio=metrics.get("toc_chunk_ratio", 0.0),
    )


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return [str(item) for item in data["questions"]]
    raise ValueError("Evaluation file must be a JSON list or an object with a questions list.")


def print_table(results: list[EvaluationResult]) -> None:
    headers = [
        "question",
        "latency",
        "grounded",
        "sources",
        "pages",
        "dup",
        "refs",
        "toc",
    ]
    rows = [
        [
            result.question[:48],
            f"{result.latency_seconds:.2f}s",
            f"{result.groundedness_score:.2f}",
            str(result.source_count),
            str(result.unique_pages),
            f"{result.duplicate_page_ratio:.2f}",
            f"{result.reference_chunk_ratio:.2f}",
            f"{result.toc_chunk_ratio:.2f}",
        ]
        for result in results
    ]
    widths = [max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality.")
    parser.add_argument("--questions", help="Optional JSON file with evaluation questions.")
    parser.add_argument("--language", default="English", choices=["English", "French", "Arabic"])
    parser.add_argument("--output", help="Optional path to write JSON results.")
    args = parser.parse_args()

    engine = DocumentRAGEngine()
    questions = load_questions(args.questions)
    results = [evaluate_question(engine, question, args.language) for question in questions]

    print_table(results)
    print()
    print("Averages")
    print(f"- Latency: {mean(result.latency_seconds for result in results):.2f}s")
    print(f"- Groundedness: {mean(result.groundedness_score for result in results):.2f}")
    print(f"- Duplicate page ratio: {mean(result.duplicate_page_ratio for result in results):.2f}")
    print(f"- Reference chunk ratio: {mean(result.reference_chunk_ratio for result in results):.2f}")
    print(f"- TOC chunk ratio: {mean(result.toc_chunk_ratio for result in results):.2f}")

    if args.output:
        payload = [result.__dict__ for result in results]
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON evaluation to {args.output}")


if __name__ == "__main__":
    main()

