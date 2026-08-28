"""Dependency-free retrieval evaluation helpers for regression datasets."""

import math
from typing import Iterable


def recall_at_k(relevant: set[str], retrieved: Iterable[str], k: int) -> float:
    found = len(relevant.intersection(list(retrieved)[:k]))
    return found / len(relevant) if relevant else 0.0


def precision_at_k(relevant: set[str], retrieved: Iterable[str], k: int) -> float:
    top = list(retrieved)[:k]
    return len(relevant.intersection(top)) / len(top) if top else 0.0


def reciprocal_rank(relevant: set[str], retrieved: Iterable[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(relevance: dict[str, float], retrieved: Iterable[str], k: int) -> float:
    values = [relevance.get(item, 0.0) for item in list(retrieved)[:k]]
    dcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(relevant: set[str], retrieved: list[str], k: int = 5) -> dict[str, float]:
    """Return standard metrics for one query and a ranked result list."""

    return {
        "recall_at_k": round(recall_at_k(relevant, retrieved, k), 4),
        "precision_at_k": round(precision_at_k(relevant, retrieved, k), 4),
        "mrr": round(reciprocal_rank(relevant, retrieved), 4),
        "hit_rate": float(bool(relevant.intersection(retrieved[:k]))),
    }