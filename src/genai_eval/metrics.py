"""Metric functions for ranked retrieval results."""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(relevant_ids: Iterable[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Fraction of known relevant IDs retrieved in the top k.

    Returns 0.0 when the example has no labelled relevant IDs. Such cases
    should normally be filtered or reported separately in dataset analysis.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    found = relevant.intersection(retrieved_ids[:k])
    return len(found) / len(relevant)


def reciprocal_rank(relevant_ids: Iterable[str], retrieved_ids: Sequence[str]) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty collection."""
    return sum(values) / len(values) if values else 0.0
