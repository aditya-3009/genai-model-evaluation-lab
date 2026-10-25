"""Evaluate ranked retrieval predictions against labelled query cases."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from genai_eval.metrics import mean, recall_at_k, reciprocal_rank


def evaluate_retrieval(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute per-query metrics and aggregate means.

    Cases missing from predictions are treated as empty retrievals and remain
    visible in the report instead of being silently dropped.
    """
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    per_case: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        relevant = case["relevant_chunk_ids"]
        retrieved = list(predictions.get(case_id, []))
        row: dict[str, Any] = {
            "case_id": case_id,
            "retrieved_count": len(retrieved),
            "reciprocal_rank": reciprocal_rank(relevant, retrieved),
        }
        for k in ks:
            row[f"recall_at_{k}"] = recall_at_k(relevant, retrieved, k)
        per_case.append(row)

    summary: dict[str, Any] = {
        "case_count": len(per_case),
        "ks": list(ks),
        "mean_reciprocal_rank": mean([row["reciprocal_rank"] for row in per_case]),
        "metrics": {},
        "per_case": per_case,
        "notes": [
            "These scores apply only to the labelled dataset and provided predictions.",
            "Missing predictions are treated as empty results.",
            "A small sample dataset is illustrative and is not evidence of general model quality.",
        ],
    }
    for k in ks:
        summary["metrics"][f"recall_at_{k}"] = mean(
            [row[f"recall_at_{k}"] for row in per_case]
        )
    return summary
