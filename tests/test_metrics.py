import pytest

from genai_eval.metrics import recall_at_k, reciprocal_rank
from genai_eval.evaluate import evaluate_retrieval


def test_recall_at_k_counts_fraction_of_relevant_items_found():
    assert recall_at_k(["a", "b"], ["x", "b", "a"], 2) == 0.5


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["a", "b"], ["x", "b", "a"]) == 0.5
    assert reciprocal_rank(["a"], ["x", "y"]) == 0.0


def test_metrics_reject_nonpositive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)


def test_evaluation_keeps_cases_with_missing_predictions():
    report = evaluate_retrieval(
        [{"case_id": "case-1", "relevant_chunk_ids": ["gold"]}],
        {},
        ks=(1, 3),
    )
    assert report["case_count"] == 1
    assert report["metrics"]["recall_at_1"] == 0.0
    assert report["mean_reciprocal_rank"] == 0.0


def test_evaluation_aggregates_predictions():
    report = evaluate_retrieval(
        [
            {"case_id": "one", "relevant_chunk_ids": ["a"]},
            {"case_id": "two", "relevant_chunk_ids": ["b"]},
        ],
        {"one": ["a"], "two": ["x", "b"]},
        ks=(1, 2),
    )
    assert report["metrics"]["recall_at_1"] == 0.5
    assert report["metrics"]["recall_at_2"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75
