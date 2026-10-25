"""Command-line interface for retrieval evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genai_eval.dataset import load_dataset
from genai_eval.evaluate import evaluate_retrieval


def _load_predictions(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Predictions file must contain a JSON list")
    result: dict[str, list[str]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError(f"Prediction row {index} must contain a string case_id")
        retrieved = row.get("retrieved_chunk_ids")
        if not isinstance(retrieved, list) or not all(isinstance(item, str) for item in retrieved):
            raise ValueError(f"Prediction row {index}: retrieved_chunk_ids must be a list of strings")
        if row["case_id"] in result:
            raise ValueError(f"Duplicate prediction case_id: {row['case_id']}")
        result[row["case_id"]] = retrieved
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked retrieval predictions.")
    parser.add_argument("--dataset", required=True, help="Input labelled JSONL dataset")
    parser.add_argument("--predictions", required=True, help="Input JSON predictions")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5], help="Recall@k cutoffs")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    predictions = _load_predictions(args.predictions)
    report = evaluate_retrieval(cases, predictions, ks=args.ks)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
