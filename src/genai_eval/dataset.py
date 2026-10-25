"""Read and validate a versionable JSONL evaluation dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL cases and validate required fields and unique case IDs."""
    source = Path(path)
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: each row must be a JSON object")
            case_id = row.get("case_id")
            query = row.get("query")
            relevant = row.get("relevant_chunk_ids")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{source}:{line_number}: case_id must be a non-empty string")
            if case_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate case_id {case_id!r}")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{source}:{line_number}: query must be a non-empty string")
            if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
                raise ValueError(f"{source}:{line_number}: relevant_chunk_ids must be a list of strings")
            cases.append(row)
            seen_ids.add(case_id)
    return cases
