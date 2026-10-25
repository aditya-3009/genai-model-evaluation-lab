import pytest

from genai_eval.dataset import load_dataset


def test_load_dataset_validates_rows_and_unique_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"a","query":"question","relevant_chunk_ids":["c1"]}\n',
        encoding="utf-8",
    )
    assert load_dataset(path)[0]["case_id"] == "a"


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"a","query":"q","relevant_chunk_ids":["c1"]}\n'
        '{"case_id":"a","query":"q2","relevant_chunk_ids":["c2"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_dataset(path)


def test_load_dataset_rejects_invalid_records(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"a","query":"","relevant_chunk_ids":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="query"):
        load_dataset(path)
