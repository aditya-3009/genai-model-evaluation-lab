# GenAI Model Evaluation Lab Learning Log

## 2026-10-25: Retrieval evaluation starter

### Built
- Added a minimal Python package with no model-provider dependency.
- Defined a JSONL schema for labelled evaluation cases.
- Implemented Recall@k and reciprocal rank.
- Added aggregate evaluation output, per-case metrics, and a CLI.
- Included small illustrative data files and unit tests.

### Important interpretation note
The sample files are intentionally tiny and illustrative. Their scores only verify that the metrics pipeline works; they are not evidence that a particular model or RAG system is good.

### Next tasks
1. Run tests and inspect the generated JSON report.
2. Create a better-labelled dataset and document the annotation process.
3. Add experiment metadata: model/version, prompt hash, corpus version, retriever, top-k, date, seed, and configuration.
4. Add answer-quality labels and a review workflow.
5. Add optional provider adapters for answer text, latency, and token usage.
6. Compare prompt-only and RAG on the same held-out evaluation set.

### Commands
```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m genai_eval.cli --dataset data/sample_eval.jsonl --predictions data/sample_predictions.json
```
