# GenAI Model Evaluation Lab 🧪

A reproducible benchmarking workspace for comparing LLM and retrieval-augmented generation (RAG) systems on a small, explicitly labelled evaluation set.

> **Status:** Evaluation framework starter. No benchmark results are claimed until experiments have been executed and the results recorded.

## Why this project exists

A GenAI demo can appear convincing while still producing incorrect, ungrounded, slow, or inconsistent answers. This project aims to turn evaluation into a repeatable engineering workflow by recording datasets, configurations, metrics, and failure cases.

## Initial scope

- Store evaluation examples as versionable JSONL records.
- Validate each record before running experiments.
- Evaluate retrieval rankings using metrics such as Recall@k and reciprocal rank.
- Record answer-level correctness and groundedness labels when suitable references are available.
- Track latency and token/cost metadata when a model provider exposes them.
- Compare configurations using the same cases and report limitations clearly.
- Preserve raw experiment outputs so summaries can be regenerated.

## Current starter milestone

The first implementation focuses on **retrieval metrics** without requiring an API key or paid model. This makes the evaluation logic easy to run deterministically and independently from a particular LLM provider.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install and test:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Evaluate a retrieval run

The repository includes a tiny sample dataset and a baseline retrieved-results file. Run:

```bash
python -m genai_eval.cli \
  --dataset data/sample_eval.jsonl \
  --predictions data/sample_predictions.json
```

The command reports Recall@k and mean reciprocal rank (MRR) and can optionally write a machine-readable summary with `--output`.

## Dataset formats

Each JSONL dataset row contains:
- `case_id`: stable unique ID
- `query`: user question or retrieval query
- `relevant_chunk_ids`: expected relevant chunk IDs
- optional `tags`: grouping metadata

Predictions are JSON records containing a `case_id` and ranked `retrieved_chunk_ids` list. See the sample files for the exact schema.

## Planned roadmap

- [x] Define a labelled retrieval dataset format.
- [x] Implement Recall@k and reciprocal rank metrics.
- [x] Add command-line evaluation and a JSON report.
- [x] Add unit tests and continuous integration.
- [ ] Add experiment configuration files and run metadata.
- [ ] Add answer correctness and groundedness review schema.
- [ ] Add latency and token-usage capture with provider adapters.
- [ ] Compare prompt-only and RAG approaches on the same cases.
- [ ] Document failure taxonomy and limitations.

## Experimental integrity

- Never invent or backfill scores that have not been measured.
- Keep the dataset, model/configuration, prompt, and retrieval settings associated with each run.
- Clearly distinguish automatic metrics from human judgements.
- Avoid treating LLM-as-a-judge scores as ground truth; calibrate them against human-reviewed examples.
- Do not store API keys, secrets, or sensitive prompts in the repository.

## Project log

See [LEARNING_LOG.md](LEARNING_LOG.md) for decisions and implementation notes.

## License

A license will be selected after dependency and dataset terms are reviewed.
