# OpenAI-compatible GPT4All Server

A minimal OpenAI API-compatible server built with FastAPI and GPT4All that runs a local `.gguf` model from the `models/` directory.

## Prereqs
- Python 3.10+
- A `.gguf` model in `models/` (e.g., `mistral-7b.gguf`).

## Setup

```bash
# create and activate virtualenv (repo uses .venv)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Run the API

```bash
# Optional: explicit model selection
# export MODEL_NAME=mistral-7b.gguf
# export MODELS_DIR=$(pwd)/models

# API key (required by default). You can provide one or many, comma-separated.
export API_KEYS="sk-local-123,sk-local-456"
export REQUIRE_API_KEY=true

uvicorn main:app --reload
```

- Base URL: `http://127.0.0.1:8000`
- Health: `GET /health`
- API Docs (Swagger UI): `GET /docs`
- API Docs (ReDoc): `GET /redoc`
- OpenAPI spec: `GET /openapi.json`

## Endpoints

Public:
- `GET /` – basic server info
- `GET /health` – health check
- `GET /status` – lightweight status dashboard (HTML)
- `GET /chat/` – embedded web UI (Flask, static under `/chat/static`)

Protected (requires `Authorization: Bearer <api-key>`):
- `GET /v1/models` – OpenAI-compatible model list (from local `.gguf` files)
- `POST /v1/chat/completions` – OpenAI-compatible Chat Completions (supports `stream: true` for SSE)
- `POST /v1/completions` – OpenAI-compatible Text Completions (supports `stream: true` for SSE)
- `POST /v1/embeddings` – Embeddings using SentenceTransformers (`all-MiniLM-L6-v2` by default)
- `GET /metrics` – JSON metrics snapshot (server/process + usage counters)
- `POST /v1/training/ingest` – Ingest prompt/response training data to JSONL (optional embeddings)
- `GET /v1/training/datasets` – List available ingested JSONL datasets

## Example requests

Chat (non-streaming):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-123' \
  -d '{
    "model": "mistral-7b.gguf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "max_tokens": 64
  }' | jq
```

Chat (streaming):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-123' \
  -d '{
    "messages": [
      {"role": "user", "content": "List 3 colors."}
    ],
    "stream": true,
    "max_tokens": 64
  }'
```

Completions (prompt-based):

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-123' \
  -d '{
    "prompt": "Write a short haiku about autumn.",
    "max_tokens": 64
  }' | jq
```

Embeddings:

```bash
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-123' \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "input": ["A quick brown fox", "jumps over the lazy dog"]
  }' | jq
```

Training Ingestion (stores JSONL under `data/training/` by default):

```bash
curl -s http://127.0.0.1:8000/v1/training/ingest \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-123' \
  -d '{
    "dataset": "my-support-faq",
    "upsert": true,
    "store_embeddings": true,
    "examples": [
      {
        "prompt": "How do I reset my password?",
        "response": "Go to Settings > Account > Reset Password and follow the instructions.",
        "meta": {"source": "faq", "lang": "en"}
      },
      {
        "prompt": "What is your refund policy?",
        "response": "We offer a 30-day money-back guarantee on all purchases.",
        "meta": {"source": "policy", "lang": "en"}
      }
    ]
  }' | jq
```

List training datasets:

```bash
curl -s http://127.0.0.1:8000/v1/training/datasets \
  -H 'Authorization: Bearer sk-local-123' | jq
```

## Notes
- By default, the server will auto-detect a single `*.gguf` file in `models/`. Set `MODEL_NAME` if multiple files exist.
- Token accounting is approximate; GPT4All does not expose exact token counts for every model backend.
- This server intentionally implements a practical subset of OpenAI responses for compatibility.
- API keys: Set with `API_KEYS` (comma-separated). If `REQUIRE_API_KEY` is `false`, auth is disabled.

### Metrics & Status
- `GET /metrics` returns a JSON payload with server time, request counters, token counts, per-model stats, and process stats (`cpu_percent`, RSS, threads, uptime).
- `GET /status` provides a small live dashboard in the browser that polls `/metrics` and renders charts.

### Environment variables
- `MODELS_DIR` – directory containing `.gguf` files (default: `models/`)
- `MODEL_NAME` – the filename of the default `.gguf` model (e.g., `mistral-7b.gguf`)
- `API_KEYS` – comma-separated API keys, e.g. `"sk-local-123,sk-local-456"`
- `REQUIRE_API_KEY` – `true`/`false` (default `true`)
- `DEFAULT_API_KEY` – prefill for UI/status page
- `EMBEDDING_MODEL` – SentenceTransformers model id (default `all-MiniLM-L6-v2`)
- `EMBEDDING_DEVICE` – `cpu`, `mps` (Apple), or `cuda` where available (default `cpu`)
- `TRAINING_DATA_DIR` – where ingested datasets are saved (default: `data/training`)
- `HOST` / `PORT` – server bind (default `0.0.0.0:8000`)
- `RELOAD` – enable Uvicorn reload via `python main.py` (default `false`)

## Web UI (optional)
This repo includes an embedded chat UI mounted at `/chat` on the same server.

Open http://127.0.0.1:8000/chat in your browser.

Notes:
- Enter your API key in the UI.
- It uses the default model (`mistral-7b.gguf`) and same-origin API.