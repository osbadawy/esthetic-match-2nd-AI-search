# Aesthetic RAG Search API (v2)

A Dockerized **FastAPI** service that provides **RAG-style treatment/procedure recommendations** for medical aesthetics based on a local Excel database and precomputed embeddings.

It supports:
- Fetching **common concerns** for a selected **sub-zone**
- Searching for **recommended procedures** given a **sub-zone + concerns[] + procedure type**

---

## Repository contents

- `api_server.py` — FastAPI server exposing `/health`, `/common_concerns`, `/search`
- `rag_treatment_app.py` — Core RAG engine (Excel loader, embeddings, retrieval, mismatch detection, optional LLM rerank)
- `llm_client.py` — Local LLM abstraction (`ollama` or in-process `transformers`)
- `web_retriever.py` — Optional DuckDuckGo HTML fetcher (useful if you extend to web-augmented RAG)
- `database.xlsx` — Procedures database (expected sheet: `Procedures`)
- `treatment_embeddings.pkl` — Cached embeddings (generated if missing)
- `Dockerfile` — Container build (Uvicorn on port 8000)
- `deploy.sh` — Convenience deployment script (build + run)
- `requirements.txt` — Python dependencies

---

## How it works (high-level)

1. **Database load**: reads `database.xlsx` (`Procedures` sheet) and normalizes rows into regions/sub-zones/procedures/types.
2. **Embeddings**: uses SentenceTransformers to embed DB rows. Loads from `treatment_embeddings.pkl` when available; otherwise rebuilds and saves the cache.
3. **API**:
   - `/common_concerns`: returns common concerns for a sub-zone
   - `/search`: retrieves and (optionally) LLM-reranks candidates, returning procedure recommendations; can also return mismatch suggestions

---

## Requirements

- Python 3.11 recommended (Docker image uses `python:3.11-slim`)
- Install dependencies from `requirements.txt`

---

## Configuration (environment variables)

### Core paths
- `DB_XLSX` — path to DB Excel (default: `database.xlsx`)
- `EMB_CACHE` — embeddings cache path (default: `treatment_embeddings.pkl`)

### API / CORS
- `CORS_ALLOW_ORIGINS` — comma-separated origins (default: `*`)

### Local LLM provider (optional rerank)
- `LOCAL_LLM_PROVIDER` — `ollama` or `transformers`

**Ollama**
- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3.2:1b`)

**Transformers**
- `HF_LLM_MODEL` (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `HF_MAX_NEW_TOKENS` (default: `220`)
- `TORCH_NUM_THREADS` (default: `2`)

### Quality gates / tuning
- `MIN_ISSUE_CHARS` — minimum issue/problem length used by the RAG engine (default: `5`)

---

## Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Health check:
- `GET http://localhost:8000/health`

---

## Run with Docker

### Build
```bash
docker build -t aesthetic-rag-api:v2 .
```

### Run
```bash
docker run -d --name aesthetic-rag-api --restart unless-stopped   \
  -p 8010:8000 \
  -v $(pwd)/database.xlsx:/app/database.xlsx \
  -v $(pwd)/treatment_embeddings.pkl:/app/treatment_embeddings.pkl \
  -e LOCAL_LLM_PROVIDER=transformers \
  -e HF_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  -e HF_MAX_NEW_TOKENS=220 \
  -e TORCH_NUM_THREADS=2 \
  -e MIN_ISSUE_CHARS=5 \
  -e CORS_ALLOW_ORIGINS=* \
  aesthetic-rag-api:v2
```

---

## Deployment script

```bash
sudo ./deploy.sh
```

What it does:
- checks required files exist
- rebuilds the Docker image
- runs the container on host port **8010** → container port **8000**
- verifies `GET /health`

> Note: If you change `database.xlsx`, you may want to delete `treatment_embeddings.pkl` so embeddings rebuild against the updated data.

---

## API Reference

### 1) Health
**GET** `/health`

Response:
```json
{"status":"ok","service":"aesthetic-rag-api","version":"2.0.0"}
```

---

### 2) Common concerns for a sub-zone
**POST** `/common_concerns`

Body:
```json
{
  "sub_zone": "eyes"
}
```

Response:
```json
{
  "sub_zone": "eyes",
  "common_concerns": ["...", "..."]
}
```

- `sub_zone` is validated against an allowlist and supports fuzzy matching (e.g., `tear trough`, `teartrough`, `tearTrough`).

---

### 3) Search procedures
**POST** `/search`

Body:
```json
{
  "sub_zone": "eyes",
  "concerns": ["dark circles", "hollowing"],
  "procedure_type": "non_surgical",
  "retrieval_k": 12,
  "final_k": 5
}
```

Response (success):
```json
{
  "mismatch": false,
  "notice": "",
  "recommended_procedures": ["Procedure A", "Procedure B"],
  "suggested_region_subzones": []
}
```

Response (mismatch example):
```json
{
  "mismatch": true,
  "notice": "Your selected sub-zone does not match your concerns.",
  "recommended_procedures": [],
  "suggested_region_subzones": [{"region":"Face","sub_zone":"..."}]
}
```

- `procedure_type` accepts: `surgical`, `non_surgical`, `both`

---

## Data expectations (`database.xlsx`)

The RAG engine expects (minimum) columns:
- `procedure_title`
- `main_zone`
- `treatment_type`

It also supports sub-zones via:
- `face_subzone` and/or `body_subzone` (preferred)

---
