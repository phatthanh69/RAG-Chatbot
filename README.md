# RAG Chatbot Platform

> Retrieval-Augmented Generation chatbot with hybrid search, dedicated Q&A management, and Gemini-powered automation for enterprise document workflows.

This README consolidates every piece of documentation in the repository into a single, maintainable reference. Use it as your source of truth for architecture, setup, deployment, evaluation, and troubleshooting.

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Core Components](#architecture--core-components)
3. [Configuration & Environment](#configuration--environment)
4. [Local Setup & Quick Start](#local-setup--quick-start)
5. [Deployment & Migration Playbook](#deployment--migration-playbook)
6. [Hybrid Ensemble Search](#hybrid-ensemble-search)
7. [LLM Pattern Automation](#llm-pattern-automation)
8. [Testing, Evaluation & Reporting](#testing-evaluation--reporting)
9. [CLI Utilities & Scripts](#cli-utilities--scripts)
10. [Troubleshooting & Support](#troubleshooting--support)
11. [Changelog Snapshot](#changelog-snapshot)

---

## System Overview

- **Hybrid Retrieval** combining BM25 and semantic vector search.
- **Dedicated Q&A management** with CRUD APIs and hybrid similarity search.
- **LLM pattern automation** that extracts regex patterns from documents using Gemini 2.5 Pro.
- **Markdown processing pipeline** for PDF/Markdown ingestion, chunking, embedding, and storage.
- **Evaluation toolkit** to benchmark chatbot responses and generate clean Excel reports.

The platform targets production scenarios that require explainable answers, citation tracking, and flexible deployment.

---

## Architecture & Core Components

### Service-Oriented Layout
- `app/services/chatbot_service.py` – Conversation orchestration, retrieval, formatting.
- `app/services/ensemble_retriever_service.py` – Hybrid BM25 + vector fusion.
- `app/services/model_pattern_service.py` – Gemini-driven pattern extraction.
- `app/services/database_service.py` – PostgreSQL + pgvector operations.
- `app/services/document_service.py` & `enhanced_markdown_service.py` – Document ingestion workflows.

### Retrieval Pipeline (`src/`)
- `enhanced_chat.py` – Core RAG logic, retrieval orchestration, prompt builds.
- `simple_vector_store.py` – JSONL-backed vector store (memory-friendly).
- `chunker.py`, `extractor.py`, `cleaner.py` – PDF/Markdown preprocessing.
- `metadata_ranker.py` – Section-aware reranking for precise answers.

### Database & Models (`app/models/`)
- `DocumentChunk`, `ChatSession`, `ChatMessage` for document and conversation storage.
- `ModelPattern` for LLM-generated regex patterns with metadata.
- Q&A tables (`qa_items`, `qa_collections`) powering dedicated knowledge bases.

### Frontend & API
- Flask app factory in `app/__init__.py` with blueprint-driven APIs under `/api`.
- Static single-page UI in `static/` for fast prototyping.

---

## Configuration & Environment

Configuration is centralized in `src/config` with environment-first utilities.

### Environment Files

```bash
cp config.env.example .env
```

Key variables:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection (pgvector enabled). |
| `GOOGLE_API_KEY` / `GOOGLE_CLOUD_PROJECT` | Gemini & Vertex AI access. |
| `GENERATION_MODEL`, `EMBEDDING_MODEL` | LLM & embedding model IDs. |
| `DEFAULT_TOP_K`, `DEFAULT_MIN_SCORE` | Retrieval defaults. |
| `CHAT_SESSIONS_DIR`, `DATA_DIR` | Storage locations (auto-created). |

### Config Utilities

```python
from src.config import config, paths, ensure_environment_setup

ensure_environment_setup()
print(paths.DATA_DIR)
print(config.CHUNK_SIZE)
```

`ensure_environment_setup()` will load environment variables, create required directories, and validate the configuration.

---

## Local Setup & Quick Start

### Prerequisites
- Python 3.10+
- Conda (recommended) or virtualenv.
- Docker (optional) for PostgreSQL.

### Development Environment

```bash
conda activate ai  # or your environment name
pip install -r requirements.txt

# Start PostgreSQL (with pgvector)


# Initialize database schema
python scripts/setup_database.py
python scripts/import_qa_data.py  # optional Q&A seed

# Launch Flask API
python run.py
```

### CLI Launchers
- `python start_chatbot.py` – Auto-detect embedded vector store.
- `python enhanced_chat.py your_file_embedded.jsonl` – Manual vector store.
- `run_chatbot.bat` – Windows double-click helper.

### Chatbot Commands
- `!help`, `!config`, `!history`, `!stats`, `!search`, `!save`, `exit`.
- Example tweak: `!set top_k 8`, `!set min_score 0.15`, `!set show_sources false`.

---

## Deployment & Migration Playbook

### BM25 Vietnamese Optimization (Production Update)

Performance improvements:
- BM25 initialization now 85–99% faster via pre-tokenized content.
- Underthesea-powered tokenization stored alongside original text.

Deployment steps:
```bash
pip install underthesea==6.8.4
alembic upgrade head
python scripts/populate_tokenized_content.py
```

Database changes:
- Adds `tokenized_content` TEXT column to `document_chunks`.
- Migration file: `migrations/versions/001_add_tokenized_content.py`.

Rollback:
```bash
alembic downgrade -1
pip uninstall underthesea
```

Verification:
1. Upload a document → confirm `tokenized_content` populated.
2. Benchmark BM25 search → latency should drop drastically.
3. Check logs for tokenization warnings.

### Team Migration Checklist
1. `git pull` latest branch.
2. Install new dependency (`underthesea`).
3. Run migrations & populate script.
4. Validate BM25 service:

```bash
python -c "
from app import create_app
from app.services.bm25_service import BM25Service

app = create_app()
with app.app_context():
    bm25 = BM25Service()
    status = bm25.initialize_retriever()
    print('BM25 Service Status:', '✅' if status else '❌')
"
```

---

## Hybrid Ensemble Search

Ensemble search fuses BM25 keyword matching with semantic vector similarity for balanced results.

### Environment Configuration

```bash
USE_ENSEMBLE_RETRIEVER=true
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
FUSION_METHOD=rrf   # or "weighted"
RRF_K=60
```

Guidelines:
- Raise `BM25_WEIGHT` (0.4–0.5) for keyword-heavy queries.
- Raise `VECTOR_WEIGHT` (0.7–0.8) for semantic questions.
- `FUSION_METHOD=rrf` works best when both rankers are reliable; `weighted` favors direct score blending.

### Runtime Controls
- `GET /api/chatbot/ensemble/config` – Inspect live config.
- `POST /api/chatbot/ensemble/config` – Update weights and method.
- `POST /api/chatbot/ensemble/toggle` – Enable/disable ensemble.

### Service Usage

```python
from app.services.ensemble_retriever_service import EnsembleRetrieverService

ensemble = EnsembleRetrieverService(bm25_weight=0.3, vector_weight=0.7, enable_rrf=True)
ensemble.initialize()
results = ensemble.search(query="BAS system monitoring", embedding=question_embedding, limit=10, min_score=0.5)
```

Monitoring tips:
- Log messages report BM25/vector counts and RRF scores.
- Metadata includes `bm25_score`, `vector_score`, `rrf_score`, and `search_method`.

---

## LLM Pattern Automation

Gemini 2.5 Pro extracts regex patterns from document headings, stores them in PostgreSQL, and exposes them through the chatbot.

### Key Components
- `app/models/model_pattern.py` – Pattern schema with confidence & usage metrics.
- `app/services/model_pattern_service.py` – LLM analysis, validation, caching.
- `populate_llm_patterns.py` – One-shot pattern generation script.
- `simple_pattern_manager.py` – CLI for inspection and manual refresh.

### Workflow
1. Extract headings → send to Gemini 2.5 Pro.
2. Validate regex + confidence threshold (default 0.6).
3. Deduplicate similar patterns, persist to DB.
4. Chatbot loads patterns on-demand and caches them.

### Commands

```bash
conda activate ai
python populate_llm_patterns.py
python simple_pattern_manager.py  # optional health check

# API-based refresh
curl -X POST http://localhost:5000/admin/patterns/refresh
```

Health endpoint: `GET /health/patterns` returns pattern counts, categories, confidence, last refresh, and status.

Performance snapshot:
- Active patterns: ≥20 (goal >85% coverage).
- Average confidence: ~0.84.
- Generation time: 2–3 minutes for ~90 headings.

---

## Testing, Evaluation & Reporting

The `TEST_CHATBOT` toolkit automates evaluation and reporting for chatbot answers.

### Evaluate Responses

```bash
python TEST_CHATBOT/evaluate_chatbot.py --input_file sample_questions.csv --method api --output_file results.csv

# Alternative direct mode (requires local chatbot running)
python TEST_CHATBOT/evaluate_chatbot.py --method direct
```

Output columns: `question`, `session_id`, `method`, `success`, `response`, `original_response`, `question_type`, `confidence`, `error`.

### Formatting Options
- `--uppercase-bold` – Uppercase bold segments (default on).
- `--format-headings` – Highlight numbered headings.
- `--clean-whitespace` – Remove duplicate blank lines.

### CSV → Excel Helpers

```bash
# Recommended markdown cleaning script
python TEST_CHATBOT/markdown_excel.py sample_questions_evaluation_results.csv --mode clean --output qa_clean.xlsx

# Other modes
python TEST_CHATBOT/markdown_excel.py ... --mode brackets
python TEST_CHATBOT/markdown_excel.py ... --mode uppercase

# Minimal converters
python TEST_CHATBOT/simple_csv_to_excel.py results.csv output.xlsx
python TEST_CHATBOT/csv_to_excel.py --input sample_questions.csv --run-evaluation --output qa_results.xlsx
```

Mode guidance:
- `clean` – remove `**` markers; best for formal reports.
- `brackets` – replace `**text**` with 〖text〗.
- `uppercase` – convert emphasis to uppercase for slides/presentations.

Dependencies:

```bash
pip install pandas openpyxl
```

---

## CLI Utilities & Scripts

| Script | Purpose |
| --- | --- |
| `scripts/setup_database.py` | Create core tables and seed admin data. |
| `scripts/import_qa_data.py` | Bulk import Q&A pairs. |
| `scripts/populate_tokenized_content.py` | Backfill BM25 tokenized columns. |
| `populate_llm_patterns.py` | Generate LLM patterns on demand. |
| `simple_pattern_manager.py` | Inspect pattern health from CLI. |
| `markdown_cli.py` | Batch process Markdown file ingestion. |

---

## Troubleshooting & Support

### Common Issues

| Symptom | Resolution |
| --- | --- |
| `Working outside of application context` | Prefer `--method api` for evaluation scripts. |
| Missing embeddings | Verify `.jsonl` files contain `content` + `embedding` keys. |
| Gemini quota errors | Check Google AI Studio usage or fall back to stored patterns. |
| BM25 initialization failure | Ensure migrations ran and tokenized content exists. |
| No ensemble results | Confirm `USE_ENSEMBLE_RETRIEVER=true` and BM25/vector services initialized. |

### Debug Helpers

```python
# Inspect first vector-store entry
from simple_vector_store import SimpleVectorStore
store = SimpleVectorStore.from_jsonl('your_embedded_file.jsonl')
print(len(store))
print(store.items[0].keys() if store.items else 'No items')

# Validate chatbot pattern loading
from app import create_app
from app.services.chatbot_service import ChatbotService

app = create_app()
with app.app_context():
    chatbot = ChatbotService()
    patterns = chatbot._get_model_patterns()
    print(f"Loaded {len(patterns)} patterns")
```

### Support Channels
- Run `!help` inside the chatbot for in-app guidance.
- Check `logs/` directory for runtime diagnostics.
- Contact the project maintainers for deployment assistance.

---

## Changelog Snapshot

| Date | Update |
| --- | --- |
| 2025-09-25 | Gemini-based LLM pattern automation (manual refresh workflow). |
| 2025-09-10 | BM25 Vietnamese tokenization optimization with `underthesea`. |
| 2025-08-15 | Hybrid ensemble retriever (BM25 + vector) enabled by default. |

---

