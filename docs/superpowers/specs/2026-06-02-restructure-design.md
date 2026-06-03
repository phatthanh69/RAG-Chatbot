# Restructure Design — Domain-Oriented `ragbot/` Package (Phase 1)

**Date:** 2026-06-02
**Status:** Approved for planning
**Scope:** File/folder restructure + dead-code cleanup only. RAG quality work is a separate Phase 2 spec.

---

## 1. Goal & Guardrails

Reorganize the codebase into a single domain-oriented `ragbot/` package so that the
near-term roadmap areas each have an obvious home:

- **Better retrieval / RAG quality** → `ragbot/retrieval/`
- **More data sources / ingestion** → `ragbot/ingestion/`
- **Admin / analytics / multi-tenant** → `ragbot/api/admin_routes.py` + `ragbot/admin/` (grows in later phases)

**Hard guardrails — this phase changes NONE of the following:**

- Runtime behavior of the chatbot (same answers for the same inputs).
- Database schema or Alembic migrations.
- Public HTTP API routes/paths (blueprint URLs stay identical).
- The frontend in `static/`.
- Deployment config (`Dockerfile`, `docker-compose.yml`, `gunicorn.conf.py`, `init.sql`, `alembic.ini`).

This is a **mechanical reorganization + decomposition**, not a behavior change.

### Context (why this is the cheap moment)

- Solo developer, **not yet in production** → free to do a clean big-bang move and break
  internal import paths in one pass.
- No automated tests exist today → a smoke-test scaffold is added as part of this work to
  make the move (and all future work) verifiable.
- `codegraph` is installed and indexes this repo (`.codegraph/codegraph.db`). It is used to
  verify the import graph before and after the move. **After any code change, re-run
  `codegraph sync`** so the index stays current.

---

## 2. Current State (verified)

The repo is a working-but-messy Vietnamese RAG chatbot for customer sales support
(answering questions about solutions/devices for sale).

**Stack:** Flask + Gunicorn · PostgreSQL + pgvector · Gemini (generation + embeddings) ·
BM25 + vector ensemble with RRF fusion · `underthesea` for Vietnamese tokenization.

**Two layers that mostly complement each other:**

- `app/` — Flask app factory, blueprints, SQLAlchemy models, DB-backed services.
- `src/` — low-level utilities (chunking, cleaning, embedding, LLM API, markdown extraction,
  plus the legacy `enhanced_chat.py` retrieval/generation engine).

**Problems this restructure fixes:**

1. `app/services/chatbot_service.py` is a **2,364-line god-object** (session mgmt +
   normalization + retrieval coordination + generation + formatting + logging).
2. **Config sprawl:** `app/core/config.py` (live) vs `src/config/settings.py` + `paths.py`
   + `utils.py` (legacy duplicate).
3. **Dead/superseded code** left in the tree (see §4).
4. **No tests** (README references a `TEST_CHATBOT/` directory that does not exist).
5. Loose ops scripts scattered at the repo root.
6. The `src` vs `app` boundary is by *layer*, not by *domain*, so each roadmap feature would
   be split awkwardly across both packages.

**Live-vs-dead was verified with `codegraph` + grep** — a few first-glance "dead" modules are
actually live and must NOT be deleted (see §4).

---

## 3. Target Structure

```
ragbot/
  __init__.py
  app.py                 # create_app factory            (was app/__init__.py)
  config.py              # SINGLE config source          (merges app/core/config.py + src/config/*)
  extensions.py          # CORS/sqlalchemy init           (was app/core/extensions.py)

  api/                   # thin Flask blueprints (URLs unchanged)
    __init__.py          # register_blueprints            (was app/api/routes.py)
    chat_routes.py       #                                (was app/api/chatbot_routes.py)
    document_routes.py   #                                (was app/api/document_routes.py)
    health_routes.py     #                                (was app/api/health_routes.py)
    admin_routes.py      # NEW empty stub — landing spot for analytics/multi-tenant roadmap

  chat/                  # conversation orchestration  ← splits the 2,364-line god-object
    __init__.py
    orchestrator.py      # top-level orchestration kept from chatbot_service
    normalization.py     # question normalization / rewriting
    prompt_builder.py    # prompt construction
    session.py           # chat-session lifecycle
    formatting.py        # response formatting
    rag_engine.py        # retrieve + generate_answer     (was src/enhanced_chat.py — moved whole)

  retrieval/             # ← "RAG quality" work lands here
    __init__.py
    ensemble.py          #                                (was app/services/ensemble_retriever_service.py)
    bm25.py              #                                (was app/services/bm25_service.py)
    vector_search.py     #                                (was app/services/vector_search_service.py)
    metadata_ranker.py   #                                (was src/metadata_ranker.py)
    simple_vector_store.py #                              (was src/simple_vector_store.py)
    graph_rag.py         #                                (was src/graph_RAG.py)
    graph_rag_service.py #                                (was app/services/graph_rag_service.py)

  ingestion/             # ← "more data sources" work lands here
    __init__.py
    extractor.py         #                                (was src/extractor.py)
    cleaner.py           #                                (was src/cleaner.py)
    chunker.py           #                                (was src/chunker.py)
    embedder.py          #                                (was src/embedder.py)
    markdown_extractor.py #                               (was src/markdown_extractor.py)
    markdown_service.py  #                                (was enhanced_markdown_service.py)
    document_service.py  #                                (was app/services/document_service.py)

  llm/
    __init__.py
    client.py            #                                (was src/llm/api.py)
    direct_model.py      #                                (was app/services/direct_model_service.py)
    pattern_service.py   #                                (was app/services/model_pattern_service.py)

  models/                # SQLAlchemy ORM — schema UNCHANGED
    __init__.py
    base.py · chat.py · document.py · model_pattern.py

  db/
    __init__.py
    database_service.py  #                                (was app/services/database_service.py)

  utils/
    __init__.py
    text.py              # NEW — shared normalize/dedupe extracted from the god-object
    response_helpers.py  #                                (was app/utils/response_helpers.py)
    calculations.py      #                                (was src/utils/calculations.py)

tests/                   # NEW
  __init__.py
  conftest.py            # Flask app fixture, test config
  test_smoke.py          # app boots, blueprints register, key services import, sample retrieval

scripts/                 # ops CLIs (existing + moved loose root scripts)
  setup_database.py · import_qa_data.py · populate_tokenized_content.py · start_services.py
  markdown_cli.py        # moved from root
  simple_pattern_manager.py # moved from root
  populate_llm_patterns.py  # moved from root
  check_db.py            # moved from root

run.py                   # thin entry → from ragbot.app import create_app
# UNCHANGED: migrations/, static/, alembic.ini, gunicorn.conf.py, Dockerfile,
#            docker-compose.yml, init.sql, requirements.txt, pyproject.toml, uv.lock
```

### Rationale for specific calls

- **Single `ragbot/` package, organized by domain** (not by `src`/`app` layer) — chosen so
  each roadmap item has one obvious home instead of being split across two packages.
- **Loose CLIs → `scripts/`** (not a `ragbot/cli/` package) — they are ops tools, not library
  code imported by the app.
- **`admin_routes.py` stub added now** — an empty placeholder blueprint so the analytics /
  multi-tenant roadmap has a landing spot; contains no logic in this phase.
- **`rag_engine.py` moved whole** — `src/enhanced_chat.py` mixes retrieval and generation;
  splitting those internals is deferred to Phase 2. This phase only relocates it.

---

## 4. Dead Code — Delete vs Keep (verified)

**Delete (confirmed unreferenced or superseded):**

| File | Reason |
| --- | --- |
| `main.py` | Hello-world stub, unused. |
| `src/text2json.py` | Zero references anywhere. |
| `markdown_document_service.py` | Only self-reference; superseded by `enhanced_markdown_service.py`. |
| `scripts/extract_entities_manual.py` | 2,192-line standalone Neo4j tool; never imported by the app. |
| `src/config/` (`settings.py`, `paths.py`, `utils.py`, `__init__.py`) | Legacy duplicate config; merged into `ragbot/config.py`. |

Git history preserves all deleted files.

**Do NOT delete — verified LIVE (corrects an earlier first-glance assessment):**

| File | Why it is live |
| --- | --- |
| `src/extractor.py` | Imported by `src/chunker.py` and `app/services/document_service.py`. |
| `src/metadata_ranker.py` | Lazily imported by `src/enhanced_chat.py` (`create_smart_ranker`). |
| `src/simple_vector_store.py` | Imported by `src/enhanced_chat.py`. |
| `src/graph_RAG.py` | Imported by `app/services/graph_rag_service.py`. |
| `enhanced_markdown_service.py` | Imported by `markdown_cli.py` and `app/services/document_service.py`. |

---

## 5. Execution Sequence (risk-ordered; each step verified green before the next)

1. **Scaffold** `ragbot/` and `tests/`. Add `conftest.py` + `test_smoke.py`:
   - `create_app()` returns a Flask app without error.
   - All blueprints register and expected URL rules exist.
   - Key services import cleanly (chat orchestrator, ensemble, document service, llm client).
   - One sample retrieval path runs end-to-end (mocked LLM / fixture data as needed).
2. **Mechanical moves** — relocate files into their domain folders and fix all import paths.
   Re-run `codegraph sync`; smoke tests must stay green.
3. **Delete dead code** per §4.
4. **Consolidate config** — merge `app/core/config.py` + `src/config/*` into a single
   `ragbot/config.py`; update all imports.
5. **Split the god-object** `chatbot_service.py` → `chat/` submodules
   (`orchestrator`, `normalization`, `prompt_builder`, `session`, `formatting`).
   **Highest-risk step — done last, after everything else is green.**
6. **Extract shared utils** — pull duplicated text normalization / dedupe logic from chat +
   ensemble into `ragbot/utils/text.py`; update call sites.
7. **Final verification** — `codegraph sync` (assert no dangling imports), smoke tests green,
   manual app boot with a sample customer question.

---

## 6. Verification Strategy

- **pytest smoke-test scaffold** (`tests/`) is the primary safety net, introduced in step 1 and
  kept green through every subsequent step.
- **`codegraph sync` + status** after structural changes to catch dangling/broken imports
  (the index currently reports 63 files / 1,129 nodes / "up to date").
- **Manual boot** of the Flask app and a sample customer question as the final acceptance check.

---

## 7. Out of Scope (Phase 2 and beyond)

- Any retrieval-quality change (reranking, query rewriting strategy, fusion tuning).
- Prompt engineering / answer-quality tuning.
- DB schema or migration changes.
- Splitting `rag_engine.py`'s retrieve-vs-generate internals (moved whole here; refactored later).
- New ingestion sources, channels, or analytics dashboards (structure makes room for them; no
  implementation here).

---

## 8. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Big-bang move breaks an import not caught by smoke tests. | `codegraph sync` + status after each structural step; smoke tests cover app boot + key imports. |
| God-object split (step 5) changes behavior subtly. | Done last, in isolation, with smoke tests green before and after; behavior-preserving extraction only. |
| Hidden runtime imports (string-based / lazy) missed by static analysis. | Grep for module names as strings; manual boot + sample question as final check. |
| Config merge drops a used setting. | Diff the union of keys from both config sources before deleting `src/config/`. |
