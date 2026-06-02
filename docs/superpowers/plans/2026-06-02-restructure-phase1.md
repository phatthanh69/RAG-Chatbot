# Restructure Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the `src/` + `app/` codebase into a single domain-oriented `ragbot/` package, delete verified-dead code, consolidate config, split the 2,364-line `chatbot_service` god-object, and add a pytest smoke-test scaffold — with zero change to runtime behavior, DB schema, API routes, or the frontend.

**Architecture:** A behavior-preserving move. Files relocate into domain folders (`ingestion`, `retrieval`, `chat`, `llm`, `models`, `db`, `api`, `config`, `utils`); imports are rewritten via an explicit module-path map; a pytest smoke scaffold + `codegraph sync` gate every structural step. The god-object is decomposed last via "extract pure function" (real TDD) and "extract collaborator class" (smoke-guarded) refactors, leaving `ChatbotService` as a thin facade.

**Tech Stack:** Python 3.13 · Flask + Gunicorn · SQLAlchemy + PostgreSQL/pgvector · Gemini (google-genai) · BM25 (`rank-bm25`) + `underthesea` · pytest (new) · `codegraph` (index/verify).

**Spec:** `docs/superpowers/specs/2026-06-02-restructure-design.md`

---

## Reference: Module-Path Rename Map

This is the single source of truth for every import rewrite. `OLD module → NEW module`:

| Old | New |
| --- | --- |
| `app` (`from app import create_app`) | `ragbot.app` |
| `app.__init__:create_app` | `ragbot.app:create_app` |
| `app.core.config` (`Config`, `get_config`) | `ragbot.config` |
| `app.core.extensions` | `ragbot.extensions` |
| `app.api.routes` | `ragbot.api` (package `__init__`) |
| `app.api.chatbot_routes` | `ragbot.api.chat_routes` |
| `app.api.document_routes` | `ragbot.api.document_routes` |
| `app.api.health_routes` | `ragbot.api.health_routes` |
| `app.models.base` | `ragbot.models.base` |
| `app.models.chat` | `ragbot.models.chat` |
| `app.models.document` | `ragbot.models.document` |
| `app.models.model_pattern` | `ragbot.models.model_pattern` |
| `app.models` (`__init__`) | `ragbot.models` |
| `app.services.chatbot_service` | `ragbot.chat.orchestrator` |
| `app.services.database_service` | `ragbot.db.database_service` |
| `app.services.ensemble_retriever_service` | `ragbot.retrieval.ensemble` |
| `app.services.bm25_service` | `ragbot.retrieval.bm25` |
| `app.services.vector_search_service` | `ragbot.retrieval.vector_search` |
| `app.services.graph_rag_service` | `ragbot.retrieval.graph_rag_service` |
| `app.services.document_service` | `ragbot.ingestion.document_service` |
| `app.services.direct_model_service` | `ragbot.llm.direct_model` |
| `app.services.model_pattern_service` | `ragbot.llm.pattern_service` |
| `app.utils.response_helpers` | `ragbot.utils.response_helpers` |
| `src.chunker` | `ragbot.ingestion.chunker` |
| `src.cleaner` | `ragbot.ingestion.cleaner` |
| `src.extractor` | `ragbot.ingestion.extractor` |
| `src.embedder` | `ragbot.ingestion.embedder` |
| `src.markdown_extractor` | `ragbot.ingestion.markdown_extractor` |
| `src.enhanced_chat` | `ragbot.chat.rag_engine` |
| `src.simple_vector_store` | `ragbot.retrieval.simple_vector_store` |
| `src.metadata_ranker` | `ragbot.retrieval.metadata_ranker` |
| `src.graph_RAG` | `ragbot.retrieval.graph_rag` |
| `src.llm.api` | `ragbot.llm.client` |
| `src.config` (`config`, `paths`, utils) | `ragbot.config` |
| `src.utils.calculations` | `ragbot.utils.calculations` |
| `enhanced_markdown_service` (root) | `ragbot.ingestion.markdown_service` |

**Deleted (no new home):** `main.py`, `src/text2json.py`, `markdown_document_service.py`, `scripts/extract_entities_manual.py`.

**Config collision note:** `app.core.config` exposes `Config` (class), `get_config()` (returns a Flask config *class*), and an internal `config` dict (env→class). `src.config` exposes `config` (an `AppConfig` *instance* with `.CHUNK_SIZE` etc.), `paths` (a `PathConfig` instance), and env utils. The new `ragbot/config/` package preserves BOTH: it re-exports `Config`/`get_config` (Flask) and `config`/`paths` (settings). The internal env→class dict is renamed `_CONFIG_MAP` to free the name `config`.

---

## Conventions for this plan

- Run all commands from repo root `/home/phamlethanhphat/code/RAG-Chatbot`.
- "Move a file" = `git mv OLD NEW` (preserves history), then rewrite imports.
- After every task that moves/creates/deletes a `.py` file: run `codegraph sync` then `codegraph status` and confirm `✓ Index is up to date`.
- Test runner: `pytest -q`. The smoke suite must stay green from Task 2 onward (except the documented red→green transition in Task 13).
- These move-tasks relocate **existing** code; where a step says "move the body verbatim", copy the exact current method/function text — do not rewrite logic.

---

## Phase 0 — Tooling & Smoke Scaffold

### Task 1: Add pytest + dev dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest>=8.0.0
```

- [ ] **Step 2: Create pytest config**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 3: Install**

Run: `pip install pytest>=8.0.0`
Expected: `Successfully installed pytest-...`

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest is wired.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini
git commit -m "chore: add pytest dev dependency and config"
```

---

### Task 2: Smoke-test scaffold against the CURRENT layout

This establishes the safety net using a layout-agnostic entrypoint shim, so the same tests keep working before and after the move.

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/_entry.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create test package marker**

Create `tests/__init__.py` (empty file).

- [ ] **Step 2: Create a single-point entry shim**

Create `tests/_entry.py` — the ONLY place that names the app factory's import path. Updated once, in Task 12.

```python
"""Single indirection point for the app factory import path.

Updated exactly once during the restructure (app factory move).
Keeping every test pointed here means the move flips one line, not many.
"""

# CURRENT layout (pre-move). Task 12 changes this line to:
#   from ragbot.app import create_app
from app import create_app  # noqa: F401

# Expected blueprint URL prefixes — behavior contract, must not change.
EXPECTED_URL_PREFIXES = ["/api/health", "/api/documents", "/api/chatbot"]
```

- [ ] **Step 3: Create conftest with a Testing-config app fixture**

Create `tests/conftest.py`:

```python
import os

import pytest

# Force the in-repo Testing config and avoid real external calls at import time.
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used")

from tests._entry import create_app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
```

- [ ] **Step 4: Write the smoke tests**

Create `tests/test_smoke.py`:

```python
from tests._entry import EXPECTED_URL_PREFIXES


def test_app_factory_boots(app):
    assert app is not None
    assert app.name


def test_expected_blueprints_registered(app):
    rules = {r.rule for r in app.url_map.iter_rules()}
    # Every API area must expose at least one rule under its prefix.
    for prefix in EXPECTED_URL_PREFIXES:
        assert any(rule.startswith(prefix) for rule in rules), (
            f"No route registered under {prefix}. Rules: {sorted(rules)}"
        )


def test_root_and_api_index_routes_exist(app):
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/" in rules
    assert "/api" in rules
```

- [ ] **Step 5: Run the smoke suite — expect PASS on current layout**

Run: `pytest tests/test_smoke.py -v`
Expected: 3 passed. (If `create_app()` needs a DB at import time it should not — it defers DB. If it fails, STOP and report; do not proceed.)

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: add smoke-test scaffold for app factory and routes"
```

---

## Phase 1 — Build `ragbot/` and Move Files (dependency order)

Move order respects the import graph: config → leaf utils/llm → models → db → ingestion → retrieval → chat (whole) → api → app factory → entrypoints. Each task ends green.

### Task 3: Create `ragbot/` package + consolidate config

`ragbot/config/` is built FIRST because nearly everything depends on it. It absorbs both `app/core/config.py` and `src/config/`.

**Files:**
- Create: `ragbot/__init__.py`
- Create: `ragbot/config/__init__.py`
- Create: `ragbot/config/flask_config.py`
- Move: `src/config/settings.py` → `ragbot/config/settings.py`
- Move: `src/config/paths.py` → `ragbot/config/paths.py`
- Move: `src/config/utils.py` → `ragbot/config/utils.py`

- [ ] **Step 1: Create the package marker**

Create `ragbot/__init__.py`:

```python
"""ragbot — domain-oriented RAG chatbot package."""
```

- [ ] **Step 2: Move the Flask config body**

```bash
git mv app/core/config.py ragbot/config/flask_config.py
```

Then in `ragbot/config/flask_config.py`, rename the module-level env→class dict `config` to `_CONFIG_MAP` (it collides with the settings singleton). Change exactly these two spots:

```python
# was: config = {
_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "development")
    # was: return config.get(env, config["default"])
    return _CONFIG_MAP.get(env, _CONFIG_MAP["default"])
```

- [ ] **Step 3: Move the settings/paths/utils modules**

```bash
git mv src/config/settings.py ragbot/config/settings.py
git mv src/config/paths.py    ragbot/config/paths.py
git mv src/config/utils.py    ragbot/config/utils.py
git rm src/config/__init__.py
```

- [ ] **Step 4: Fix intra-config imports**

`ragbot/config/utils.py` and `settings.py`/`paths.py` may reference each other via `from .paths import ...` / `from .settings import ...`. Those relative imports still resolve inside the new package — verify by reading the top of each file; if any uses `from src.config...` or `from config...`, change it to a relative `from .paths import paths` / `from .settings import config` form.

- [ ] **Step 5: Write the consolidated package `__init__`**

Create `ragbot/config/__init__.py`:

```python
"""Unified configuration package.

Exposes two interfaces, both previously split across app/core/config.py and
src/config/:
  - Flask app config:  Config, get_config  (Flask config *classes*)
  - Runtime settings:  config, paths       (singletons with .CHUNK_SIZE, .DATA_DIR, ...)
"""

from ragbot.config.flask_config import (
    Config,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    get_config,
)
from ragbot.config.paths import PathConfig, paths
from ragbot.config.settings import AppConfig, config
from ragbot.config.utils import (
    ensure_environment_setup,
    load_environment,
    validate_environment,
)

__all__ = [
    "Config",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
    "get_config",
    "PathConfig",
    "paths",
    "AppConfig",
    "config",
    "ensure_environment_setup",
    "load_environment",
    "validate_environment",
]
```

Note: if `ragbot/config/utils.py` exports more names that other modules import, add them here. Verify by grepping later in Task 12.

- [ ] **Step 6: Verify config package imports both interfaces**

Run:
```bash
python -c "from ragbot.config import Config, get_config, config, paths; print(get_config().__name__, type(config).__name__, type(paths).__name__)"
```
Expected: prints something like `TestingConfig AppConfig PathConfig` (env-dependent) with no ImportError.

- [ ] **Step 7: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: consolidate config into ragbot/config package"
```

---

### Task 4: Move leaf utilities and the LLM client

`src/utils/calculations.py` and `src/llm/api.py` are leaves (no internal deps except each other-free).

**Files:**
- Move: `src/utils/calculations.py` → `ragbot/utils/calculations.py`
- Move: `src/llm/api.py` → `ragbot/llm/client.py`
- Create: `ragbot/utils/__init__.py`, `ragbot/llm/__init__.py`

- [ ] **Step 1: Create package markers**

Create empty `ragbot/utils/__init__.py` and `ragbot/llm/__init__.py`.

- [ ] **Step 2: Move the files**

```bash
git mv src/utils/calculations.py ragbot/utils/calculations.py
git mv src/llm/api.py ragbot/llm/client.py
git rm -f src/utils/__init__.py src/llm/__init__.py 2>/dev/null || true
```

- [ ] **Step 3: Fix imports inside moved files**

In `ragbot/llm/client.py`, rewrite any `from src.config import ...` → `from ragbot.config import ...`. `calculations.py` has no internal imports (verified: pure numpy). Confirm by reading its top.

- [ ] **Step 4: Verify**

Run:
```bash
python -c "import ragbot.utils.calculations as c; print(hasattr(c,'cosine_similarity'))"
python -c "import ragbot.llm.client as l; print(hasattr(l,'init_genai_client'))"
```
Expected: `True` then `True`.

- [ ] **Step 5: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move calculations and llm client into ragbot"
```

---

### Task 5: Move SQLAlchemy models

Models depend only on each other and `db`.

**Files:**
- Move: `app/models/base.py`, `chat.py`, `document.py`, `model_pattern.py`, `__init__.py` → `ragbot/models/`

- [ ] **Step 1: Move the directory**

```bash
git mv app/models ragbot/models
```

- [ ] **Step 2: Rewrite intra-models imports**

In each moved file, replace `from app.models.base import db` → `from ragbot.models.base import db`, and in `ragbot/models/__init__.py` replace the three `from app.models.<x>` lines with `from ragbot.models.<x>`.

- [ ] **Step 3: Verify**

Run: `python -c "from ragbot.models import db, ChatMessage, ChatSession, Document, DocumentChunk, ModelPattern; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move SQLAlchemy models into ragbot/models"
```

---

### Task 6: Move the database service into `ragbot/db`

**Files:**
- Create: `ragbot/db/__init__.py`
- Move: `app/services/database_service.py` → `ragbot/db/database_service.py`

- [ ] **Step 1: Create marker + move**

```bash
mkdir -p ragbot/db && touch ragbot/db/__init__.py
git mv app/services/database_service.py ragbot/db/database_service.py
git add ragbot/db/__init__.py
```

- [ ] **Step 2: Rewrite imports in the moved file**

Replace: `from app.models.base import db` → `from ragbot.models.base import db`; `from app.models.chat import ...` → `from ragbot.models.chat import ...`; `from app.models.document import ...` → `from ragbot.models.document import ...`; `from app.models.model_pattern import ...` → `from ragbot.models.model_pattern import ...`.

- [ ] **Step 3: Verify**

Run: `python -c "from ragbot.db.database_service import DatabaseService; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move database_service into ragbot/db"
```

---

### Task 7: Move ingestion modules

Order within: `extractor` → `cleaner` → `embedder` → `chunker` → `markdown_extractor` → `markdown_service` → `document_service` (document_service depends on all).

**Files:**
- Create: `ragbot/ingestion/__init__.py`
- Move the 7 modules per the rename map.

- [ ] **Step 1: Create marker**

Create empty `ragbot/ingestion/__init__.py`.

- [ ] **Step 2: Move src-side ingestion files**

```bash
git mv src/extractor.py          ragbot/ingestion/extractor.py
git mv src/cleaner.py            ragbot/ingestion/cleaner.py
git mv src/embedder.py          ragbot/ingestion/embedder.py
git mv src/chunker.py            ragbot/ingestion/chunker.py
git mv src/markdown_extractor.py ragbot/ingestion/markdown_extractor.py
git mv enhanced_markdown_service.py ragbot/ingestion/markdown_service.py
git mv app/services/document_service.py ragbot/ingestion/document_service.py
```

- [ ] **Step 3: Rewrite imports inside each moved file**

Apply per the rename map:
- `chunker.py`: `from src.cleaner import normalize_text` → `from ragbot.ingestion.cleaner import normalize_text`; `from src.config import config, paths` → `from ragbot.config import config, paths`; `from src.extractor import (` → `from ragbot.ingestion.extractor import (`.
- `embedder.py`: `from src.llm.api import init_genai_client` → `from ragbot.llm.client import init_genai_client`.
- `markdown_extractor.py`: `from src.cleaner import normalize_text` → `from ragbot.ingestion.cleaner import normalize_text`; `from src.config import config` → `from ragbot.config import config`.
- `markdown_service.py`: `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`; `from app.utils.response_helpers import get_vietnam_time` → `from ragbot.utils.response_helpers import get_vietnam_time`; `from src.cleaner import normalize_text` → `from ragbot.ingestion.cleaner import normalize_text`; `from src.embedder import (` → `from ragbot.ingestion.embedder import (`; `from src.markdown_extractor import (` → `from ragbot.ingestion.markdown_extractor import (`.
- `document_service.py`: `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`; `from app.services.model_pattern_service import ModelPatternAnalysisService` → `from ragbot.llm.pattern_service import ModelPatternAnalysisService`; `from app.utils.response_helpers import get_vietnam_time` → `from ragbot.utils.response_helpers import get_vietnam_time`; `from src import chunker, cleaner, embedder, extractor` → `from ragbot.ingestion import chunker, cleaner, embedder, extractor`; and the deferred `from enhanced_markdown_service import EnhancedMarkdownDocumentService` → `from ragbot.ingestion.markdown_service import EnhancedMarkdownDocumentService`.

> Note: `document_service.py` imports `ragbot.utils.response_helpers` (moved in Task 10) and `ragbot.llm.pattern_service` (moved in Task 9). The import lines are correct now; the modules land before the final smoke run. Verify with a syntax-only check this task, full import in Task 11.

- [ ] **Step 4: Syntax-check the moved ingestion files**

Run: `python -m py_compile ragbot/ingestion/*.py && echo OK`
Expected: `OK` (compiles even though some target modules arrive later).

- [ ] **Step 5: Verify leaf ingestion imports resolve now**

Run: `python -c "from ragbot.ingestion import extractor, cleaner, embedder, chunker, markdown_extractor; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move ingestion modules into ragbot/ingestion"
```

---

### Task 8: Move retrieval modules

`enhanced_chat`/`metadata_ranker`/`simple_vector_store` are interdependent (`metadata_ranker` imports `RetrievalResult` from `enhanced_chat`; `enhanced_chat` imports `simple_vector_store`). `enhanced_chat` becomes `ragbot/chat/rag_engine.py` but its retrieval collaborators live in `retrieval/`.

**Files:**
- Create: `ragbot/retrieval/__init__.py`, `ragbot/chat/__init__.py`
- Move: ensemble, bm25, vector_search, graph_rag_service, simple_vector_store, metadata_ranker, graph_RAG, and enhanced_chat→chat/rag_engine.

- [ ] **Step 1: Create markers**

Create empty `ragbot/retrieval/__init__.py` and `ragbot/chat/__init__.py`.

- [ ] **Step 2: Move the files**

```bash
git mv app/services/ensemble_retriever_service.py ragbot/retrieval/ensemble.py
git mv app/services/bm25_service.py               ragbot/retrieval/bm25.py
git mv app/services/vector_search_service.py      ragbot/retrieval/vector_search.py
git mv app/services/graph_rag_service.py          ragbot/retrieval/graph_rag_service.py
git mv src/simple_vector_store.py                 ragbot/retrieval/simple_vector_store.py
git mv src/metadata_ranker.py                     ragbot/retrieval/metadata_ranker.py
git mv src/graph_RAG.py                           ragbot/retrieval/graph_rag.py
git mv src/enhanced_chat.py                        ragbot/chat/rag_engine.py
```

- [ ] **Step 3: Rewrite imports inside moved files**

- `ensemble.py`: `from app.models.document import DocumentChunk` → `from ragbot.models.document import DocumentChunk`; `from app.services.bm25_service import BM25Service` → `from ragbot.retrieval.bm25 import BM25Service`; `from app.services.vector_search_service import VectorSearchService` → `from ragbot.retrieval.vector_search import VectorSearchService`.
- `bm25.py`: `from app.models.base import db` → `from ragbot.models.base import db`; `from app.models.document import Document, DocumentChunk` → `from ragbot.models.document import Document, DocumentChunk`.
- `vector_search.py`: `from app.models.base import db` → `from ragbot.models.base import db`; `from app.models.document import Document, DocumentChunk` → `from ragbot.models.document import Document, DocumentChunk`.
- `graph_rag_service.py`: `from src.graph_RAG import GraphRAGService` → `from ragbot.retrieval.graph_rag import GraphRAGService`; `from src.llm.api import init_genai_client` → `from ragbot.llm.client import init_genai_client`.
- `simple_vector_store.py`: `from src.utils.calculations import cosine_similarity` → `from ragbot.utils.calculations import cosine_similarity`.
- `metadata_ranker.py`: `from src.enhanced_chat import RetrievalResult` → `from ragbot.chat.rag_engine import RetrievalResult`.
- `rag_engine.py` (was enhanced_chat): `from src.llm.api import init_genai_client` → `from ragbot.llm.client import init_genai_client`; `from src.simple_vector_store import SimpleVectorStore` → `from ragbot.retrieval.simple_vector_store import SimpleVectorStore`; `from src.utils.calculations import cosine_similarity` → `from ragbot.utils.calculations import cosine_similarity`; and the two lazy `from src.metadata_ranker import create_smart_ranker` → `from ragbot.retrieval.metadata_ranker import create_smart_ranker`.

- [ ] **Step 4: Syntax-check + import-verify**

Run: `python -m py_compile ragbot/retrieval/*.py ragbot/chat/rag_engine.py && echo OK`
Expected: `OK`.

Run: `python -c "from ragbot.retrieval import bm25, vector_search, ensemble, simple_vector_store, metadata_ranker, graph_rag, graph_rag_service; from ragbot.chat import rag_engine; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move retrieval modules + rag_engine into ragbot"
```

---

### Task 9: Move the LLM-side services (direct model + pattern)

**Files:**
- Move: `app/services/direct_model_service.py` → `ragbot/llm/direct_model.py`
- Move: `app/services/model_pattern_service.py` → `ragbot/llm/pattern_service.py`

- [ ] **Step 1: Move**

```bash
git mv app/services/direct_model_service.py ragbot/llm/direct_model.py
git mv app/services/model_pattern_service.py ragbot/llm/pattern_service.py
```

- [ ] **Step 2: Rewrite imports**

- `direct_model.py`: `from app.core.extensions import db` → `from ragbot.extensions import db`. *(Note: `ragbot.extensions` arrives in Task 11; until then this module imports lazily. If it imports `db` at module top, temporarily point it at `from ragbot.models.base import db` which is the same object and always available — preferred.)* Use `from ragbot.models.base import db`.
- `pattern_service.py`: `from app.core.extensions import db` → `from ragbot.models.base import db`; `from app.models.model_pattern import ModelPattern` → `from ragbot.models.model_pattern import ModelPattern`; `from src.enhanced_chat import generate_answer` → `from ragbot.chat.rag_engine import generate_answer`; `from src.llm.api import init_genai_client` → `from ragbot.llm.client import init_genai_client`.

- [ ] **Step 3: Verify**

Run: `python -c "from ragbot.llm.pattern_service import ModelPatternAnalysisService; from ragbot.llm.direct_model import *; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move direct_model and pattern services into ragbot/llm"
```

---

### Task 10: Move `response_helpers` + move the chatbot god-object (whole, not yet split)

The god-object moves as ONE file first (`ragbot/chat/orchestrator.py`); the split happens in Phase 4. This keeps the move and the decomposition as separate, independently-verifiable changes.

**Files:**
- Move: `app/utils/response_helpers.py` → `ragbot/utils/response_helpers.py`
- Move: `app/services/chatbot_service.py` → `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Move both**

```bash
git mv app/utils/response_helpers.py ragbot/utils/response_helpers.py
git rm -f app/utils/__init__.py 2>/dev/null || true
git mv app/services/chatbot_service.py ragbot/chat/orchestrator.py
```

- [ ] **Step 2: Rewrite imports in `orchestrator.py`**

The file has a try/except dual-import block for `src.enhanced_chat` and `src.llm.api`, plus three `app.services` imports. Replace:
- both `from src.enhanced_chat import (` occurrences → `from ragbot.chat.rag_engine import (`
- both `from src.llm.api import init_genai_client` → `from ragbot.llm.client import init_genai_client`
- `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`
- `from app.services.ensemble_retriever_service import EnsembleRetrieverService` → `from ragbot.retrieval.ensemble import EnsembleRetrieverService`
- `from app.services.vector_search_service import VectorSearchService` → `from ragbot.retrieval.vector_search import VectorSearchService`

Also delete the now-obsolete `sys.path` manipulation comment block if present (lines ~16–41 do `try/except` path juggling "to avoid circular imports"); collapse to the single direct import form since paths are now clean. If unsure, leave the try/except but fix both branches' module paths.

- [ ] **Step 3: Verify response_helpers + orchestrator import**

Run: `python -c "from ragbot.utils.response_helpers import error_response, success_response, get_vietnam_time; print('ok')"`
Expected: `ok`.

Run: `python -c "from ragbot.chat.orchestrator import ChatbotService; print('ok')"`
Expected: `ok` (constructing it may need DB/LLM, but importing the class must not).

- [ ] **Step 4: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move response_helpers and chatbot_service (whole) into ragbot/chat"
```

---

### Task 11: Move API blueprints + extensions

**Files:**
- Create: `ragbot/api/__init__.py` (was `app/api/routes.py`)
- Move: `chatbot_routes.py`→`chat_routes.py`, `document_routes.py`, `health_routes.py`
- Move: `app/core/extensions.py` → `ragbot/extensions.py`
- Create: `ragbot/api/admin_routes.py` (empty stub)

- [ ] **Step 1: Move extensions**

```bash
git mv app/core/extensions.py ragbot/extensions.py
```

Rewrite in `ragbot/extensions.py`: `from app.models.base import db` → `from ragbot.models.base import db`; `from app.services.chatbot_service import ChatbotService` → `from ragbot.chat.orchestrator import ChatbotService`.

- [ ] **Step 2: Move route modules**

```bash
git mv app/api/chatbot_routes.py ragbot/api/chat_routes.py
git mv app/api/document_routes.py ragbot/api/document_routes.py
git mv app/api/health_routes.py  ragbot/api/health_routes.py
git mv app/api/routes.py         ragbot/api/__init__.py
```

- [ ] **Step 3: Rewrite imports in route files**

- `chat_routes.py`: `from app.core.extensions import get_chatbot_service` → `from ragbot.extensions import get_chatbot_service`; `from app.services.chatbot_service import ChatbotService` → `from ragbot.chat.orchestrator import ChatbotService`; `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`; `from app.services.vector_search_service import VectorSearchService` → `from ragbot.retrieval.vector_search import VectorSearchService`; `from app.utils.response_helpers import error_response, success_response` → `from ragbot.utils.response_helpers import error_response, success_response`.
- `document_routes.py`: `from app.core.config import Config` → `from ragbot.config import Config`; `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`; `from app.services.document_service import DocumentService` → `from ragbot.ingestion.document_service import DocumentService`; `from app.utils.response_helpers import ...` → `from ragbot.utils.response_helpers import ...`.
- `health_routes.py`: `from app.utils.response_helpers import get_vietnam_time` → `from ragbot.utils.response_helpers import get_vietnam_time`.
- `ragbot/api/__init__.py` (was routes.py): `from app.api.document_routes import document_bp` → `from ragbot.api.document_routes import document_bp`; `from app.api.chatbot_routes import chatbot_bp` → `from ragbot.api.chat_routes import chatbot_bp`; `from app.api.health_routes import health_bp` → `from ragbot.api.health_routes import health_bp`. **Do not change the `url_prefix` strings** (`/api/health`, `/api/documents`, `/api/chatbot`).

- [ ] **Step 4: Create the admin stub**

Create `ragbot/api/admin_routes.py`:

```python
"""Admin / analytics blueprint — placeholder for the multi-tenant/analytics roadmap.

Intentionally empty of business logic in Phase 1. Register it in ragbot/api/__init__.py
when the first admin endpoint is added.
"""

from flask import Blueprint

admin_bp = Blueprint("admin", __name__)
```

(Do NOT register `admin_bp` yet — registering an empty blueprint adds no rules and the smoke test asserts the existing three prefixes only.)

- [ ] **Step 5: Verify API + extensions import**

Run: `python -c "from ragbot.api import register_blueprints; from ragbot.extensions import init_extensions, get_chatbot_service; from ragbot.api.admin_routes import admin_bp; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move API blueprints and extensions into ragbot"
```

---

### Task 12: Move the app factory and flip the test entrypoint

**Files:**
- Create: `ragbot/app.py` (was `app/__init__.py`)
- Modify: `tests/_entry.py`

- [ ] **Step 1: Move the factory**

```bash
git mv app/__init__.py ragbot/app.py
```

- [ ] **Step 2: Rewrite imports in `ragbot/app.py`**

`from app.api.routes import register_blueprints` → `from ragbot.api import register_blueprints`; `from app.core.config import Config` → `from ragbot.config import Config`; `from app.core.extensions import init_extensions` → `from ragbot.extensions import init_extensions`. The static-folder path logic computes `os.path.dirname(current_dir)` — since `ragbot/app.py` sits one level below repo root just like `app/__init__.py` did, `static/` resolves identically. Verify the computed `static_dir` still points at repo-root `static/`.

- [ ] **Step 3: Remove the now-empty `app/` tree**

```bash
git rm -r app/core 2>/dev/null || true
# app/ should now be empty of .py files; remove leftovers:
find app -type f -name '*.py' -print
rmdir app/api app/services app 2>/dev/null || true
```
Expected: the `find` prints nothing (all moved). If it prints a file, move it per the rename map before continuing.

- [ ] **Step 4: Flip the single test entry indirection**

In `tests/_entry.py`, change:

```python
# from app import create_app
from ragbot.app import create_app  # noqa: F401
```

- [ ] **Step 5: Run the FULL smoke suite against the new layout**

Run: `pytest -q`
Expected: 3 passed. This is the green that proves the move preserved app boot + routes.

- [ ] **Step 6: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: move app factory to ragbot.app; smoke suite green on new layout"
```

---

### Task 13: Update entrypoints and ops scripts to the new paths

**Files:**
- Modify: `run.py`, `migrations/env.py`, `check_db.py`, `populate_llm_patterns.py`, `simple_pattern_manager.py`, `markdown_cli.py`
- Modify: `scripts/import_qa_data.py`, `scripts/populate_tokenized_content.py`, `scripts/setup_database.py`
- Move loose CLIs into `scripts/`

- [ ] **Step 1: Rewrite imports in entrypoints**

- `run.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.core.config import get_config` → `from ragbot.config import get_config`.
- `migrations/env.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.models.base import db` → `from ragbot.models.base import db`.
- `check_db.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.core.extensions import db` → `from ragbot.extensions import db`.
- `populate_llm_patterns.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.models.model_pattern import ModelPattern` → `from ragbot.models.model_pattern import ModelPattern`; `from app.services.model_pattern_service import ModelPatternAnalysisService` → `from ragbot.llm.pattern_service import ModelPatternAnalysisService`.
- `markdown_cli.py`: `from enhanced_markdown_service import EnhancedMarkdownDocumentService` → `from ragbot.ingestion.markdown_service import EnhancedMarkdownDocumentService`.
- `simple_pattern_manager.py`: rewrite any `from app...`/`from src...` to the mapped paths (grep it first).
- `scripts/import_qa_data.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.models.base import db` → `from ragbot.models.base import db`; `from app.services.database_service import DatabaseService` → `from ragbot.db.database_service import DatabaseService`.
- `scripts/populate_tokenized_content.py`: `from app import create_app` → `from ragbot.app import create_app`; `from app.services.bm25_service import BM25Service` → `from ragbot.retrieval.bm25 import BM25Service`.
- `scripts/setup_database.py`: `from app.models.base import db` → `from ragbot.models.base import db`; `from app.core.config import Config` → `from ragbot.config import Config`; `from app import create_app` → `from ragbot.app import create_app`.

- [ ] **Step 2: Move loose CLIs into `scripts/`**

```bash
git mv markdown_cli.py            scripts/markdown_cli.py
git mv simple_pattern_manager.py scripts/simple_pattern_manager.py
git mv populate_llm_patterns.py  scripts/populate_llm_patterns.py
git mv check_db.py               scripts/check_db.py
```

(`markdown_cli.py` and others use absolute `ragbot.` imports, so moving them into `scripts/` is import-safe as long as they're run from repo root.)

- [ ] **Step 3: Verify entrypoint imports compile**

Run:
```bash
python -m py_compile run.py migrations/env.py scripts/*.py && echo OK
python -c "import run" 2>&1 | head -3
```
Expected: `OK`; `import run` may print logging but must not raise ImportError.

- [ ] **Step 4: Full smoke + manual boot check**

Run: `pytest -q`
Expected: 3 passed.

- [ ] **Step 5: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: point entrypoints and ops scripts at ragbot paths"
```

---

## Phase 2 — Delete Dead Code

### Task 14: Remove verified-dead files

**Files:**
- Delete: `main.py`, `src/text2json.py`, `markdown_document_service.py`, `scripts/extract_entities_manual.py`, and any now-empty `src/` remnants.

- [ ] **Step 1: Confirm zero references before deleting**

Run:
```bash
grep -rn --include='*.py' -E "text2json|markdown_document_service|extract_entities_manual" . | grep -v '.codegraph' | grep -v '^./scripts/extract_entities_manual.py' | grep -v '^./markdown_document_service.py'
```
Expected: no output (no external references). If anything prints, STOP and investigate.

- [ ] **Step 2: Delete**

```bash
git rm main.py src/text2json.py markdown_document_service.py scripts/extract_entities_manual.py
# remove the now-empty src tree
find src -type f | sort   # expect: nothing
rm -rf src
```

- [ ] **Step 3: Verify nothing references `src` or `app` anymore**

Run:
```bash
grep -rn --include='*.py' -E "^(from|import) (app|src)([. ]|$)" . | grep -v '.codegraph' | grep -v '/tests/_entry.py'
```
Expected: no output. (Confirms the rename map was applied everywhere.)

- [ ] **Step 4: Smoke + codegraph + commit**

```bash
pytest -q
codegraph sync && codegraph status
git add -A && git commit -m "chore: delete verified-dead code and the obsolete src/ tree"
```

---

## Phase 3 — Final config cleanup verification

### Task 15: Confirm no setting was lost in the config merge

**Files:**
- Inspect: `ragbot/config/` (read-only verification)

- [ ] **Step 1: Diff the union of config keys against git history**

Run:
```bash
git show HEAD~12:app/core/config.py | grep -oE "^[[:space:]]+[A-Z_]+ =" | sort -u > /tmp/old_flask_keys.txt
grep -oE "^[[:space:]]+[A-Z_]+ =" ragbot/config/flask_config.py | sort -u > /tmp/new_flask_keys.txt
diff /tmp/old_flask_keys.txt /tmp/new_flask_keys.txt && echo "NO FLASK KEYS LOST"
```
Expected: `NO FLASK KEYS LOST` (only difference allowed is the intentional `config` → `_CONFIG_MAP` rename — verify that's the sole diff if any).

- [ ] **Step 2: Confirm settings/paths attributes still resolve for ingestion**

Run:
```bash
python -c "from ragbot.config import config, paths; print(config.CHUNK_SIZE, config.CHUNK_OVERLAP); print(bool(paths))"
```
Expected: prints chunk sizes and `True` with no AttributeError.

- [ ] **Step 3: No commit needed unless a fix was required.** If a key was missing, add it to `ragbot/config/flask_config.py`, re-run, then:

```bash
git add -A && git commit -m "fix: restore config key dropped during merge"
```

---

## Phase 4 — Decompose the `ChatbotService` god-object

`ragbot/chat/orchestrator.py` is ~2,364 lines. We extract in safety order: pure functions first (real TDD), then collaborator classes (smoke + targeted tests). `ChatbotService` stays as the public facade so external imports (`from ragbot.chat.orchestrator import ChatbotService`) are unchanged.

> Method line references below are from the file as moved in Task 10 (numbers shift as you edit — locate by method name).

### Task 16: Extract pure text helpers to `ragbot/utils/text.py` (TDD)

`normalize_text` and `dedupe_preserve_order` are `@staticmethod` pure functions, also re-implemented inside `ensemble.py`.

**Files:**
- Create: `ragbot/utils/text.py`
- Create: `tests/test_text_utils.py`
- Modify: `ragbot/chat/orchestrator.py`, `ragbot/retrieval/ensemble.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_text_utils.py`:

```python
from ragbot.utils.text import dedupe_preserve_order, normalize_text


def test_normalize_text_strips_and_lowercases_whitespace():
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_normalize_text_handles_empty():
    assert normalize_text("") == ""


def test_dedupe_preserve_order_keeps_first_occurrence_order():
    assert dedupe_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_dedupe_preserve_order_empty():
    assert dedupe_preserve_order([]) == []
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

Run: `pytest tests/test_text_utils.py -q`
Expected: FAIL — `ModuleNotFoundError: ragbot.utils.text`.

- [ ] **Step 3: Create `ragbot/utils/text.py` by moving the bodies verbatim**

Copy the exact bodies of `ChatbotService.normalize_text` and `ChatbotService.dedupe_preserve_order` from `orchestrator.py` into module-level functions. Adjust the normalize implementation only if the existing one differs from the test contract — if it does, KEEP the existing behavior and fix the test to match (behavior preservation beats the assumed contract). Example shape:

```python
"""Pure text helpers shared by chat orchestration and retrieval."""

import re


def normalize_text(s: str) -> str:
    # <verbatim body from ChatbotService.normalize_text>
    ...


def dedupe_preserve_order(items):
    # <verbatim body from ChatbotService.dedupe_preserve_order>
    ...
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_text_utils.py -q`
Expected: PASS (after aligning tests to the real preserved behavior if needed).

- [ ] **Step 5: Delegate from the call sites**

In `orchestrator.py`, replace the two `@staticmethod` definitions with thin delegators (so internal `self.normalize_text(...)` / `ChatbotService.normalize_text(...)` calls keep working), OR replace all internal callers with the imported function. Minimal-risk: keep the static methods as one-line delegators:

```python
from ragbot.utils.text import normalize_text as _normalize_text
from ragbot.utils.text import dedupe_preserve_order as _dedupe

class ChatbotService:
    @staticmethod
    def normalize_text(s: str) -> str:
        return _normalize_text(s)

    @staticmethod
    def dedupe_preserve_order(items):
        return _dedupe(items)
```

In `ensemble.py`, replace its private duplicate(s) with `from ragbot.utils.text import dedupe_preserve_order` and call that.

- [ ] **Step 6: Run full smoke + new tests**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 7: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: extract shared text helpers to ragbot/utils/text with tests"
```

---

### Task 17: Extract prompt building to `ragbot/chat/prompt_builder.py` (TDD)

`_build_prompt(self, question, sources, context)` and `_get_recent_context(self, history, last_n=3)` are near-pure (read inputs, no mutation of `self`).

**Files:**
- Create: `ragbot/chat/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`
- Modify: `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Read the two method bodies** in `orchestrator.py` to confirm they reference only their args (and module constants, not `self.<state>`). If `_build_prompt` reads any `self.X`, pass `X` as a parameter.

- [ ] **Step 2: Write failing tests**

Create `tests/test_prompt_builder.py`:

```python
from ragbot.chat.prompt_builder import build_prompt, get_recent_context


def test_get_recent_context_returns_last_n():
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    out = get_recent_context(history, last_n=2)
    assert "q4" in out and "q3" in out
    assert "q0" not in out


def test_get_recent_context_empty():
    assert get_recent_context([], last_n=3) == ""


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt("What is BAS?", sources=[], context="BAS = Building Automation")
    assert "What is BAS?" in prompt
    assert "BAS = Building Automation" in prompt
```

- [ ] **Step 3: Run — expect FAIL**

Run: `pytest tests/test_prompt_builder.py -q`
Expected: FAIL — module missing.

- [ ] **Step 4: Create the module from verbatim bodies**

Move the bodies of `_build_prompt` and `_get_recent_context` into module functions `build_prompt(question, sources, context)` and `get_recent_context(history, last_n=3)`. Align the tests to the real output format if the assertions above don't match the existing template (preserve behavior; adjust assertions).

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_prompt_builder.py -q`
Expected: PASS.

- [ ] **Step 6: Delegate in orchestrator**

Replace `_build_prompt`/`_get_recent_context` method bodies with delegators:

```python
from ragbot.chat.prompt_builder import build_prompt, get_recent_context

class ChatbotService:
    def _build_prompt(self, question, sources, context):
        return build_prompt(question, sources, context)

    def _get_recent_context(self, history, last_n=3):
        return get_recent_context(history, last_n)
```

- [ ] **Step 7: Full suite + codegraph + commit**

```bash
pytest -q
codegraph sync && codegraph status
git add -A && git commit -m "refactor: extract prompt building to ragbot/chat/prompt_builder with tests"
```

---

### Task 18: Extract classification into `ragbot/chat/classification.py` (collaborator class)

`_classify_question_type`, `_fallback_question_classification`, `_analyze_heading_and_rewrite` form question-understanding. `_fallback_question_classification` is rule-based and unit-testable; the others may call the genai client and heading context.

**Files:**
- Create: `ragbot/chat/classification.py`
- Create: `tests/test_classification.py`
- Modify: `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Read the three methods** and list every `self.X` they touch (e.g. `self._get_genai_client()`, `self._get_heading_context()`, logging helpers). These become constructor dependencies of a `QuestionClassifier`.

- [ ] **Step 2: Write a failing test for the pure fallback path**

Create `tests/test_classification.py`:

```python
from ragbot.chat.classification import QuestionClassifier


def test_fallback_classification_returns_dict_with_type():
    # genai_client/heading_provider not needed for the rule-based fallback
    clf = QuestionClassifier(genai_client=None, heading_provider=lambda **k: {})
    result = clf.fallback_classify("Giá của thiết bị BAS là bao nhiêu?")
    assert isinstance(result, dict)
    assert "question_type" in result or "type" in result
```

(Adjust the asserted key to the real key returned by `_fallback_question_classification` — read it first.)

- [ ] **Step 3: Run — expect FAIL**

Run: `pytest tests/test_classification.py -q`
Expected: FAIL — module missing.

- [ ] **Step 4: Build `QuestionClassifier`**

Create `ragbot/chat/classification.py` with a class whose methods are the verbatim bodies, with `self._get_genai_client()` → `self._genai_client`, `self._get_heading_context(...)` → `self._heading_provider(...)`, and logging calls → injected `self._log` (default no-op). Signature:

```python
class QuestionClassifier:
    def __init__(self, genai_client=None, heading_provider=None, logger=None):
        self._genai_client = genai_client
        self._heading_provider = heading_provider or (lambda **k: {})
        self._log = logger or (lambda *a, **k: None)

    def classify(self, question, ...):       # was _classify_question_type
        ...
    def fallback_classify(self, question):    # was _fallback_question_classification
        ...
    def analyze_and_rewrite(self, question, ...):  # was _analyze_heading_and_rewrite
        ...
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_classification.py -q`
Expected: PASS.

- [ ] **Step 6: Wire into orchestrator**

In `ChatbotService.__init__`, construct `self._classifier = QuestionClassifier(genai_client=self._get_genai_client(), heading_provider=self._get_heading_context, logger=self._log_classification_result)` (use the real dep accessors). Replace the three method bodies with delegators to `self._classifier`. Keep method names/signatures identical so internal callers in `ask_question` are unchanged.

- [ ] **Step 7: Full suite + codegraph + commit**

```bash
pytest -q
codegraph sync && codegraph status
git add -A && git commit -m "refactor: extract QuestionClassifier from chatbot orchestrator"
```

---

### Task 19: Extract session/history into `ragbot/chat/session.py` (collaborator class)

`get_or_create_session`, `_update_session_active_headings`, `_update_session_active_entity`, `_add_to_history_db` manage session + history via `DatabaseService`.

**Files:**
- Create: `ragbot/chat/session.py`
- Modify: `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Identify deps** — these methods use `self.db_service` (a `DatabaseService`) and possibly logging. Confirm by reading them.

- [ ] **Step 2: Build `SessionManager`**

Create `ragbot/chat/session.py`:

```python
class SessionManager:
    """Chat-session lifecycle and history persistence."""

    def __init__(self, db_service, logger=None):
        self._db = db_service
        self._log = logger or (lambda *a, **k: None)

    def get_or_create_session(self, ...):           # verbatim body, self.db_service -> self._db
        ...
    def update_active_headings(self, ...):           # was _update_session_active_headings
        ...
    def update_active_entity(self, ...):             # was _update_session_active_entity
        ...
    def add_to_history(self, ...):                   # was _add_to_history_db
        ...
```

- [ ] **Step 3: Wire into orchestrator** — in `__init__`, `self._sessions = SessionManager(self.db_service, logger=self._log_process_step)`. Replace the four method bodies with delegators of identical signature.

- [ ] **Step 4: Smoke — these need DB, so import-level guard only**

Run: `python -c "from ragbot.chat.session import SessionManager; print('ok')"`
Expected: `ok`.

Run: `pytest -q`
Expected: all pass (smoke imports the orchestrator, which now constructs `SessionManager`).

- [ ] **Step 5: codegraph + commit**

```bash
codegraph sync && codegraph status
git add -A && git commit -m "refactor: extract SessionManager from chatbot orchestrator"
```

---

### Task 20: Extract source formatting + process-logging into `ragbot/chat/formatting.py`

`_process_single_source`, `_convert_sources_to_dict` shape retrieval results; the `_log_*` helpers are cross-cutting logging.

**Files:**
- Create: `ragbot/chat/formatting.py`
- Create: `tests/test_formatting.py`
- Modify: `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Read `_convert_sources_to_dict`** — it transforms a list of source objects to dicts; unit-testable with a small fake source.

- [ ] **Step 2: Write a failing test**

Create `tests/test_formatting.py`:

```python
from ragbot.chat.formatting import convert_sources_to_dict


class _FakeSource:
    def __init__(self):
        self.content = "BAS controls HVAC"
        self.metadata = {"source": "doc1.pdf"}
        self.score = 0.91


def test_convert_sources_to_dict_shape():
    out = convert_sources_to_dict([_FakeSource()])
    assert isinstance(out, list) and isinstance(out[0], dict)
```

(Align attribute names to what `_process_single_source`/`_convert_sources_to_dict` actually read — inspect first; the fake must match.)

- [ ] **Step 3: Run — expect FAIL**

Run: `pytest tests/test_formatting.py -q`
Expected: FAIL — module missing.

- [ ] **Step 4: Move the bodies**

Create `ragbot/chat/formatting.py` with `process_single_source(...)` and `convert_sources_to_dict(sources)` as module functions (verbatim bodies; replace any `self.normalize_text` with the imported helper). Move the `_log_*` helpers into the same module as functions taking a `logger` (or keep them as a tiny `ProcessLogger` class) — whichever requires fewer call-site changes; if they reference `self`, prefer a `ProcessLogger` class constructed in `__init__`.

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_formatting.py -q`
Expected: PASS.

- [ ] **Step 6: Delegate from orchestrator**, keeping method signatures stable.

- [ ] **Step 7: Full suite + codegraph + commit**

```bash
pytest -q
codegraph sync && codegraph status
git add -A && git commit -m "refactor: extract source formatting and process logging from orchestrator"
```

---

### Task 21: Trim the orchestrator + final size check

After Tasks 16–20, `orchestrator.py` should be substantially smaller (the heavy `ask_question`, `_vector_search`, `_get_question_embedding`, `_check_qa_similarity`, `_generate_answer_with_sources`, config/stats methods remain as the facade — that is acceptable for Phase 1).

**Files:**
- Modify: `ragbot/chat/orchestrator.py`

- [ ] **Step 1: Remove dead imports** left behind in `orchestrator.py` (e.g. `re`, `json` if no longer used after extractions). Run:

```bash
python -m pyflakes ragbot/chat/orchestrator.py || pip install pyflakes && python -m pyflakes ragbot/chat/orchestrator.py
```
Expected: no "imported but unused" for the modules you removed. Fix any reported unused imports.

- [ ] **Step 2: Record the new line count**

Run: `wc -l ragbot/chat/orchestrator.py`
Expected: meaningfully below 2,364 (the extracted modules account for the difference). No hard target — this is informational.

- [ ] **Step 3: Full suite + codegraph + commit**

```bash
pytest -q
codegraph sync && codegraph status
git add -A && git commit -m "refactor: trim chatbot orchestrator after god-object decomposition"
```

---

## Phase 5 — Final Verification & Handoff

### Task 22: End-to-end verification

- [ ] **Step 1: Full test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 2: codegraph integrity — no dangling references**

Run: `codegraph sync && codegraph status`
Expected: `✓ Index is up to date`; file count reflects the new tree (no `app/`, no `src/`).

Run: `grep -rn --include='*.py' -E "^(from|import) (app|src)([. ]|$)" . | grep -v '.codegraph' | grep -v 'tests/_entry.py'`
Expected: no output.

- [ ] **Step 3: Manual app boot (requires DB available)**

Run: `python run.py` in one shell; in another:
```bash
curl -s http://localhost:5000/api | head
curl -s -X POST http://localhost:5000/api/chatbot/ask -H 'Content-Type: application/json' -d '{"question":"Các thiết bị BAS gồm những gì?"}' | head -c 400
```
Expected: `/api` returns the JSON index; the chatbot endpoint returns a normal answer payload (same shape as before the restructure). If DB/LLM creds are unavailable in this environment, record that this manual step is deferred to a machine that has them, and rely on the smoke suite + codegraph as the gate.

- [ ] **Step 4: Update README structure references**

In `README.md`, update the "Architecture & Core Components" section paths (`app/services/...`, `src/...`) to the new `ragbot/...` paths. This is documentation only — no code.

- [ ] **Step 5: Final commit**

```bash
git add -A && git commit -m "docs: update README architecture paths to ragbot/ layout"
```

- [ ] **Step 6: Final codegraph sync (per user requirement)**

Run: `codegraph sync && codegraph status`
Expected: `✓ Index is up to date`. **This satisfies the standing requirement to re-run codegraph after updates.**

---

## Self-Review Notes (already reconciled against the spec)

- **§3 target structure:** realized, with the documented refinement that config is a `ragbot/config/` *package* (not a single file) to preserve the dual `Config`/`get_config` + `config`/`paths` interfaces. `db/` added for `database_service` (spec showed it under `db/`).
- **§4 delete list:** Task 14 deletes exactly `main.py`, `src/text2json.py`, `markdown_document_service.py`, `scripts/extract_entities_manual.py`. Verified-live files (`extractor`, `metadata_ranker`, `simple_vector_store`) are *moved*, not deleted (Tasks 7–8).
- **§5 execution sequence:** Phase 0–1 = scaffold + moves; Phase 2 = delete; Phase 3 = config verify (config itself consolidated early in Task 3 for import-safety — a deliberate, documented reordering from spec step 4); Phase 4 = god-object split (last, as required); Phase 5 = final verify.
- **§6 verification:** smoke suite (Task 2 onward) + `codegraph sync` after every structural task + manual boot (Task 22).
- **Placeholder scan:** every code step shows real code or a verbatim-move instruction with the exact target; refactor-move tasks explicitly state "move the body verbatim" because reproducing 400-line existing method bodies in the plan would be noise, not signal.
- **Type/name consistency:** `create_app`, `register_blueprints`, `init_extensions`, `get_chatbot_service`, `ChatbotService`, `DatabaseService`, `EnsembleRetrieverService`, `ModelPatternAnalysisService`, blueprint vars (`document_bp`, `chatbot_bp`, `health_bp`, `admin_bp`) and URL prefixes (`/api/health`, `/api/documents`, `/api/chatbot`) are used identically across tasks.
