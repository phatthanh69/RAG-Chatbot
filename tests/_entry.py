"""Single indirection point for the app factory import path.

Updated exactly once during the restructure (app factory move).
Keeping every test pointed here means the move flips one line, not many.
"""

# CURRENT layout (pre-move). Task 12 changes this line to:
#   from ragbot.app import create_app
from app import create_app  # noqa: F401

# Expected blueprint URL prefixes — behavior contract, must not change.
EXPECTED_URL_PREFIXES = ["/api/health", "/api/documents", "/api/chatbot"]
