"""Admin / analytics blueprint — placeholder for the multi-tenant/analytics roadmap.

Intentionally empty of business logic in Phase 1. Register it in ragbot/api/__init__.py
when the first admin endpoint is added.
"""

from flask import Blueprint

admin_bp = Blueprint("admin", __name__)
