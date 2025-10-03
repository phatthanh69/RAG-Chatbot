"""Merge heads for model patterns

Revision ID: fe7cf02dfd99
Revises: 35d66f9d4ae7, create_model_patterns_table
Create Date: 2025-09-25 15:38:49.030361

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fe7cf02dfd99'
down_revision = ('35d66f9d4ae7', 'create_model_patterns_table')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
