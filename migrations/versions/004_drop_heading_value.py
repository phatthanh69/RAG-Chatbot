"""Drop heading_value column from document_chunks

Revision ID: 004_drop_heading_value
Revises: 003_add_heading_fields
Create Date: 2025-09-22

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "004_drop_heading_value"
down_revision = "003_add_heading_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop heading_value if exists (Postgres)
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS heading_value")


def downgrade() -> None:
    # Recreate heading_value column if downgrading
    op.add_column(
        "document_chunks",
        sa.Column("heading_value", sa.String(length=1000), nullable=True),
    )
