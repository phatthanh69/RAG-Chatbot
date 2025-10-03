"""Drop entity and section columns from document_chunks

Revision ID: 35d66f9d4ae7
Revises: 004_drop_heading_value
Create Date: 2025-09-23 09:28:15.868228

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "35d66f9d4ae7"
down_revision = "004_drop_heading_value"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop entity and section columns from document_chunks
    op.drop_column("document_chunks", "entity")
    op.drop_column("document_chunks", "section")


def downgrade() -> None:
    # Add back entity and section columns
    op.add_column(
        "document_chunks", sa.Column("entity", sa.String(length=500), nullable=True)
    )
    op.add_column(
        "document_chunks", sa.Column("section", sa.String(length=500), nullable=True)
    )
