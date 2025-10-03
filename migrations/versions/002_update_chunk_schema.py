"""Update document_chunks schema to use entity, section, is_heading

Revision ID: 002_update_chunk_schema
Revises: 001_add_tokenized_content
Create Date: 2025-09-11

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "002_update_chunk_schema"
down_revision = "001_add_tokenized_content"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use Postgres-safe IF NOT EXISTS for idempotency
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS entity VARCHAR(500)"
    )
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section VARCHAR(500)"
    )
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS is_heading BOOLEAN"
    )

    # Set default values for existing data (safe even if column existed)
    op.execute("UPDATE document_chunks SET is_heading = FALSE WHERE is_heading IS NULL")

    # Make is_heading NOT NULL if column exists
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("document_chunks")}
    if "is_heading" in cols:
        op.alter_column("document_chunks", "is_heading", nullable=False)

    # Drop old columns if they exist
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS is_section")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS section_title")


def downgrade() -> None:
    # Add back old columns
    op.add_column(
        "document_chunks",
        sa.Column("is_section", sa.Boolean(), nullable=False, default=False),
    )
    op.add_column(
        "document_chunks", sa.Column("section_title", sa.String(500), nullable=True)
    )

    # Drop new columns
    op.drop_column("document_chunks", "entity")
    op.drop_column("document_chunks", "section")
    op.drop_column("document_chunks", "is_heading")
