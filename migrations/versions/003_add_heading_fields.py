"""Add hierarchical heading fields to document_chunks

Revision ID: 003_add_heading_fields
Revises: 002_update_chunk_schema
Create Date: 2025-09-22

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "003_add_heading_fields"
down_revision = "002_update_chunk_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns for hierarchical headings
    op.add_column(
        "document_chunks", sa.Column("heading_id", sa.String(length=100), nullable=True)
    )
    op.add_column(
        "document_chunks",
        sa.Column("heading_title", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("heading_parent_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "document_chunks", sa.Column("heading_level", sa.Integer(), nullable=True)
    )
    op.add_column(
        "document_chunks",
        sa.Column("heading_value", sa.String(length=1000), nullable=True),
    )

    # Optional: create index to speed up queries by heading
    op.create_index("ix_document_chunks_heading_id", "document_chunks", ["heading_id"])
    op.create_index(
        "ix_document_chunks_heading_parent_id", "document_chunks", ["heading_parent_id"]
    )
    op.create_index(
        "ix_document_chunks_heading_level", "document_chunks", ["heading_level"]
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index("ix_document_chunks_heading_level", table_name="document_chunks")
    op.drop_index("ix_document_chunks_heading_parent_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_heading_id", table_name="document_chunks")

    # Drop columns
    op.drop_column("document_chunks", "heading_value")
    op.drop_column("document_chunks", "heading_level")
    op.drop_column("document_chunks", "heading_parent_id")
    op.drop_column("document_chunks", "heading_title")
    op.drop_column("document_chunks", "heading_id")
