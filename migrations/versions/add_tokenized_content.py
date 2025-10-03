"""Add tokenized_content column to document_chunks

Revision ID: 001_add_tokenized_content
Revises:
Create Date: 2025-09-10

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "001_add_tokenized_content"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add tokenized_content column to document_chunks table
    op.add_column(
        "document_chunks", sa.Column("tokenized_content", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    # Remove tokenized_content column from document_chunks table
    op.drop_column("document_chunks", "tokenized_content")
