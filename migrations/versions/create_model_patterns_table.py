# Create model patterns table migration

"""Create model_patterns table

Revision ID: create_model_patterns_table
Revises:
Create Date: 2025-09-25

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "create_model_patterns_table"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create model_patterns table"""
    op.create_table(
        "model_patterns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pattern_regex", sa.String(255), nullable=False),
        sa.Column("pattern_name", sa.String(100), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("examples", postgresql.JSONB(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=False, default=0.0),
        sa.Column("usage_count", sa.Integer(), nullable=False, default=0),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("extraction_method", sa.String(50), nullable=False, default="llm"),
        sa.Column("llm_analysis_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for performance
    op.create_index("idx_model_patterns_category", "model_patterns", ["category"])
    op.create_index("idx_model_patterns_active", "model_patterns", ["is_active"])
    op.create_index(
        "idx_model_patterns_confidence", "model_patterns", ["confidence_score"]
    )


def downgrade():
    """Drop model_patterns table"""
    op.drop_index("idx_model_patterns_confidence")
    op.drop_index("idx_model_patterns_active")
    op.drop_index("idx_model_patterns_category")
    op.drop_table("model_patterns")
