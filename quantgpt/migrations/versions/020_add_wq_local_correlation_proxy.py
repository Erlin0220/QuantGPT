"""add local WQ candidate correlation proxy evidence

Revision ID: 020
Revises: 019
Create Date: 2026-08-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_candidates", sa.Column("local_correlation", sa.Float(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("local_correlation_alpha_id", sa.String(50), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("local_correlation_samples", sa.Integer(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("local_correlation_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_research_candidates", "local_correlation_at")
    op.drop_column("wq_research_candidates", "local_correlation_samples")
    op.drop_column("wq_research_candidates", "local_correlation_alpha_id")
    op.drop_column("wq_research_candidates", "local_correlation")
