"""add WQ candidate validation and live-field metadata

Revision ID: 014
Revises: 013
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_candidates", sa.Column("data_fields", sa.JSON(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("dataset_id", sa.String(100), nullable=True))
    op.add_column(
        "wq_research_candidates",
        sa.Column("validation_status", sa.String(30), server_default="research_pass", nullable=False),
    )
    op.add_column("wq_research_candidates", sa.Column("robustness_score", sa.Float(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("novelty_score", sa.Float(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("validation_details", sa.JSON(), nullable=True))
    op.add_column("wq_research_trials", sa.Column("data_fields", sa.JSON(), nullable=True))
    op.add_column("wq_research_trials", sa.Column("dataset_id", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_research_trials", "dataset_id")
    op.drop_column("wq_research_trials", "data_fields")
    op.drop_column("wq_research_candidates", "validation_details")
    op.drop_column("wq_research_candidates", "novelty_score")
    op.drop_column("wq_research_candidates", "robustness_score")
    op.drop_column("wq_research_candidates", "validation_status")
    op.drop_column("wq_research_candidates", "dataset_id")
    op.drop_column("wq_research_candidates", "data_fields")
