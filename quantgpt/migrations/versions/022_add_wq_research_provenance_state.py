"""add WQ research provenance state

Revision ID: 022
Revises: 021
Create Date: 2026-08-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("wq_research_trials", "wq_research_candidates"):
        op.add_column(table, sa.Column("dataset_category", sa.String(100), nullable=True))
        op.add_column(table, sa.Column("provenance_state", sa.String(20), nullable=True))
        op.add_column(table, sa.Column("provenance_reason", sa.String(200), nullable=True))


def downgrade() -> None:
    for table in ("wq_research_candidates", "wq_research_trials"):
        op.drop_column(table, "provenance_reason")
        op.drop_column(table, "provenance_state")
        op.drop_column(table, "dataset_category")
