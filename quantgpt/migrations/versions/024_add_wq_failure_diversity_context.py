"""add WQ failure diagnosis and diversity context

Revision ID: 024
Revises: 023
Create Date: 2026-08-11
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("wq_research_trials", "wq_research_candidates"):
        op.add_column(table, sa.Column("failure_signature", sa.JSON(), nullable=True))
        op.add_column(table, sa.Column("diversity_case", sa.JSON(), nullable=True))


def downgrade() -> None:
    for table in ("wq_research_candidates", "wq_research_trials"):
        op.drop_column(table, "diversity_case")
        op.drop_column(table, "failure_signature")
