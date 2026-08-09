"""add structured WQ research failure taxonomy

Revision ID: 017
Revises: 016
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_trials", sa.Column("failure_stage", sa.String(30), nullable=True))
    op.add_column("wq_research_trials", sa.Column("failure_reason", sa.String(60), nullable=True))
    op.add_column("wq_research_trials", sa.Column("failure_reasons", sa.JSON(), nullable=True))
    op.add_column("wq_research_trials", sa.Column("failure_evidence", sa.JSON(), nullable=True))
    op.create_index("ix_wq_research_trials_failure_stage", "wq_research_trials", ["failure_stage"])
    op.create_index("ix_wq_research_trials_failure_reason", "wq_research_trials", ["failure_reason"])


def downgrade() -> None:
    op.drop_index("ix_wq_research_trials_failure_reason", table_name="wq_research_trials")
    op.drop_index("ix_wq_research_trials_failure_stage", table_name="wq_research_trials")
    op.drop_column("wq_research_trials", "failure_evidence")
    op.drop_column("wq_research_trials", "failure_reasons")
    op.drop_column("wq_research_trials", "failure_reason")
    op.drop_column("wq_research_trials", "failure_stage")
