"""add WQ candidate funnel stage events

Revision ID: 019
Revises: 018
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wq_research_stage_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("account", sa.String(20), nullable=False),
        sa.Column("lineage_id", sa.String(40), nullable=False),
        sa.Column("parent_lineage_id", sa.String(40), nullable=True),
        sa.Column("source_run_id", sa.String(100), nullable=True),
        sa.Column("stage", sa.String(40), nullable=False),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("failure_stage", sa.String(30), nullable=True),
        sa.Column("failure_reason", sa.String(60), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("account", "lineage_id", "parent_lineage_id", "source_run_id", "stage", "outcome", "created_at"):
        op.create_index(f"ix_wq_research_stage_events_{column}", "wq_research_stage_events", [column])
    op.create_index(
        "ix_wq_stage_events_account_stage",
        "wq_research_stage_events",
        ["account", "stage", "outcome"],
    )
    op.create_index(
        "ix_wq_stage_events_lineage_stage",
        "wq_research_stage_events",
        ["lineage_id", "stage"],
    )


def downgrade() -> None:
    op.drop_table("wq_research_stage_events")
