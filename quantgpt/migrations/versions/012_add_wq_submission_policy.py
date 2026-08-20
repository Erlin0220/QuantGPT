"""add WQ submission budget, candidate queue, and delayed-points ledger

Revision ID: 012
Revises: 011
Create Date: 2026-08-08
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wq_research_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("account", sa.String(20), server_default="primary", nullable=False),
        sa.Column("alpha_id", sa.String(50), nullable=False),
        sa.Column("expression", sa.Text(), nullable=False),
        sa.Column("region", sa.String(10), server_default="USA", nullable=False),
        sa.Column("universe", sa.String(20), server_default="TOP3000", nullable=False),
        sa.Column("delay", sa.Integer(), server_default="1", nullable=False),
        sa.Column("decay", sa.Integer(), server_default="0", nullable=False),
        sa.Column("neutralization", sa.String(30), server_default="SUBINDUSTRY", nullable=False),
        sa.Column("truncation", sa.Float(), server_default="0.08", nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("fitness", sa.Float(), nullable=True),
        sa.Column("returns", sa.Float(), nullable=True),
        sa.Column("turnover", sa.Float(), nullable=True),
        sa.Column("priority_score", sa.Float(), server_default="0", nullable=False),
        sa.Column("tag", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), server_default="queued", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_wq_research_candidates_account", "wq_research_candidates", ["account"])
    op.create_index("ix_wq_research_candidates_alpha_id", "wq_research_candidates", ["alpha_id"])
    op.create_index("ix_wq_candidates_account_alpha", "wq_research_candidates", ["account", "alpha_id"], unique=True)
    op.create_index(
        "ix_wq_candidates_account_status_priority",
        "wq_research_candidates",
        ["account", "status", "priority_score"],
    )

    op.create_table(
        "wq_submission_attempts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("account", sa.String(20), server_default="primary", nullable=False),
        sa.Column("alpha_id", sa.String(50), nullable=False),
        sa.Column("submission_day", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), server_default="RESERVED", nullable=False),
        sa.Column("score_state", sa.String(20), server_default="PENDING", nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("points_at_reservation", sa.Float(), nullable=True),
        sa.Column("points_status_at_reservation", sa.String(30), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_wq_submission_attempts_account", "wq_submission_attempts", ["account"])
    op.create_index("ix_wq_submission_attempts_alpha_id", "wq_submission_attempts", ["alpha_id"])
    op.create_index("ix_wq_submission_attempts_submission_day", "wq_submission_attempts", ["submission_day"])
    op.create_index("ix_wq_attempts_account_day", "wq_submission_attempts", ["account", "submission_day"])
    op.create_index(
        "ix_wq_attempts_account_alpha_day",
        "wq_submission_attempts",
        ["account", "alpha_id", "submission_day"],
        unique=True,
    )

    op.create_table(
        "wq_submission_states",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("account", sa.String(20), nullable=False),
        sa.Column("daily_budget", sa.Integer(), server_default="2", nullable=False),
        sa.Column("last_observed_points", sa.Float(), nullable=True),
        sa.Column("last_points_status", sa.String(30), nullable=True),
        sa.Column("last_settled_points", sa.Float(), nullable=True),
        sa.Column("last_settled_delta", sa.Float(), nullable=True),
        sa.Column("last_settled_submission_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("untracked_active_gap", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_wq_submission_states_account", "wq_submission_states", ["account"], unique=True)


def downgrade() -> None:
    op.drop_table("wq_submission_states")
    op.drop_table("wq_submission_attempts")
    op.drop_table("wq_research_candidates")
