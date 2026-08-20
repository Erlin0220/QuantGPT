"""add WQ autonomous research memory and candidate lineage

Revision ID: 013
Revises: 012
Create Date: 2026-08-08
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_candidates", sa.Column("family", sa.String(50), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("hypothesis", sa.Text(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("parent_expression", sa.Text(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("generation", sa.Integer(), server_default="0", nullable=False))
    op.add_column("wq_research_candidates", sa.Column("mutation_type", sa.String(50), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("structure_signature", sa.Text(), nullable=True))

    op.create_table(
        "wq_research_trials",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("account", sa.String(20), server_default="primary", nullable=False),
        sa.Column("alpha_id", sa.String(50), nullable=True),
        sa.Column("expression", sa.Text(), nullable=False),
        sa.Column("expression_normalized", sa.Text(), nullable=False),
        sa.Column("family", sa.String(50), server_default="unknown", nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("parent_expression", sa.Text(), nullable=True),
        sa.Column("generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("mutation_type", sa.String(50), nullable=True),
        sa.Column("status", sa.String(30), server_default="researched", nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("fitness", sa.Float(), nullable=True),
        sa.Column("returns", sa.Float(), nullable=True),
        sa.Column("turnover", sa.Float(), nullable=True),
        sa.Column("self_correlation_failed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("mutation_targets", sa.JSON(), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=True),
        sa.Column("tag", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_wq_research_trials_account", "wq_research_trials", ["account"])
    op.create_index("ix_wq_research_trials_alpha_id", "wq_research_trials", ["alpha_id"])
    op.create_index("ix_wq_research_trials_family", "wq_research_trials", ["family"])
    op.create_index("ix_wq_research_trials_status", "wq_research_trials", ["status"])
    op.create_index("ix_wq_trials_account_family", "wq_research_trials", ["account", "family"])
    op.create_index("ix_wq_trials_account_expr", "wq_research_trials", ["account", "expression_normalized"])
    op.create_index("ix_wq_trials_account_status", "wq_research_trials", ["account", "status"])


def downgrade() -> None:
    op.drop_table("wq_research_trials")
    op.drop_column("wq_research_candidates", "structure_signature")
    op.drop_column("wq_research_candidates", "mutation_type")
    op.drop_column("wq_research_candidates", "generation")
    op.drop_column("wq_research_candidates", "parent_expression")
    op.drop_column("wq_research_candidates", "hypothesis")
    op.drop_column("wq_research_candidates", "family")
