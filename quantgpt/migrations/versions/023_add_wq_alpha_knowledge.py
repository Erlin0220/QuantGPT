"""add WQ Alpha knowledge sources and cards

Revision ID: 023
Revises: 022
Create Date: 2026-08-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wq_knowledge_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_key", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("authors", sa.JSON(), nullable=True),
        sa.Column("published_year", sa.Integer(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("access_scope", sa.String(50), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wq_knowledge_sources_source_key", "wq_knowledge_sources", ["source_key"], unique=True)
    op.create_index("ix_wq_knowledge_sources_source_type", "wq_knowledge_sources", ["source_type"])
    op.create_index("ix_wq_knowledge_sources_external_id", "wq_knowledge_sources", ["external_id"])

    op.create_table(
        "wq_knowledge_cards",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("card_key", sa.String(255), nullable=False),
        sa.Column("concept", sa.String(120), nullable=False),
        sa.Column("family", sa.String(80), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("mechanism", sa.JSON(), nullable=True),
        sa.Column("scope", sa.JSON(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("source_keys", sa.JSON(), nullable=True),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.Column("operators", sa.JSON(), nullable=True),
        sa.Column("expression_templates", sa.JSON(), nullable=True),
        sa.Column("failure_modes", sa.JSON(), nullable=True),
        sa.Column("mutation_strategies", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wq_knowledge_cards_card_key", "wq_knowledge_cards", ["card_key"], unique=True)
    op.create_index("ix_wq_knowledge_cards_concept", "wq_knowledge_cards", ["concept"])
    op.create_index("ix_wq_knowledge_cards_family", "wq_knowledge_cards", ["family"])
    op.create_index("ix_wq_knowledge_cards_status", "wq_knowledge_cards", ["status"])
    op.create_index("ix_wq_knowledge_cards_family_status", "wq_knowledge_cards", ["family", "status"])
    op.create_index("ix_wq_knowledge_cards_status_confidence", "wq_knowledge_cards", ["status", "confidence"])

    op.add_column("wq_research_trials", sa.Column("knowledge_card_ids", sa.JSON(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("knowledge_card_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_research_candidates", "knowledge_card_ids")
    op.drop_column("wq_research_trials", "knowledge_card_ids")
    op.drop_index("ix_wq_knowledge_cards_status_confidence", table_name="wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_cards_family_status", table_name="wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_cards_status", table_name="wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_cards_family", table_name="wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_cards_concept", table_name="wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_cards_card_key", table_name="wq_knowledge_cards")
    op.drop_table("wq_knowledge_cards")
    op.drop_index("ix_wq_knowledge_sources_external_id", table_name="wq_knowledge_sources")
    op.drop_index("ix_wq_knowledge_sources_source_type", table_name="wq_knowledge_sources")
    op.drop_index("ix_wq_knowledge_sources_source_key", table_name="wq_knowledge_sources")
    op.drop_table("wq_knowledge_sources")
