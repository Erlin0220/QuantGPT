"""add WQ candidate ACTIVE probability calibration

Revision ID: 021
Revises: 020
Create Date: 2026-08-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_candidates", sa.Column("active_probability", sa.Float(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("confidence_tier", sa.String(1), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("probability_support", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("wq_research_candidates", sa.Column("probability_provenance", sa.String(200), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("calibration_details", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_research_candidates", "calibration_details")
    op.drop_column("wq_research_candidates", "probability_provenance")
    op.drop_column("wq_research_candidates", "probability_support")
    op.drop_column("wq_research_candidates", "confidence_tier")
    op.drop_column("wq_research_candidates", "active_probability")
