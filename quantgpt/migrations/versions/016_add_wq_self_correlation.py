"""persist BRAIN self-correlation on WQ candidates

Revision ID: 016
Revises: 015
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_research_candidates", sa.Column("self_correlation", sa.Float(), nullable=True))
    op.add_column("wq_research_candidates", sa.Column("sc_status", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_research_candidates", "sc_status")
    op.drop_column("wq_research_candidates", "self_correlation")
