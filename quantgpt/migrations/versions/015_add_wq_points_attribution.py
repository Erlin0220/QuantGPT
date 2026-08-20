"""add conservative WQ points attribution metadata

Revision ID: 015
Revises: 014
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wq_submission_attempts", sa.Column("attributed_points_share", sa.Float(), nullable=True))
    op.add_column("wq_submission_attempts", sa.Column("attribution_confidence", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("wq_submission_attempts", "attribution_confidence")
    op.drop_column("wq_submission_attempts", "attributed_points_share")
