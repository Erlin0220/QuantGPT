"""add restart-safe WQ research lineage metadata

Revision ID: 018
Revises: 017
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    ("lineage_id", sa.String(40)),
    ("parent_lineage_id", sa.String(40)),
    ("operator_pattern", sa.Text()),
    ("operators", sa.JSON()),
    ("mutation_reason", sa.Text()),
    ("planner_strategy", sa.String(100)),
    ("allocation_cell", sa.String(200)),
    ("source_run_id", sa.String(100)),
)


def upgrade() -> None:
    for table in ("wq_research_trials", "wq_research_candidates"):
        for name, column_type in _COLUMNS:
            op.add_column(table, sa.Column(name, column_type, nullable=True))
        op.create_index(f"ix_{table}_lineage_id", table, ["lineage_id"])
        op.create_index(f"ix_{table}_parent_lineage_id", table, ["parent_lineage_id"])


def downgrade() -> None:
    for table in ("wq_research_candidates", "wq_research_trials"):
        op.drop_index(f"ix_{table}_parent_lineage_id", table_name=table)
        op.drop_index(f"ix_{table}_lineage_id", table_name=table)
        for name, _ in reversed(_COLUMNS):
            op.drop_column(table, name)
