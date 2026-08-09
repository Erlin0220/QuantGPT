"""Database engine, session factory, and FastAPI dependency."""

import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./quantgpt.db")
        kwargs: dict = {"echo": False}
        if "postgresql" in url:
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10
        elif "sqlite" in url:
            from sqlalchemy.pool import StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        _engine = create_async_engine(url, **kwargs)
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def get_db():
    """FastAPI dependency that yields an async DB session."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables (dev convenience). Use Alembic for production."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_add_columns)
    logger.info("Database tables created/verified")


def _migrate_add_columns(connection):
    """Add columns that were added after initial table creation."""
    import sqlalchemy as sa
    inspector = sa.inspect(connection)
    _add_column_if_missing(
        connection, inspector, "submitted_alphas", "tag",
        "ALTER TABLE submitted_alphas ADD COLUMN tag VARCHAR(100)",
    )
    _add_column_if_missing(
        connection, inspector, "wq_submission_states", "untracked_active_gap",
        "ALTER TABLE wq_submission_states ADD COLUMN untracked_active_gap INTEGER NOT NULL DEFAULT 0",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "family",
        "ALTER TABLE wq_research_candidates ADD COLUMN family VARCHAR(50)",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "hypothesis",
        "ALTER TABLE wq_research_candidates ADD COLUMN hypothesis TEXT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "parent_expression",
        "ALTER TABLE wq_research_candidates ADD COLUMN parent_expression TEXT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "generation",
        "ALTER TABLE wq_research_candidates ADD COLUMN generation INTEGER NOT NULL DEFAULT 0",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "mutation_type",
        "ALTER TABLE wq_research_candidates ADD COLUMN mutation_type VARCHAR(50)",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "structure_signature",
        "ALTER TABLE wq_research_candidates ADD COLUMN structure_signature TEXT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "data_fields",
        "ALTER TABLE wq_research_candidates ADD COLUMN data_fields JSON",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "dataset_id",
        "ALTER TABLE wq_research_candidates ADD COLUMN dataset_id VARCHAR(100)",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "validation_status",
        "ALTER TABLE wq_research_candidates ADD COLUMN validation_status VARCHAR(30) NOT NULL DEFAULT 'research_pass'",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "robustness_score",
        "ALTER TABLE wq_research_candidates ADD COLUMN robustness_score FLOAT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "novelty_score",
        "ALTER TABLE wq_research_candidates ADD COLUMN novelty_score FLOAT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_candidates", "validation_details",
        "ALTER TABLE wq_research_candidates ADD COLUMN validation_details JSON",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_trials", "data_fields",
        "ALTER TABLE wq_research_trials ADD COLUMN data_fields JSON",
    )
    _add_column_if_missing(
        connection, inspector, "wq_research_trials", "dataset_id",
        "ALTER TABLE wq_research_trials ADD COLUMN dataset_id VARCHAR(100)",
    )
    _add_column_if_missing(
        connection, inspector, "wq_submission_attempts", "attributed_points_share",
        "ALTER TABLE wq_submission_attempts ADD COLUMN attributed_points_share FLOAT",
    )
    _add_column_if_missing(
        connection, inspector, "wq_submission_attempts", "attribution_confidence",
        "ALTER TABLE wq_submission_attempts ADD COLUMN attribution_confidence FLOAT",
    )


def _add_column_if_missing(connection, inspector, table, column, ddl):
    import sqlalchemy as sa
    if inspector.has_table(table):
        cols = [c["name"] for c in inspector.get_columns(table)]
        if column not in cols:
            connection.execute(sa.text(ddl))
            logger.info(f"Migration: added column {table}.{column}")


async def close_db():
    """Close the engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
