import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

import quantgpt.db as db


@pytest.mark.asyncio
async def test_file_sqlite_uses_wal_busy_timeout_and_non_static_pool(monkeypatch, tmp_path):
    database_path = tmp_path / "quantgpt-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    db._engine = None
    db._session_factory = None

    engine = db._get_engine()
    try:
        assert not isinstance(engine.sync_engine.pool, StaticPool)
        async with engine.connect() as connection:
            journal_mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()
            busy_timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar_one()
        assert str(journal_mode).lower() == "wal"
        assert busy_timeout == 30000
    finally:
        await engine.dispose()
        db._engine = None
        db._session_factory = None


@pytest.mark.asyncio
async def test_memory_sqlite_keeps_static_pool(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    db._engine = None
    db._session_factory = None

    engine = db._get_engine()
    try:
        assert isinstance(engine.sync_engine.pool, StaticPool)
        async with engine.connect() as connection:
            busy_timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar_one()
        assert busy_timeout == 30000
    finally:
        await engine.dispose()
        db._engine = None
        db._session_factory = None
