"""Shared task lifecycle helpers for MCP tools.

MCP calls can either execute inline or enqueue long-running background work.
All task records share the same in-memory store and DB persistence used by the
HTTP API so callers can submit a job, return immediately, and poll later.
"""

from __future__ import annotations

import logging
import os
import time
import uuid as uuid_mod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .auth import _DEV_USER_ID
from .task_store import (
    MAX_ACTIVE_TASKS,
    persist_task_to_db,
    persist_task_to_db_async,
    sanitize_task_response,
    tasks,
    tasks_lock,
)

logger = logging.getLogger(__name__)

_DEV_USER_ID_STR = str(_DEV_USER_ID)
_FINAL_STATUSES = {"completed", "failed", "cancelled", "iteration_completed"}
_MCP_BACKGROUND_WORKERS = max(
    1,
    min(
        MAX_ACTIVE_TASKS,
        int(os.environ.get("QUANTGPT_MCP_BACKGROUND_WORKERS", "8")),
    ),
)
_background_executor = ThreadPoolExecutor(
    max_workers=_MCP_BACKGROUND_WORKERS,
    thread_name_prefix="mcp-task",
)


class MCPTaskCapacityError(RuntimeError):
    """Raised when accepting another task would exceed the shared task cap."""


async def start_mcp_task(
    task_type: str,
    expression: str | None,
    params: dict,
    *,
    status: str = "running",
) -> str:
    task_id = uuid_mod.uuid4().hex[:12]
    params = {**params, "source": "mcp"}
    task = {
        "task_id": task_id,
        "user_id": _DEV_USER_ID_STR,
        "session_id": None,
        "status": status,
        "task_type": task_type,
        "cancelled": False,
        "params": params,
        "expression": expression,
        "created_at": time.time(),
    }
    with tasks_lock:
        active = sum(1 for existing in tasks.values() if existing.get("status") not in _FINAL_STATUSES)
        if active >= MAX_ACTIVE_TASKS:
            raise MCPTaskCapacityError(f"Too many active tasks ({active}/{MAX_ACTIVE_TASKS}); retry later")
        tasks[task_id] = task

    try:
        await persist_task_to_db_async(task_id, _DEV_USER_ID_STR, task)
    except Exception as e:
        logger.error(f"[{task_id}] MCP task initial persist error: {e}")

    return task_id


def update_mcp_task(task_id: str, *, persist: bool = False, **updates: Any) -> dict | None:
    """Update an MCP task from either the event loop or a worker thread."""
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return None
        if task.get("cancelled") and "status" in updates:
            updates = {k: v for k, v in updates.items() if k != "status"}
        task.update(updates)
        snapshot = dict(task)

    if persist:
        try:
            persist_task_to_db(task_id, _DEV_USER_ID_STR, snapshot)
        except Exception as e:
            logger.error(f"[{task_id}] MCP task update persist error: {e}")
    return snapshot


def is_mcp_task_cancelled(task_id: str) -> bool:
    """Return the cooperative cancellation flag for a background task."""
    with tasks_lock:
        task = tasks.get(task_id)
        return bool(task and task.get("cancelled"))


def _apply_completion(
    task_id: str,
    result: dict | None = None,
    error: str | None = None,
    expression: str | None = None,
) -> dict:
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            logger.warning(f"[{task_id}] task evicted from memory, reconstructing for DB persist")
            task = {
                "task_id": task_id,
                "user_id": _DEV_USER_ID_STR,
                "session_id": None,
                "status": "failed" if error else "completed",
                "task_type": "mcp_unknown",
                "cancelled": False,
                "params": {"source": "mcp", "note": "reconstructed after memory eviction"},
                "expression": expression,
                "created_at": time.time(),
            }
            tasks[task_id] = task

        task["completed_at"] = time.time()
        if task.get("cancelled"):
            task["status"] = "cancelled"
            task.setdefault("error", "任务已取消")
        else:
            task["status"] = "failed" if error else "completed"
            if error:
                task["error"] = error
            else:
                task.pop("error", None)
                task["progress"] = 100
                if task.get("progress_total") is not None:
                    task["progress_current"] = task["progress_total"]
                task["progress_message"] = "任务完成"
        if expression:
            task["expression"] = expression
        if result is not None:
            task["result"] = result
        return dict(task)


async def complete_mcp_task(
    task_id: str,
    result: dict | None = None,
    error: str | None = None,
    expression: str | None = None,
):
    task = _apply_completion(task_id, result, error, expression)
    try:
        await persist_task_to_db_async(task_id, _DEV_USER_ID_STR, task)
    except Exception as e:
        logger.error(f"[{task_id}] MCP task persist error: {e}")


def complete_mcp_task_sync(
    task_id: str,
    result: dict | None = None,
    error: str | None = None,
    expression: str | None = None,
):
    """Complete an MCP task from a background worker thread."""
    task = _apply_completion(task_id, result, error, expression)
    try:
        persist_task_to_db(task_id, _DEV_USER_ID_STR, task)
    except Exception as e:
        logger.error(f"[{task_id}] MCP background task persist error: {e}")


def start_mcp_background_task(
    task_id: str,
    worker: Callable[..., dict],
    *args: Any,
    expression: str | None = None,
    **kwargs: Any,
) -> Future:
    """Run a long MCP operation in the bounded shared worker executor."""

    def _runner() -> None:
        update_mcp_task(task_id, status="running", started_at=time.time())
        try:
            result = worker(task_id, *args, **kwargs)
            error = None
            if isinstance(result, dict) and result.get("ok") is False:
                error = str(result.get("error") or "background task failed")
            complete_mcp_task_sync(task_id, result, error, expression)
        except Exception as e:
            logger.exception(f"[{task_id}] MCP background worker crashed")
            complete_mcp_task_sync(task_id, None, str(e), expression)

    return _background_executor.submit(_runner)


def shutdown_mcp_background_executor(*, wait: bool = False) -> None:
    """Stop accepting MCP background work during application shutdown."""
    _background_executor.shutdown(wait=wait, cancel_futures=True)


async def cancel_mcp_task(task_id: str) -> dict | None:
    """Request cooperative cancellation and persist it immediately."""
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return None
        if task.get("status") in _FINAL_STATUSES:
            safe = {k: v for k, v in task.items() if k != "user_id"}
            return sanitize_task_response(dict(safe))
        task["cancelled"] = True
        task["status"] = "cancelled"
        task["error"] = "任务已取消"
        task["completed_at"] = time.time()
        snapshot = dict(task)

    try:
        await persist_task_to_db_async(task_id, _DEV_USER_ID_STR, snapshot)
    except Exception as e:
        logger.error(f"[{task_id}] MCP task cancellation persist error: {e}")

    safe = {k: v for k, v in snapshot.items() if k != "user_id"}
    return sanitize_task_response(dict(safe))


async def get_mcp_task_snapshot(task_id: str) -> dict | None:
    """Read task status from memory, falling back to the persisted DB record."""
    with tasks_lock:
        task = tasks.get(task_id)
        if task:
            safe = {k: v for k, v in task.items() if k != "user_id"}
            return sanitize_task_response(dict(safe))

    from sqlalchemy import select

    from .db import _get_session_factory
    from .models import Task as TaskModel

    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(TaskModel).where(
                TaskModel.id == task_id,
                TaskModel.user_id == uuid_mod.UUID(_DEV_USER_ID_STR),
            )
        )
        db_task = result.scalar_one_or_none()
        if not db_task:
            return None

        response = {
            "task_id": db_task.id,
            "status": db_task.status,
            "task_type": db_task.task_type,
            "params": db_task.params,
            "expression": db_task.expression,
            "result": db_task.result,
            "error": db_task.error,
            "created_at": db_task.created_at.isoformat() if db_task.created_at else None,
            "updated_at": db_task.updated_at.isoformat() if db_task.updated_at else None,
        }
        return sanitize_task_response(response)
