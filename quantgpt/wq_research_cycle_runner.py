"""Repo-local WorldQuant research-cycle runner.

The runner owns lifecycle control and persistence for one outer research cycle.
Production flows still expect callers to supply Skill-reviewed candidates.  The
explicit ``local-llm-poc`` command is a bounded adapter test that lets the local
OpenCode Go / DeepSeek model generate the next Skill-reviewed batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psutil
from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from .auth import _DEV_USER_ID
from .db import _get_session_factory
from .models import Task as TaskModel
from .wq_active_session import ensure_active_first_session, get_active_first_session, record_research_transition
from .wq_autonomous_research import (
    REQUIRED_WQ_SKILL_CHAIN,
    run_autonomous_research,
    validate_skill_candidate_contract,
)
from .wq_brain_client import get_client, is_configured
from .wq_control_tower import build_research_control_tower
from .wq_local_candidate_generator import LocalCandidateGenerationError, generate_local_skill_batch
from .wq_research_memory import load_research_memory_sync, record_research_trials_sync
from .wq_submission_policy import (
    _run_coro_sync,
    get_submission_policy_status,
    record_research_candidates_sync,
)

_TASK_TYPE = "wq_research_cycle"
_RUNNING_TASK_STATUS = "iterating"
_DEFAULT_TARGET_SIMULATIONS = 100
_DEFAULT_BUDGET_MINUTES = 50
_BEIJING = ZoneInfo("Asia/Shanghai")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def generate_research_cycle_id(now: datetime | None = None) -> str:
    current = now or _now_utc()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(_BEIJING).strftime("%Y%m%d-%H%M%S")


def _task_id(account: str, cycle_id: str) -> str:
    digest = hashlib.sha1(f"{account}:{cycle_id}".encode(), usedforsecurity=False).hexdigest()[:10]
    return f"rc{digest}"


def new_research_cycle_state(
    *,
    account: str,
    cycle_id: str,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
    now: datetime | None = None,
) -> dict[str, Any]:
    started = now or _now_utc()
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    started = started.astimezone(timezone.utc)
    target = max(1, int(target_simulations))
    budget = max(1, int(budget_minutes))
    return {
        "version": 1,
        "account": account,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "started_at": started.isoformat(),
        "deadline_at": (started + timedelta(minutes=budget)).isoformat(),
        "target_simulations": target,
        "budget_minutes": budget,
        "counters": {
            "iterations": 0,
            "simulations": 0,
            "total_brain_simulations": 0,
            "candidates": 0,
            "failed": 0,
            "invalid": 0,
        },
        "batches": [],
        "batch_inflight": False,
        "batch_started_at": None,
        "next_action": "generate_skill_candidate_batch",
        "stop_reason": None,
        "last_error": None,
        "updated_at": started.isoformat(),
    }


def set_research_cycle_inflight(
    state: dict[str, Any],
    inflight: bool,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(state))
    current = now or _now_utc()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    updated["batch_inflight"] = bool(inflight)
    updated["batch_started_at"] = current.isoformat() if inflight else None
    updated["updated_at"] = current.isoformat()
    return updated


def research_cycle_progress(state: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    current = now or _now_utc()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    started = _parse_time(state.get("started_at")) or current
    deadline = _parse_time(state.get("deadline_at")) or (
        started + timedelta(minutes=max(1, int(state.get("budget_minutes") or _DEFAULT_BUDGET_MINUTES)))
    )
    completed_at = _parse_time(state.get("completed_at"))
    persisted_stop_reason = str(state.get("stop_reason") or "")
    if completed_at is not None:
        current = min(current, completed_at)
    elif str(state.get("status") or "").upper() == "COMPLETED" and persisted_stop_reason == "budget_exhausted":
        # Legacy cycles completed before ``completed_at`` was persisted should
        # still report the configured budget duration rather than aging forever.
        current = min(current, deadline)
    counters = dict(state.get("counters") or {})
    simulations = max(0, int(counters.get("simulations") or 0))
    target = max(1, int(state.get("target_simulations") or _DEFAULT_TARGET_SIMULATIONS))
    elapsed_seconds = max(0.0, (current - started).total_seconds())
    remaining_seconds = max(0.0, (deadline - current).total_seconds())
    if simulations >= target:
        stop_reason = "target_reached"
    elif current >= deadline:
        stop_reason = "budget_exhausted"
    elif str(state.get("status") or "").upper() == "FAILED":
        stop_reason = str(state.get("stop_reason") or "hard_failure")
    else:
        stop_reason = None
    return {
        "simulations": simulations,
        "target_simulations": target,
        "remaining_simulations": max(0, target - simulations),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 2),
        "remaining_minutes": round(remaining_seconds / 60.0, 2),
        "stop_reason": stop_reason,
        "should_stop": stop_reason is not None,
    }


def apply_research_batch_to_cycle(
    state: dict[str, Any],
    result: dict[str, Any],
    *,
    source_run_id: str,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(state))
    started = started_at or _now_utc()
    ended = ended_at or _now_utc()
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    summary = dict(result.get("summary") or {})
    simulated = max(0, int(summary.get("simulated") or 0))
    total_simulations = max(simulated, int(summary.get("total_simulations") or simulated))
    candidate_count = len(result.get("candidates") or [])
    failed_count = len(result.get("failed") or [])
    invalid_count = len(result.get("invalid") or [])
    counters = dict(updated.get("counters") or {})
    if simulated > 0:
        counters["iterations"] = int(counters.get("iterations") or 0) + 1
    counters["simulations"] = int(counters.get("simulations") or 0) + simulated
    counters["total_brain_simulations"] = int(counters.get("total_brain_simulations") or 0) + total_simulations
    counters["candidates"] = int(counters.get("candidates") or 0) + candidate_count
    counters["failed"] = int(counters.get("failed") or 0) + failed_count
    counters["invalid"] = int(counters.get("invalid") or 0) + invalid_count
    updated["counters"] = counters
    batches = list(updated.get("batches") or [])
    batches.append(
        {
            "source_run_id": source_run_id,
            "tag": result.get("tag"),
            "started_at": started.astimezone(timezone.utc).isoformat(),
            "ended_at": ended.astimezone(timezone.utc).isoformat(),
            "duration_seconds": round(max(0.0, (ended - started).total_seconds()), 3),
            "simulations": simulated,
            "total_brain_simulations": total_simulations,
            "candidates": candidate_count,
            "failed": failed_count,
            "invalid": invalid_count,
            "error": error,
        }
    )
    updated["batches"] = batches[-50:]
    updated["batch_inflight"] = False
    updated["batch_started_at"] = None
    updated["last_error"] = error
    if candidate_count > 0:
        updated["next_action"] = "advance_candidate_to_submission_gate"
    elif simulated > 0:
        updated["next_action"] = "failure_diagnosis_then_generate_next_skill_batch"
    else:
        updated["next_action"] = "diagnose_execution_then_generate_replacement_batch"
    updated["updated_at"] = ended.astimezone(timezone.utc).isoformat()
    progress = research_cycle_progress(updated, now=ended)
    if progress["stop_reason"]:
        updated["status"] = "COMPLETED" if progress["stop_reason"] in {"target_reached", "budget_exhausted"} else "FAILED"
        updated["stop_reason"] = progress["stop_reason"]
        updated["completed_at"] = ended.astimezone(timezone.utc).isoformat()
        updated["next_action"] = "cycle_complete"
    return updated


async def _get_research_cycle_task(account: str, cycle_id: str) -> TaskModel | None:
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(TaskModel).where(TaskModel.id == _task_id(account, cycle_id)))
        return result.scalar_one_or_none()


async def _get_latest_research_cycle_task(account: str) -> TaskModel | None:
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(TaskModel)
            .where(TaskModel.user_id == _DEV_USER_ID, TaskModel.task_type == _TASK_TYPE)
            .order_by(TaskModel.created_at.desc())
            .limit(20)
        )
        for task in result.scalars().all():
            if str((task.params or {}).get("account") or "primary") == account:
                return task
    return None


async def ensure_research_cycle(
    account: str,
    cycle_id: str,
    *,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
) -> dict[str, Any]:
    task = await _get_research_cycle_task(account, cycle_id)
    if task is not None:
        return {"task_id": task.id, **dict(task.result or {})}
    state = new_research_cycle_state(
        account=account,
        cycle_id=cycle_id,
        target_simulations=target_simulations,
        budget_minutes=budget_minutes,
    )
    task_id = _task_id(account, cycle_id)
    factory = _get_session_factory()
    async with factory() as session:
        session.add(
            TaskModel(
                id=task_id,
                user_id=_DEV_USER_ID,
                session_id=None,
                status=_RUNNING_TASK_STATUS,
                task_type=_TASK_TYPE,
                params={
                    "account": account,
                    "cycle_id": cycle_id,
                    "target_simulations": state["target_simulations"],
                    "budget_minutes": state["budget_minutes"],
                },
                result=state,
            )
        )
        await session.commit()
    return {"task_id": task_id, **state}


async def update_research_cycle(cycle_id: str, account: str, state: dict[str, Any]) -> dict[str, Any]:
    task_id = _task_id(account, cycle_id)
    clean = {key: value for key, value in dict(state).items() if key != "task_id"}
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(TaskModel).where(TaskModel.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            return {"task_id": task_id, **clean, "persistence_error": "cycle_not_found"}
        task.result = clean
        task.status = "failed" if clean.get("status") == "FAILED" else "completed" if clean.get("status") == "COMPLETED" else _RUNNING_TASK_STATUS
        task.error = clean.get("last_error") if task.status == "failed" else None
        task.updated_at = _now_utc()
        await session.commit()
    return {"task_id": task_id, **clean}


async def get_research_cycle(account: str = "primary", cycle_id: str | None = None) -> dict[str, Any] | None:
    task = await (_get_research_cycle_task(account, cycle_id) if cycle_id else _get_latest_research_cycle_task(account))
    if task is None:
        return None
    return {"task_id": task.id, **dict(task.result or {})}


def ensure_research_cycle_sync(
    account: str,
    cycle_id: str,
    *,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
) -> dict[str, Any]:
    return dict(
        _run_coro_sync(
            ensure_research_cycle(
                account,
                cycle_id,
                target_simulations=target_simulations,
                budget_minutes=budget_minutes,
            )
        )
    )


def get_research_cycle_sync(account: str = "primary", cycle_id: str | None = None) -> dict[str, Any] | None:
    value = _run_coro_sync(get_research_cycle(account, cycle_id))
    return dict(value) if value else None


def update_research_cycle_sync(cycle_id: str, account: str, state: dict[str, Any]) -> dict[str, Any]:
    return dict(_run_coro_sync(update_research_cycle(cycle_id, account, state)))


async def get_inflight_research_cycle(account: str = "primary") -> dict[str, Any] | None:
    cycle = await get_research_cycle(account)
    if not cycle or not bool(cycle.get("batch_inflight")):
        return None
    return cycle


def get_inflight_research_cycle_sync(account: str = "primary") -> dict[str, Any] | None:
    value = _run_coro_sync(get_inflight_research_cycle(account))
    return dict(value) if value else None


def _active_mcp_research_task() -> dict[str, Any] | None:
    from .mcp_task_helper import get_active_mcp_task

    value = _run_coro_sync(get_active_mcp_task("wq_research"))
    return dict(value) if value else None


class ResearchCycleBusyError(RuntimeError):
    pass


def _runner_runtime_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".scratch" / "wq-research-cycle" / ".runtime"


def _runner_lock_path(account: str) -> Path:
    return _runner_runtime_dir() / f"{account}.lock"


def _runner_lock_owner_status(lock_path: Path) -> bool | None:
    """Return True for a live owner, False for a dead/missing owner, None if unreadable."""
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        pid_text, _token, acquired_text = text.split(":", 2)
        pid = int(pid_text)
        acquired_at = float(acquired_text)
    except (TypeError, ValueError):
        return None
    try:
        process = psutil.Process(pid)
        created_at = float(process.create_time())
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return bool(psutil.pid_exists(pid))
    # A reused PID created after the lock was written is not the original owner.
    return created_at <= acquired_at + 5.0


def _recover_stale_cycle_inflight_sync(account: str, cycle: dict[str, Any]) -> dict[str, Any]:
    if not bool(cycle.get("batch_inflight")):
        return cycle
    batch_started = _parse_time(cycle.get("batch_started_at"))
    if batch_started is not None and (_now_utc() - batch_started).total_seconds() < 30:
        return cycle
    owner_status = _runner_lock_owner_status(_runner_lock_path(account))
    if owner_status is not False:
        return cycle
    recovered = set_research_cycle_inflight(cycle, False)
    recovered["last_error"] = "stale_batch_inflight_recovered_after_runner_exit"
    recovered["next_action"] = "diagnose_execution_then_generate_replacement_batch"
    return update_research_cycle_sync(str(cycle.get("cycle_id") or ""), account, recovered)


@contextmanager
def _runner_process_lock(account: str, *, budget_minutes: int):
    runtime_dir = _runner_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _runner_lock_path(account)
    token = f"{os.getpid()}:{uuid.uuid4().hex}:{time.time()}"
    stale_seconds = max(900, int(budget_minutes) * 60 + 600)
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            owner_status = _runner_lock_owner_status(lock_path)
            if owner_status is True:
                raise ResearchCycleBusyError(f"local research runner already active for account={account}")
            try:
                age_seconds = max(0.0, time.time() - lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if owner_status is None and age_seconds <= stale_seconds:
                raise ResearchCycleBusyError(f"local research runner lock owner is unknown for account={account}")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
            acquired = True
            break
    if not acquired:
        raise ResearchCycleBusyError(f"unable to acquire local research runner lock for account={account}")
    try:
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _stamp_source_run(result: dict[str, Any], source_run_id: str) -> dict[str, Any]:
    result["source_run_id"] = source_run_id
    seen_ids: set[int] = set()

    def stamp(item: object) -> None:
        if not isinstance(item, dict) or id(item) in seen_ids:
            return
        seen_ids.add(id(item))
        metadata = dict(item.get("research_meta") or {})
        metadata["source_run_id"] = source_run_id
        item["research_meta"] = metadata

    for key in ("results", "candidates", "ready_candidates", "failed", "invalid"):
        for item in result.get(key) or []:
            stamp(item)
    for generation in result.get("generations") or []:
        if not isinstance(generation, dict):
            continue
        for key in ("results", "candidates", "failed", "invalid"):
            for item in generation.get(key) or []:
                stamp(item)
    stamp(result.get("best"))
    return result


def _runner_candidate_contract_error(candidate: dict[str, Any]) -> str | None:
    error = validate_skill_candidate_contract(candidate)
    if error:
        return error
    chain = {str(value) for value in (candidate.get("skill_chain") or []) if str(value)}
    if "wq-economic-hypothesis" not in chain:
        return "skill_chain missing wq-economic-hypothesis"
    if not str(candidate.get("family") or "").strip():
        return "missing family"
    if not [value for value in (candidate.get("data_fields") or []) if str(value).strip()]:
        return "missing data_fields"
    if "knowledge_card_ids" not in candidate:
        return "missing knowledge_card_ids"
    return None


def _submission_required(policy: dict[str, Any]) -> bool:
    remaining_slots = max(0, int(policy.get("remaining_submission_slots") or 0))
    candidates = list(policy.get("submission_candidate_top") or [])
    return (
        remaining_slots > 0
        and bool(candidates)
        and not bool(policy.get("submission_frozen"))
        and not bool(policy.get("submission_reconciliation_required"))
    )


def _current_cycle_trial_evidence(memory: dict[str, Any], cycle: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    source_run_ids = {
        str(batch.get("source_run_id") or "")
        for batch in (cycle.get("batches") or [])
        if str(batch.get("source_run_id") or "")
    }
    if not source_run_ids:
        return []

    evidence: list[dict[str, Any]] = []
    for trial in memory.get("recent_trials") or []:
        if str(trial.get("source_run_id") or "") not in source_run_ids:
            continue
        evidence.append(
            {
                "source_run_id": trial.get("source_run_id"),
                "expression": trial.get("expression"),
                "family": trial.get("family"),
                "status": trial.get("status"),
                "sharpe": trial.get("sharpe"),
                "fitness": trial.get("fitness"),
                "returns": trial.get("returns"),
                "turnover": trial.get("turnover"),
                "failure_stage": trial.get("failure_stage"),
                "failure_reason": trial.get("failure_reason"),
                "failure_reasons": list(trial.get("failure_reasons") or []),
                "failure_evidence": trial.get("failure_evidence"),
                "mutation_targets": list(trial.get("mutation_targets") or []),
                "dataset_id": trial.get("dataset_id"),
                "operator_pattern": trial.get("operator_pattern"),
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def _planner_context(
    memory: dict[str, Any],
    policy: dict[str, Any],
    control: dict[str, Any],
    cycle: dict[str, Any],
) -> dict[str, Any]:
    scheduler = dict(control.get("scheduler") or {})
    campaign = dict(policy.get("active_campaign") or {})
    return {
        "required_skill_chain": ["wq-economic-hypothesis", *REQUIRED_WQ_SKILL_CHAIN],
        "research_strategy": policy.get("research_strategy"),
        "remaining_active_target": policy.get("remaining_active_target"),
        "remaining_submission_slots": policy.get("remaining_submission_slots"),
        "submission_frozen": bool(policy.get("submission_frozen")),
        "submission_reconciliation_required": bool(policy.get("submission_reconciliation_required")),
        "submission_candidates": list(policy.get("submission_candidate_top") or [])[:5],
        "inventory": policy.get("inventory") or {},
        "next_focus": scheduler.get("next_focus"),
        "selected_cells": list(scheduler.get("selected_cells") or [])[:8],
        "failure_counts": (control.get("failures") or {}).get("reason_counts") or {},
        "dominant_bottleneck_stage": (control.get("failures") or {}).get("dominant_bottleneck_stage"),
        "avoid_structure_signatures": list(campaign.get("avoid_structure_signatures") or [])[:20],
        "research_memory_positive": list((control.get("research_memory") or {}).get("positive") or [])[:12],
        "research_memory_negative": list((control.get("research_memory") or {}).get("negative") or [])[:12],
        "recent_round_audits": list(control.get("round_audits") or [])[:5],
        "current_cycle_trial_evidence": _current_cycle_trial_evidence(memory, cycle),
        "learning_maturity": memory.get("learning_maturity") or {},
    }


def _cycle_decision(cycle: dict[str, Any], policy: dict[str, Any], *, now: datetime | None = None) -> str:
    progress = research_cycle_progress(cycle, now=now)
    if progress["stop_reason"] == "target_reached":
        return "TARGET_REACHED"
    if progress["stop_reason"] == "budget_exhausted":
        return "BUDGET_EXHAUSTED"
    if progress["stop_reason"]:
        return "HARD_STOP"
    if bool(policy.get("submission_reconciliation_required")):
        return "RECONCILIATION_REQUIRED"
    if _submission_required(policy):
        return "SUBMISSION_REQUIRED"
    return "NEEDS_SKILL_BATCH"


def _effective_cycle_decision(
    cycle: dict[str, Any],
    policy: dict[str, Any],
    *,
    submission_deferred: bool = False,
    now: datetime | None = None,
) -> str:
    decision = _cycle_decision(cycle, policy, now=now)
    if submission_deferred and decision in {"SUBMISSION_REQUIRED", "RECONCILIATION_REQUIRED"}:
        return "NEEDS_SKILL_BATCH"
    return decision


def _reconcile_cycle_from_active_session_sync(
    account: str,
    cycle: dict[str, Any],
    active_session: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recover primary-Simulation progress committed before a worker interruption.

    ACTIVE-session progress is written immediately after a real research batch.
    The repo-local runner writes its richer batch ledger immediately afterwards.
    If a worker exits between those writes, the ACTIVE session is the restart-safe
    deterministic witness for primary Simulation/iteration/Candidate counts.
    Reconciliation is monotonic: it can only move runner counters upward.
    """
    if not active_session:
        return cycle
    active_cycle = active_session.get("research_cycle") or {}
    cycle_id = str(cycle.get("cycle_id") or "")
    if not cycle_id or str(active_cycle.get("cycle_id") or "") != cycle_id:
        return cycle

    counters = dict(cycle.get("counters") or {})
    runner_simulations = max(0, int(counters.get("simulations") or 0))
    runner_iterations = max(0, int(counters.get("iterations") or 0))
    runner_candidates = max(0, int(counters.get("candidates") or 0))
    active_simulations = max(0, int(active_cycle.get("simulations") or 0))
    active_iterations = max(0, int(active_cycle.get("iterations") or 0))
    active_candidates = max(0, int(active_cycle.get("candidates") or 0))
    if (
        active_simulations <= runner_simulations
        and active_iterations <= runner_iterations
        and active_candidates <= runner_candidates
    ):
        return cycle

    updated = json.loads(json.dumps(cycle))
    updated_counters = dict(updated.get("counters") or {})
    updated_counters["simulations"] = max(runner_simulations, active_simulations)
    updated_counters["iterations"] = max(runner_iterations, active_iterations)
    updated_counters["candidates"] = max(runner_candidates, active_candidates)
    updated_counters["total_brain_simulations"] = max(
        int(updated_counters.get("total_brain_simulations") or 0),
        updated_counters["simulations"],
    )
    updated["counters"] = updated_counters
    recoveries = list(updated.get("recovery_events") or [])
    recoveries.append({
        "source": "active_first_session_research_cycle",
        "recovered_at": _now_utc().isoformat(),
        "simulations_before": runner_simulations,
        "simulations_after": updated_counters["simulations"],
        "iterations_before": runner_iterations,
        "iterations_after": updated_counters["iterations"],
        "candidates_before": runner_candidates,
        "candidates_after": updated_counters["candidates"],
    })
    updated["recovery_events"] = recoveries[-20:]
    updated["updated_at"] = _now_utc().isoformat()
    progress = research_cycle_progress(updated)
    if progress["stop_reason"]:
        updated["status"] = "COMPLETED" if progress["stop_reason"] in {"target_reached", "budget_exhausted"} else "FAILED"
        updated["stop_reason"] = progress["stop_reason"]
        if not updated.get("completed_at"):
            if progress["stop_reason"] == "budget_exhausted":
                updated["completed_at"] = str(updated.get("deadline_at") or _now_utc().isoformat())
            else:
                updated["completed_at"] = _now_utc().isoformat()
        updated["next_action"] = "cycle_complete"
    return update_research_cycle_sync(cycle_id, account, updated)


def _finalize_cycle_if_stopped_sync(account: str, cycle: dict[str, Any]) -> dict[str, Any]:
    progress = research_cycle_progress(cycle)
    stop_reason = progress.get("stop_reason")
    if not stop_reason or str(cycle.get("status") or "").upper() in {"COMPLETED", "FAILED"}:
        return cycle
    updated = json.loads(json.dumps(cycle))
    updated["status"] = "COMPLETED" if stop_reason in {"target_reached", "budget_exhausted"} else "FAILED"
    updated["stop_reason"] = stop_reason
    if not updated.get("completed_at"):
        updated["completed_at"] = (
            str(updated.get("deadline_at") or _now_utc().isoformat())
            if stop_reason == "budget_exhausted"
            else _now_utc().isoformat()
        )
    updated["next_action"] = "cycle_complete"
    updated["updated_at"] = _now_utc().isoformat()
    return update_research_cycle_sync(str(updated.get("cycle_id") or ""), account, updated)


def research_cycle_snapshot(
    *,
    account: str = "primary",
    cycle_id: str | None = None,
    create: bool = False,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
) -> dict[str, Any]:
    if create:
        cycle_id = cycle_id or generate_research_cycle_id()
        cycle = ensure_research_cycle_sync(
            account,
            cycle_id,
            target_simulations=target_simulations,
            budget_minutes=budget_minutes,
        )
    else:
        cycle = get_research_cycle_sync(account, cycle_id)
        if cycle is None:
            return {"ok": False, "status": "research_cycle_not_found", "cycle_id": cycle_id}
        cycle_id = str(cycle.get("cycle_id") or cycle_id or "")
    cycle = _recover_stale_cycle_inflight_sync(account, cycle)
    active_session = _run_coro_sync(get_active_first_session(account))
    cycle = _reconcile_cycle_from_active_session_sync(account, cycle, active_session)
    cycle = _finalize_cycle_if_stopped_sync(account, cycle)
    policy = dict(_run_coro_sync(get_submission_policy_status(account)))
    memory = load_research_memory_sync(account)
    memory["inventory"] = policy.get("inventory") or {}
    memory["submission"] = policy
    control = build_research_control_tower(memory, policy)
    progress = research_cycle_progress(cycle)
    return {
        "ok": True,
        "status": "BATCH_INFLIGHT" if bool(cycle.get("batch_inflight")) else _cycle_decision(cycle, policy),
        "cycle_id": cycle_id,
        "research_cycle": cycle,
        "progress": progress,
        "active_first_session": active_session,
        "planner_context": _planner_context(memory, policy, control, cycle),
    }


def _run_skill_batch_locked(
    *,
    account: str,
    cycle_id: str,
    skill_candidates: list[dict[str, Any]],
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
    goal: str = "maximize robust low-correlation WorldQuant candidates",
    tag: str | None = None,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_simulations: int = 20,
    family_count: int = 3,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    submission_deferred: bool = False,
) -> dict[str, Any]:
    cycle = ensure_research_cycle_sync(
        account,
        cycle_id,
        target_simulations=target_simulations,
        budget_minutes=budget_minutes,
    )
    policy = dict(_run_coro_sync(get_submission_policy_status(account)))
    decision = _effective_cycle_decision(cycle, policy, submission_deferred=submission_deferred)
    if decision != "NEEDS_SKILL_BATCH":
        snapshot = research_cycle_snapshot(account=account, cycle_id=cycle_id)
        snapshot["batch_executed"] = False
        return snapshot

    candidates = list(skill_candidates or [])
    if not candidates:
        return {
            "ok": False,
            "status": "skill_generation_required",
            "cycle_id": cycle_id,
            "required_skill_chain": ["wq-economic-hypothesis", *REQUIRED_WQ_SKILL_CHAIN],
        }
    contract_errors = [
        {"index": index, "error": error}
        for index, candidate in enumerate(candidates)
        if (error := _runner_candidate_contract_error(candidate))
    ]
    if contract_errors:
        return {
            "ok": False,
            "status": "skill_candidate_contract_invalid",
            "cycle_id": cycle_id,
            "details": contract_errors,
        }
    if not is_configured(account):
        return {"ok": False, "status": "wq_not_configured", "cycle_id": cycle_id}

    request_id = f"research-cycle-{uuid.uuid4().hex[:12]}"
    source_run_id = f"rcr-{uuid.uuid4().hex[:12]}"
    batch_started = _now_utc()
    cycle_deadline = _parse_time(cycle.get("deadline_at"))

    def cycle_budget_exhausted() -> bool:
        return bool(cycle_deadline and _now_utc() >= cycle_deadline)
    client = get_client(account, request_id=request_id)
    params = {
        "account": account,
        "skill_candidates": candidates[:40],
        "chatgpt_expressions": [],
        "allow_deterministic_fallback": False,
        "goal": goal,
        "tag": tag or f"research-cycle-{cycle_id}",
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "max_simulations": max(4, min(40, int(max_simulations))),
        "generations": 1,
        "family_count": max(1, min(4, int(family_count))),
        "min_sharpe": min_sharpe,
        "min_fitness": min_fitness,
        "research_cycle_id": cycle_id,
        "request_id": request_id,
        "submission_deferred": bool(submission_deferred),
    }
    try:
        if not client.authenticate():
            return {
                "ok": False,
                "status": "brain_authentication_failed",
                "cycle_id": cycle_id,
                "error": client.last_error or {"error": "WQ BRAIN authentication failed"},
                "batch_executed": False,
            }

        policy_before = dict(_run_coro_sync(get_submission_policy_status(account)))
        active_session = _run_coro_sync(ensure_active_first_session(account, policy_before))
        active_session_id = (active_session or {}).get("session_id")
        memory = load_research_memory_sync(account)
        memory["inventory"] = policy_before.get("inventory") or {}
        memory["submission"] = policy_before
        memory["research_learning"] = build_research_control_tower(
            memory,
            policy_before,
            research_budget=params["max_simulations"],
        )
        result = run_autonomous_research(
            client,
            memory=memory,
            skill_candidates=params["skill_candidates"],
            chatgpt_expressions=[],
            allow_deterministic_fallback=False,
            goal=goal,
            tag=params["tag"],
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            max_simulations=params["max_simulations"],
            generations=1,
            family_count=params["family_count"],
            min_sharpe=min_sharpe,
            min_fitness=min_fitness,
            check_cancelled=cycle_budget_exhausted,
        )
        _stamp_source_run(result, source_run_id)
        result["research_trials_saved"] = record_research_trials_sync(
            account,
            result,
            default_family="",
            hypothesis=goal,
            tag=result.get("tag"),
        )
        result["candidate_queue_saved"] = record_research_candidates_sync(
            account,
            result.get("candidates", []),
            settings=result.get("settings", {}),
            tag=result.get("tag"),
        )
        if active_session_id:
            result["active_first_session"] = _run_coro_sync(
                record_research_transition(active_session_id, result, params)
            )
        batch_ended = _now_utc()
        updated_cycle = apply_research_batch_to_cycle(
            cycle,
            result,
            source_run_id=source_run_id,
            started_at=batch_started,
            ended_at=batch_ended,
        )
        cycle = update_research_cycle_sync(cycle_id, account, updated_cycle)
    except Exception as exc:
        batch_ended = _now_utc()
        failed_cycle = apply_research_batch_to_cycle(
            cycle,
            {},
            source_run_id=source_run_id,
            started_at=batch_started,
            ended_at=batch_ended,
            error=f"{type(exc).__name__}: {exc}",
        )
        update_research_cycle_sync(cycle_id, account, failed_cycle)
        return {
            "ok": False,
            "status": "batch_failed",
            "cycle_id": cycle_id,
            "error": f"{type(exc).__name__}: {exc}",
            "batch_executed": True,
            "research_cycle": failed_cycle,
        }
    finally:
        client.close()

    snapshot = research_cycle_snapshot(account=account, cycle_id=cycle_id)
    snapshot.update(
        {
            "batch_executed": True,
            "source_run_id": source_run_id,
            "batch_result": {
                "ok": result.get("ok"),
                "tag": result.get("tag"),
                "summary": result.get("summary") or {},
                "best": result.get("best"),
                "candidates": list(result.get("candidates") or [])[:10],
                "failed": list(result.get("failed") or [])[:20],
                "invalid": list(result.get("invalid") or [])[:20],
                "skill_plan_rejections": list(result.get("skill_plan_rejections") or [])[:20],
                "adaptive_allocation": result.get("adaptive_allocation") or {},
            },
        }
    )
    return snapshot


def run_skill_batch(
    *,
    account: str,
    cycle_id: str,
    skill_candidates: list[dict[str, Any]],
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
    goal: str = "maximize robust low-correlation WorldQuant candidates",
    tag: str | None = None,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_simulations: int = 20,
    family_count: int = 3,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    submission_deferred: bool = False,
) -> dict[str, Any]:
    try:
        with _runner_process_lock(account, budget_minutes=budget_minutes):
            active_mcp = _active_mcp_research_task()
            if active_mcp:
                return {
                    "ok": False,
                    "status": "research_singleflight_busy",
                    "cycle_id": cycle_id,
                    "active_task": active_mcp,
                    "batch_executed": False,
                }
            cycle = ensure_research_cycle_sync(
                account,
                cycle_id,
                target_simulations=target_simulations,
                budget_minutes=budget_minutes,
            )
            inflight_cycle = set_research_cycle_inflight(cycle, True)
            update_research_cycle_sync(cycle_id, account, inflight_cycle)
            active_mcp = _active_mcp_research_task()
            if active_mcp:
                update_research_cycle_sync(cycle_id, account, set_research_cycle_inflight(inflight_cycle, False))
                return {
                    "ok": False,
                    "status": "research_singleflight_busy",
                    "cycle_id": cycle_id,
                    "active_task": active_mcp,
                    "batch_executed": False,
                }
            try:
                return _run_skill_batch_locked(
                    account=account,
                    cycle_id=cycle_id,
                    skill_candidates=skill_candidates,
                    target_simulations=target_simulations,
                    budget_minutes=budget_minutes,
                    goal=goal,
                    tag=tag,
                    region=region,
                    universe=universe,
                    delay=delay,
                    decay=decay,
                    neutralization=neutralization,
                    truncation=truncation,
                    max_simulations=max_simulations,
                    family_count=family_count,
                    min_sharpe=min_sharpe,
                    min_fitness=min_fitness,
                    submission_deferred=submission_deferred,
                )
            finally:
                latest = get_research_cycle_sync(account, cycle_id) or inflight_cycle
                if bool(latest.get("batch_inflight")):
                    update_research_cycle_sync(cycle_id, account, set_research_cycle_inflight(latest, False))
    except ResearchCycleBusyError as exc:
        return {
            "ok": False,
            "status": "research_singleflight_busy",
            "cycle_id": cycle_id,
            "error": str(exc),
            "batch_executed": False,
        }


def _background_python_executable() -> str:
    executable = Path(sys.executable)
    if os.name == "nt":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return str(executable)


def _batch_worker_command(
    *,
    account: str,
    cycle_id: str,
    batch_file: str,
    target_simulations: int,
    budget_minutes: int,
    goal: str,
    tag: str | None,
    region: str,
    universe: str,
    delay: int,
    decay: int,
    neutralization: str,
    truncation: float,
    max_simulations: int,
    family_count: int,
    min_sharpe: float,
    min_fitness: float,
    submission_deferred: bool,
) -> list[str]:
    command = [
        _background_python_executable(),
        "-m",
        "quantgpt.wq_research_cycle_runner",
        "--account",
        account,
        "execute-batch",
        "--cycle-id",
        cycle_id,
        "--batch-file",
        str(Path(batch_file).resolve()),
        "--target-simulations",
        str(target_simulations),
        "--budget-minutes",
        str(budget_minutes),
        "--goal",
        goal,
        "--region",
        region,
        "--universe",
        universe,
        "--delay",
        str(delay),
        "--decay",
        str(decay),
        "--neutralization",
        neutralization,
        "--truncation",
        str(truncation),
        "--max-simulations",
        str(max_simulations),
        "--family-count",
        str(family_count),
        "--min-sharpe",
        str(min_sharpe),
        "--min-fitness",
        str(min_fitness),
    ]
    if tag:
        command.extend(["--tag", tag])
    if submission_deferred:
        command.append("--submission-deferred")
    return command


def enqueue_skill_batch(
    *,
    account: str,
    cycle_id: str,
    batch_file: str,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
    goal: str = "maximize robust low-correlation WorldQuant candidates",
    tag: str | None = None,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_simulations: int = 20,
    family_count: int = 3,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    submission_deferred: bool = False,
) -> dict[str, Any]:
    candidates = _load_skill_candidates(batch_file)
    contract_errors = [
        {"index": index, "error": error}
        for index, candidate in enumerate(candidates)
        if (error := _runner_candidate_contract_error(candidate))
    ]
    if not candidates:
        return {"ok": False, "status": "skill_generation_required", "cycle_id": cycle_id}
    if contract_errors:
        return {
            "ok": False,
            "status": "skill_candidate_contract_invalid",
            "cycle_id": cycle_id,
            "details": contract_errors,
        }

    try:
        with _runner_process_lock(account, budget_minutes=budget_minutes):
            cycle = ensure_research_cycle_sync(
                account,
                cycle_id,
                target_simulations=target_simulations,
                budget_minutes=budget_minutes,
            )
            cycle = _recover_stale_cycle_inflight_sync(account, cycle)
            if bool(cycle.get("batch_inflight")):
                return {
                    "ok": True,
                    "status": "BATCH_INFLIGHT",
                    "cycle_id": cycle_id,
                    "coalesced": True,
                    "research_cycle": cycle,
                }
            active_mcp = _active_mcp_research_task()
            if active_mcp:
                return {
                    "ok": False,
                    "status": "research_singleflight_busy",
                    "cycle_id": cycle_id,
                    "active_task": active_mcp,
                    "batch_executed": False,
                }
            policy = dict(_run_coro_sync(get_submission_policy_status(account)))
            decision = _effective_cycle_decision(
                cycle,
                policy,
                submission_deferred=submission_deferred,
            )
            if decision != "NEEDS_SKILL_BATCH":
                snapshot = research_cycle_snapshot(account=account, cycle_id=cycle_id)
                snapshot["batch_executed"] = False
                return snapshot

            reserved = set_research_cycle_inflight(cycle, True)
            reserved["batch_request_file"] = str(Path(batch_file).resolve())
            reserved["next_action"] = "wait_for_batch_completion"
            reserved = update_research_cycle_sync(cycle_id, account, reserved)
            runtime_dir = _runner_runtime_dir()
            runtime_dir.mkdir(parents=True, exist_ok=True)
            log_path = runtime_dir / f"{cycle_id}-worker.log"
            command = _batch_worker_command(
                account=account,
                cycle_id=cycle_id,
                batch_file=batch_file,
                target_simulations=target_simulations,
                budget_minutes=budget_minutes,
                goal=goal,
                tag=tag,
                region=region,
                universe=universe,
                delay=delay,
                decay=decay,
                neutralization=neutralization,
                truncation=truncation,
                max_simulations=max_simulations,
                family_count=family_count,
                min_sharpe=min_sharpe,
                min_fitness=min_fitness,
                submission_deferred=submission_deferred,
            )
            creationflags = 0
            popen_kwargs: dict[str, Any] = {}
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            else:
                popen_kwargs["start_new_session"] = True
            try:
                with log_path.open("ab") as log_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=Path(__file__).resolve().parents[1],
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=log_handle,
                        close_fds=True,
                        creationflags=creationflags,
                        **popen_kwargs,
                    )
            except Exception as exc:
                failed = set_research_cycle_inflight(reserved, False)
                failed["last_error"] = f"worker_spawn_failed:{type(exc).__name__}:{exc}"
                failed["next_action"] = "diagnose_execution_then_generate_replacement_batch"
                update_research_cycle_sync(cycle_id, account, failed)
                return {
                    "ok": False,
                    "status": "worker_spawn_failed",
                    "cycle_id": cycle_id,
                    "error": str(exc),
                }
            reserved["worker_pid"] = process.pid
            reserved["worker_log"] = str(log_path)
            update_research_cycle_sync(cycle_id, account, reserved)
            return {
                "ok": True,
                "status": "BATCH_INFLIGHT",
                "cycle_id": cycle_id,
                "worker_pid": process.pid,
                "worker_log": str(log_path),
                "coalesced": False,
            }
    except ResearchCycleBusyError:
        # Duplicate launch attempts should coalesce onto the already-reserved batch
        # rather than causing a scheduled task to fail while the real worker runs.
        for _ in range(10):
            cycle = get_research_cycle_sync(account, cycle_id)
            if cycle and bool(cycle.get("batch_inflight")):
                return {
                    "ok": True,
                    "status": "BATCH_INFLIGHT",
                    "cycle_id": cycle_id,
                    "coalesced": True,
                    "research_cycle": cycle,
                }
            time.sleep(0.1)
        return {
            "ok": False,
            "status": "research_singleflight_busy",
            "cycle_id": cycle_id,
            "batch_executed": False,
        }


def _load_skill_candidates(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("skill_candidates") or payload.get("candidates") or []
    if not isinstance(payload, list):
        raise ValueError("batch file must contain a JSON list or {skill_candidates:[...]}")
    return [item for item in payload if isinstance(item, dict)]


def run_local_llm_poc(
    *,
    account: str = "primary",
    cycle_id: str | None = None,
    target_simulations: int = 12,
    budget_minutes: int = 15,
    max_batches: int = 3,
    batch_size: int = 2,
    max_simulations_per_batch: int = 4,
    model: str | None = None,
    thinking: str = "disabled",
    reasoning_effort: str = "max",
    goal: str = "maximize robust low-correlation WorldQuant candidates",
) -> dict[str, Any]:
    """Run a bounded research-only loop with local LLM candidate generation.

    This deliberately never performs formal submission.  Submission/reconciliation
    priority is deferred only for this PoC so candidate generation → BRAIN
    Simulation → failure evidence can be exercised end to end.
    """
    resolved_cycle_id = cycle_id or generate_research_cycle_id()
    snapshot = research_cycle_snapshot(
        account=account,
        cycle_id=resolved_cycle_id,
        create=True,
        target_simulations=target_simulations,
        budget_minutes=budget_minutes,
    )
    if not snapshot.get("ok"):
        return snapshot

    poc_batches: list[dict[str, Any]] = []
    limit = max(1, min(5, int(max_batches)))
    for batch_index in range(1, limit + 1):
        snapshot = research_cycle_snapshot(account=account, cycle_id=resolved_cycle_id)
        if not snapshot.get("ok"):
            break
        progress = dict(snapshot.get("progress") or {})
        if progress.get("should_stop") or snapshot.get("status") == "BATCH_INFLIGHT":
            break

        planner_context = dict(snapshot.get("planner_context") or {})
        try:
            generation = generate_local_skill_batch(
                planner_context,
                batch_size=batch_size,
                model=model,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
        except LocalCandidateGenerationError as exc:
            return {
                "ok": False,
                "status": "local_llm_generation_failed",
                "cycle_id": resolved_cycle_id,
                "error": str(exc),
                "poc_batches": poc_batches,
                "research_cycle": snapshot.get("research_cycle"),
                "progress": progress,
            }

        candidates = list(generation.get("skill_candidates") or [])
        contract_errors = [
            {"index": index, "error": error}
            for index, candidate in enumerate(candidates)
            if (error := _runner_candidate_contract_error(candidate))
        ]
        if contract_errors:
            return {
                "ok": False,
                "status": "local_llm_candidate_contract_invalid",
                "cycle_id": resolved_cycle_id,
                "details": contract_errors,
                "generation": {key: value for key, value in generation.items() if key != "skill_candidates"},
                "poc_batches": poc_batches,
            }

        batch_result = run_skill_batch(
            account=account,
            cycle_id=resolved_cycle_id,
            skill_candidates=candidates,
            target_simulations=target_simulations,
            budget_minutes=budget_minutes,
            goal=goal,
            max_simulations=max(4, min(20, int(max_simulations_per_batch))),
            family_count=3,
            submission_deferred=True,
        )
        poc_batches.append(
            {
                "batch_index": batch_index,
                "generation": {key: value for key, value in generation.items() if key != "skill_candidates"},
                "generated_candidates": [
                    {"expression": item.get("expression"), "family": item.get("family")}
                    for item in candidates
                ],
                "run_status": batch_result.get("status"),
                "source_run_id": batch_result.get("source_run_id"),
                "summary": (batch_result.get("batch_result") or {}).get("summary") or {},
                "progress": batch_result.get("progress") or {},
            }
        )
        snapshot = batch_result
        if not batch_result.get("ok") or not batch_result.get("batch_executed"):
            break
        if (batch_result.get("progress") or {}).get("should_stop"):
            break

    final_snapshot = research_cycle_snapshot(account=account, cycle_id=resolved_cycle_id)
    return {
        "ok": bool(final_snapshot.get("ok")),
        "status": final_snapshot.get("status"),
        "cycle_id": resolved_cycle_id,
        "model": model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash",
        "thinking": thinking,
        "reasoning_effort": reasoning_effort if thinking == "enabled" else None,
        "formal_submission": False,
        "poc_batches": poc_batches,
        "research_cycle": final_snapshot.get("research_cycle"),
        "progress": final_snapshot.get("progress"),
        "planner_context": final_snapshot.get("planner_context"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantGPT repo-local WorldQuant research-cycle runner")
    parser.add_argument("--account", default="primary")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--cycle-id")
    start.add_argument("--target-simulations", type=int, default=_DEFAULT_TARGET_SIMULATIONS)
    start.add_argument("--budget-minutes", type=int, default=_DEFAULT_BUDGET_MINUTES)

    status = subparsers.add_parser("status")
    status.add_argument("--cycle-id")

    local_poc = subparsers.add_parser("local-llm-poc")
    local_poc.add_argument("--cycle-id")
    local_poc.add_argument("--target-simulations", type=int, default=12)
    local_poc.add_argument("--budget-minutes", type=int, default=15)
    local_poc.add_argument("--max-batches", type=int, default=3)
    local_poc.add_argument("--batch-size", type=int, default=2)
    local_poc.add_argument("--max-simulations-per-batch", type=int, default=4)
    local_poc.add_argument("--model")
    local_poc.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    local_poc.add_argument("--reasoning-effort", choices=["high", "max"], default="max")
    local_poc.add_argument("--goal", default="maximize robust low-correlation WorldQuant candidates")

    def add_batch_arguments(batch_parser: argparse.ArgumentParser) -> None:
        batch_parser.add_argument("--cycle-id", required=True)
        batch_parser.add_argument("--batch-file", required=True)
        batch_parser.add_argument("--target-simulations", type=int, default=_DEFAULT_TARGET_SIMULATIONS)
        batch_parser.add_argument("--budget-minutes", type=int, default=_DEFAULT_BUDGET_MINUTES)
        batch_parser.add_argument("--goal", default="maximize robust low-correlation WorldQuant candidates")
        batch_parser.add_argument("--tag")
        batch_parser.add_argument("--region", default="USA")
        batch_parser.add_argument("--universe", default="TOP3000")
        batch_parser.add_argument("--delay", type=int, default=1)
        batch_parser.add_argument("--decay", type=int, default=0)
        batch_parser.add_argument("--neutralization", default="SUBINDUSTRY")
        batch_parser.add_argument("--truncation", type=float, default=0.08)
        batch_parser.add_argument("--max-simulations", type=int, default=20)
        batch_parser.add_argument("--family-count", type=int, default=3)
        batch_parser.add_argument("--min-sharpe", type=float, default=1.25)
        batch_parser.add_argument("--min-fitness", type=float, default=1.0)
        batch_parser.add_argument(
            "--submission-deferred",
            action="store_true",
            help="Continue research only after the caller explicitly deferred an otherwise-prioritized submission/reconciliation action.",
        )

    add_batch_arguments(subparsers.add_parser("run-batch"))
    add_batch_arguments(subparsers.add_parser("execute-batch", help=argparse.SUPPRESS))
    return parser


def start_research_cycle_snapshot(
    *,
    account: str = "primary",
    cycle_id: str | None = None,
    target_simulations: int = _DEFAULT_TARGET_SIMULATIONS,
    budget_minutes: int = _DEFAULT_BUDGET_MINUTES,
) -> dict[str, Any]:
    with _runner_process_lock(account, budget_minutes=budget_minutes):
        if cycle_id is None:
            latest = get_research_cycle_sync(account)
            if latest is not None:
                latest = _recover_stale_cycle_inflight_sync(account, latest)
                latest = _finalize_cycle_if_stopped_sync(account, latest)
                if str(latest.get("status") or "").upper() == "RUNNING":
                    snapshot = research_cycle_snapshot(account=account, cycle_id=str(latest.get("cycle_id") or ""))
                    snapshot["coalesced_existing_cycle"] = True
                    return snapshot
            cycle_id = generate_research_cycle_id()
        return research_cycle_snapshot(
            account=account,
            cycle_id=cycle_id,
            create=True,
            target_simulations=target_simulations,
            budget_minutes=budget_minutes,
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        output = start_research_cycle_snapshot(
            account=args.account,
            cycle_id=args.cycle_id,
            target_simulations=args.target_simulations,
            budget_minutes=args.budget_minutes,
        )
    elif args.command == "status":
        output = research_cycle_snapshot(account=args.account, cycle_id=args.cycle_id)
    elif args.command == "local-llm-poc":
        output = run_local_llm_poc(
            account=args.account,
            cycle_id=args.cycle_id,
            target_simulations=args.target_simulations,
            budget_minutes=args.budget_minutes,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
            max_simulations_per_batch=args.max_simulations_per_batch,
            model=args.model,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            goal=args.goal,
        )
    elif args.command == "run-batch":
        output = enqueue_skill_batch(
            account=args.account,
            cycle_id=args.cycle_id,
            batch_file=args.batch_file,
            target_simulations=args.target_simulations,
            budget_minutes=args.budget_minutes,
            goal=args.goal,
            tag=args.tag,
            region=args.region,
            universe=args.universe,
            delay=args.delay,
            decay=args.decay,
            neutralization=args.neutralization,
            truncation=args.truncation,
            max_simulations=args.max_simulations,
            family_count=args.family_count,
            min_sharpe=args.min_sharpe,
            min_fitness=args.min_fitness,
            submission_deferred=args.submission_deferred,
        )
    else:
        output = run_skill_batch(
            account=args.account,
            cycle_id=args.cycle_id,
            skill_candidates=_load_skill_candidates(args.batch_file),
            target_simulations=args.target_simulations,
            budget_minutes=args.budget_minutes,
            goal=args.goal,
            tag=args.tag,
            region=args.region,
            universe=args.universe,
            delay=args.delay,
            decay=args.decay,
            neutralization=args.neutralization,
            truncation=args.truncation,
            max_simulations=args.max_simulations,
            family_count=args.family_count,
            min_sharpe=args.min_sharpe,
            min_fitness=args.min_fitness,
            submission_deferred=args.submission_deferred,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0 if output.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
