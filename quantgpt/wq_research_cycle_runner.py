"""Repo-local WorldQuant research-cycle runner.

The runner owns lifecycle control and persistence for one outer research cycle.
It deliberately does not generate Alpha hypotheses or call an LLM: callers must
supply Skill-reviewed candidates for each batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
            .order_by(TaskModel.updated_at.desc())
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


@contextmanager
def _runner_process_lock(account: str, *, budget_minutes: int):
    runtime_dir = Path(__file__).resolve().parents[1] / ".scratch" / "wq-research-cycle" / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / f"{account}.lock"
    token = f"{os.getpid()}:{uuid.uuid4().hex}:{time.time()}"
    stale_seconds = max(900, int(budget_minutes) * 60 + 600)
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age_seconds <= stale_seconds:
                raise ResearchCycleBusyError(f"local research runner already active for account={account}")
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


def _planner_context(memory: dict[str, Any], policy: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
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
    policy = dict(_run_coro_sync(get_submission_policy_status(account)))
    memory = load_research_memory_sync(account)
    memory["inventory"] = policy.get("inventory") or {}
    memory["submission"] = policy
    control = build_research_control_tower(memory, policy)
    active_session = _run_coro_sync(get_active_first_session(account))
    progress = research_cycle_progress(cycle)
    return {
        "ok": True,
        "status": _cycle_decision(cycle, policy),
        "cycle_id": cycle_id,
        "research_cycle": cycle,
        "progress": progress,
        "active_first_session": active_session,
        "planner_context": _planner_context(memory, policy, control),
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


def _load_skill_candidates(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("skill_candidates") or payload.get("candidates") or []
    if not isinstance(payload, list):
        raise ValueError("batch file must contain a JSON list or {skill_candidates:[...]}")
    return [item for item in payload if isinstance(item, dict)]


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

    run_batch = subparsers.add_parser("run-batch")
    run_batch.add_argument("--cycle-id", required=True)
    run_batch.add_argument("--batch-file", required=True)
    run_batch.add_argument("--target-simulations", type=int, default=_DEFAULT_TARGET_SIMULATIONS)
    run_batch.add_argument("--budget-minutes", type=int, default=_DEFAULT_BUDGET_MINUTES)
    run_batch.add_argument("--goal", default="maximize robust low-correlation WorldQuant candidates")
    run_batch.add_argument("--tag")
    run_batch.add_argument("--region", default="USA")
    run_batch.add_argument("--universe", default="TOP3000")
    run_batch.add_argument("--delay", type=int, default=1)
    run_batch.add_argument("--decay", type=int, default=0)
    run_batch.add_argument("--neutralization", default="SUBINDUSTRY")
    run_batch.add_argument("--truncation", type=float, default=0.08)
    run_batch.add_argument("--max-simulations", type=int, default=20)
    run_batch.add_argument("--family-count", type=int, default=3)
    run_batch.add_argument("--min-sharpe", type=float, default=1.25)
    run_batch.add_argument("--min-fitness", type=float, default=1.0)
    run_batch.add_argument(
        "--submission-deferred",
        action="store_true",
        help="Continue research only after the caller explicitly deferred an otherwise-prioritized submission/reconciliation action.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        cycle_id = args.cycle_id or generate_research_cycle_id()
        output = research_cycle_snapshot(
            account=args.account,
            cycle_id=cycle_id,
            create=True,
            target_simulations=args.target_simulations,
            budget_minutes=args.budget_minutes,
        )
    elif args.command == "status":
        output = research_cycle_snapshot(account=args.account, cycle_id=args.cycle_id)
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
