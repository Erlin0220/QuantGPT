"""Persistent ACTIVE-first research session state for WorldQuant workflows.

The server intentionally does not generate Alpha hypotheses.  This state machine
only enforces whether an ACTIVE-first session may stop, and persists counters
across MCP calls / ChatGPT turns in the existing Task table.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .auth import _DEV_USER_ID
from .db import _get_session_factory
from .models import Task as TaskModel
from .wq_submission_policy import submission_day

_TASK_TYPE = "wq_active_session"
_RUNNING_STATUS = "iterating"
_FINAL_STATUSES = {"completed", "failed", "cancelled", "iteration_completed"}
# Compatibility checkpoint retained for telemetry and older persisted sessions.
# A persistent ACTIVE-first session must not become stoppable merely because it
# crossed this many research calls; caller-level time/simulation budgets own
# per-run stopping, while this state machine owns ACTIVE/platform safety gates.
_DEFAULT_MAX_ITERATIONS = 4


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _route_counts(skill_candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"new_hypotheses": 0, "repairs": 0, "diversifications": 0}
    for candidate in skill_candidates:
        chain = {str(value) for value in (candidate.get("skill_chain") or []) if str(value)}
        if "wq-alpha-repair" in chain:
            counts["repairs"] += 1
        elif "wq-alpha-diversify" in chain:
            counts["diversifications"] += 1
        else:
            counts["new_hypotheses"] += 1
    return counts


def _base_counters() -> dict[str, int]:
    return {
        "iterations": 0,
        "simulations": 0,
        "repairs": 0,
        "diversifications": 0,
        "new_hypotheses": 0,
        "candidates": 0,
        "formal_submissions": 0,
        "active": 0,
        "terminal_failures": 0,
    }


def _policy_has_submission_candidate(policy: dict[str, Any]) -> bool:
    """Return whether policy already exposes a candidate that may use a daily slot."""
    candidates = policy.get("submission_candidate_top") or []
    fallback_count = max(0, _int(policy.get("fallback_submission_candidate_count")))
    raw_slots = policy.get("remaining_submission_slots")
    remaining_slots = (
        max(0, _int(raw_slots))
        if raw_slots is not None
        else max(0, _int(policy.get("remaining_active_target")))
    )
    return bool(candidates or fallback_count) and remaining_slots > 0 and not bool(
        policy.get("submission_frozen") or policy.get("submission_reconciliation_required")
    )


def new_active_first_state(
    *,
    account: str,
    policy: dict[str, Any],
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any]:
    remaining = max(0, _int(policy.get("remaining_active_target")))
    submission_ready = remaining > 0 and _policy_has_submission_candidate(policy)
    return {
        "version": 1,
        "account": account,
        "submission_day": submission_day(),
        "status": "RUNNING" if remaining > 0 else "COMPLETED",
        "phase": "NEEDS_SUBMISSION" if submission_ready else "NEEDS_RESEARCH" if remaining > 0 else "DONE",
        "session_stop_allowed": remaining == 0,
        "end_reason": "submission_candidate_available" if submission_ready else None if remaining > 0 else "daily_active_target_reached",
        "next_action": "advance_candidate_to_submission_gate" if submission_ready else "generate_skill_candidate" if remaining > 0 else "replenishment",
        "remaining_active_target": remaining,
        "campaign": deepcopy(policy.get("active_campaign") or {}),
        "max_iterations": max(1, int(max_iterations)),
        "counters": _base_counters(),
        "last_event": "session_started",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _budget_exhausted(state: dict[str, Any]) -> bool:
    counters = state.get("counters") or {}
    return _int(counters.get("iterations")) >= max(1, _int(state.get("max_iterations"), _DEFAULT_MAX_ITERATIONS))


def apply_policy(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state)
    remaining = max(0, _int(policy.get("remaining_active_target")))
    updated["remaining_active_target"] = remaining
    if policy.get("active_campaign") is not None:
        updated["campaign"] = deepcopy(policy.get("active_campaign") or {})
    if remaining == 0:
        updated.update({
            "status": "COMPLETED",
            "phase": "DONE",
            "session_stop_allowed": True,
            "end_reason": "daily_active_target_reached",
            "next_action": "replenishment",
            "last_event": "active_target_observed",
        })
    elif _policy_has_submission_candidate(policy):
        # Submission-ready platform/recovered candidates outrank more research.
        # This is especially important for ACTIVE_FILL, where local robustness
        # evidence is advisory and the official Submission Gate/SC is final.
        updated.update({
            "status": "RUNNING",
            "phase": "NEEDS_SUBMISSION",
            "session_stop_allowed": False,
            "end_reason": "submission_candidate_available",
            "next_action": "advance_candidate_to_submission_gate",
            "last_event": "submission_candidate_observed",
        })
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def apply_research_result(
    state: dict[str, Any],
    result: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(state)
    counters = dict(updated.get("counters") or _base_counters())
    summary = dict(result.get("summary") or {})
    simulations = _int(summary.get("simulated"))
    candidates = len(result.get("candidates") or [])
    routes = _route_counts([item for item in (params.get("skill_candidates") or []) if isinstance(item, dict)])

    if simulations > 0:
        counters["iterations"] = _int(counters.get("iterations")) + 1
    counters["simulations"] = _int(counters.get("simulations")) + simulations
    counters["candidates"] = _int(counters.get("candidates")) + candidates
    for key, value in routes.items():
        counters[key] = _int(counters.get(key)) + value
    updated["counters"] = counters
    updated["last_event"] = "research_iteration"

    if _int(updated.get("remaining_active_target")) <= 0:
        updated.update({
            "status": "COMPLETED",
            "phase": "DONE",
            "session_stop_allowed": True,
            "end_reason": "daily_active_target_reached",
            "next_action": "replenishment",
        })
    elif candidates > 0:
        updated.update({
            "status": "RUNNING",
            "phase": "NEEDS_SUBMISSION",
            "session_stop_allowed": False,
            "end_reason": "candidate_found",
            "next_action": "advance_candidate_to_submission_gate",
        })
    elif simulations > 0:
        checkpoint_reached = _budget_exhausted(updated)
        updated.update({
            "status": "RUNNING",
            "phase": "CONTINUE_REQUIRED",
            "session_stop_allowed": False,
            "end_reason": (
                "iteration_checkpoint_reached_target_remaining"
                if checkpoint_reached
                else "iteration_exhausted_no_candidate"
            ),
            "next_action": "failure_diagnosis_then_allocate_next_iteration",
        })
    else:
        updated.update({
            "status": "RUNNING",
            "phase": "CONTINUE_REQUIRED",
            "session_stop_allowed": False,
            "end_reason": "no_simulation_result",
            "next_action": "diagnose_execution_or_generate_replacement",
        })

    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def _submission_statuses(result: dict[str, Any]) -> list[str]:
    statuses: list[str] = []
    for entry in (result.get("results") or {}).values():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("final_status") or entry.get("status") or "").upper()
        if status:
            statuses.append(status)
    return statuses


def apply_submission_result(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state)
    counters = dict(updated.get("counters") or _base_counters())
    statuses = _submission_statuses(result)
    active = max(_int(result.get("active")), statuses.count("ACTIVE"))
    sc_fail = max(_int(result.get("sc_fail")), statuses.count("SC_FAIL"))
    other_fail = statuses.count("OTHER_FAIL")
    uncertain = max(
        _int(result.get("timeout")),
        sum(1 for status in statuses if status in {"SC_PENDING", "SUBMIT_UNKNOWN", "TIMEOUT", "SUBMITTING"}),
    )
    formal = active + sc_fail + other_fail + uncertain

    counters["formal_submissions"] = _int(counters.get("formal_submissions")) + formal
    counters["active"] = _int(counters.get("active")) + active
    counters["terminal_failures"] = _int(counters.get("terminal_failures")) + sc_fail + other_fail
    updated["counters"] = counters
    updated["last_event"] = "formal_submission"

    if active > 0:
        updated["remaining_active_target"] = max(0, _int(updated.get("remaining_active_target")) - active)
        if _int(updated.get("remaining_active_target")) == 0:
            updated.update({
                "status": "COMPLETED",
                "phase": "DONE",
                "session_stop_allowed": True,
                "end_reason": "daily_active_target_reached",
                "next_action": "replenishment",
            })
        else:
            updated.update({
                "status": "RUNNING",
                "phase": "CONTINUE_REQUIRED",
                "session_stop_allowed": False,
                "end_reason": "active_observed_target_remaining",
                "next_action": "continue_active_target_fill",
            })
    elif uncertain > 0:
        updated.update({
            "status": "COMPLETED",
            "phase": "PLATFORM_INFLIGHT_OR_UNKNOWN",
            "session_stop_allowed": True,
            "end_reason": "platform_inflight_or_unknown",
            "next_action": "reconcile_platform_before_more_submissions",
        })
    elif sc_fail + other_fail > 0:
        updated.update({
            "status": "RUNNING",
            "phase": "CONTINUE_REQUIRED",
            "session_stop_allowed": False,
            "end_reason": (
                "formal_submission_failed_after_iteration_checkpoint"
                if _budget_exhausted(updated)
                else "formal_submission_failed"
            ),
            "next_action": "failure_diagnosis_then_allocate_next_iteration",
        })
    else:
        # Local policy blocks / candidate readiness issues are not legitimate
        # ACTIVE-first stop conditions.  They require another candidate or a bug fix.
        updated.update({
            "status": "RUNNING",
            "phase": "CONTINUE_REQUIRED",
            "session_stop_allowed": False,
            "end_reason": "submission_not_formally_attempted",
            "next_action": "diagnose_submission_gate_or_generate_replacement",
        })

    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def apply_hard_stop(state: dict[str, Any], reason: str) -> dict[str, Any]:
    updated = deepcopy(state)
    updated.update({
        "status": "FAILED",
        "phase": "HARD_STOP",
        "session_stop_allowed": True,
        "end_reason": reason or "hard_task_failure",
        "next_action": "stop_for_hard_failure",
        "last_event": "hard_failure",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return updated


def _task_status_for_state(state: dict[str, Any]) -> str:
    status = str(state.get("status") or "RUNNING").upper()
    if status == "FAILED":
        return "failed"
    if bool(state.get("session_stop_allowed")):
        return "completed"
    return _RUNNING_STATUS


async def _latest_session_task(account: str) -> TaskModel | None:
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(TaskModel)
            .where(
                TaskModel.user_id == _DEV_USER_ID,
                TaskModel.task_type == _TASK_TYPE,
            )
            .order_by(TaskModel.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


async def get_active_first_session(account: str = "primary") -> dict[str, Any] | None:
    task = await _latest_session_task(account)
    if task is None:
        return None
    state = dict(task.result or {})
    if str((task.params or {}).get("account") or "primary") != account:
        return None
    return {"session_id": task.id, **state}


async def ensure_active_first_session(
    account: str,
    policy: dict[str, Any],
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any] | None:
    remaining = max(0, _int(policy.get("remaining_active_target")))
    if remaining == 0:
        current = await get_active_first_session(account)
        if current and not current.get("session_stop_allowed"):
            return await update_active_first_session(current["session_id"], apply_policy(current, policy))
        return current

    day = submission_day()
    task = await _latest_session_task(account)
    if task is not None:
        params = dict(task.params or {})
        state = dict(task.result or {})
        if params.get("account") == account and params.get("submission_day") == day and task.status not in _FINAL_STATUSES:
            state = apply_policy(state, policy)
            return await update_active_first_session(task.id, state)

    state = new_active_first_state(account=account, policy=policy, max_iterations=max_iterations)
    task_id = f"af{uuid.uuid4().hex[:10]}"
    factory = _get_session_factory()
    async with factory() as session:
        session.add(TaskModel(
            id=task_id,
            user_id=_DEV_USER_ID,
            session_id=None,
            status=_RUNNING_STATUS,
            task_type=_TASK_TYPE,
            params={
                "source": "wq_active_first_state_machine",
                "account": account,
                "submission_day": day,
                "max_iterations": state["max_iterations"],
            },
            result=state,
        ))
        await session.commit()
    return {"session_id": task_id, **state}


async def update_active_first_session(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in dict(state).items() if key != "session_id"}
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(TaskModel).where(TaskModel.id == session_id))
        task = result.scalar_one_or_none()
        if task is None:
            return {"session_id": session_id, **clean, "persistence_error": "session_not_found"}
        task.result = clean
        task.status = _task_status_for_state(clean)
        task.error = clean.get("end_reason") if task.status == "failed" else None
        task.updated_at = datetime.now(timezone.utc)
        await session.commit()
    return {"session_id": session_id, **clean}


async def record_research_transition(
    session_id: str | None,
    result: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if not session_id:
        return None
    current = await get_active_first_session(str(params.get("account") or "primary"))
    if not current or current.get("session_id") != session_id:
        return None
    return await update_active_first_session(session_id, apply_research_result(current, result, params))


async def record_submission_transition(
    session_id: str | None,
    result: dict[str, Any],
    account: str = "primary",
) -> dict[str, Any] | None:
    if not session_id:
        return None
    current = await get_active_first_session(account)
    if not current or current.get("session_id") != session_id:
        return None
    return await update_active_first_session(session_id, apply_submission_result(current, result))


async def record_hard_stop(session_id: str | None, account: str, reason: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    current = await get_active_first_session(account)
    if not current or current.get("session_id") != session_id:
        return None
    return await update_active_first_session(session_id, apply_hard_stop(current, reason))
