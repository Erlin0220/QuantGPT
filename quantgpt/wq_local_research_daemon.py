"""Durable local Pi Agent WorldQuant research daemon."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .wq_brain_client import get_client
from .wq_brain_service import run_daily_submission_gate
from .wq_research_cycle_runner import run_local_llm_research

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaemonDecision:
    delay_seconds: float
    reason: str
    completed_cycle: bool = False
    fatal: bool = False


def classify_daemon_result(result: dict[str, Any]) -> DaemonDecision:
    """Choose the next daemon action from one local-research invocation."""
    status = str(result.get("status") or "")
    progress = dict(result.get("progress") or {})
    if progress.get("should_stop"):
        return DaemonDecision(2.0, str(progress.get("stop_reason") or "cycle_complete"), completed_cycle=True)
    if status in {"BATCH_INFLIGHT", "research_singleflight_busy"}:
        return DaemonDecision(15.0, status)
    if status == "SUBMISSION_ATTEMPTED":
        return DaemonDecision(2.0, status)
    if status in {"SUBMISSION_NOT_REQUIRED", "NO_SUBMISSION_CANDIDATE"}:
        return DaemonDecision(1.0, status)
    if status == "RECONCILIATION_REQUIRED":
        return DaemonDecision(15.0, status)
    if status == "local_llm_generation_failed":
        return DaemonDecision(3.0, status)
    if status in {"wq_not_configured", "brain_authentication_failed"}:
        return DaemonDecision(300.0, status)
    if not result.get("ok"):
        return DaemonDecision(60.0, status or "local_research_failed")
    if status == "NEEDS_SKILL_BATCH":
        return DaemonDecision(1.0, "continue_cycle")
    return DaemonDecision(15.0, status or "idle")


def compact_daemon_event(result: dict[str, Any], decision: DaemonDecision) -> dict[str, Any]:
    """Keep daemon logs useful without dumping full planner/candidate payloads."""
    cycle = dict(result.get("research_cycle") or {})
    local_llm = dict(result.get("local_llm_state") or cycle.get("local_llm") or {})
    rounds = [item for item in (result.get("local_llm_rounds_this_run") or []) if isinstance(item, dict)]
    last_round = rounds[-1] if rounds else None
    return {
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
        "cycle_id": result.get("cycle_id"),
        "progress": result.get("progress") or {},
        "last_error": cycle.get("last_error"),
        "local_llm": {
            "model": result.get("model") or local_llm.get("model"),
            "reasoning_policy": result.get("reasoning_policy"),
            "round_count": len(local_llm.get("rounds") or []),
            "last_round": {
                "round_id": last_round.get("round_id"),
                "status": last_round.get("status"),
                "run_status": last_round.get("run_status"),
                "reasoning_mode": last_round.get("reasoning_mode"),
                "source_run_id": last_round.get("source_run_id"),
                "summary": last_round.get("summary") or {},
            }
            if last_round
            else None,
        },
        "next": {
            "delay_seconds": decision.delay_seconds,
            "reason": decision.reason,
            "completed_cycle": decision.completed_cycle,
        },
    }


def run_submission_gate(*, account: str, trigger: dict[str, Any]) -> dict[str, Any]:
    """Resolve a production submission/reconciliation gate for the local daemon."""
    from .wq_active_session import ensure_active_first_session, record_submission_transition
    from .wq_submission_policy import _run_coro_sync, get_submission_policy_status

    request_id = f"local-daemon-submit-{uuid.uuid4().hex[:12]}"
    client = get_client(account, request_id=request_id)
    active_session_id: str | None = None
    try:
        policy = dict(_run_coro_sync(get_submission_policy_status(account)))
        active_session = _run_coro_sync(ensure_active_first_session(account, policy))
        active_session_id = str((active_session or {}).get("session_id") or "") or None
        if not client.authenticate():
            return {
                "ok": False,
                "status": "brain_authentication_failed",
                "request_id": request_id,
                "cycle_id": trigger.get("cycle_id"),
                "progress": trigger.get("progress") or {},
                "error": client.last_error or {"error": "WQ BRAIN authentication failed"},
            }

        result = run_daily_submission_gate(client, account, request_id=request_id)
        response = {
            **result,
            "cycle_id": trigger.get("cycle_id"),
            "progress": trigger.get("progress") or {},
            "trigger_status": trigger.get("status"),
        }
        if active_session_id:
            try:
                response["active_first_session"] = _run_coro_sync(
                    record_submission_transition(active_session_id, response, account)
                )
            except Exception as exc:
                logger.warning("Failed to advance ACTIVE-first submission session: %s", exc)
                response["active_first_session"] = {"error": str(exc)}
        return response
    except Exception as exc:
        logger.exception("WQ local submission gate failed")
        return {
            "ok": False,
            "status": "submission_gate_failed",
            "request_id": request_id,
            "cycle_id": trigger.get("cycle_id"),
            "progress": trigger.get("progress") or {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        client.close()


def run_local_research_daemon(
    *,
    account: str = "primary",
    target_simulations: int = 100,
    budget_minutes: int = 50,
    batch_size: int = 8,
    max_simulations_per_batch: int = 8,
    reasoning_policy: str = "max",
    max_cycles: int = 0,
) -> None:
    """Continuously run ACTIVE-first submission plus restart-safe research cycles."""
    completed_cycles = 0
    while max_cycles <= 0 or completed_cycles < max_cycles:
        result = run_local_llm_research(
            account=account,
            target_simulations=target_simulations,
            budget_minutes=budget_minutes,
            max_batches=50,
            batch_size=batch_size,
            max_simulations_per_batch=max_simulations_per_batch,
            reasoning_policy=reasoning_policy,
            submission_deferred=False,
        )
        if str(result.get("status") or "") in {"SUBMISSION_REQUIRED", "RECONCILIATION_REQUIRED"}:
            result = run_submission_gate(account=account, trigger=result)
        decision = classify_daemon_result(result)
        logger.info("WQ local research daemon event %s", json.dumps(compact_daemon_event(result, decision), ensure_ascii=False))
        if decision.completed_cycle:
            completed_cycles += 1
        if decision.fatal:
            raise RuntimeError(f"WQ local research daemon stopped: {decision.reason}")
        if max_cycles > 0 and completed_cycles >= max_cycles:
            break
        time.sleep(decision.delay_seconds)
