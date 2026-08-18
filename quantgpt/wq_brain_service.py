"""Shared WQ BRAIN business logic — the single source of truth.

All WQ BRAIN operations (single simulate, batch sweep, submit-by-ids,
check-alphas, list-alphas) live here as **sync** functions.

MCP tools call them via `asyncio.to_thread(service_fn, ...)`.
HTTP routes call them directly from background `threading.Thread`.

wq_brain_client.py (low-level HTTP transport) is the only dependency.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests

from .wq_brain_client import HTTP_TIMEOUT, build_wq_error
from .wq_operator_registry import WQ_FALLBACK_OPERATORS, validate_wq_expression

logger = logging.getLogger(__name__)


def _resolve_request_id(client, request_id: str | None = None) -> str | None:
    resolved = request_id or getattr(client, "request_id", None)
    if resolved and getattr(client, "request_id", None) != resolved:
        client.request_id = resolved
        session = getattr(client, "_session", None)
        if session is not None and hasattr(session, "request_id"):
            session.request_id = resolved
    return resolved


def _with_request_id(payload: dict, request_id: str | None) -> dict:
    if request_id:
        payload.setdefault("request_id", request_id)
    return payload


def _service_error(
    client,
    message: str,
    *,
    request_id: str | None = None,
    kind: str = "service_error",
    http_status: int | None = None,
    retryable: bool = False,
    **extra,
) -> dict:
    resolved = _resolve_request_id(client, request_id)
    return build_wq_error(
        message,
        request_id=resolved,
        layer="wq_service",
        kind=kind,
        http_status=http_status,
        retryable=retryable,
        **extra,
    )


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


_GLOBAL_SIMULATION_LIMIT = _env_int("WQ_SIM_GLOBAL_CONCURRENCY", 4)
_GLOBAL_SIMULATION_SLOTS = threading.BoundedSemaphore(_GLOBAL_SIMULATION_LIMIT)
_GLOBAL_SLOT_STATE = threading.local()
_ADAPTIVE_CONCURRENCY_LOCK = threading.Lock()
_ADAPTIVE_CONCURRENCY_HINT: int | None = None


def _acquire_global_simulation_slot(
    check_cancelled: Callable[[], bool] | None = None,
) -> bool | None:
    """Acquire the process-wide BRAIN Simulation slot, re-entrantly per thread."""
    if getattr(_GLOBAL_SLOT_STATE, "held", False):
        return False
    while True:
        if check_cancelled and check_cancelled():
            return None
        if _GLOBAL_SIMULATION_SLOTS.acquire(timeout=1.0):
            _GLOBAL_SLOT_STATE.held = True
            return True


def _release_global_simulation_slot(acquired: bool | None) -> None:
    if acquired:
        _GLOBAL_SLOT_STATE.held = False
        _GLOBAL_SIMULATION_SLOTS.release()


def _reset_adaptive_concurrency_hint() -> None:
    """Reset the cross-task hint; primarily useful for deterministic tests."""
    global _ADAPTIVE_CONCURRENCY_HINT
    with _ADAPTIVE_CONCURRENCY_LOCK:
        _ADAPTIVE_CONCURRENCY_HINT = None


def _remember_adaptive_concurrency(final: int, throttle_events: int, maximum: int) -> None:
    """Carry the last stable BRAIN concurrency into later tasks in this process."""
    global _ADAPTIVE_CONCURRENCY_HINT
    if not _env_int("WQ_SIM_CONCURRENCY_LEARN", 1, minimum=0):
        return
    stable = max(1, min(int(final), int(maximum), _GLOBAL_SIMULATION_LIMIT))
    with _ADAPTIVE_CONCURRENCY_LOCK:
        if throttle_events or _ADAPTIVE_CONCURRENCY_HINT is None:
            _ADAPTIVE_CONCURRENCY_HINT = stable
        else:
            _ADAPTIVE_CONCURRENCY_HINT = max(_ADAPTIVE_CONCURRENCY_HINT, stable)


def _concurrency_settings(total: int) -> tuple[int, int, int]:
    configured_initial = min(_env_int("WQ_SIM_CONCURRENCY", 3), _GLOBAL_SIMULATION_LIMIT)
    # This account consistently accepts three concurrent Simulations but rejects
    # the fourth with CONCURRENT_SIMULATION_LIMIT. Keep 3 as the safe default;
    # operators can explicitly opt back into probing 4 via the environment.
    maximum = min(max(configured_initial, _env_int("WQ_SIM_CONCURRENCY_MAX", 3)), _GLOBAL_SIMULATION_LIMIT)
    with _ADAPTIVE_CONCURRENCY_LOCK:
        learned = _ADAPTIVE_CONCURRENCY_HINT
    initial = min(maximum, learned) if learned is not None else configured_initial
    growth_waves = _env_int("WQ_SIM_CONCURRENCY_GROWTH_WAVES", 2)
    return min(initial, total), min(maximum, total), growth_waves


def _fork_worker_client(client):
    fork = getattr(client, "fork_authenticated", None)
    if callable(fork):
        return fork(), True
    return client, False


def _run_adaptive_parallel(
    client,
    items: list[Any],
    worker: Callable[[Any, Any, Callable[[], None]], dict],
    *,
    on_progress: Callable[[int, int, Any], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[tuple[Any, dict]], dict, bool]:
    """Run BRAIN simulations in small adaptive waves.

    Starts conservatively, increases after clean waves, and backs off when any
    worker observes BRAIN concurrency/rate limiting.  A worker gets its own
    authenticated client session when the client supports ``fork_authenticated``.
    """
    total = len(items)
    if total == 0:
        return (
            [],
            {
                "initial": 0,
                "max": 0,
                "peak": 0,
                "final": 0,
                "throttle_events": 0,
                "global_limit": _GLOBAL_SIMULATION_LIMIT,
            },
            False,
        )

    current, maximum, growth_waves = _concurrency_settings(total)
    initial = current
    peak = current
    clean_waves = 0
    throttle_events = 0
    completed = 0
    cursor = 0
    outputs: list[tuple[Any, dict]] = []
    cancelled = False

    while cursor < total:
        if check_cancelled and check_cancelled():
            cancelled = True
            break

        wave = items[cursor : cursor + current]
        cursor += len(wave)
        throttled = threading.Event()

        def mark_throttled() -> None:
            throttled.set()

        def run_one(item: Any) -> dict:
            acquired = _acquire_global_simulation_slot(check_cancelled)
            if acquired is None:
                return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}

            worker_client = None
            owns_client = False
            try:
                worker_client, owns_client = _fork_worker_client(client)
                return worker(worker_client, item, mark_throttled)
            except Exception as exc:
                logger.exception("Parallel WQ simulation failed")
                return {"ok": False, "error": str(exc)}
            finally:
                try:
                    if owns_client and worker_client is not None:
                        close = getattr(worker_client, "close", None)
                        if callable(close):
                            close()
                finally:
                    _release_global_simulation_slot(acquired)

        logger.info(
            "WQ adaptive simulation wave: size=%s concurrency=%s remaining=%s",
            len(wave),
            current,
            total - cursor,
        )
        with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="wq-sim") as pool:
            futures = {pool.submit(run_one, item): item for item in wave}
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                outputs.append((item, result))
                completed += 1
                if on_progress:
                    on_progress(completed, total, item)

        if check_cancelled and check_cancelled():
            cancelled = True
            break

        if throttled.is_set():
            throttle_events += 1
            previous = current
            current = max(1, current - 1)
            clean_waves = 0
            logger.info("WQ adaptive concurrency backoff: %s -> %s", previous, current)
        else:
            clean_waves += 1
            if clean_waves >= growth_waves and current < maximum:
                previous = current
                current += 1
                peak = max(peak, current)
                clean_waves = 0
                logger.info("WQ adaptive concurrency increase: %s -> %s", previous, current)

    _remember_adaptive_concurrency(current, throttle_events, maximum)
    stats = {
        "initial": initial,
        "max": maximum,
        "peak": peak,
        "final": current,
        "throttle_events": throttle_events,
        "global_limit": _GLOBAL_SIMULATION_LIMIT,
        "learned_for_next_task": current,
    }
    return outputs, stats, cancelled


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------


def safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fitness_to_grade(fitness: float | None) -> str:
    if fitness is None:
        return "D"
    if fitness >= 1.0:
        return "A"
    if fitness >= 0.5:
        return "B"
    if fitness >= 0.25:
        return "C"
    return "D"


def parse_is_metrics(is_data: dict) -> dict:
    return {
        "sharpe": safe_float(is_data.get("sharpe")),
        "fitness": safe_float(is_data.get("fitness")),
        "returns": safe_float(is_data.get("returns")),
        "turnover": safe_float(is_data.get("turnover")),
    }


def _build_wq_result_block(sharpe, fitness, returns_val, turnover, grade):
    return {
        "backtest_summary": {
            "long_short_sharpe": sharpe,
            "wq_fitness": fitness,
            "rank_ic_mean": None,
            "turnover": turnover,
            "wq_rating": grade,
        },
        "wq_brain": {
            "wq_sharpe": sharpe,
            "wq_fitness": fitness,
            "wq_returns": returns_val,
            "wq_turnover": turnover,
            "wq_rating": grade,
        },
        "interpretation": {"rating": grade},
    }


def prepare_wq_expression(client, expression: str) -> tuple[str | None, str | None, str]:
    """Canonicalize and validate an expression against the current BRAIN catalog."""
    source = "live"
    try:
        supported = client.list_operator_names()
        if not supported:
            raise RuntimeError("empty operator catalog")
    except Exception as e:
        logger.warning(f"Falling back to bundled WQ operator catalog: {e}")
        supported = WQ_FALLBACK_OPERATORS
        source = "fallback"

    validation = validate_wq_expression(expression, supported)
    if not validation.ok:
        return None, validation.error, source
    return validation.expression, None, source


# ---------------------------------------------------------------------------
# Service functions (sync, stateless — caller manages client lifecycle)
# ---------------------------------------------------------------------------


def run_single_simulation(
    client,
    expression: str,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    auto_submit: bool = False,
    user_id: str | None = None,
    tag: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
    submission_guard: Callable[[str], dict | bool] | None = None,
    submission_result_callback: Callable[[str, dict], None] | None = None,
    request_id: str | None = None,
) -> dict:
    """Simulate one expression and optionally auto-submit. Returns result dict."""
    request_id = _resolve_request_id(client, request_id)
    canonical_expression, validation_error, operator_catalog_source = prepare_wq_expression(client, expression)
    if validation_error:
        return _service_error(client, validation_error, request_id=request_id, kind="validation_error")
    assert canonical_expression is not None
    expression = canonical_expression

    acquired = _acquire_global_simulation_slot(check_cancelled)
    if acquired is None:
        return _service_error(
            client,
            "WQ simulation cancelled",
            request_id=request_id,
            kind="cancelled",
            cancelled=True,
        )
    try:
        result = client.simulate(
            expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            progress_callback=progress_callback,
            cancel_callback=check_cancelled,
        )
    finally:
        _release_global_simulation_slot(acquired)

    if not result.get("ok"):
        error_result = {
            "ok": False,
            "cancelled": bool(result.get("cancelled")),
            "error": result.get("error", "Simulation failed"),
        }
        for key in ("request_id", "layer", "kind", "http_status", "retryable"):
            if key in result:
                error_result[key] = result.get(key)
        return _with_request_id(error_result, request_id)

    alpha_id = result.get("alpha_id")
    is_data = result.get("is", {})
    m = parse_is_metrics(is_data)
    grade = fitness_to_grade(m["fitness"])

    submitted = False
    submission_blocked: dict | None = None
    if auto_submit and alpha_id and grade == "A":
        submission_blocked = {
            "allowed": False,
            "reason": "auto_submit_disabled_use_candidate_pipeline",
            "detail": "simulate/research first, persist a validated Candidate, then use submit-by-id",
        }

    if submitted and alpha_id and user_id:
        _track_alpha(
            user_id=user_id,
            alpha_id=alpha_id,
            expression=expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            tag=tag,
            metrics=m,
        )

    out = {
        "ok": True,
        "request_id": request_id,
        "expression": expression,
        "alpha_id": alpha_id,
        "is_metrics": is_data,
        "oos_metrics": result.get("oos", {}),
        "settings": result.get("settings", {}),
        "submitted": submitted,
        "submission_blocked": submission_blocked,
        "simulation_id": result.get("simulation_id"),
        "operator_catalog_source": operator_catalog_source,
    }
    out.update(_build_wq_result_block(m["sharpe"], m["fitness"], m["returns"], m["turnover"], grade))
    return out


def run_batch_simulation(
    client,
    expression: str,
    regions: list[str],
    delays: list[int],
    universes: list[str],
    neutralizations: list[str],
    decay: int = 0,
    truncation: float = 0.08,
    auto_submit: bool = False,
    user_id: str | None = None,
    tag: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
    submission_guard: Callable[[str], dict | bool] | None = None,
    submission_result_callback: Callable[[str, dict], None] | None = None,
    request_id: str | None = None,
) -> dict:
    """Sweep expression over region×delay×universe×neutralization grid. Returns result dict."""
    request_id = _resolve_request_id(client, request_id)
    canonical_expression, validation_error, _operator_catalog_source = prepare_wq_expression(client, expression)
    if validation_error:
        return _service_error(client, validation_error, request_id=request_id, kind="validation_error")
    assert canonical_expression is not None
    expression = canonical_expression

    combos = list(itertools.product(regions, delays, universes, neutralizations))

    def run_combo(worker_client, combo, mark_throttled):
        region, delay_val, universe, neut = combo
        key = f"{region}_D{delay_val}_{universe}_{neut}"

        def on_sim_progress(_pct: int, message: str) -> None:
            if "并发限制" in message or "速率限制" in message or "CONCURRENT_SIMULATION_LIMIT" in message:
                mark_throttled()

        sim_result = worker_client.simulate(
            expression,
            region=region,
            universe=universe,
            delay=delay_val,
            decay=decay,
            neutralization=neut,
            truncation=truncation,
            progress_callback=on_sim_progress,
            cancel_callback=check_cancelled,
        )

        if not sim_result.get("ok"):
            failed = {
                "key": key,
                "region": region,
                "delay": delay_val,
                "universe": universe,
                "neutralization": neut,
                "status": "failed",
                "error": sim_result.get("error", "unknown"),
            }
            for field in ("request_id", "layer", "kind", "http_status", "retryable"):
                if field in sim_result:
                    failed[field] = sim_result.get(field)
            return failed

        alpha_id = sim_result.get("alpha_id")
        is_data = sim_result.get("is", {})
        m = parse_is_metrics(is_data)
        grade = fitness_to_grade(m["fitness"])

        submitted = False
        submission_blocked: dict | None = None
        if auto_submit and alpha_id and grade == "A":
            submission_blocked = {
                "allowed": False,
                "reason": "auto_submit_disabled_use_candidate_pipeline",
                "detail": "simulate/research first, persist a validated Candidate, then use submit-by-id",
            }

        if submitted and alpha_id and user_id:
            _track_alpha(
                user_id=user_id,
                alpha_id=alpha_id,
                expression=expression,
                region=region,
                universe=universe,
                delay=delay_val,
                decay=decay,
                neutralization=neut,
                truncation=truncation,
                tag=tag,
                metrics=m,
            )

        return {
            "key": key,
            "region": region,
            "delay": delay_val,
            "universe": universe,
            "neutralization": neut,
            "status": "completed",
            "alpha_id": alpha_id,
            "sharpe": m["sharpe"],
            "fitness": m["fitness"],
            "returns": m["returns"],
            "turnover": m["turnover"],
            "submitted": submitted,
            "submission_blocked": submission_blocked,
            "rating": grade,
        }

    def report_progress(current: int, total: int, combo) -> None:
        if not on_progress:
            return
        region, delay_val, universe, neut = combo
        on_progress(current, total, f"{region}_D{delay_val}_{universe}_{neut}")

    completed, concurrency, cancelled = _run_adaptive_parallel(
        client,
        combos,
        run_combo,
        on_progress=report_progress,
        check_cancelled=check_cancelled,
    )
    sub_results = {result["key"]: result for _combo, result in completed}
    out = _aggregate_batch_result(expression, len(combos), sub_results)
    out["concurrency"] = concurrency
    out["cancelled"] = cancelled
    return _with_request_id(out, request_id)


def reconcile_submission_uncertainty(client, account: str = "primary") -> dict[str, Any]:
    """Reconcile fail-closed formal-submit outcomes before allowing another reservation."""
    from .wq_submission_policy import (
        get_submission_reservation_recovery_ids_sync,
        reconcile_candidate_platform_statuses_sync,
        reconcile_submission_reservations_sync,
    )

    recovery_ids = get_submission_reservation_recovery_ids_sync(account, limit=100)
    if not recovery_ids:
        return {"ok": True, "checked": 0, "reconciled": 0, "remaining": []}

    checked = run_check_alphas(client, recovery_ids)
    platform_alphas = checked.get("alphas", {})
    reconciled = reconcile_submission_reservations_sync(account, platform_alphas)
    candidates_reconciled = reconcile_candidate_platform_statuses_sync(account, platform_alphas)
    remaining = get_submission_reservation_recovery_ids_sync(account, limit=100)
    return {
        "ok": not remaining,
        "checked": len(recovery_ids),
        "reconciled": reconciled,
        "candidates_reconciled": candidates_reconciled,
        "remaining": remaining,
        "platform": checked,
    }


def _explicit_platform_check_failure(result: dict[str, Any]) -> str | None:
    """Return a non-SC BRAIN check that explicitly rejected an unsubmitted Alpha."""
    try:
        status_code = int(result.get("status_code") or 0)
    except (TypeError, ValueError):
        status_code = 0
    if status_code != 403:
        return None
    detail = str(result.get("detail") or "")
    for match in re.finditer(
        r'"name"\s*:\s*"([^"]+)"(?:(?!\}\s*,\s*\{).)*?"result"\s*:\s*"FAIL"',
        detail,
        flags=re.IGNORECASE,
    ):
        check_name = str(match.group(1) or "").upper()
        if check_name and check_name != "SELF_CORRELATION":
            return check_name
    return None


def run_submit_by_ids(
    client,
    alpha_ids: list[str],
    on_progress: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
    on_each_done: Callable[[str, dict], None] | None = None,
    submission_guard: Callable[[str], dict | bool] | None = None,
    submission_result_callback: Callable[[str, dict], None] | None = None,
    request_id: str | None = None,
) -> dict:
    """Submit a list of already-simulated alphas. Returns summary dict."""
    request_id = _resolve_request_id(client, request_id)
    results: dict[str, dict] = {}
    active = sc_fail = timeout = blocked = 0

    for i, alpha_id in enumerate(alpha_ids):
        if check_cancelled and check_cancelled():
            break

        if on_progress:
            on_progress(i + 1, len(alpha_ids), alpha_id)

        decision = submission_guard(alpha_id) if submission_guard else {"allowed": True}
        allowed = decision if isinstance(decision, bool) else bool(decision.get("allowed"))
        if not allowed:
            blocked += 1
            entry = build_wq_error(
                "submission blocked by local daily policy",
                request_id=request_id,
                layer="wq_service",
                kind="policy_blocked",
                retryable=False,
                final_status="POLICY_BLOCKED",
                detail="submission blocked by local daily policy",
                submission_policy=decision if isinstance(decision, dict) else {"allowed": False},
            )
            results[alpha_id] = entry
            if on_each_done:
                on_each_done(alpha_id, entry)
            continue

        if i > 0:
            time.sleep(5)

        result = client.submit_alpha(alpha_id)
        entry: dict[str, Any] = {
            "ok": result.get("ok", False),
            "request_id": result.get("request_id") or request_id,
            "detail": result.get("detail", ""),
            "platform_status": result.get("platform_status", ""),
            "status_code": result.get("status_code"),
        }
        for field in ("layer", "kind", "http_status", "retryable"):
            if field in result:
                entry[field] = result.get(field)
        if result.get("sc_value") is not None:
            entry["sc_value"] = result["sc_value"]
            entry["sc_limit"] = result.get("sc_limit")

        if result.get("ok"):
            active += 1
            entry["final_status"] = "ACTIVE"
        elif "SC FAIL" in result.get("detail", ""):
            sc_fail += 1
            entry["final_status"] = "SC_FAIL"
        elif platform_failure := _explicit_platform_check_failure(result):
            entry["final_status"] = "OTHER_FAIL"
            entry["confirmed_not_submitted"] = True
            entry["platform_check_failure"] = platform_failure
        elif result.get("confirmed_not_submitted") or str(result.get("platform_status") or "").upper() == "UNSUBMITTED":
            entry["final_status"] = "UNSUBMITTED_CONFIRMED"
            entry["confirmed_not_submitted"] = True
        else:
            timeout += 1
            entry["final_status"] = "SUBMIT_UNKNOWN"
            entry["submission_uncertain"] = True

        results[alpha_id] = entry

        if submission_result_callback:
            submission_result_callback(alpha_id, entry)

        if on_each_done:
            on_each_done(alpha_id, entry)

    return {
        "request_id": request_id,
        "total": len(alpha_ids),
        "active": active,
        "sc_fail": sc_fail,
        "timeout": timeout,
        "blocked": blocked,
        "results": results,
    }


def _normalize_level(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value.upper() if value else None
    if isinstance(value, dict):
        for key in ("level", "name", "id", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().upper()
    return None


def run_account_status(
    client,
    *,
    allow_alpha_count_fallback: bool = True,
    request_id: str | None = None,
) -> dict:
    """Return the BRAIN account progress needed by autonomous research loops.

    ``allow_alpha_count_fallback=False`` keeps latency bounded for MCP status calls.
    BRAIN's alpha-summary endpoint may omit submission-state counts; enumerating the
    full alpha history is useful for offline reconciliation but must not delay the
    Points/level truth path.
    """
    request_id = _resolve_request_id(client, request_id)
    user_info = client.get_user_info()
    if not user_info:
        last_error = getattr(client, "last_error", None)
        if isinstance(last_error, dict):
            return _with_request_id(dict(last_error), request_id)
        return _service_error(
            client,
            "Failed to fetch BRAIN user info",
            request_id=request_id,
            kind="account_status_unavailable",
            retryable=True,
        )

    user_id = str(user_info.get("id") or "").strip()
    competitions = client.get_user_competitions(user_id) if user_id else {}
    alpha_summary = client.get_user_alpha_summary()
    if allow_alpha_count_fallback and not any(
        key in alpha_summary for key in ("active", "unsubmitted", "decommissioned")
    ):
        fallback_counts = {"active": 0, "unsubmitted": 0, "decommissioned": 0}
        offset = 0
        for _ in range(100):
            page = run_list_alphas(client, limit=100, offset=offset)
            if not page.get("ok"):
                fallback_counts = {}
                break
            alphas = page.get("alphas", [])
            for alpha in alphas:
                status = str(alpha.get("status") or "").lower()
                if status in fallback_counts:
                    fallback_counts[status] += 1
            if len(alphas) < 100:
                break
            offset += 100
        alpha_summary = fallback_counts

    competition_results = competitions.get("results", []) if isinstance(competitions, dict) else []
    if not isinstance(competition_results, list):
        competition_results = []
    challenge = next(
        (
            item
            for item in competition_results
            if isinstance(item, dict) and str(item.get("id", "")).lower() == "challenge"
        ),
        {},
    )
    challenge = challenge if isinstance(challenge, dict) else {}
    raw_leaderboard = challenge.get("leaderboard")
    leaderboard = raw_leaderboard if isinstance(raw_leaderboard, dict) else {}
    raw_progress = challenge.get("progress")
    progress = raw_progress if isinstance(raw_progress, dict) else {}
    raw_progress_score = progress.get("score")
    progress_score = raw_progress_score if isinstance(raw_progress_score, dict) else {}

    challenge_scoring = str(challenge.get("scoring") or "").strip().upper()
    leaderboard_score = safe_float(leaderboard.get("score"))
    leaderboard_score_is_points = challenge_scoring != "PERFORMANCE"
    points = leaderboard_score if leaderboard_score_is_points else None
    points_source = "challenge.leaderboard.score" if points is not None else None
    if points is None:
        genius_data = user_info.get("geniusLevel")
        if isinstance(genius_data, dict):
            for key in ("points", "score"):
                points = safe_float(genius_data.get(key))
                if points is not None:
                    points_source = f"users.self.geniusLevel.{key}"
                    break

    genius_level = (
        _normalize_level(user_info.get("geniusLevel"))
        or _normalize_level(leaderboard.get("level"))
        or "NONE"
    )
    next_genius_level = _normalize_level(progress.get("level"))
    consultant_level = _normalize_level(user_info.get("level")) or "NONE"
    onboarding = user_info.get("onboarding")
    if consultant_level != "NONE":
        consultant_status = "ACTIVE"
    elif onboarding:
        consultant_status = "ONBOARDING"
    else:
        consultant_status = "NOT_CONSULTANT"

    def count(name: str) -> int | None:
        value = alpha_summary.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    active = count("active")
    unsubmitted = count("unsubmitted")
    decommissioned = count("decommissioned")
    submitted = None if active is None or decommissioned is None else active + decommissioned

    leaderboard_alpha_count = safe_float(leaderboard.get("alphas"))
    if leaderboard_alpha_count is not None and float(leaderboard_alpha_count).is_integer():
        leaderboard_alpha_count = int(leaderboard_alpha_count)
    leaderboard_rank = safe_float(leaderboard.get("rank"))
    if leaderboard_rank is not None and float(leaderboard_rank).is_integer():
        leaderboard_rank = int(leaderboard_rank)
    active_alpha_gap = None
    if active is not None and leaderboard_alpha_count is not None:
        active_alpha_gap = max(0, active - leaderboard_alpha_count)
    total_alphas = None
    if active is not None and unsubmitted is not None and decommissioned is not None:
        total_alphas = active + unsubmitted + decommissioned

    target_points = 10_000
    points_value = int(points) if points is not None and float(points).is_integer() else points
    leaderboard_score_value = (
        int(leaderboard_score)
        if leaderboard_score is not None and float(leaderboard_score).is_integer()
        else leaderboard_score
    )
    points_remaining = None if points is None else max(0, target_points - points)
    if points_remaining is not None and float(points_remaining).is_integer():
        points_remaining = int(points_remaining)
    gold_reached = genius_level in {"GOLD", "GRANDMASTER"} or consultant_level in {"GOLD", "GRANDMASTER"}
    if points is None:
        # Challenge PERFORMANCE leaderboards expose a performance metric (for
        # example 0.36), not cumulative Challenge Points. Keep the Points
        # channel explicitly unresolved so callers can fall back to the last
        # trusted Points observation without corrupting the local ledger.
        points_status = "SYNC_UNKNOWN" if challenge_scoring == "PERFORMANCE" and leaderboard_score is not None else "UNAVAILABLE"
    elif active is None or leaderboard_alpha_count is None:
        # Points can only be considered settled when both platform ACTIVE count
        # and Challenge leaderboard Alpha count are known. Treat missing count
        # evidence as an explicit sync-unknown state instead of assuming gap=0.
        points_status = "SYNC_UNKNOWN"
    elif active_alpha_gap:
        points_status = "LEADERBOARD_LAGGING"
    else:
        points_status = "CURRENT"

    goal_reached = bool(
        points is not None
        and points >= target_points
        and (gold_reached or consultant_status in {"ONBOARDING", "ACTIVE"})
    )

    return {
        "ok": True,
        "request_id": request_id,
        "points": points_value,
        "points_source": points_source,
        "points_status": points_status,
        "target_points": target_points,
        "points_remaining": points_remaining,
        "genius_level": genius_level,
        "next_genius_level": next_genius_level,
        "next_level_points_remaining": safe_float(progress_score.get("remaining")),
        "gold_reached": gold_reached,
        "consultant_status": consultant_status,
        "consultant_level": consultant_level,
        "goal_reached": goal_reached,
        "challenge": {
            "status": challenge.get("status"),
            "scoring": challenge.get("scoring"),
            "sign_up_date": challenge.get("signUpDate"),
            "submissions": challenge.get("submissions"),
        },
        "leaderboard": {
            "rank": leaderboard_rank,
            "score": leaderboard_score_value,
            "score_semantics": challenge_scoring or None,
            "level": _normalize_level(leaderboard.get("level")),
            "alpha_count": leaderboard_alpha_count,
            "active_alpha_gap": active_alpha_gap,
        },
        "alpha_counts": {
            "total": total_alphas,
            "submitted": submitted,
            "active": active,
            "unsubmitted": unsubmitted,
            "decommissioned": decommissioned,
        },
    }


def run_check_alphas(
    client,
    alpha_ids: list[str],
    *,
    request_id: str | None = None,
) -> dict:
    """Check platform status of multiple alphas. Returns summary + per-alpha dict."""
    request_id = _resolve_request_id(client, request_id)
    results: dict[str, dict] = {}

    for alpha_id in alpha_ids:
        data = client.check_alpha_status(alpha_id)
        if not data.get("ok"):
            failed = {"ok": False, "error": data.get("error", "not found")}
            for field in ("request_id", "layer", "kind", "http_status", "retryable"):
                if field in data:
                    failed[field] = data.get(field)
            results[alpha_id] = _with_request_id(failed, request_id)
            continue

        is_data = data.get("is", {})
        checks = is_data.get("checks", [])
        sc_check = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)

        results[alpha_id] = {
            "ok": True,
            "status": data.get("status"),
            "grade": data.get("grade"),
            "dateCreated": data.get("dateCreated"),
            "sharpe": safe_float(is_data.get("sharpe")),
            "fitness": safe_float(is_data.get("fitness")),
            "returns": safe_float(is_data.get("returns")),
            "turnover": safe_float(is_data.get("turnover")),
            "sc_result": sc_check.get("result") if sc_check else None,
            "sc_value": sc_check.get("value") if sc_check else None,
        }

    summary = {
        "total": len(alpha_ids),
        "active": sum(1 for r in results.values() if r.get("status") == "ACTIVE"),
        "unsubmitted": sum(1 for r in results.values() if r.get("status") == "UNSUBMITTED"),
        "sc_fail": sum(1 for r in results.values() if r.get("sc_result") == "FAIL"),
        "sc_pending": sum(1 for r in results.values() if r.get("sc_result") == "PENDING"),
    }
    return {"request_id": request_id, "summary": summary, "alphas": results}


def run_daily_submission_gate(
    client,
    account: str = "primary",
    *,
    request_id: str | None = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Consume the daily ACTIVE-first submission gate using the tracked Candidate Inventory.

    This is the reusable production path for non-MCP workers. It reconciles any
    uncertain prior submission first, refreshes platform status for the highest
    ranked fallback-eligible candidates, optionally performs configured
    robustness revalidation, then reserves and formally submits only the slots
    still required by the daily ACTIVE target.
    """
    from .wq_submission_policy import (
        _run_coro_sync,
        finalize_submission_attempt_sync,
        get_candidate_robustness_revalidation_payloads_sync,
        get_submission_policy_status,
        reconcile_candidate_platform_statuses_sync,
        record_research_candidates_sync,
        reserve_submission_sync,
    )

    request_id = _resolve_request_id(client, request_id)
    recovery = reconcile_submission_uncertainty(client, account)
    if not recovery.get("ok"):
        return {
            "ok": False,
            "status": "RECONCILIATION_REQUIRED",
            "request_id": request_id,
            "submission_recovery": recovery,
        }

    policy = dict(_run_coro_sync(get_submission_policy_status(account)))
    remaining_target = max(0, int(policy.get("remaining_active_target") or 0))
    remaining_slots = max(0, int(policy.get("remaining_submission_slots") or 0))
    if remaining_target <= 0 or remaining_slots <= 0:
        return {
            "ok": True,
            "status": "SUBMISSION_NOT_REQUIRED",
            "request_id": request_id,
            "submission_recovery": recovery,
            "submission_policy": policy,
        }

    candidate_ids = [
        str(item.get("alpha_id") or "").strip()
        for item in (policy.get("submission_candidate_top") or [])
        if str(item.get("alpha_id") or "").strip()
    ]
    candidate_ids = list(dict.fromkeys(candidate_ids))[: max(remaining_target, remaining_slots, 1) * 3]
    if not candidate_ids:
        return {
            "ok": True,
            "status": "NO_SUBMISSION_CANDIDATE",
            "request_id": request_id,
            "submission_recovery": recovery,
            "submission_policy": policy,
        }

    preflight = run_check_alphas(client, candidate_ids, request_id=request_id)
    reconcile_candidate_platform_statuses_sync(account, preflight.get("alphas", {}))
    policy = dict(_run_coro_sync(get_submission_policy_status(account)))

    robustness_revalidation: dict[str, dict[str, Any]] = {}
    if os.environ.get("WQ_REQUIRE_PRE_SUBMIT_ROBUSTNESS", "0").strip() == "1":
        from .wq_autonomous_research import validate_candidate_robustness

        pending_ids = [
            str(item.get("alpha_id") or "").strip()
            for item in (policy.get("submission_candidate_top") or [])
            if str(item.get("alpha_id") or "").strip()
        ]
        pending = get_candidate_robustness_revalidation_payloads_sync(account, pending_ids)
        for payload in pending:
            if check_cancelled and check_cancelled():
                break
            alpha_id = str(payload.get("alpha_id") or "").strip()
            settings = dict(payload.get("settings") or {})
            robustness = validate_candidate_robustness(
                client,
                payload,
                region=str(settings.get("region") or "USA"),
                universe=str(settings.get("universe") or "TOP3000"),
                delay=int(settings.get("delay") or 1),
                decay=int(settings.get("decay") or 0),
                neutralization=str(settings.get("neutralization") or "SUBINDUSTRY"),
                truncation=float(settings.get("truncation") or 0.08),
                check_cancelled=check_cancelled,
            )
            merged_validation = dict(payload.get("validation") or {})
            merged_validation.update(robustness)
            payload["validation"] = merged_validation
            saved = record_research_candidates_sync(
                account,
                [payload],
                settings=settings,
                tag=payload.get("tag"),
            )
            robustness_revalidation[alpha_id] = {
                "status": robustness.get("status"),
                "robustness_score": robustness.get("robustness_score"),
                "validation_simulations": robustness.get("validation_simulations"),
                "candidate_rows_saved": saved,
            }
        policy = dict(_run_coro_sync(get_submission_policy_status(account)))
    else:
        robustness_revalidation["_policy"] = {
            "status": "advisory_not_blocking",
            "reason": "daily ACTIVE target uses official BRAIN SC as the final fallback filter",
        }

    remaining_target = max(0, int(policy.get("remaining_active_target") or 0))
    remaining_slots = max(0, int(policy.get("remaining_submission_slots") or 0))
    submit_limit = min(remaining_target, remaining_slots)
    submit_ids = [
        str(item.get("alpha_id") or "").strip()
        for item in (policy.get("submission_candidate_top") or [])
        if str(item.get("alpha_id") or "").strip()
    ]
    submit_ids = list(dict.fromkeys(submit_ids))[:submit_limit]
    if not submit_ids:
        return {
            "ok": True,
            "status": "NO_SUBMISSION_CANDIDATE",
            "request_id": request_id,
            "submission_recovery": recovery,
            "preflight": preflight,
            "robustness_revalidation": robustness_revalidation,
            "submission_policy": policy,
        }

    result = run_submit_by_ids(
        client,
        submit_ids,
        check_cancelled=check_cancelled,
        submission_guard=lambda alpha_id: reserve_submission_sync(account, alpha_id),
        submission_result_callback=lambda alpha_id, entry: finalize_submission_attempt_sync(account, alpha_id, entry),
        request_id=request_id,
    )
    return {
        "ok": True,
        "status": "SUBMISSION_ATTEMPTED",
        "request_id": request_id,
        "submission_recovery": recovery,
        "preflight": preflight,
        "robustness_revalidation": robustness_revalidation,
        "submission_policy": policy,
        "submitted_alpha_ids": submit_ids,
        **result,
    }


def run_list_alphas(
    client,
    limit: int = 100,
    offset: int = 0,
    min_fitness: float | None = None,
    status_filter: str | None = None,
    request_id: str | None = None,
) -> dict:
    """List alphas from the platform with optional filtering."""
    request_id = _resolve_request_id(client, request_id)
    s = client._get_session()
    params = {"limit": min(limit, 100), "offset": offset, "order": "-dateCreated"}
    normalized_status = str(status_filter or "").strip().upper()
    if normalized_status:
        params["status"] = normalized_status

    try:
        r = s.get(
            "https://api.worldquantbrain.com/users/self/alphas",
            params=params,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        return _service_error(
            client,
            f"Failed to list BRAIN alphas: {exc}",
            request_id=request_id,
            kind=type(exc).__name__,
            retryable=True,
        )
    if r.status_code != 200:
        return _service_error(
            client,
            f"HTTP {r.status_code}: {r.text[:300]}",
            request_id=request_id,
            kind="http_error",
            http_status=r.status_code,
            retryable=r.status_code in {408, 425, 429, 500, 502, 503, 504},
        )

    data = r.json()
    raw_alphas = data if isinstance(data, list) else data.get("results", [])

    alphas = []
    for a in raw_alphas:
        code = a.get("regular", {})
        expr = code.get("code", "") if isinstance(code, dict) else str(code)
        settings = a.get("settings", {})
        is_data = a.get("is", {})
        fitness = safe_float(is_data.get("fitness"))
        alpha_status = a.get("status", "")

        if min_fitness is not None and (fitness is None or fitness < min_fitness):
            continue
        if normalized_status and alpha_status.upper() != normalized_status:
            continue

        alphas.append(
            {
                "alpha_id": a.get("id"),
                "expression": expr,
                "status": alpha_status,
                "dateCreated": a.get("dateCreated"),
                "neutralization": settings.get("neutralization"),
                "sharpe": safe_float(is_data.get("sharpe")),
                "fitness": fitness,
                "returns": safe_float(is_data.get("returns")),
                "turnover": safe_float(is_data.get("turnover")),
            }
        )

    return {"ok": True, "request_id": request_id, "total": len(alphas), "alphas": alphas}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_batch_result(expression: str, total_combos: int, sub_results: dict) -> dict:
    all_failed = all(s.get("status") == "failed" for s in sub_results.values())
    if all_failed:
        first_error = next(
            (s.get("error", "unknown") for s in sub_results.values()),
            "all simulations failed",
        )
        return {
            "ok": False,
            "expression": expression,
            "total_combinations": total_combos,
            "best_fitness": None,
            "best_key": None,
            "submittable_count": 0,
            "sub_results": sub_results,
            "error": first_error,
        }

    best_fitness = -999.0
    best_key = None
    submittable_count = 0

    for key, sub in sub_results.items():
        if sub.get("status") != "completed":
            continue
        fitness = sub.get("fitness")
        if fitness is not None and fitness >= 1.0:
            submittable_count += 1
        if fitness is not None and fitness > best_fitness:
            best_fitness = fitness
            best_key = key

    best_sub = sub_results.get(best_key, {}) if best_key else {}
    best_fit = round(best_fitness, 4) if best_fitness > -999 else None
    best_grade = fitness_to_grade(best_fit)

    out: dict[str, Any] = {
        "ok": True,
        "expression": expression,
        "total_combinations": total_combos,
        "best_fitness": best_fit,
        "best_key": best_key,
        "submittable_count": submittable_count,
        "sub_results": sub_results,
    }
    out.update(
        _build_wq_result_block(
            best_sub.get("sharpe"),
            best_fit,
            best_sub.get("returns"),
            best_sub.get("turnover"),
            best_grade,
        )
    )
    return out


def _track_alpha(
    user_id, alpha_id, expression, region, universe, delay, decay, neutralization, truncation, tag, metrics
):
    try:
        from .alpha_tracker import record_submitted_alpha_sync

        record_submitted_alpha_sync(
            user_id=user_id,
            alpha_id=alpha_id,
            expression=expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            sharpe=metrics["sharpe"],
            fitness=metrics["fitness"],
            returns=metrics["returns"],
            turnover=metrics["turnover"],
            tag=tag,
        )
    except Exception as e:
        logger.warning(f"Alpha tracking failed for {alpha_id}: {e}")
