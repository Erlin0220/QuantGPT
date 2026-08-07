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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .wq_brain_client import HTTP_TIMEOUT
from .wq_operator_registry import WQ_FALLBACK_OPERATORS, validate_wq_expression

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


_GLOBAL_SIMULATION_LIMIT = _env_int("WQ_SIM_GLOBAL_CONCURRENCY", 4)
_GLOBAL_SIMULATION_SLOTS = threading.BoundedSemaphore(_GLOBAL_SIMULATION_LIMIT)
_GLOBAL_SLOT_STATE = threading.local()


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


def _concurrency_settings(total: int) -> tuple[int, int, int]:
    initial = min(_env_int("WQ_SIM_CONCURRENCY", 3), _GLOBAL_SIMULATION_LIMIT)
    maximum = min(max(initial, _env_int("WQ_SIM_CONCURRENCY_MAX", 4)), _GLOBAL_SIMULATION_LIMIT)
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

    stats = {
        "initial": initial,
        "max": maximum,
        "peak": peak,
        "final": current,
        "throttle_events": throttle_events,
        "global_limit": _GLOBAL_SIMULATION_LIMIT,
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
) -> dict:
    """Simulate one expression and optionally auto-submit. Returns result dict."""
    canonical_expression, validation_error, operator_catalog_source = prepare_wq_expression(client, expression)
    if validation_error:
        return {"ok": False, "error": validation_error}
    assert canonical_expression is not None
    expression = canonical_expression

    acquired = _acquire_global_simulation_slot(check_cancelled)
    if acquired is None:
        return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}
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
        return {
            "ok": False,
            "cancelled": bool(result.get("cancelled")),
            "error": result.get("error", "Simulation failed"),
        }

    alpha_id = result.get("alpha_id")
    is_data = result.get("is", {})
    m = parse_is_metrics(is_data)
    grade = fitness_to_grade(m["fitness"])

    submitted = False
    if auto_submit and alpha_id and grade == "A":
        submit_result = client.submit_alpha(alpha_id)
        submitted = submit_result.get("ok", False)

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
        "expression": expression,
        "alpha_id": alpha_id,
        "is_metrics": is_data,
        "oos_metrics": result.get("oos", {}),
        "settings": result.get("settings", {}),
        "submitted": submitted,
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
) -> dict:
    """Sweep expression over region×delay×universe×neutralization grid. Returns result dict."""
    canonical_expression, validation_error, _operator_catalog_source = prepare_wq_expression(client, expression)
    if validation_error:
        return {"ok": False, "error": validation_error}
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
            return {
                "key": key,
                "region": region,
                "delay": delay_val,
                "universe": universe,
                "neutralization": neut,
                "status": "failed",
                "error": sim_result.get("error", "unknown"),
            }

        alpha_id = sim_result.get("alpha_id")
        is_data = sim_result.get("is", {})
        m = parse_is_metrics(is_data)
        grade = fitness_to_grade(m["fitness"])

        submitted = False
        if auto_submit and alpha_id and grade == "A":
            submit_result = worker_client.submit_alpha(alpha_id)
            submitted = submit_result.get("ok", False)

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
    return out


def run_submit_by_ids(
    client,
    alpha_ids: list[str],
    on_progress: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
    on_each_done: Callable[[str, dict], None] | None = None,
) -> dict:
    """Submit a list of already-simulated alphas. Returns summary dict."""
    results: dict[str, dict] = {}
    active = sc_fail = timeout = 0

    for i, alpha_id in enumerate(alpha_ids):
        if check_cancelled and check_cancelled():
            break

        if i > 0:
            time.sleep(5)

        if on_progress:
            on_progress(i + 1, len(alpha_ids), alpha_id)

        result = client.submit_alpha(alpha_id)
        entry: dict[str, Any] = {
            "ok": result.get("ok", False),
            "detail": result.get("detail", ""),
            "platform_status": result.get("platform_status", ""),
            "status_code": result.get("status_code"),
        }
        if result.get("sc_value") is not None:
            entry["sc_value"] = result["sc_value"]
            entry["sc_limit"] = result.get("sc_limit")

        if result.get("ok"):
            active += 1
            entry["final_status"] = "ACTIVE"
        elif "SC FAIL" in result.get("detail", ""):
            sc_fail += 1
            entry["final_status"] = "SC_FAIL"
        elif result.get("platform_status") == "TIMEOUT":
            timeout += 1
            entry["final_status"] = "SC_PENDING"
        else:
            entry["final_status"] = "OTHER_FAIL"

        results[alpha_id] = entry

        if on_each_done:
            on_each_done(alpha_id, entry)

    return {
        "total": len(alpha_ids),
        "active": active,
        "sc_fail": sc_fail,
        "timeout": timeout,
        "results": results,
    }


def run_check_alphas(client, alpha_ids: list[str]) -> dict:
    """Check platform status of multiple alphas. Returns summary + per-alpha dict."""
    results: dict[str, dict] = {}

    for alpha_id in alpha_ids:
        data = client.check_alpha_status(alpha_id)
        if not data.get("ok"):
            results[alpha_id] = {"ok": False, "error": data.get("error", "not found")}
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
    return {"summary": summary, "alphas": results}


def run_list_alphas(
    client,
    limit: int = 100,
    offset: int = 0,
    min_fitness: float | None = None,
    status_filter: str | None = None,
) -> dict:
    """List alphas from the platform with optional filtering."""
    s = client._get_session()
    r = s.get(
        "https://api.worldquantbrain.com/users/self/alphas",
        params={"limit": min(limit, 100), "offset": offset, "order": "-dateCreated"},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}

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
        if status_filter and alpha_status.upper() != status_filter.upper():
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

    return {"ok": True, "total": len(alphas), "alphas": alphas}


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
