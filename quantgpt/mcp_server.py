"""FastMCP server for factor backtesting.

Provides tools for Agent-driven backtest workflow:
- list_operators: Show available factor expression operators
- list_universes: Show available stock universes
- validate_expression: Check expression syntax
- run_backtest: Execute full backtest pipeline
- score_factor: Compute composite factor quality score
- diagnose_factor: Diagnose factor issues and suggest mutations
- run_anti_overfit: Run anti-overfit detection
- run_rolling_validation: Walk-forward rolling validation
"""

import asyncio
import gzip
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import cast

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .expression_parser import __doc__ as _expr_module_doc
from .expression_parser import parse_expression
from .fundamental_data import ALL_FUNDAMENTAL_NAMES
from .market_data import BENCHMARK_CODES, UNIVERSES, MarketDataFetcher, fetch_benchmark_returns, get_universe
from .mcp_task_helper import (
    MCPTaskAlreadyRunningError,
    MCPTaskCapacityError,
    cancel_mcp_task,
    complete_mcp_task,
    get_active_mcp_task,
    get_mcp_task_snapshot,
    is_mcp_task_cancelled,
    start_mcp_background_task,
    start_mcp_task,
    update_mcp_task,
)
from .report import generate_report
from .task_executor import _run_backtest_in_process, get_executor
from .wq_autonomous_research import run_autonomous_research
from .wq_brain_service import (
    prepare_wq_expression,
    reconcile_submission_uncertainty,
    run_account_status,
    run_batch_simulation,
    run_check_alphas,
    run_list_alphas,
    run_single_simulation,
    run_submit_by_ids,
)
from .wq_operator_registry import WQ_FALLBACK_OPERATORS
from .wq_operator_registry import validate_wq_expression as validate_wq_catalog_expression
from .wq_research_agent import run_research_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

_DEFAULT_MCP_ALLOWED_HOSTS = "localhost,localhost:8003,127.0.0.1,127.0.0.1:8003"
_MCP_ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("QUANTGPT_MCP_ALLOWED_HOSTS", _DEFAULT_MCP_ALLOWED_HOSTS).split(",")
    if host.strip()
]
_WQ_RESEARCH_STALE_SECONDS = max(
    120,
    int(os.environ.get("QUANTGPT_WQ_RESEARCH_STALE_SECONDS", "900")),
)

mcp = FastMCP(
    "quantgpt",
    instructions=(
        "QuantGPT — A 股因子回测与 WorldQuant BRAIN 研究服务。"
        "所有回测、验证、因子值计算、WQ 模拟、研究和正式提交均为异步任务：调用后立即返回 task_id，"
        "使用 get_task_status 轮询直到 completed/failed/cancelled；需要中止时调用 cancel_task。"
        "禁止在单次 MCP 调用中等待重任务完成。因子值结果使用 get_factor_values_page 分页读取。"
    ),
    streamable_http_path="/",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=_MCP_ALLOWED_HOSTS,
    ),
)


def _enrich_with_fundamentals(expression: str, market_df, stock_codes: list, start_date: str, end_date: str):
    """Conditionally fetch and merge fundamental data if the expression uses fundamental vars."""
    from .fundamental_data import detect_fundamental_vars, enrich_market_data

    fund_vars = detect_fundamental_vars(expression)
    return enrich_market_data(market_df, fund_vars, stock_codes, start_date, end_date)


def _fetch_data_for_market(universe: str, start_date: str, end_date: str):
    """Fetch market data and stock codes. Returns (market_df, stock_codes)."""
    stock_codes = get_universe(universe, date=start_date)
    fetcher = MarketDataFetcher()
    market_df = fetcher.fetch_stocks(stock_codes, start_date, end_date)
    return market_df, stock_codes


def _fetch_benchmark_for_market(benchmark: str, start_date: str, end_date: str):
    """Fetch benchmark returns."""
    return fetch_benchmark_returns(benchmark, start_date, end_date)


async def _enqueue_mcp_tool(
    task_type: str,
    expression: str | None,
    params: dict,
    worker,
    *,
    legacy_wq_poll: bool = False,
    singleflight: bool = False,
    stale_after_seconds: int | None = None,
    **response_extra,
) -> str:
    """Atomically accept a bounded background task and return its polling contract."""
    try:
        task_id = await start_mcp_task(
            task_type,
            expression,
            params,
            status="pending",
            singleflight=singleflight,
            stale_after_seconds=stale_after_seconds,
        )
    except MCPTaskAlreadyRunningError as exc:
        task = exc.task
        response = {
            "task_id": task.get("task_id"),
            "status": task.get("status", "running"),
            "async": True,
            "new_task": False,
            "reused_existing": True,
            "message": "已有同类研究任务正在执行，本次未创建重复任务",
            "poll_with": "get_task_status",
            "poll_after_seconds": 10,
            **response_extra,
        }
        if legacy_wq_poll:
            response["legacy_poll"] = {
                "tool": "wq_brain_check_alphas",
                "alpha_ids": [f"task:{task.get('task_id')}"],
            }
        return json.dumps(response, ensure_ascii=False)
    except MCPTaskCapacityError as exc:
        return json.dumps({"error": str(exc), "retryable": True}, ensure_ascii=False)

    try:
        start_mcp_background_task(
            task_id,
            worker,
            params,
            expression=expression,
        )
    except Exception as exc:
        await complete_mcp_task(task_id, error=f"Failed to enqueue background task: {exc}", expression=expression)
        return json.dumps({"error": str(exc), "task_id": task_id}, ensure_ascii=False)

    return _async_task_response(
        task_id,
        legacy_wq_poll=legacy_wq_poll,
        **response_extra,
    )


def _load_local_factor_data(task_id: str, params: dict):
    expression = params["expression"]
    update_mcp_task(task_id, status="fetching_data", progress=5, progress_message="下载行情数据", persist=True)
    market_df, stock_codes = _fetch_data_for_market(params["universe"], params["start_date"], params["end_date"])
    if market_df is None or len(market_df) == 0:
        return None, stock_codes, "No market data available. Check date range and stock codes."
    if is_mcp_task_cancelled(task_id):
        return None, stock_codes, "任务已取消"
    update_mcp_task(task_id, progress=20, progress_message="加载基本面数据")
    market_df = _enrich_with_fundamentals(
        expression,
        market_df,
        stock_codes,
        params["start_date"],
        params["end_date"],
    )
    return market_df, stock_codes, None


def _run_local_backtest_process(task_id: str, params: dict, *, cost_rate=None):
    market_df, stock_codes, error = _load_local_factor_data(task_id, params)
    if error:
        return None, stock_codes, {"ok": False, "cancelled": error == "任务已取消", "error": error}

    update_mcp_task(task_id, status="backtesting", progress=30, progress_message="执行因子回测", persist=True)
    submit_kwargs = {
        "neutralize_industry": params["neutralize_industry"],
        "neutralize_cap": params["neutralize_cap"],
    }
    if cost_rate is not None:
        submit_kwargs["cost_rate"] = cost_rate
    future = get_executor().submit_cpu_work(
        _run_backtest_in_process,
        market_df,
        params["expression"],
        params["n_groups"],
        params["holding_period"],
        **submit_kwargs,
    )
    timeout_seconds = max(1, int(os.environ.get("QUANTGPT_LOCAL_TASK_TIMEOUT", "600")))
    deadline = time.monotonic() + timeout_seconds
    while True:
        if is_mcp_task_cancelled(task_id):
            future.cancel()
            return None, stock_codes, {"ok": False, "cancelled": True, "error": "任务已取消"}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            future.cancel()
            return (
                None,
                stock_codes,
                {
                    "ok": False,
                    "error": f"Local backtest timed out after {timeout_seconds}s",
                },
            )
        try:
            result = future.result(timeout=min(1.0, remaining))
            break
        except FutureTimeoutError:
            if future.done():
                raise
    if is_mcp_task_cancelled(task_id):
        return None, stock_codes, {"ok": False, "cancelled": True, "error": "任务已取消"}
    update_mcp_task(task_id, progress=70, progress_message="回测完成，整理结果")
    return result, stock_codes, None


def _run_local_backtest_mcp_task(task_id: str, params: dict) -> dict:
    result, stock_codes, error = _run_local_backtest_process(task_id, params)
    if error:
        return error
    assert result is not None

    anti_overfit_result = None
    factor_df = result.get("_factor_df")
    if factor_df is not None and len(factor_df) > 100:
        try:
            from .anti_overfit import run_anti_overfit as _run_ao

            update_mcp_task(task_id, progress=75, progress_message="执行反过拟合检查")
            anti_overfit_result = _run_ao(factor_df, params["holding_period"])
        except Exception as exc:
            logger.warning("Anti-overfit analysis failed: %s", exc)

    bm_returns = None
    try:
        update_mcp_task(task_id, progress=85, progress_message="加载基准并生成报告")
        bm_returns = _fetch_benchmark_for_market(params["benchmark"], params["start_date"], params["end_date"])
    except Exception as exc:
        logger.warning("Benchmark fetch failed: %s", exc)

    report_result = generate_report(
        result["ls_returns"],
        benchmark_returns=bm_returns,
        title=f"Factor: {params['expression']}",
    )
    return {
        "ok": True,
        "report_path": report_result["report_path"],
        "metrics": report_result["metrics"],
        "backtest_summary": {
            "long_short_sharpe": result["long_short_sharpe"],
            "long_short_annual": result.get("long_short_annual", 0),
            "top_group_sharpe": result.get("top_group_sharpe", 0),
            "monotonicity_score": result["monotonicity_score"],
            "spread": result["spread"],
            "group_returns": result["group_returns"],
            "ic_mean": result.get("ic_mean", 0),
            "rank_ic_mean": result.get("rank_ic_mean", 0),
            "ic_ir": result.get("ic_ir", 0),
            "ic_win_rate": result.get("ic_win_rate", 0),
            "turnover": result.get("turnover", 0),
            "cost_adjusted": result.get("cost_adjusted", False),
            "cost_rate": result.get("cost_rate", 0),
            "total_cost_drag": result.get("total_cost_drag", 0),
        },
        "wq_brain": result.get("wq_brain", {}),
        "anti_overfit": anti_overfit_result,
        "params": {**params, "stock_count": len(stock_codes)},
    }


def _run_local_score_mcp_task(task_id: str, params: dict) -> dict:
    from .iteration import compute_factor_score

    result, _stock_codes, error = _run_local_backtest_process(task_id, params)
    if error:
        return error
    assert result is not None
    bm_returns = None
    try:
        bm_returns = _fetch_benchmark_for_market(params["benchmark"], params["start_date"], params["end_date"])
    except Exception:
        pass
    update_mcp_task(task_id, progress=85, progress_message="计算因子评分")
    report_result = generate_report(result["ls_returns"], benchmark_returns=bm_returns, title="Factor Score")
    scoring = compute_factor_score(
        backtest_summary={
            "long_short_sharpe": result["long_short_sharpe"],
            "monotonicity_score": result["monotonicity_score"],
            "spread": result["spread"],
            "ic_mean": result.get("ic_mean", 0),
            "rank_ic_mean": result.get("rank_ic_mean", 0),
            "ic_ir": result.get("ic_ir", 0),
            "ic_win_rate": result.get("ic_win_rate", 0),
        },
        report_metrics=report_result["metrics"],
    )
    return {
        "ok": True,
        "score": scoring["score"],
        "grade": scoring["grade"],
        "component_scores": scoring["component_scores"],
        "key_metrics": {
            "ic_mean": result.get("ic_mean", 0),
            "ic_ir": result.get("ic_ir", 0),
            "monotonicity": result["monotonicity_score"],
            "top_group_sharpe": result.get("top_group_sharpe", 0),
            "turnover": result.get("turnover", 0),
            "sharpe": report_result["metrics"].get("sharpe", 0),
            "max_drawdown": report_result["metrics"].get("max_drawdown", 0),
        },
        "interpretation": {"rating": scoring["grade"]},
    }


def _run_local_anti_overfit_mcp_task(task_id: str, params: dict) -> dict:
    from .anti_overfit import run_anti_overfit as _run_ao

    result, _stock_codes, error = _run_local_backtest_process(task_id, params, cost_rate=0)
    if error:
        return error
    assert result is not None
    factor_df = result.get("_factor_df")
    if factor_df is None or len(factor_df) < 100:
        return {"ok": False, "error": "Insufficient factor data for anti-overfit analysis."}
    update_mcp_task(task_id, progress=85, progress_message="执行反过拟合分析")
    return {"ok": True, **_run_ao(factor_df, params["holding_period"])}


def _run_local_rolling_validation_mcp_task(task_id: str, params: dict) -> dict:
    from .rolling_validator import run_rolling_validation as _run_rv

    result, _stock_codes, error = _run_local_backtest_process(task_id, params, cost_rate=0)
    if error:
        return error
    assert result is not None
    factor_df = result.get("_factor_df")
    if factor_df is None or len(factor_df) < 100:
        return {"ok": False, "error": "Insufficient factor data for rolling validation."}
    update_mcp_task(task_id, progress=85, progress_message="执行滚动验证")
    return {"ok": True, **_run_rv(factor_df, params["holding_period"])}


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FACTOR_VALUES_DIR = _PROJECT_ROOT / "reports" / "mcp_factor_values"
_MAX_FACTOR_VALUE_ARTIFACTS = max(1, int(os.environ.get("QUANTGPT_MAX_FACTOR_VALUE_ARTIFACTS", "100")))
_MAX_FACTOR_VALUE_ARTIFACT_BYTES = max(
    1,
    int(os.environ.get("QUANTGPT_MAX_FACTOR_VALUE_ARTIFACT_MB", "64")) * 1024 * 1024,
)
_MAX_FACTOR_VALUE_TOTAL_BYTES = max(
    _MAX_FACTOR_VALUE_ARTIFACT_BYTES,
    int(os.environ.get("QUANTGPT_MAX_FACTOR_VALUE_TOTAL_MB", "512")) * 1024 * 1024,
)
_FACTOR_VALUE_ARTIFACT_TTL_SECONDS = max(
    60,
    int(os.environ.get("QUANTGPT_FACTOR_VALUE_ARTIFACT_TTL_SECONDS", "86400")),
)


def _cleanup_factor_value_artifacts() -> None:
    if not _FACTOR_VALUES_DIR.exists():
        return
    now = time.time()
    files: list[tuple[Path, os.stat_result]] = []
    for path in _FACTOR_VALUES_DIR.glob("*.jsonl.gz"):
        try:
            metadata = path.stat()
        except OSError:
            continue
        if now - metadata.st_mtime > _FACTOR_VALUE_ARTIFACT_TTL_SECONDS:
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to remove expired factor artifact: %s", path)
            continue
        files.append((path, metadata))

    files.sort(key=lambda item: item[1].st_mtime, reverse=True)
    retained_bytes = 0
    for index, (path, metadata) in enumerate(files):
        keep = (
            index < _MAX_FACTOR_VALUE_ARTIFACTS and retained_bytes + metadata.st_size <= _MAX_FACTOR_VALUE_TOTAL_BYTES
        )
        if keep:
            retained_bytes += metadata.st_size
            continue
        try:
            path.unlink()
        except OSError:
            logger.warning("Failed to remove stale factor artifact: %s", path)


def _run_compute_factor_values_mcp_task(task_id: str, params: dict) -> dict:
    from datetime import date, timedelta

    import numpy as np

    from .backtest import api_context

    expression = params["expression"]
    try:
        factor_fn = parse_expression(expression)
    except Exception as exc:
        return {"ok": False, "error": f"Invalid expression: {exc}"}

    with api_context():
        update_mcp_task(task_id, status="fetching_data", progress=5, progress_message="加载股票池", persist=True)
        stocks = get_universe(params["universe"])
        if not stocks:
            return {"ok": False, "error": f"Empty universe: {params['universe']}"}

        end_dt = params["end_date"] or date.today().isoformat()
        start_dt = params["start_date"] or (date.fromisoformat(end_dt) - timedelta(days=365)).isoformat()
        d_start = date.fromisoformat(start_dt)
        d_end = date.fromisoformat(end_dt)
        if d_end < d_start:
            return {"ok": False, "error": "end_date must be on or after start_date"}
        if (d_end - d_start).days > 750:
            return {"ok": False, "error": "Date range too large (max 750 days)"}

        fetch_start = (d_start - timedelta(days=260)).isoformat()
        update_mcp_task(task_id, progress=15, progress_message="下载行情数据")
        df = MarketDataFetcher().fetch_stocks(stocks, fetch_start, end_dt)
        if df is None or df.empty:
            return {"ok": False, "error": "No market data available for this universe/date range"}
        if is_mcp_task_cancelled(task_id):
            return {"ok": False, "cancelled": True, "error": "任务已取消"}

        update_mcp_task(task_id, status="computing", progress=45, progress_message="计算因子值", persist=True)
        try:
            df["factor_value"] = factor_fn(df)
        except Exception as exc:
            return {"ok": False, "error": f"Expression evaluation failed: {exc}"}

        result_df = cast(
            pd.DataFrame,
            df.loc[df["trade_date"] >= start_dt, ["trade_date", "stock_code", "factor_value"]],
        ).dropna(subset=["factor_value"])
        grouped = result_df.groupby("trade_date")
        total_days = grouped.ngroups

        _FACTOR_VALUES_DIR.mkdir(parents=True, exist_ok=True)
        artifact = _FACTOR_VALUES_DIR / f"{task_id}.jsonl.gz"
        temporary = artifact.with_suffix(".jsonl.gz.tmp")
        total_values = 0
        written_days = 0
        try:
            with gzip.open(temporary, "wt", encoding="utf-8") as output:
                for index, (trade_date, group) in enumerate(grouped, start=1):
                    if is_mcp_task_cancelled(task_id):
                        return {"ok": False, "cancelled": True, "error": "任务已取消"}
                    stock_values = cast(pd.Series, group["stock_code"])
                    factor_values = cast(pd.Series, group["factor_value"])
                    values = {
                        str(stock_code): round(float(factor_value), 6)
                        for stock_code, factor_value in zip(stock_values, factor_values, strict=False)
                        if np.isfinite(factor_value)
                    }
                    if values:
                        output.write(
                            json.dumps(
                                {
                                    "date": str(trade_date),
                                    "values": values,
                                    "count": len(values),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        output.write("\n")
                        written_days += 1
                        total_values += len(values)
                    if index == total_days or index % 20 == 0:
                        progress = 60 + int(index * 35 / max(1, total_days))
                        update_mcp_task(
                            task_id,
                            progress=min(progress, 95),
                            progress_message=f"写入分页结果 {index}/{total_days}",
                        )
            artifact_size = temporary.stat().st_size
            if artifact_size > _MAX_FACTOR_VALUE_ARTIFACT_BYTES:
                return {
                    "ok": False,
                    "error": (
                        "Factor-values artifact exceeds configured limit "
                        f"({_MAX_FACTOR_VALUE_ARTIFACT_BYTES // (1024 * 1024)} MB)"
                    ),
                }
            temporary.replace(artifact)
        finally:
            if temporary.exists():
                temporary.unlink()

        _cleanup_factor_value_artifacts()
        relative_path = artifact.relative_to(_PROJECT_ROOT).as_posix()
        return {
            "ok": True,
            "expression": expression,
            "universe": params["universe"],
            "start_date": start_dt,
            "end_date": end_dt,
            "trading_days": written_days,
            "total_values": total_values,
            "result_storage": {
                "format": "gzip-jsonl",
                "path": relative_path,
                "page_tool": "get_factor_values_page",
                "default_page_size": 5,
                "max_page_size": 20,
            },
        }


# Dummy DataFrame for expression validation (includes fundamental columns)
_VALIDATION_DUMMY = pd.DataFrame(
    {
        "open": [1.0, 2.0, 3.0],
        "high": [1.1, 2.1, 3.1],
        "low": [0.9, 1.9, 2.9],
        "close": [1.0, 2.0, 3.0],
        "volume": [100, 200, 300],
        "amount": [100, 400, 900],
        "pct_change": [0, 100, 50],
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        **{name: [1.0, 1.1, 1.2] for name in ALL_FUNDAMENTAL_NAMES},
    }
)


@mcp.tool()
def list_operators() -> str:
    """返回因子表达式支持的全部操作符及用法说明。Agent 据此生成因子表达式。"""
    return _expr_module_doc or _OPERATORS_DOC


@mcp.tool()
def list_universes() -> str:
    """返回可用股票池列表及说明。"""
    a_share_info = {
        "small_scale": f"5 只蓝筹股（快速测试）: {UNIVERSES['small_scale']}",
        "hs300": "沪深300成分股（动态获取）",
        "csi500": "中证500成分股（动态获取）",
        "csi1000": "中证1000成分股（派生: 全A - HS300 - CSI500, 取前1000）",
        "csi2000": "中证2000成分股（派生: 全A - HS300 - CSI500 - CSI1000, 取前2000）",
    }
    a_share_benchmarks = {k: v["name"] for k, v in BENCHMARK_CODES.items()}
    return json.dumps(
        {
            "universes": a_share_info,
            "benchmarks": a_share_benchmarks,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
async def list_wq_operators() -> str:
    """返回当前 WorldQuant BRAIN 账号实际可用的 FASTEXPR 算子目录。

    优先读取 BRAIN /operators 实时目录；账号未配置或目录读取失败时返回内置 fallback。
    WQ Alpha 生成前应优先调用本工具，而不是复用 A 股本地算子列表。
    """
    from .wq_brain_client import get_client
    from .wq_brain_client import is_configured as _wq_configured

    if _wq_configured("primary"):
        client = get_client("primary")
        try:
            authenticated = await asyncio.to_thread(client.authenticate)
            if authenticated:
                operators = await asyncio.to_thread(client.list_operators)
                return json.dumps(
                    {
                        "source": "live",
                        "count": len(operators),
                        "operators": [
                            {
                                "name": item.get("name"),
                                "category": item.get("category"),
                                "definition": item.get("definition"),
                            }
                            for item in operators
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"WQ operator catalog fetch failed, using fallback: {e}")
        finally:
            await asyncio.to_thread(client.close)

    return json.dumps(
        {
            "source": "fallback",
            "count": len(WQ_FALLBACK_OPERATORS),
            "operators": sorted(WQ_FALLBACK_OPERATORS),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
async def wq_brain_data_catalog(
    account: str = "primary",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    dataset_id: str | None = None,
    search: str | None = None,
    limit: int = 100,
) -> str:
    """读取当前 BRAIN 账号可用的 Data Explorer 数据集和 MATRIX Data Fields。

    Autonomous Research 使用同一实时目录生成新 Alpha，避免依赖硬编码字段。
    可用 dataset_id / search 缩小字段范围；本工具只读，不会创建 Simulation 或提交 Alpha。
    """
    from .wq_brain_client import get_client
    from .wq_brain_client import is_configured as _wq_configured

    if account not in {"primary", "alt"}:
        return json.dumps({"error": "account 必须是 primary 或 alt"}, ensure_ascii=False)
    if not _wq_configured(account):
        return json.dumps({"error": f"WQ BRAIN 未配置 (account={account})"}, ensure_ascii=False)
    limit = max(1, min(200, int(limit)))
    client = get_client(account)
    try:
        if not await asyncio.to_thread(client.authenticate):
            return json.dumps({"error": "WQ BRAIN 认证失败"}, ensure_ascii=False)
        try:
            datasets = await asyncio.to_thread(
                client.list_datasets,
                region=region,
                universe=universe,
                delay=delay,
                limit=100,
            )
        except Exception as exc:
            logger.warning("WQ dataset catalog fetch failed: %s", exc)
            datasets = []
        fields = await asyncio.to_thread(
            client.list_data_fields,
            region=region,
            universe=universe,
            delay=delay,
            dataset_id=dataset_id,
            search=search,
            limit=limit,
        )
        matrix_fields = [item for item in fields if str(item.get("type") or "MATRIX").upper() == "MATRIX"]
        return json.dumps(
            {
                "source": "live",
                "account": account,
                "scope": {"region": region, "universe": universe, "delay": delay},
                "dataset_filter": dataset_id,
                "search": search,
                "dataset_count": len(datasets),
                "datasets": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "category": item.get("category"),
                        "description": item.get("description"),
                    }
                    for item in datasets
                ],
                "field_count": len(matrix_fields),
                "fields": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "type": item.get("type"),
                        "dataset": item.get("dataset"),
                        "description": item.get("description"),
                    }
                    for item in matrix_fields
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)[:500]}, ensure_ascii=False)
    finally:
        await asyncio.to_thread(client.close)


@mcp.tool()
async def kb_upsert_source(
    source_type: str,
    title: str,
    content: str = "",
    source_key: str | None = None,
    authors: list[str] | None = None,
    published_year: int | None = None,
    url: str | None = None,
    external_id: str | None = None,
    access_scope: str | None = None,
    metadata: dict | None = None,
) -> str:
    """写入/更新 Alpha Knowledge 原始来源。

    适合接收 book-fetch/仓颉蒸馏前的教材文本、arXiv/SSRN 摘要或合法全文、WorldQuant 官方资料。
    原始来源与 BRAIN 实验记忆分开保存；不会因为写入来源而直接影响 Alpha 搜索。
    """
    from .wq_knowledge import upsert_knowledge_source

    try:
        result = await upsert_knowledge_source(
            source_type=source_type,
            source_key=source_key,
            title=title,
            content=content,
            authors=authors,
            published_year=published_year,
            url=url,
            external_id=external_id,
            access_scope=access_scope,
            metadata=metadata,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def kb_upsert_card(card: dict) -> str:
    """写入一张结构化 Alpha Knowledge Card。

    Active Card 必须至少引用两个不同 source_key；单来源卡自动降级为 draft，避免把单篇论文/单本书当成确定事实。
    """
    from .wq_knowledge import upsert_knowledge_card

    try:
        result = await upsert_knowledge_card(card)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def kb_search_alpha_knowledge(
    query: str = "",
    family: str | None = None,
    operators: list[str] | None = None,
    limit: int = 12,
    include_drafts: bool = False,
) -> str:
    """检索多源 Alpha Knowledge，并叠加真实 BRAIN trial 的经验反馈排序。"""
    from .wq_knowledge import search_alpha_knowledge

    try:
        result = await search_alpha_knowledge(
            query=query,
            families=[family] if family else None,
            operators=operators,
            limit=limit,
            include_drafts=include_drafts,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def kb_arxiv_search(query: str, limit: int = 10, ingest: bool = False) -> str:
    """通过 arXiv 官方 API 搜索论文元数据/摘要；ingest=true 时写入 Knowledge Sources。"""
    from .wq_knowledge import fetch_arxiv_sources

    try:
        result = await fetch_arxiv_sources(query, limit=limit, ingest=ingest)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def kb_distill_sources(
    source_keys: list[str],
    concept: str,
    family: str = "unknown",
    research_goal: str = "WorldQuant BRAIN Alpha research",
) -> str:
    """把至少两个已存来源交叉蒸馏为一张证据约束的 Alpha Knowledge Card。

    蒸馏只允许使用指定来源内容，必须保留相互冲突的证据，并输出 WQ operator / expression template / failure / mutation 信息。
    """
    from .wq_knowledge import cross_distill_sources

    try:
        result = await cross_distill_sources(
            source_keys,
            concept=concept,
            family=family,
            research_goal=research_goal,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def kb_explain_alpha(
    alpha_id: str | None = None,
    expression: str | None = None,
    account: str = "primary",
) -> str:
    """解释一个 Alpha 的知识血缘：使用过哪些 Knowledge Card，以及这些卡驱动后的真实 trial 表现。"""
    from .wq_knowledge import explain_alpha_knowledge

    try:
        result = await explain_alpha_knowledge(account=account, alpha_id=alpha_id, expression=expression)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def validate_expression(expression: str, mode: str = "local") -> str:
    """验证因子表达式语法是否正确。返回 OK 或错误信息。

    Args:
        expression: 因子表达式
        mode: "local"（本地回测验证，默认）或 "wq"（WQ BRAIN 提交验证，放宽字段/算子限制）
    """

    depth = 0
    for i, ch in enumerate(expression):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return f"ERROR: 括号不平衡：位置 {i} 处多余的右括号 ')'"
    if depth > 0:
        return f"ERROR: 括号不平衡：缺少 {depth} 个右括号 ')'"

    try:
        if mode == "wq":
            from .wq_brain_client import get_client
            from .wq_brain_client import is_configured as _wq_configured

            canonical = expression
            catalog_source = "fallback"
            if _wq_configured("primary"):
                client = get_client("primary")
                try:
                    if client.authenticate():
                        canonical, error, catalog_source = prepare_wq_expression(client, expression)
                        if error:
                            return f"ERROR: {error}"
                    else:
                        validation = validate_wq_catalog_expression(expression, WQ_FALLBACK_OPERATORS)
                        if not validation.ok:
                            return f"ERROR: {validation.error}"
                        canonical = validation.expression
                finally:
                    client.close()
            else:
                validation = validate_wq_catalog_expression(expression, WQ_FALLBACK_OPERATORS)
                if not validation.ok:
                    return f"ERROR: {validation.error}"
                canonical = validation.expression

            if canonical != expression:
                return (
                    "OK: expression is valid for WQ BRAIN submission; "
                    f"canonical_expression={canonical}; operator_catalog={catalog_source}"
                )
            return f"OK: expression is valid for WQ BRAIN submission; operator_catalog={catalog_source}"

        func = parse_expression(expression, mode=mode)
        func(_VALIDATION_DUMMY)
        return "OK: expression is valid"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
async def run_backtest(
    expression: str,
    universe: str = "hs300",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    n_groups: int = 5,
    holding_period: int = 5,
    benchmark: str = "hs300",
    neutralize_industry: bool = True,
    neutralize_cap: bool = True,
) -> str:
    """执行因子回测,生成 QuantStats HTML 报告。

    Args:
        expression: 因子表达式,如 "rank(close/ts_mean(close, 20))"
        universe: 股票池名称 (small_scale/hs300/csi500/csi1000/csi2000)
        start_date: 回测起始日期 YYYY-MM-DD
        end_date: 回测结束日期 YYYY-MM-DD
        n_groups: 分组数量
        holding_period: 持仓周期(交易日)
        benchmark: 基准 (hs300/zz500/sz50/csi1000)
        neutralize_industry: 行业中性化(默认开启)
        neutralize_cap: 市值中性化(默认开启)

    Returns:
        JSON string with report_path, metrics, group_returns, anti_overfit.
    """
    params = {
        "expression": expression,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "n_groups": n_groups,
        "holding_period": holding_period,
        "benchmark": benchmark,
        "neutralize_industry": neutralize_industry,
        "neutralize_cap": neutralize_cap,
    }
    return await _enqueue_mcp_tool("backtest", expression, params, _run_local_backtest_mcp_task)


@mcp.tool()
async def score_factor(
    expression: str,
    universe: str = "hs300",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    n_groups: int = 5,
    holding_period: int = 5,
    benchmark: str = "hs300",
    neutralize_industry: bool = True,
    neutralize_cap: bool = True,
) -> str:
    """执行因子回测并返回综合评分(0-100)和等级(A/B/C/D)。

    比 run_backtest 更轻量,不生成 HTML 报告,专注评分。

    Args:
        expression: 因子表达式
        universe: 股票池 (small_scale/hs300/csi500/csi1000/csi2000)
        start_date: 起始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        n_groups: 分组数量
        holding_period: 持仓周期(交易日)
        benchmark: 基准 (hs300/zz500/sz50/csi1000)
        neutralize_industry: 行业中性化(默认开启)
        neutralize_cap: 市值中性化(默认开启)

    Returns:
        JSON with score, grade, component_scores, key metrics.
    """
    params = {
        "expression": expression,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "n_groups": n_groups,
        "holding_period": holding_period,
        "benchmark": benchmark,
        "neutralize_industry": neutralize_industry,
        "neutralize_cap": neutralize_cap,
    }
    return await _enqueue_mcp_tool("score", expression, params, _run_local_score_mcp_task)


@mcp.tool()
def diagnose_factor(
    expression: str,
    ic_mean: float = 0.0,
    ic_ir: float = 0.0,
    monotonicity_score: float = 0.0,
    score: float = 50.0,
) -> str:
    """诊断因子问题并推荐突变策略。

    根据因子的 IC/IR/单调性/评分,判断失败模式(IC为零、IC为负、嵌套过深等),
    返回推荐的改进策略和定向 LLM 提示词。

    Args:
        expression: 当前因子表达式
        ic_mean: IC 均值
        ic_ir: IC 信息比率
        monotonicity_score: 分组单调性 (0-1)
        score: 综合评分 (0-100)

    Returns:
        JSON with diagnosis strategy, reason, and suggested mutation prompt.
    """
    from .mutation_engine import MutationEngine

    try:
        engine = MutationEngine(
            expression=expression,
            metrics={
                "backtest_summary": {
                    "ic_mean": ic_mean,
                    "ic_ir": ic_ir,
                    "monotonicity_score": monotonicity_score,
                },
                "report_metrics": {},
            },
            score=score,
        )
        diagnosis = engine.diagnose_failure()
        sys_prompt, user_prompt = engine.build_mutation_prompt()

        output = {
            "strategy": diagnosis.strategy.value,
            "reason": diagnosis.reason,
            "details": diagnosis.details,
            "mutation_prompt": {
                "system": sys_prompt[:500] + "..." if len(sys_prompt) > 500 else sys_prompt,
                "user": user_prompt,
            },
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Diagnose failed: {traceback.format_exc()}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def run_anti_overfit(
    expression: str,
    universe: str = "hs300",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    holding_period: int = 5,
    neutralize_industry: bool = True,
    neutralize_cap: bool = True,
) -> str:
    """对因子执行反过拟合检测(4项测试)。

    测试项: IC稳定性、子样本压力、安慰剂检验、半衰期估计。
    返回总分(0-100)和各测试通过情况。

    Args:
        expression: 因子表达式
        universe: 股票池 (small_scale/hs300/csi500/csi1000/csi2000)
        start_date: 起始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        holding_period: 持仓周期(交易日)
        neutralize_industry: 行业中性化(默认开启)
        neutralize_cap: 市值中性化(默认开启)

    Returns:
        JSON with score, recommendation, and per-test details.
    """
    params = {
        "expression": expression,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "n_groups": 5,
        "holding_period": holding_period,
        "benchmark": "hs300",
        "neutralize_industry": neutralize_industry,
        "neutralize_cap": neutralize_cap,
    }
    return await _enqueue_mcp_tool("anti_overfit", expression, params, _run_local_anti_overfit_mcp_task)


@mcp.tool()
async def run_rolling_validation(
    expression: str,
    universe: str = "hs300",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    holding_period: int = 5,
    neutralize_industry: bool = True,
    neutralize_cap: bool = True,
) -> str:
    """对因子执行滚动验证(Walk-Forward)。

    将数据切分为多个 训练/验证/测试 窗口(默认 3年/1年/1年,步长3个月),
    计算每个窗口的 IC/IR,评估因子在样本外的衰减情况。

    Args:
        expression: 因子表达式
        universe: 股票池 (small_scale/hs300/csi500/csi1000/csi2000)
        start_date: 起始日期(建议≥5年数据)
        end_date: 结束日期
        holding_period: 持仓周期(交易日)
        neutralize_industry: 行业中性化(默认开启)
        neutralize_cap: 市值中性化(默认开启)

    Returns:
        JSON with composite score, per-window results, decay analysis.
    """
    params = {
        "expression": expression,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "n_groups": 5,
        "holding_period": holding_period,
        "benchmark": "hs300",
        "neutralize_industry": neutralize_industry,
        "neutralize_cap": neutralize_cap,
    }
    return await _enqueue_mcp_tool(
        "rolling_validation",
        expression,
        params,
        _run_local_rolling_validation_mcp_task,
    )


def _run_wq_single_mcp_task(task_id: str, params: dict) -> dict:
    from .wq_brain_client import get_client
    from .wq_submission_policy import finalize_submission_attempt_sync, reserve_submission_sync

    client = get_client("primary")
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}

        update_mcp_task(
            task_id, status="simulating", progress=0, progress_message="Simulation 已提交，等待 BRAIN", persist=True
        )

        def on_progress(pct: int, message: str) -> None:
            update_mcp_task(task_id, status="simulating", progress=pct, progress_message=message)

        return run_single_simulation(
            client,
            params["expression"],
            region=params["region"],
            universe=params["universe"],
            delay=params["delay"],
            decay=params["decay"],
            neutralization=params["neutralization"],
            truncation=params["truncation"],
            auto_submit=params["auto_submit"],
            tag=params["tag"],
            progress_callback=on_progress,
            check_cancelled=lambda: is_mcp_task_cancelled(task_id),
            submission_guard=lambda alpha_id: reserve_submission_sync("primary", alpha_id),
            submission_result_callback=lambda alpha_id, result: finalize_submission_attempt_sync("primary", alpha_id, result),
        )
    finally:
        client.close()


def _run_wq_batch_mcp_task(task_id: str, params: dict) -> dict:
    from .wq_brain_client import get_client
    from .wq_submission_policy import finalize_submission_attempt_sync, reserve_submission_sync

    client = get_client("primary")
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}

        def on_progress(current: int, total: int, key: str) -> None:
            pct = int((current - 1) * 100 / total) if total else 0
            update_mcp_task(
                task_id,
                status="simulating",
                progress=pct,
                progress_current=current,
                progress_total=total,
                progress_message=f"模拟参数组合 {current}/{total}: {key}",
            )

        update_mcp_task(
            task_id, status="simulating", progress=0, progress_message="开始参数网格 Simulation", persist=True
        )
        return run_batch_simulation(
            client,
            params["expression"],
            regions=params["regions"],
            delays=params["delays"],
            universes=params["universes"],
            neutralizations=params["neutralizations"],
            decay=params["decay"],
            truncation=params["truncation"],
            auto_submit=params["auto_submit"],
            tag=params["tag"],
            on_progress=on_progress,
            check_cancelled=lambda: is_mcp_task_cancelled(task_id),
            submission_guard=lambda alpha_id: reserve_submission_sync("primary", alpha_id),
            submission_result_callback=lambda alpha_id, result: finalize_submission_attempt_sync("primary", alpha_id, result),
        )
    finally:
        client.close()


def _run_wq_research_mcp_task(task_id: str, params: dict) -> dict:
    from .wq_brain_client import get_client

    client = get_client("primary")
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}
        update_mcp_task(
            task_id, status="researching", progress=0, progress_message="执行 WQ Alpha 批量研究", persist=True
        )

        def on_progress(current: int, total: int, stage: str) -> None:
            pct = int((current - 1) * 100 / total) if total else 0
            update_mcp_task(
                task_id,
                status="researching",
                progress=pct,
                progress_current=current,
                progress_total=total,
                progress_message=f"研究进度 {current}/{total}: {stage[:160]}",
            )

        result = run_research_batch(
            client,
            params["expressions"],
            goal=params["goal"],
            tag=params["tag"],
            region=params["region"],
            universe=params["universe"],
            delay=params["delay"],
            decay=params["decay"],
            neutralization=params["neutralization"],
            truncation=params["truncation"],
            max_candidates=params["max_candidates"],
            min_sharpe=params["min_sharpe"],
            min_fitness=params["min_fitness"],
            skip_existing=params["skip_existing"],
            sweep_top_n=params["sweep_top_n"],
            sweep_universes=params["sweep_universes"],
            sweep_neutralizations=params["sweep_neutralizations"],
            on_progress=on_progress,
            check_cancelled=lambda: is_mcp_task_cancelled(task_id),
        )
        try:
            from .wq_research_memory import record_research_trials_sync

            result["research_trials_saved"] = record_research_trials_sync(
                "primary",
                result,
                default_family="",
                hypothesis=params["goal"],
                tag=result.get("tag"),
            )
        except Exception as exc:
            logger.warning("Failed to persist WQ research memory: %s", exc)
            result["research_trials_saved"] = 0
        try:
            from .wq_submission_policy import record_research_candidates_sync

            result["candidate_queue_saved"] = record_research_candidates_sync(
                "primary",
                result.get("candidates", []),
                settings=result.get("settings", {}),
                tag=result.get("tag"),
            )
        except Exception as exc:
            logger.warning("Failed to persist WQ research candidates: %s", exc)
            result["candidate_queue_saved"] = 0
        return result
    finally:
        client.close()


def _run_wq_autonomous_research_mcp_task(task_id: str, params: dict) -> dict:
    from .wq_brain_client import get_client
    from .wq_control_tower import build_research_control_tower
    from .wq_research_memory import load_research_memory_sync, record_research_trials_sync
    from .wq_submission_policy import _run_coro_sync, get_submission_policy_status, record_research_candidates_sync

    client = get_client("primary")
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}

        update_mcp_task(
            task_id,
            status="researching",
            progress=0,
            progress_message="读取研究记忆并规划低覆盖 Alpha 家族",
            persist=True,
        )
        memory = load_research_memory_sync("primary")
        submission_policy = _run_coro_sync(get_submission_policy_status("primary"))
        memory["inventory"] = submission_policy.get("inventory") or {}
        memory["submission"] = submission_policy
        memory["research_learning"] = build_research_control_tower(memory, submission_policy)

        def on_progress(current: int, total: int, stage: str) -> None:
            pct = int(current * 100 / total) if total else 0
            update_mcp_task(
                task_id,
                status="researching",
                progress=min(99, pct),
                progress_current=current,
                progress_total=total,
                progress_message=f"自主研究 {current}/{total}: {stage[:160]}",
            )

        result = run_autonomous_research(
            client,
            memory=memory,
            goal=params["goal"],
            tag=params["tag"],
            region=params["region"],
            universe=params["universe"],
            delay=params["delay"],
            decay=params["decay"],
            neutralization=params["neutralization"],
            truncation=params["truncation"],
            max_simulations=params["max_simulations"],
            generations=params["generations"],
            family_count=params["family_count"],
            min_sharpe=params["min_sharpe"],
            min_fitness=params["min_fitness"],
            on_progress=on_progress,
            check_cancelled=lambda: is_mcp_task_cancelled(task_id),
        )
        try:
            result["research_trials_saved"] = record_research_trials_sync(
                "primary",
                result,
                default_family="",
                hypothesis=params["goal"],
                tag=result.get("tag"),
            )
        except Exception as exc:
            logger.warning("Failed to persist autonomous WQ research memory: %s", exc)
            result["research_trials_saved"] = 0
        try:
            result["candidate_queue_saved"] = record_research_candidates_sync(
                "primary",
                result.get("candidates", []),
                settings=result.get("settings", {}),
                tag=result.get("tag"),
            )
        except Exception as exc:
            logger.warning("Failed to persist autonomous WQ candidates: %s", exc)
            result["candidate_queue_saved"] = 0
        return result
    finally:
        client.close()


def _run_wq_submit_by_ids_mcp_task(task_id: str, params: dict) -> dict:
    from .wq_autonomous_research import validate_candidate_robustness
    from .wq_brain_client import get_client
    from .wq_submission_policy import (
        _run_coro_sync,
        finalize_submission_attempt_sync,
        get_candidate_robustness_revalidation_payloads_sync,
        get_submission_policy_status,
        reconcile_candidate_platform_statuses,
        record_research_candidates_sync,
        reserve_submission_sync,
    )

    client = get_client(params["account"])
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}

        update_mcp_task(
            task_id,
            status="finalizing",
            progress=0,
            progress_message="正式提交前对账历史不确定提交状态",
            persist=True,
        )
        submission_recovery = reconcile_submission_uncertainty(client, params["account"])
        if not submission_recovery.get("ok"):
            return {
                "ok": False,
                "error": "存在尚未与 BRAIN 对账完成的正式提交；为避免超额提交，本次提交已冻结",
                "submission_recovery": submission_recovery,
            }

        update_mcp_task(
            task_id,
            status="finalizing",
            progress=0,
            progress_message="正式提交前刷新 BRAIN 指标与 SC 状态",
            persist=True,
        )
        preflight = run_check_alphas(client, params["alpha_ids"])
        _run_coro_sync(
            reconcile_candidate_platform_statuses(
                params["account"],
                preflight.get("alphas", {}),
            )
        )

        robustness_revalidation: dict[str, dict] = {}
        submission_policy = _run_coro_sync(get_submission_policy_status(params["account"]))
        require_pre_submit_robustness = os.environ.get("WQ_REQUIRE_PRE_SUBMIT_ROBUSTNESS", "0").strip() == "1"
        pending_robustness = (
            get_candidate_robustness_revalidation_payloads_sync(
                params["account"],
                params["alpha_ids"],
            )
            if require_pre_submit_robustness
            else []
        )
        if not require_pre_submit_robustness:
            robustness_revalidation["_policy"] = {
                "status": "advisory_not_blocking",
                "reason": "daily ACTIVE target uses official BRAIN SC as the final fallback filter",
                "remaining_active_target": submission_policy.get("remaining_active_target"),
            }
        for index, payload in enumerate(pending_robustness, start=1):
            if is_mcp_task_cancelled(task_id):
                break
            alpha_id = str(payload.get("alpha_id") or "")
            settings = dict(payload.get("settings") or {})
            update_mcp_task(
                task_id,
                status="finalizing",
                progress=0,
                progress_message=f"提交前稳健性复核 {index}/{len(pending_robustness)}: {alpha_id}",
            )
            robustness = validate_candidate_robustness(
                client,
                payload,
                region=str(settings.get("region") or "USA"),
                universe=str(settings.get("universe") or "TOP3000"),
                delay=int(settings.get("delay") or 1),
                decay=int(settings.get("decay") or 0),
                neutralization=str(settings.get("neutralization") or "SUBINDUSTRY"),
                truncation=float(settings.get("truncation") or 0.08),
                check_cancelled=lambda: is_mcp_task_cancelled(task_id),
            )
            merged_validation = dict(payload.get("validation") or {})
            merged_validation.update(robustness)
            payload["validation"] = merged_validation
            saved = record_research_candidates_sync(
                params["account"],
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

        def on_progress(current: int, total: int, alpha_id: str) -> None:
            pct = int((current - 1) * 100 / total) if total else 0
            update_mcp_task(
                task_id,
                status="submitting",
                progress=pct,
                progress_current=current,
                progress_total=total,
                progress_message=f"提交 Alpha {current}/{total}: {alpha_id}",
            )

        update_mcp_task(task_id, status="submitting", progress=0, progress_message="开始正式提交 Alpha", persist=True)
        result = run_submit_by_ids(
            client,
            params["alpha_ids"],
            on_progress=on_progress,
            check_cancelled=lambda: is_mcp_task_cancelled(task_id),
            submission_guard=lambda alpha_id: reserve_submission_sync(params["account"], alpha_id),
            submission_result_callback=lambda alpha_id, entry: finalize_submission_attempt_sync(params["account"], alpha_id, entry),
        )
        return {
            "ok": True,
            "submission_recovery": submission_recovery,
            "preflight": preflight,
            "robustness_revalidation": robustness_revalidation,
            **result,
        }
    finally:
        client.close()


def _run_wq_finalize_mcp_task(task_id: str, params: dict) -> dict:
    from .routes.wq_brain_batch import _finalize_alpha_statuses
    from .wq_brain_client import get_client

    client = get_client(params["account"])
    try:
        update_mcp_task(task_id, status="authenticating", progress=0, progress_message="认证 WQ BRAIN", persist=True)
        if not client.authenticate():
            return {"ok": False, "error": "WQ BRAIN 认证失败"}
        update_mcp_task(task_id, status="finalizing", progress=0, progress_message="查询最终 SC 状态", persist=True)
        result = _finalize_alpha_statuses(client, params["alpha_ids"], None)
        return {"ok": True, **result}
    finally:
        client.close()


def _async_task_response(
    task_id: str,
    *,
    legacy_wq_poll: bool = False,
    **extra,
) -> str:
    """Build the stable response returned by every long-running MCP tool."""
    response = {
        "task_id": task_id,
        "status": "pending",
        "async": True,
        **extra,
        "poll_with": "get_task_status",
        "poll_after_seconds": 10,
    }
    if legacy_wq_poll:
        response["legacy_poll"] = {
            "tool": "wq_brain_check_alphas",
            "alpha_ids": [f"task:{task_id}"],
        }
    return json.dumps(response, ensure_ascii=False)


@mcp.tool()
async def get_task_status(task_id: str) -> str:
    """查询异步任务状态和最终结果。

    WQ 长任务会立即返回 task_id。使用本工具轮询；当 status 为 completed 时读取 result，
    status 为 failed 时读取 error。pending/running/authenticating/simulating/researching/submitting/finalizing
    均表示后台任务仍在执行。
    """
    task = await get_mcp_task_snapshot(task_id)
    if task is None:
        return json.dumps({"error": f"Task not found: {task_id}"}, ensure_ascii=False)

    status = task.get("status")
    task["done"] = status in ("completed", "failed", "cancelled", "iteration_completed")
    if not task["done"]:
        task["poll_after_seconds"] = 10
    return json.dumps(task, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """取消一个正在后台执行的异步任务。

    取消是协作式的：WQ Simulation 会在下一次轮询/等待点停止；已经提交给
    WorldQuant 的远端请求可能仍会在平台侧完成，但 QuantGPT 不再继续阻塞等待。
    """
    task = await cancel_mcp_task(task_id)
    if task is None:
        return json.dumps({"error": f"Task not found: {task_id}"}, ensure_ascii=False)
    return json.dumps(task, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def wq_brain_submit(
    expression: str,
    tag: str,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    auto_submit: bool = False,
) -> str:
    """提交因子表达式到 WorldQuant BRAIN 平台进行真实模拟。

    与 run_backtest（本地 A 股回测）不同，此工具调用 WQ BRAIN 真实 API，
    在美股 TOP3000 等市场上评估因子。返回样本内/样本外指标和提交资格。

    需要在 .env 中配置 WQ_BRAIN_EMAIL 和 WQ_BRAIN_PASSWORD。

    Args:
        expression: FASTEXPR 表达式 (如 "rank(close/open)")
        tag: 提交者标识 (如 "agent-lowcorr-0506")，用于追踪哪个 agent 提交
        region: 市场区域 (当前仅 USA 可用)
        universe: WQ Universe (TOP3000, TOP500 等)
        delay: 信号延迟 (0 或 1)
        decay: Alpha 衰减 (0-20)
        neutralization: 中性化 (SUBINDUSTRY, INDUSTRY, SECTOR, MARKET, NONE)
        truncation: 权重截断 (0-0.5)
        auto_submit: 已弃用；正式提交必须走 Research/Candidate → wq_brain_submit_by_ids

    Returns:
        JSON with task_id. Simulation runs in the background; poll with get_task_status.
    """
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured("primary"):
        return json.dumps({"error": "WQ BRAIN 未配置 — 请设置 WQ_BRAIN_EMAIL 和 WQ_BRAIN_PASSWORD"})
    if auto_submit:
        return json.dumps(
            {
                "error": "auto_submit 已停用；请先研究并进入 Candidate Queue，再调用 wq_brain_submit_by_ids",
                "reason": "auto_submit_disabled_use_candidate_pipeline",
            },
            ensure_ascii=False,
        )

    params = {
        "expression": expression,
        "tag": tag,
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "auto_submit": auto_submit,
    }
    return await _enqueue_mcp_tool(
        "wq_brain_submit",
        expression,
        params,
        _run_wq_single_mcp_task,
        legacy_wq_poll=True,
    )


@mcp.tool()
async def wq_brain_batch_submit(
    expression: str,
    tag: str,
    regions: list[str] | None = None,
    delays: list[int] | None = None,
    universes: list[str] | None = None,
    neutralizations: list[str] | None = None,
    decay: int = 0,
    truncation: float = 0.08,
    auto_submit: bool = False,
) -> str:
    """批量扫描因子表达式在多个参数组合下的 WQ BRAIN 表现。

    在 region × delay × universe × neutralization 的网格上逐一模拟，
    返回每个组合的 IS 指标和最优组合。适合找出同一表达式的最佳参数。

    Args:
        expression: FASTEXPR 表达式
        tag: 提交者标识 (如 "agent-lowcorr-0506")，用于追踪哪个 agent 提交
        regions: 市场区域列表 (默认 ["USA"])
        delays: 信号延迟列表 (默认 [1])
        universes: Universe 列表 (默认 ["TOP3000"])
        neutralizations: 中性化列表 (默认 ["SUBINDUSTRY"])
        decay: Alpha 衰减 (0-20, 共用)
        truncation: 权重截断 (0-0.5, 共用)
        auto_submit: 已弃用；正式提交必须走 Research/Candidate → wq_brain_submit_by_ids

    Returns:
        JSON with task_id. The parameter sweep runs in the background; poll with get_task_status.
    """
    from .wq_brain_client import is_configured as _wq_configured

    regions = regions or ["USA"]
    delays = delays or [1]
    universes = universes or ["TOP3000"]
    neutralizations = neutralizations or ["SUBINDUSTRY"]

    if not _wq_configured("primary"):
        return json.dumps({"error": "WQ BRAIN 未配置 — 请设置 WQ_BRAIN_EMAIL 和 WQ_BRAIN_PASSWORD"})
    if auto_submit:
        return json.dumps(
            {
                "error": "auto_submit 已停用；请先研究并进入 Candidate Queue，再调用 wq_brain_submit_by_ids",
                "reason": "auto_submit_disabled_use_candidate_pipeline",
            },
            ensure_ascii=False,
        )

    total = len(regions) * len(delays) * len(universes) * len(neutralizations)
    if total > 36:
        return json.dumps({"error": f"组合数 {total} 超过上限 36"})

    params = {
        "expression": expression,
        "tag": tag,
        "regions": regions,
        "delays": delays,
        "universes": universes,
        "neutralizations": neutralizations,
        "decay": decay,
        "truncation": truncation,
        "auto_submit": auto_submit,
    }
    return await _enqueue_mcp_tool(
        "wq_brain_batch",
        expression,
        params,
        _run_wq_batch_mcp_task,
        legacy_wq_poll=True,
        total_combinations=total,
    )


@mcp.tool()
async def wq_brain_research(
    expressions: list[str],
    tag: str = "wq-research",
    goal: str = "",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_candidates: int = 20,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    skip_existing: bool = True,
    sweep_top_n: int = 0,
    sweep_universes: list[str] | None = None,
    sweep_neutralizations: list[str] | None = None,
) -> str:
    """批量研究 WQ Alpha：去重、真实 Simulation、筛选、诊断和可选参数扫描。

    本工具是 Autonomous WQ Research 的确定性执行层：调用方/ChatGPT 负责生成候选表达式，
    QuantGPT 负责使用真实 BRAIN 数据进行验证和淘汰。无论结果多好，本工具都不会正式提交 Alpha。

    Args:
        expressions: 候选 FASTEXPR 列表，默认最多研究 20 个。
        tag: 本轮研究标识。
        goal: 研究目标/假设说明，仅用于记录结果。
        region/universe/delay/decay/neutralization/truncation: BRAIN Simulation 参数。
        min_sharpe/min_fitness: 候选筛选阈值。
        skip_existing: 跳过账号最近已研究过的完全相同表达式。
        sweep_top_n: 对排名前 N 个候选执行额外参数扫描，0 表示关闭。
        sweep_universes/sweep_neutralizations: 参数扫描候选集合。

    Returns:
        JSON with task_id. Research runs in the background; poll with get_task_status.
    """
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured("primary"):
        return json.dumps({"error": "WQ BRAIN 未配置 — 请设置 WQ_BRAIN_EMAIL 和 WQ_BRAIN_PASSWORD"})
    if not expressions:
        return json.dumps({"error": "expressions 不能为空"})
    if max_candidates < 1 or max_candidates > 50:
        return json.dumps({"error": "max_candidates 必须在 1~50 之间"})
    if sweep_top_n < 0 or sweep_top_n > 5:
        return json.dumps({"error": "sweep_top_n 必须在 0~5 之间"})

    params = {
        "expressions": expressions[:max_candidates],
        "tag": tag,
        "goal": goal,
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "max_candidates": max_candidates,
        "min_sharpe": min_sharpe,
        "min_fitness": min_fitness,
        "skip_existing": skip_existing,
        "sweep_top_n": sweep_top_n,
        "sweep_universes": sweep_universes,
        "sweep_neutralizations": sweep_neutralizations,
    }
    return await _enqueue_mcp_tool(
        "wq_research",
        expressions[0],
        params,
        _run_wq_research_mcp_task,
        legacy_wq_poll=True,
        singleflight=True,
        stale_after_seconds=_WQ_RESEARCH_STALE_SECONDS,
        requested=len(params["expressions"]),
    )


@mcp.tool()
async def wq_brain_autonomous_research(
    goal: str = "maximize robust low-correlation WorldQuant candidates",
    tag: str = "wq-autonomous",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_simulations: int = 20,
    generations: int = 2,
    family_count: int = 3,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
) -> str:
    """自主规划并研究 WQ Alpha，不需要调用方手工提供 expressions。

    自动读取 Research Memory、近期 BRAIN Alpha 和账号实时 Data Explorer 字段；在可用时
    使用现有 DeepSeek/OpenAI-compatible provider 生成少量结构创新 FASTEXPR，并以真实
    BRAIN Simulation 验证。``max_simulations`` 是主研究 generations 的 Simulation 预算；
    Robustness Validation 使用独立、显式上报的有界预算。Primary Pass 还会经过有限的跨
    Universe/Neutralization Robustness Funnel，只有 READY Candidate 才进入正式候选库存。工具永远不会正式提交
    Alpha，正式提交仍由 Submission Gate 控制。

    Returns:
        JSON with task_id. Uses the same single-flight ``wq_research`` gate as manual research.
    """
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured("primary"):
        return json.dumps({"error": "WQ BRAIN 未配置 — 请设置 WQ_BRAIN_EMAIL 和 WQ_BRAIN_PASSWORD"})
    if max_simulations < 4 or max_simulations > 40:
        return json.dumps({"error": "max_simulations 必须在 4~40 之间"})
    if generations < 1 or generations > 3:
        return json.dumps({"error": "generations 必须在 1~3 之间"})
    if family_count < 1 or family_count > 4:
        return json.dumps({"error": "family_count 必须在 1~4 之间"})

    params = {
        "goal": goal,
        "tag": tag,
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "max_simulations": max_simulations,
        "generations": generations,
        "family_count": family_count,
        "min_sharpe": min_sharpe,
        "min_fitness": min_fitness,
    }
    return await _enqueue_mcp_tool(
        "wq_research",
        None,
        params,
        _run_wq_autonomous_research_mcp_task,
        legacy_wq_poll=True,
        singleflight=True,
        stale_after_seconds=_WQ_RESEARCH_STALE_SECONDS,
        autonomous=True,
        max_simulations=max_simulations,
        generations=generations,
    )


@mcp.tool()
async def wq_brain_submit_by_ids(
    alpha_ids: list[str],
    account: str = "primary",
) -> str:
    """批量提交已模拟的 alpha（通过 alpha_id 直接提交，无需重新模拟）。

    用于提交之前模拟过但未正式提交的 A 级 alpha。逐个处理，
    每个 alpha 等待 SC 检查结果（最长 120s）。

    Args:
        alpha_ids: 要提交的 alpha_id 列表（最多 50 个）
        account: WQ 账号（提交只能用 'primary'）

    Returns:
        JSON with task_id. Submission and SC polling run in the background.
    """
    from .wq_brain_client import is_configured as _wq_configured

    if account != "primary":
        return json.dumps({"error": "Alpha 提交仅允许 primary 账号"})
    if not _wq_configured(account):
        return json.dumps({"error": "WQ BRAIN 未配置"})
    if len(alpha_ids) > 50:
        return json.dumps({"error": f"alpha_ids 数量 {len(alpha_ids)} 超过上限 50"})

    params = {"alpha_ids": alpha_ids, "account": account}
    return await _enqueue_mcp_tool(
        "wq_brain_submit_by_ids",
        None,
        params,
        _run_wq_submit_by_ids_mcp_task,
        legacy_wq_poll=True,
        total=len(alpha_ids),
    )


@mcp.tool()
async def wq_brain_account_status(account: str = "primary") -> str:
    """读取当前 WorldQuant BRAIN 账号的 Points、等级和 Alpha 数量。

    这是 Autonomous WQ Research 的停止条件/进度工具。Points 优先使用 BRAIN Challenge
    leaderboard score；Gold 使用平台实际 genius level，progress.level 仅表示下一目标等级。

    Args:
        account: WQ 账号 ('primary' 或 'alt')

    Returns:
        JSON with points, target_points=10000, genius_level, consultant status,
        goal_reached and platform Alpha counts.
    """
    from .wq_brain_client import get_client
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured(account):
        return json.dumps({"error": f"WQ BRAIN 未配置 (account={account})"})

    client = get_client(account)
    platform_candidate_backfill = None
    reservation_recovery = None
    try:
        authenticated = await asyncio.to_thread(client.authenticate)
        if not authenticated:
            return json.dumps({"error": "WQ BRAIN 认证失败"})
        result = await asyncio.to_thread(run_account_status, client)
        if account == "primary" and result.get("ok"):
            platform_candidate_backfill = await asyncio.to_thread(
                run_list_alphas,
                client,
                limit=100,
                offset=0,
                min_fitness=1.0,
                status_filter="UNSUBMITTED",
            )
            from .wq_submission_policy import get_submission_reservation_recovery_ids

            recovery_ids = await get_submission_reservation_recovery_ids(account)
            if recovery_ids:
                reservation_recovery = await asyncio.to_thread(run_check_alphas, client, recovery_ids)
    finally:
        await asyncio.to_thread(client.close)

    if not result.get("ok"):
        return json.dumps({"error": result.get("error", "unknown")}, ensure_ascii=False)
    result["account"] = account
    try:
        from .wq_submission_policy import (
            observe_account_status,
            reconcile_candidate_platform_statuses,
            reconcile_submission_reservations,
            record_platform_candidates,
        )

        if platform_candidate_backfill and platform_candidate_backfill.get("ok"):
            result["platform_candidates_recovered"] = await record_platform_candidates(
                account,
                platform_candidate_backfill.get("alphas", []),
            )
        if reservation_recovery and reservation_recovery.get("alphas"):
            recovery_alphas = reservation_recovery.get("alphas", {})
            result["submission_reservations_reconciled"] = await reconcile_submission_reservations(
                account,
                recovery_alphas,
            )
            result["reservation_candidates_reconciled"] = await reconcile_candidate_platform_statuses(
                account,
                recovery_alphas,
            )
        result["submission_policy"] = await observe_account_status(account, result)
    except Exception as exc:
        logger.warning("Failed to reconcile WQ submission policy: %s", exc)
        result["submission_policy"] = {"error": str(exc)}

    if account == "primary":
        try:
            active_research = await get_active_mcp_task(
                "wq_research",
                stale_after_seconds=_WQ_RESEARCH_STALE_SECONDS,
            )
            active_summary = None
            if active_research:
                active_summary = {
                    key: active_research.get(key)
                    for key in (
                        "task_id",
                        "status",
                        "created_at",
                        "updated_at",
                        "progress",
                        "progress_current",
                        "progress_total",
                        "progress_message",
                    )
                    if active_research.get(key) is not None
                }
            result["research_gate"] = {
                "mode": "singleflight",
                "can_start_new": active_summary is None,
                "active_task": active_summary,
                "stale_after_seconds": _WQ_RESEARCH_STALE_SECONDS,
            }
            from .wq_control_tower import build_research_control_tower
            from .wq_research_memory import load_research_memory

            memory = await load_research_memory(account, limit=2000)
            result["research_learning"] = build_research_control_tower(
                memory,
                result.get("submission_policy") or {},
            )
            result["research_memory"] = {
                "trials": memory.get("trials", 0),
                "family_counts": memory.get("family_counts", {}),
                "candidate_family_counts": memory.get("candidate_family_counts", {}),
                "self_correlation_family_counts": memory.get("self_correlation_family_counts", {}),
                "status_counts": memory.get("status_counts", {}),
                "failure_stage_counts": memory.get("failure_stage_counts", {}),
                "failure_reason_counts": memory.get("failure_reason_counts", {}),
                "metadata_completeness": memory.get("metadata_completeness", {}),
                "provenance": memory.get("provenance", {}),
                "field_registry_count": len(memory.get("field_registry") or {}),
                "learning_maturity": memory.get("learning_maturity", {}),
                "research_memory_guidance": memory.get("research_memory_guidance", {}),
                "knowledge_guidance": {
                    "policy": (memory.get("knowledge_guidance") or {}).get("policy"),
                    "active_cards": len((memory.get("knowledge_guidance") or {}).get("cards") or []),
                    "preferred_templates": len((memory.get("knowledge_guidance") or {}).get("preferred_templates") or []),
                    "family_confidence": (memory.get("knowledge_guidance") or {}).get("family_confidence", {}),
                },
                "candidate_funnel": memory.get("candidate_funnel", {}),
                "learning_funnel": memory.get("learning_funnel", {}),
                "research_cells": memory.get("research_cells", []),
                "local_correlation_risk": memory.get("local_correlation_risk", {}),
                "overfitting_evidence": memory.get("overfitting_evidence", {}),
                "active_conversion": memory.get("active_conversion", {}),
                "family_points_feedback": memory.get("family_points_feedback", {}),
                "dataset_points_feedback": memory.get("dataset_points_feedback", {}),
                "operator_points_feedback": memory.get("operator_points_feedback", {}),
                "points_feedback_coverage": memory.get("points_feedback_coverage", {}),
                "points_feedback_gate": memory.get("points_feedback_gate", {}),
                "points_attribution_rule": memory.get("points_attribution_rule"),
            }
        except Exception as exc:
            logger.warning("Failed to read WQ research gate: %s", exc)
            result["research_gate"] = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def wq_brain_list_alphas(
    account: str = "primary",
    limit: int = 100,
    offset: int = 0,
    min_fitness: float | None = None,
    status_filter: str | None = None,
) -> str:
    """列出 WQ BRAIN 平台上的所有 alpha（包括已模拟未提交的）。

    可按 fitness 下限和状态过滤。返回 alpha_id、表达式、指标。

    Args:
        account: WQ 账号 ('primary' 或 'alt')
        limit: 返回数量上限（最大 100）
        offset: 分页偏移
        min_fitness: 最低 fitness 过滤（如 1.0 只看 A 级）
        status_filter: 状态过滤（如 'UNSUBMITTED' 或 'ACTIVE'）

    Returns:
        JSON with alpha list, each containing alpha_id, expression, metrics.
    """
    from .wq_brain_client import get_client
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured(account):
        return json.dumps({"error": f"WQ BRAIN 未配置 (account={account})"})

    client = get_client(account)
    authenticated = await asyncio.to_thread(client.authenticate)
    if not authenticated:
        return json.dumps({"error": "WQ BRAIN 认证失败"})

    result = await asyncio.to_thread(
        run_list_alphas,
        client,
        limit=limit,
        offset=offset,
        min_fitness=min_fitness,
        status_filter=status_filter,
    )
    await asyncio.to_thread(client.close)

    if not result.get("ok"):
        return json.dumps({"error": result.get("error", "unknown")})

    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def wq_brain_check_alphas(
    alpha_ids: list[str],
    account: str = "primary",
) -> str:
    """批量查询 alpha 在 WQ BRAIN 平台上的状态。

    返回每个 alpha 的状态（ACTIVE/UNSUBMITTED）、SC 检查结果、指标。

    Args:
        alpha_ids: 要查询的 alpha_id 列表（最多 50 个）
        account: WQ 账号 ('primary' 或 'alt')

    Returns:
        JSON with summary and per-alpha status. For backward-compatible task
        polling, pass a single pseudo id in the form ``task:<task_id>``.
    """
    if len(alpha_ids) == 1 and str(alpha_ids[0]).startswith("task:"):
        task_id = str(alpha_ids[0])[5:]
        task = await get_mcp_task_snapshot(task_id)
        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"}, ensure_ascii=False)
        status = task.get("status")
        task["done"] = status in ("completed", "failed", "cancelled", "iteration_completed")
        if not task["done"]:
            task["poll_after_seconds"] = 10
        return json.dumps({"task": task}, ensure_ascii=False, indent=2, default=str)

    from .wq_brain_client import get_client
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured(account):
        return json.dumps({"error": f"WQ BRAIN 未配置 (account={account})"})
    if len(alpha_ids) > 50:
        return json.dumps({"error": f"alpha_ids 数量 {len(alpha_ids)} 超过上限 50"})

    client = get_client(account)
    authenticated = await asyncio.to_thread(client.authenticate)
    if not authenticated:
        return json.dumps({"error": "WQ BRAIN 认证失败"})

    result = await asyncio.to_thread(run_check_alphas, client, alpha_ids)
    await asyncio.to_thread(client.close)

    try:
        from .wq_submission_policy import reconcile_candidate_platform_statuses

        result["candidate_queue_reconciled"] = await reconcile_candidate_platform_statuses(
            account,
            result.get("alphas") or {},
        )
    except Exception as exc:
        logger.warning("Failed to reconcile WQ candidate queue from platform checks: %s", exc)
        result["candidate_queue_reconciled"] = 0

    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def wq_brain_finalize_submissions(
    alpha_ids: list[str],
    account: str = "primary",
) -> str:
    """查询已提交 alpha 的最终 SC 检查结果。

    提交 alpha 后 SC 检查可能需要数小时。初次提交 SC 超时的 alpha 用此工具查询最终结果。
    会自动更新 DB 中已解决 alpha 的状态（ACTIVE / SC_FAIL）。

    Args:
        alpha_ids: 要查询最终状态的 alpha_id 列表（最多 100 个）
        account: WQ 账号 ('primary' 或 'alt')

    Returns:
        JSON with task_id. Final SC status collection runs in the background.
    """
    from .wq_brain_client import is_configured as _wq_configured

    if not _wq_configured(account):
        return json.dumps({"error": f"WQ BRAIN 未配置 (account={account})"})
    if len(alpha_ids) > 100:
        return json.dumps({"error": f"alpha_ids 数量 {len(alpha_ids)} 超过上限 100"})

    params = {"alpha_ids": alpha_ids, "account": account}
    return await _enqueue_mcp_tool(
        "wq_brain_finalize",
        None,
        params,
        _run_wq_finalize_mcp_task,
        legacy_wq_poll=True,
        total=len(alpha_ids),
    )


@mcp.tool()
async def compute_factor_values(
    expression: str,
    universe: str = "csi500",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """异步计算因子截面值，并把大结果写入分页文件。

    用于下游组合构建、因子库上传或外部分析。
    立即返回 task_id。任务完成后使用 get_factor_values_page 分页读取，避免巨大 MCP JSON。

    Args:
        expression: 因子表达式（如 rank(ts_mean(close/open, 10))）
        universe: 股票池（hs300 / csi500 / csi1000 / csi2000）
        start_date: 起始日期 YYYY-MM-DD（默认 end_date 前 365 天）
        end_date: 截止日期 YYYY-MM-DD（默认今天）
    """
    params = {
        "expression": expression,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
    }
    return await _enqueue_mcp_tool(
        "compute_factor_values",
        expression,
        params,
        _run_compute_factor_values_mcp_task,
    )


@mcp.tool()
async def get_factor_values_page(
    task_id: str,
    page: int = 1,
    page_size: int = 5,
) -> str:
    """分页读取 compute_factor_values 的结果，每条记录对应一个交易日。"""
    if page < 1:
        return json.dumps({"error": "page must be >= 1"}, ensure_ascii=False)
    if page_size < 1 or page_size > 20:
        return json.dumps({"error": "page_size must be between 1 and 20"}, ensure_ascii=False)

    task = await get_mcp_task_snapshot(task_id)
    if task is None:
        return json.dumps({"error": f"Task not found: {task_id}"}, ensure_ascii=False)
    if task.get("task_type") != "compute_factor_values":
        return json.dumps({"error": f"Task {task_id} is not a factor-values task"}, ensure_ascii=False)
    if task.get("status") != "completed":
        return json.dumps(
            {
                "task_id": task_id,
                "status": task.get("status"),
                "error": task.get("error"),
            },
            ensure_ascii=False,
        )

    result = task.get("result") or {}
    relative_path = (result.get("result_storage") or {}).get("path")
    if not relative_path:
        return json.dumps({"error": "Factor-values artifact is missing from task result"}, ensure_ascii=False)
    artifact = (_PROJECT_ROOT / relative_path).resolve()
    artifact_root = _FACTOR_VALUES_DIR.resolve()
    if not artifact.is_relative_to(artifact_root) or not artifact.is_file():
        return json.dumps({"error": "Factor-values artifact is unavailable"}, ensure_ascii=False)

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    rows = []
    with gzip.open(artifact, "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start_index:
                continue
            if index >= end_index:
                break
            rows.append(json.loads(line))

    trading_days = int(result.get("trading_days", 0))
    total_pages = math.ceil(trading_days / page_size) if trading_days else 0
    return json.dumps(
        {
            "task_id": task_id,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "trading_days": trading_days,
            "has_more": page < total_pages,
            "data": rows,
        },
        ensure_ascii=False,
        default=str,
    )


# Operator documentation fallback
_OPERATORS_DOC = """
因子表达式操作符:

一元函数: rank, zscore, sign, log, abs, scale, tanh, sigmoid, exp, sqrt
时序函数: ts_mean, ts_std, ts_max, ts_min, ts_sum, ts_shift, ts_delta, ts_rank, ts_argmax, ts_argmin, decay_linear, product
  用法: ts_mean(close, 20) — 20日均值
双列时序: ts_corr(col1, col2, N), ts_cov(col1, col2, N)
二元函数: power, max, min
条件函数: clip(expr, lo, hi), where(cond, true_val, false_val)
算术运算: +, -, *, /, ^
比较运算: >, <, >=, <=, ==, !=
特殊变量: vwap, returns, adv{N} (如 adv20)
可用列名: open, high, low, close, volume, amount, pct_change
别名: delta=ts_delta, delay=ts_shift, correlation=ts_corr, covariance=ts_cov
"""
