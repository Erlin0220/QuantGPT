"""Factor iteration optimization — QuantGPT
Copyright (c) 2026 Miasyster. Licensed under the MIT License.
https://github.com/Miasyster/QuantGPT

Scoring, prompt building, candidate generation.

Refactored with QuantaAlpha three-phase evolution architecture:
  Phase 1: TrajectoryAnalyzer — trajectory quality metrics
  Phase 2: MetaEvolutionSelector — adaptive strategy selection
  Phase 3: Strategy execution (Mutation / Crossover / Explore)
"""

import logging
import re
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

from .expression_parser import parse_expression
from .report import generate_report
from .task_executor import _run_backtest_in_process, get_executor

logger = logging.getLogger(__name__)


# ---- Factor scoring (unchanged) ----

def compute_factor_score(
    backtest_summary: dict,
    report_metrics: dict,
    anti_overfit_score: float | None = None,
    data_days: int | None = None,
) -> dict:
    """Compute a composite 0-100 score for a factor backtest result.

    6-component scoring with Cloud alignment as primary external target:
      IC Mean 15%, IC IR 15%, Stability 15%, Anti-Overfit 15%,
      Group BT 15%, Cloud Alignment 25%.
    """
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    ic_mean = backtest_summary.get("ic_mean", 0.0) or backtest_summary.get("rank_ic_mean", 0.0)
    ic_mean_score = min(abs(ic_mean) / 0.05, 1.0) * 100

    ic_ir = backtest_summary.get("ic_ir", 0.0)
    ic_ir_score = min(abs(ic_ir) / 1.0, 1.0) * 100

    ic_win_rate = backtest_summary.get("ic_win_rate", 0.5)
    ic_wr_sub = min(max(ic_win_rate - 0.5, 0) / 0.2, 1.0) * 100
    ls_sharpe = backtest_summary.get("long_short_sharpe", 0.0)
    ls_consistency_sub = min(abs(ls_sharpe) / 2.0, 1.0) * 100
    stability_score = ic_wr_sub * 0.6 + ls_consistency_sub * 0.4

    ao_score = _clamp(anti_overfit_score, 0, 100) if anti_overfit_score is not None else 50.0

    ls_sharpe_gb = min(max(ls_sharpe, 0) / 1.0, 1.0) * 100
    mono = backtest_summary.get("monotonicity_score", 0.0)
    mono_sub = _clamp(mono, 0, 1) * 100
    spread = backtest_summary.get("spread", 0.0)
    top_positive_sub = 100.0 if spread > 0 else 0.0
    group_bt_score = ls_sharpe_gb * 0.4 + mono_sub * 0.4 + top_positive_sub * 0.2

    # Cloud Alignment: IC Mean (30%) + IC IR (30%) + Turnover (20%) + Data Sufficiency (20%)
    cloud_ic_mean = abs(ic_mean)
    cloud_ic_ir = abs(ic_ir)
    cloud_turnover = backtest_summary.get("turnover", 0.0)
    cloud_data_days = data_days if data_days is not None else 120

    cloud_ic_mean_sub = min(cloud_ic_mean / 0.03, 1.0) * 100
    cloud_ic_ir_sub = min(cloud_ic_ir / 0.3, 1.0) * 100

    if 0.01 <= cloud_turnover <= 0.35:
        cloud_turnover_sub = 100.0
    elif cloud_turnover > 0.35:
        cloud_turnover_sub = max(0.0, 100.0 - (cloud_turnover - 0.35) / 0.35 * 100)
    else:
        cloud_turnover_sub = 0.0

    cloud_data_sub = min(cloud_data_days / 120, 1.0) * 100
    cloud_alignment_score = (cloud_ic_mean_sub * 0.30 + cloud_ic_ir_sub * 0.30
                             + cloud_turnover_sub * 0.20 + cloud_data_sub * 0.20)

    score = (ic_mean_score * 0.15 + ic_ir_score * 0.15 + stability_score * 0.15
             + ao_score * 0.15 + group_bt_score * 0.15 + cloud_alignment_score * 0.25)
    score = round(_clamp(score, 0, 100), 1)

    grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"

    capped = False
    cap_reason = None
    cagr = report_metrics.get("cagr", 0.0)
    sharpe = report_metrics.get("sharpe", 0.0)
    if cagr < 0 or sharpe < 0:
        if grade in ("A", "B"):
            grade = "C"
            score = min(score, 59.9)
            capped = True
            cap_reason = "negative_cagr" if cagr < 0 else "negative_sharpe"

    cloud_predicted_pass = (
        cloud_ic_mean >= 0.015
        and cloud_ic_ir >= 0.15
        and cloud_turnover <= 0.35
        and cloud_data_days >= 120
    )

    return {
        "score": score, "grade": grade,
        "component_scores": {
            "ic_mean": round(ic_mean_score, 1), "ic_ir": round(ic_ir_score, 1),
            "stability": round(stability_score, 1), "anti_overfit": round(ao_score, 1),
            "group_backtest": round(group_bt_score, 1),
            "cloud_alignment": round(cloud_alignment_score, 1),
        },
        "cloud_predicted_pass": cloud_predicted_pass,
        "capped": capped, "cap_reason": cap_reason,
    }


# ---- Duplicate detection ----

def _normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr.lower())


def is_duplicate_expression(expr: str, existing: list[str]) -> bool:
    norm = _normalize_expression(expr)
    return any(_normalize_expression(e) == norm for e in existing)


# ---- Single candidate evaluation ----

def _evaluate_candidate(
    expression: str, params: dict, market_df: pd.DataFrame, user_id: str,
) -> dict:
    """Run backtest + anti-overfit + report + score for a single expression."""
    n_groups = params.get("n_groups", 5)
    holding_period = params.get("holding_period", 5)
    executor = get_executor()
    future = executor.submit_cpu_work(
        _run_backtest_in_process, market_df, expression, n_groups, holding_period,
    )
    result = future.result(timeout=300)

    # Fast anti-overfit (IC stability + half-life only)
    anti_overfit_result = None
    factor_df = result.get("_factor_df")
    if factor_df is not None and len(factor_df) > 100:
        try:
            from .anti_overfit import AntiOverfitDetector
            detector = AntiOverfitDetector(factor_df, holding_period)
            t1 = detector.test_ic_stability()
            t4 = detector.test_half_life()
            fast_passed = sum(1 for t in [t1, t4] if t.passed)
            anti_overfit_result = {
                "score": fast_passed / 2 * 100,
                "recommendation": "推荐" if fast_passed == 2 else "谨慎" if fast_passed == 1 else "需改进",
                "tests": [{"name": t.name, "passed": t.passed, "details": t.details} for t in [t1, t4]],
            }
        except Exception as e:
            logger.warning(f"Anti-overfit failed: {e}")

    # Generate report
    from .market_data import fetch_benchmark_returns
    bm_returns = None
    try:
        bm_returns = fetch_benchmark_returns(
            params.get("benchmark", "hs300"),
            params.get("start_date", "2023-01-01"),
            params.get("end_date", "2025-12-31"),
        )
    except Exception:
        pass

    user_report_dir = Path(__file__).resolve().parent.parent / "reports" / user_id
    user_report_dir.mkdir(parents=True, exist_ok=True)
    report_result = generate_report(
        result["strategy_returns"], benchmark_returns=bm_returns,
        title="Factor Top-Group Backtest", output_dir=str(user_report_dir),
    )
    report_filename = Path(report_result["report_path"]).name

    # Score
    ao_val = anti_overfit_result.get("score") if anti_overfit_result else None
    backtest_summary = {
        "long_short_sharpe": result["long_short_sharpe"],
        "monotonicity_score": result["monotonicity_score"],
        "spread": result["spread"],
        "ic_mean": result.get("ic_mean", 0),
        "rank_ic_mean": result.get("rank_ic_mean", 0),
        "ic_ir": result.get("ic_ir", 0),
        "ic_win_rate": result.get("ic_win_rate", 0),
        "long_short_annual": result.get("long_short_annual", 0),
        "top_group_sharpe": result.get("top_group_sharpe", 0),
        "group_returns": result["group_returns"],
        "turnover": result.get("turnover", 0),
        "wq_fitness": result.get("wq_fitness", 0),
    }
    scoring = compute_factor_score(backtest_summary, report_result["metrics"], ao_val)

    cloud_validation = None
    if scoring["grade"] == "A" and factor_df is not None:
        try:
            from .cloud_client import auto_upload_to_cloud
            cloud_validation = auto_upload_to_cloud(
                expression=expression,
                universe=params.get("universe", "hs300"),
                factor_df=factor_df,
                claimed_ic_mean=result.get("ic_mean"),
                claimed_ic_ir=result.get("ic_ir"),
            )
        except Exception as e:
            logger.warning(f"Cloud auto-upload failed for iteration candidate: {e}")

    return {
        "expression": expression,
        "status": "success",
        "score": scoring["score"],
        "grade": scoring["grade"],
        "component_scores": scoring["component_scores"],
        "backtest_summary": backtest_summary,
        "wq_brain": result.get("wq_brain", {}),
        "anti_overfit": anti_overfit_result,
        "cloud_validation": cloud_validation,
        "report_metrics": report_result["metrics"],
        "report_url": f"/api/v1/reports/{report_filename}",
        "report_filename": report_filename,
        "metrics": {"backtest_summary": backtest_summary, "report_metrics": report_result["metrics"]},
    }


# ---- Main adaptive iteration loop ----

def _validate_expression(expr: str) -> str | None:
    """Validate expression syntax. Returns error string or None if valid."""
    from .llm_service import validate_parentheses as _validate_parentheses
    paren_err = _validate_parentheses(expr)
    if paren_err:
        return f"括号错误: {paren_err}"
    try:
        from .fundamental_data import ALL_FUNDAMENTAL_NAMES as _FN
        dummy = pd.DataFrame({
            "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9], "close": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300], "amount": [100, 400, 900],
            "pct_change": [0, 100, 50],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            **{name: [1.0, 1.1, 1.2] for name in _FN},
        })
        func = parse_expression(expr)
        func(dummy)
        return None
    except Exception as e:
        return f"表达式验证失败: {e}"


def generate_iteration_candidates(
    parent_expression: str,
    parent_metrics: dict,
    parent_score: float,
    parent_grade: str,
    params: dict,
    market_df: pd.DataFrame,
    user_id: str,
    n_candidates: int = 5,
    max_concurrent: int = 50,
    on_progress: Callable[[int, dict], None] | None = None,
    task_id: str = "",
    direction: str | None = None,
    chatgpt_expressions: list[str] | None = None,
) -> list[dict]:
    """Validate and evaluate candidate expressions authored by the ChatGPT client."""
    del parent_metrics, parent_score, parent_grade, max_concurrent

    supplied = list(chatgpt_expressions or [])[:n_candidates]
    if not supplied:
        raise RuntimeError(
            "未提供 ChatGPT 候选表达式。请先让 ChatGPT 根据 diagnose_factor/mutation_prompt 生成候选，"
            "再通过 chatgpt_expressions 提交评估。"
        )

    all_expressions = [parent_expression]
    candidates: list[dict] = []
    for i, raw_expression in enumerate(supplied):
        expression = str(raw_expression or "").strip()
        try:
            if not expression:
                raise ValueError("候选表达式为空")
            if is_duplicate_expression(expression, all_expressions):
                raise ValueError("候选表达式与父因子或本轮候选重复")
            err = _validate_expression(expression)
            if err:
                raise ValueError(err)

            all_expressions.append(expression)
            result = _evaluate_candidate(expression, params, market_df, user_id)
            result["strategy_used"] = "chatgpt_client"
            if direction:
                result["direction"] = direction
            candidates.append(result)
        except Exception as e:
            logger.error(f"[{task_id}] ChatGPT candidate {i} failed: {traceback.format_exc()}")
            result = {"expression": expression or "unknown", "status": "failed", "error": str(e), "score": 0}
            candidates.append(result)

        if on_progress:
            on_progress(len(candidates), result)

    candidates.sort(key=lambda c: (c.get("status") == "success", c.get("score", 0)), reverse=True)
    return candidates


# Legacy alias
build_iterate_prompt = None  # Removed — prompts now built inside generate_iteration_candidates
