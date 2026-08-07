"""WorldQuant BRAIN operator catalog, validation, and local-to-BRAIN aliases.

The live BRAIN ``/operators`` endpoint is the source of truth when credentials
are available. ``WQ_FALLBACK_OPERATORS`` is only an offline fallback so WQ
validation can still reject known-local-only operators without network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Snapshot of the operators returned by https://api.worldquantbrain.com/operators
# for the configured account on 2026-08-07. Live catalog results take priority.
WQ_FALLBACK_OPERATORS = frozenset(
    {
        "abs",
        "add",
        "and",
        "bucket",
        "days_from_last_change",
        "densify",
        "divide",
        "equal",
        "greater",
        "greater_equal",
        "group_backfill",
        "group_mean",
        "group_neutralize",
        "group_rank",
        "group_scale",
        "group_zscore",
        "hump",
        "if_else",
        "inverse",
        "is_nan",
        "kth_element",
        "last_diff_value",
        "less",
        "less_equal",
        "log",
        "max",
        "min",
        "multiply",
        "normalize",
        "not",
        "not_equal",
        "or",
        "power",
        "quantile",
        "rank",
        "reverse",
        "scale",
        "sign",
        "signed_power",
        "sqrt",
        "subtract",
        "trade_when",
        "ts_arg_max",
        "ts_arg_min",
        "ts_av_diff",
        "ts_backfill",
        "ts_corr",
        "ts_count_nans",
        "ts_covariance",
        "ts_decay_linear",
        "ts_delay",
        "ts_delta",
        "ts_mean",
        "ts_product",
        "ts_quantile",
        "ts_rank",
        "ts_regression",
        "ts_scale",
        "ts_std_dev",
        "ts_step",
        "ts_sum",
        "ts_zscore",
        "vec_avg",
        "vec_sum",
        "winsorize",
        "zscore",
    }
)


# QuantGPT/local spellings that have a direct BRAIN equivalent. These aliases are
# canonicalized before validation and before a simulation is sent to BRAIN.
WQ_OPERATOR_ALIASES = {
    "correlation": "ts_corr",
    "covariance": "ts_covariance",
    "decay_linear": "ts_decay_linear",
    "delay": "ts_delay",
    "delta": "ts_delta",
    "product": "ts_product",
    "sign_power": "signed_power",
    "ts_argmax": "ts_arg_max",
    "ts_argmin": "ts_arg_min",
    "ts_cov": "ts_covariance",
    "ts_shift": "ts_delay",
    "ts_std": "ts_std_dev",
    "where": "if_else",
}


WQ_OPERATOR_HINTS = {
    "clip": "use max(lo, min(hi, x))",
    "decay_linear": "use ts_decay_linear(x, d)",
    "ema": "use ts_decay_linear(x, d)",
    "exp": "use power(2.718, x)",
    "indneutralize": "use group_neutralize(x, group)",
    "product": "use ts_product(x, d)",
    "sign_power": "use signed_power(x, y)",
    "sigmoid": "use rank(x) or another available BRAIN transform",
    "sma": "use ts_mean(x, d)",
    "tanh": "use signed_power(x, 0.5) or x / (1 + abs(x))",
    "ts_argmax": "use ts_arg_max(x, d)",
    "ts_argmin": "use ts_arg_min(x, d)",
    "ts_cov": "use ts_covariance(x, y, d)",
    "ts_shift": "use ts_delay(x, d)",
    "ts_std": "use ts_std_dev(x, d)",
    "where": "use if_else(condition, true_value, false_value)",
    "wma": "use ts_decay_linear(x, d)",
}


_FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class WQValidationResult:
    expression: str
    operators: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.unsupported

    @property
    def error(self) -> str:
        if self.ok:
            return ""
        parts = []
        for operator in self.unsupported:
            hint = WQ_OPERATOR_HINTS.get(operator)
            parts.append(f"{operator} ({hint})" if hint else operator)
        return "Unsupported/inaccessible WQ BRAIN operator(s): " + ", ".join(parts)


def canonicalize_wq_expression(expression: str) -> str:
    """Translate QuantGPT/local function aliases to exact BRAIN operator names."""
    result = expression
    for alias, canonical in sorted(WQ_OPERATOR_ALIASES.items(), key=lambda item: -len(item[0])):
        result = re.sub(
            rf"\b{re.escape(alias)}(?=\s*\()",
            canonical,
            result,
            flags=re.IGNORECASE,
        )
    return result


def extract_wq_operators(expression: str) -> tuple[str, ...]:
    """Return function identifiers used by an expression, normalized to lowercase."""
    return tuple(sorted({match.group(1).lower() for match in _FUNC_RE.finditer(expression)}))


def validate_wq_expression(
    expression: str,
    supported_operators: Iterable[str] | None = None,
) -> WQValidationResult:
    """Validate operator availability against a live or fallback BRAIN catalog.

    This intentionally validates operator names only. BRAIN data-field availability
    depends on account/region/universe and is therefore left to BRAIN simulation.
    """
    canonical = canonicalize_wq_expression(expression)
    supported = {
        name.lower() for name in (supported_operators if supported_operators is not None else WQ_FALLBACK_OPERATORS)
    }
    operators = extract_wq_operators(canonical)
    unsupported = tuple(op for op in operators if op not in supported)
    return WQValidationResult(
        expression=canonical,
        operators=operators,
        unsupported=unsupported,
    )
