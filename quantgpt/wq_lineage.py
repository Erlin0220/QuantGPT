"""Stable lineage and conservative metadata recovery for WQ research."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .wq_operator_registry import canonicalize_wq_expression

_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_OPERATOR_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_BUILTIN_NAMES = {
    "true",
    "false",
    "nan",
    "inf",
    "market",
    "sector",
    "industry",
    "subindustry",
}


def canonical_expression(expression: str) -> str:
    value = str(expression or "").strip()
    if not value:
        return ""
    try:
        value = canonicalize_wq_expression(value)
    except Exception:
        pass
    return re.sub(r"\s+", "", value).lower()


def extract_expression_metadata(expression: str) -> dict[str, Any]:
    """Extract only provenance that is deterministic from the expression itself."""
    canonical = canonical_expression(expression)
    operators = [match.group(1).lower() for match in _OPERATOR_RE.finditer(canonical)]
    operator_set = set(operators)
    identifiers = [token.lower() for token in _IDENTIFIER_RE.findall(canonical)]
    fields: list[str] = []
    for token in identifiers:
        if token in operator_set or token in _BUILTIN_NAMES:
            continue
        if token.startswith("adv") and token[3:].isdigit():
            # ADV aliases are derived from volume, not a distinct source field.
            token = "volume"
        if token not in fields:
            fields.append(token)
    operator_pattern = ">".join(operators) if operators else "raw_field"
    structure_signature = re.sub(r"(?<![a-z_])\d+(?:\.\d+)?", "#", canonical)
    return {
        "canonical_expression": canonical,
        "operators": operators,
        "operator_pattern": operator_pattern,
        "structure_signature": structure_signature,
        "data_fields": fields,
    }


def classify_family_from_metadata(expression: str, data_fields: list[str] | None = None) -> str:
    """Classify from recovered field evidence before falling back to broad expression tokens."""
    metadata = extract_expression_metadata(expression)
    fields = {str(value).lower() for value in (data_fields or metadata["data_fields"])}
    tokens = fields | set(metadata["operators"])
    if any("analyst" in value or "revision" in value or "estimate" in value for value in tokens):
        return "analyst_revision"
    if any(
        key in value
        for value in tokens
        for key in ("option", "implied_vol", "historical_vol", "open_interest", "pcr_")
    ):
        return "options_volatility"
    if any("sentiment" in value or value.startswith("snt_") or "social" in value for value in tokens):
        return "sentiment"
    if any(
        key in value
        for value in fields
        for key in (
            "income",
            "cashflow",
            "cash_flow",
            "assets",
            "liabil",
            "equity",
            "revenue",
            "margin",
            "debt",
            "dividend",
            "enterprise_value",
            "market_cap",
            "quality",
            "fundamental",
            "mdf_",
        )
    ):
        return "fundamental_quality"
    if fields & {"volume", "vwap", "turnover"} or any(value.startswith("adv") for value in fields):
        return "price_volume"
    if any(value in tokens for value in ("ts_std", "std_dev", "volatility", "ts_argmax", "ts_argmin")):
        return "volatility_structure"
    if fields & {"returns", "close", "open", "high", "low"}:
        return "momentum_reversal"
    return "other" if canonical_expression(expression) else "unknown"


def recover_research_metadata(
    expression: str,
    research_meta: dict[str, Any] | None = None,
    *,
    platform_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit provenance with expression-derived evidence without inventing datasets."""
    explicit = dict(research_meta or {})
    platform = dict(platform_meta or {})
    derived = extract_expression_metadata(expression)

    explicit_fields = explicit.get("data_fields") or platform.get("data_fields") or platform.get("fields")
    fields = [str(value) for value in (explicit_fields or derived["data_fields"]) if str(value)]
    family = str(explicit.get("family") or platform.get("family") or "").strip()
    if not family or family == "unknown":
        family = classify_family_from_metadata(expression, fields)

    dataset_id = str(
        explicit.get("dataset_id")
        or platform.get("dataset_id")
        or platform.get("datasetId")
        or ""
    ).strip() or None

    return {
        **explicit,
        "family": family or "unknown",
        "dataset_id": dataset_id,
        "data_fields": fields,
        "operators": list(explicit.get("operators") or derived["operators"]),
        "operator_pattern": str(explicit.get("operator_pattern") or derived["operator_pattern"]),
        "structure_signature": str(explicit.get("structure_signature") or derived["structure_signature"]),
        "canonical_expression": derived["canonical_expression"],
    }


def lineage_id_for(
    expression: str,
    *,
    settings: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Stable lineage identity: equivalent formatting dedupes; parameter changes remain distinct."""
    # Lineage identity intentionally excludes mutable/recoverable labels such as
    # family, hypothesis and planner strategy. That makes a parent ID
    # reconstructable from its expression + simulation settings after restart,
    # while distinct expression parameters/settings still remain distinct.
    _ = metadata
    stable_identity = {
        "expression": canonical_expression(expression),
        "settings": dict(settings or {}),
    }
    payload = json.dumps(stable_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "wql_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def parent_lineage_id_for(parent_expression: str | None, *, settings: dict[str, Any] | None = None) -> str | None:
    value = str(parent_expression or "").strip()
    if not value:
        return None
    return lineage_id_for(value, settings=settings, metadata={})
