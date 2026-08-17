"""Local Pi Agent / DeepSeek candidate generation for WorldQuant research."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .pi_agent_client import PiAgentError, run_pi_prompt

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_SKILLS = [
    "wq-economic-hypothesis",
    "wq-alpha-hypothesis",
    "wq-alpha-review",
    "wq-robustness-validation",
    "wq-candidate-evidence",
]
_FAILURE_SKILLS = [
    "wq-failure-diagnosis",
    "wq-experiment-allocation",
    "wq-alpha-repair",
]
_DIVERSIFY_SKILLS = ["wq-alpha-diversify"]
_DEFAULT_MODEL = "opencode-go/deepseek-v4-flash"
_PI_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
_CORE_DATA_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "returns",
    "cap",
    "adv20",
    "adv60",
    "adv120",
}


class LocalCandidateGenerationError(RuntimeError):
    """Raised when the local-model candidate generator cannot produce a usable batch."""


def _data_field_is_grounded(field: str, planner_context_text: str) -> bool:
    """Accept core fields or exact field ids already grounded in planner evidence."""
    resolved = str(field or "").strip()
    if not resolved:
        return False
    if resolved in _CORE_DATA_FIELDS:
        return True
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(resolved)}(?![A-Za-z0-9_])", planner_context_text) is not None


def _skill_names_for_context(planner_context: dict[str, Any]) -> list[str]:
    names = list(_BASE_SKILLS)
    if planner_context.get("current_cycle_trial_evidence"):
        names.extend(_FAILURE_SKILLS)
    inventory = planner_context.get("inventory") or {}
    inventory_deficit = int(inventory.get("deficit") or inventory.get("high_confidence_deficit") or 0)
    if planner_context.get("avoid_structure_signatures") or inventory_deficit > 0:
        names.extend(_DIVERSIFY_SKILLS)
    return list(dict.fromkeys(names))


def _compact_skill_text(text: str) -> str:
    """Build a small runtime instruction pack from the actual local Skill text."""
    lines = text.splitlines()
    name = ""
    description = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("name:") and not name:
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("description:") and not description:
            description = stripped.split(":", 1)[1].strip()

    def section(prefix: str) -> list[str]:
        out: list[str] = []
        active = False
        for line in lines:
            if line.startswith("## "):
                active = line.startswith(prefix)
                continue
            if active:
                out.append(line)
        return out

    interpretation = "\n".join(line for line in section("## I ") if line.strip()).strip()[:650]
    boundary_lines = [line.strip() for line in section("## B ") if line.strip().startswith("-")][:3]
    boundary = "\n".join(boundary_lines)[:500]
    parts = [f"name: {name}" if name else "", f"description: {description[:320]}" if description else ""]
    if interpretation:
        parts.append(f"Interpretation:\n{interpretation}")
    if boundary:
        parts.append(f"Boundary:\n{boundary}")
    compact = "\n".join(part for part in parts if part).strip()
    return compact or text.strip()[:1400]


def _load_skill_context(planner_context: dict[str, Any]) -> tuple[list[str], str]:
    names = _skill_names_for_context(planner_context)
    sections: list[str] = []
    missing: list[str] = []
    max_chars = max(4000, int(os.environ.get("WQ_LOCAL_LLM_SKILL_CONTEXT_MAX_CHARS") or 12000))
    used_chars = 0
    loaded_names: list[str] = []
    for name in names:
        path = _REPO_ROOT / ".agents" / "skills" / name / "SKILL.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            missing.append(name)
            continue
        compact = _compact_skill_text(text)
        section = f"\n===== {name} =====\n{compact}\n"
        if sections and used_chars + len(section) > max_chars:
            continue
        sections.append(section)
        loaded_names.append(name)
        used_chars += len(section)
    if missing:
        raise LocalCandidateGenerationError(f"missing local WQ skill files: {', '.join(missing)}")
    if not sections:
        raise LocalCandidateGenerationError("no local WQ Skill context could be loaded")
    return loaded_names, "\n".join(sections)


def _extract_json_payload(content: str) -> Any:
    text = str(content or "").strip()
    if not text:
        raise LocalCandidateGenerationError("LLM returned empty content")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise LocalCandidateGenerationError("LLM response did not contain valid JSON")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _candidate_items_from_payload(decoded: Any) -> list[Any]:
    if isinstance(decoded, list):
        return decoded
    if not isinstance(decoded, dict):
        return []
    for key in ("skill_candidates", "candidates", "alphas"):
        value = decoded.get(key)
        if isinstance(value, list):
            return value
    if decoded.get("expression") and decoded.get("hypothesis"):
        return [decoded]
    result = decoded.get("result")
    if isinstance(result, dict):
        return _candidate_items_from_payload(result)
    return []


def _normalize_candidate(raw: dict[str, Any], planner_context: dict[str, Any] | None = None) -> dict[str, Any]:
    robustness = raw.get("robustness_plan") if isinstance(raw.get("robustness_plan"), dict) else {}
    checks = robustness.get("checks") if isinstance(robustness, dict) else None
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        checks = [{"universe": "TOP1000", "purpose": "test universe sensitivity"}]

    base_chain = [
        "wq-economic-hypothesis",
        "wq-alpha-hypothesis",
        "wq-alpha-review",
        "wq-robustness-validation",
        "wq-candidate-evidence",
    ]
    requested_chain = _string_list(raw.get("skill_chain"))
    route = str(raw.get("route") or "").strip().upper()
    evidence_expressions = {
        str(item.get("expression") or "").strip()
        for item in ((planner_context or {}).get("current_cycle_trial_evidence") or [])
        if isinstance(item, dict) and str(item.get("expression") or "").strip()
    }
    requested_parent = str(raw.get("parent_expression") or "").strip()
    repair_requested = "wq-alpha-repair" in requested_chain or route == "REPAIR"
    if repair_requested and requested_parent in evidence_expressions:
        skill_chain = [*base_chain, "wq-failure-diagnosis", "wq-experiment-allocation", "wq-alpha-repair"]
        route = "REPAIR"
    elif "wq-alpha-diversify" in requested_chain or route == "DIVERSIFY":
        skill_chain = [*base_chain, "wq-alpha-diversify"]
        route = "DIVERSIFY"
    else:
        skill_chain = base_chain
        route = "NEW_HYPOTHESIS"

    candidate = {
        "expression": str(raw.get("expression") or "").strip(),
        "hypothesis": str(raw.get("hypothesis") or "").strip(),
        "family": str(raw.get("family") or "").strip(),
        "data_fields": _string_list(raw.get("data_fields")),
        "knowledge_card_ids": _string_list(raw.get("knowledge_card_ids")),
        "skill_chain": skill_chain,
        "review_decision": "RUN",
        "review_notes": str(raw.get("review_notes") or "local Pi Agent / DSV4 review").strip(),
        "robustness_plan": {"mode": "skill_defined", "checks": checks[:4]},
        "candidate_evidence_policy": {"mode": "calibrated_evidence_hierarchy"},
        "local_llm_route": route,
    }
    for key in ("dataset_id", "dataset_category"):
        value = raw.get(key)
        if value not in (None, ""):
            candidate[key] = value
    if route == "REPAIR":
        candidate["parent_expression"] = requested_parent
        for key in ("mutation_type", "mutation_reason"):
            value = raw.get(key)
            if value not in (None, ""):
                candidate[key] = value
        if isinstance(raw.get("failure_signature"), dict):
            candidate["failure_signature"] = dict(raw["failure_signature"])
    if route == "DIVERSIFY" and isinstance(raw.get("diversity_case"), dict):
        candidate["diversity_case"] = dict(raw["diversity_case"])
    if isinstance(raw.get("settings_delta"), dict):
        candidate["settings_delta"] = dict(raw["settings_delta"])
    return candidate


def _forced_exploration_cells(items: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in (items or [])
        if isinstance(item, dict) and str(item.get("rationale") or "") == "forced_exploration"
    ]


def _candidate_fills_exploration_slot(
    candidate: dict[str, Any],
    *,
    recent_expression_blob: str,
    forced_exploration_cells: list[dict[str, Any]],
    forced_exploration_fields: list[dict[str, Any]] | None = None,
) -> bool:
    if candidate.get("local_llm_route") != "DIVERSIFY":
        return False
    changed_dimensions = {
        str(value)
        for value in ((candidate.get("diversity_case") or {}).get("changed_dimensions") or [])
    }
    if not ({"information_source", "economic_mechanism"} & changed_dimensions):
        return False

    fields = {str(value).strip() for value in (candidate.get("data_fields") or []) if str(value).strip()}
    target_field_ids = {
        str(item.get("field_id") or item.get("id") or "").strip()
        for item in (forced_exploration_fields or [])
        if isinstance(item, dict) and str(item.get("field_id") or item.get("id") or "").strip()
    }
    if target_field_ids:
        return bool(fields & target_field_ids)

    dataset_id = str(candidate.get("dataset_id") or "").strip()
    exploration_datasets = {
        str(item.get("dataset") or item.get("dataset_id") or "").strip()
        for item in forced_exploration_cells
        if str(item.get("dataset") or item.get("dataset_id") or "").strip()
    }
    if dataset_id and dataset_id in exploration_datasets:
        return True

    orthogonal_core_fields = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "returns",
        "adv20",
        "adv60",
        "adv120",
    }
    if fields & orthogonal_core_fields and all(field not in recent_expression_blob for field in fields):
        return True

    family = str(candidate.get("family") or "").strip()
    exploration_families = {
        str(item.get("family") or "").strip()
        for item in forced_exploration_cells
        if str(item.get("family") or "").strip()
    }
    return bool(family and family in exploration_families and "economic_mechanism" in changed_dimensions)


def _compact_selected_cell(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cell_key",
        "family",
        "dataset",
        "operator_pattern",
        "slots",
        "rationale",
        "active",
        "formal_submissions",
        "terminal_failures",
        "sc_fail_rate",
        "effective_allocation_score",
    )
    return {key: item.get(key) for key in keys if item.get(key) is not None}


def _allocated_context_cells(items: Any, *, force_diversify: bool) -> list[dict[str, Any]]:
    values = [item for item in (items or []) if isinstance(item, dict)]
    if not values:
        return []
    if force_diversify:
        chosen = _diverse_context_slice(values, limit=4)
    else:
        chosen = list(values[:3])
        exploration = next(
            (item for item in values if str(item.get("rationale") or "") == "forced_exploration"),
            None,
        )
        if exploration is not None and exploration not in chosen:
            chosen.append(exploration)
    return [_compact_selected_cell(item) for item in chosen]


def _diverse_context_slice(items: Any, *, limit: int) -> list[dict[str, Any]]:
    """Keep a tiny cross-dataset/family sample instead of only the top exploitation cell."""
    values = [item for item in (items or []) if isinstance(item, dict)]
    if not values:
        return []
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    seen_datasets: set[str] = set()
    seen_families: set[str] = set()

    for index, item in enumerate(values):
        dataset = str(item.get("dataset") or item.get("dataset_id") or "").strip()
        family = str(item.get("family") or "").strip()
        if dataset and dataset in seen_datasets:
            continue
        chosen.append(item)
        chosen_ids.add(index)
        if dataset:
            seen_datasets.add(dataset)
        if family:
            seen_families.add(family)
        if len(chosen) >= limit:
            return chosen

    for index, item in enumerate(values):
        if index in chosen_ids:
            continue
        family = str(item.get("family") or "").strip()
        if family and family in seen_families:
            continue
        chosen.append(item)
        chosen_ids.add(index)
        if family:
            seen_families.add(family)
        if len(chosen) >= limit:
            return chosen

    for index, item in enumerate(values):
        if index in chosen_ids:
            continue
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def _compact_planner_context(planner_context: dict[str, Any]) -> dict[str, Any]:
    """Keep only decision-relevant evidence for the next candidate batch."""
    force_diversify = bool(planner_context.get("local_llm_force_diversify"))
    selected_cells = _allocated_context_cells(
        planner_context.get("selected_cells"),
        force_diversify=force_diversify,
    )
    positive_memory = (
        _diverse_context_slice(planner_context.get("research_memory_positive"), limit=4)
        if force_diversify
        else list(planner_context.get("research_memory_positive") or [])[:1]
    )
    return {
        "research_strategy": planner_context.get("research_strategy"),
        "remaining_active_target": planner_context.get("remaining_active_target"),
        "remaining_submission_slots": planner_context.get("remaining_submission_slots"),
        "inventory": planner_context.get("inventory") or {},
        "next_focus": planner_context.get("next_focus"),
        "selected_cells": selected_cells,
        "forced_exploration_fields": list(planner_context.get("forced_exploration_fields") or [])[:24],
        "failure_counts": planner_context.get("failure_counts") or {},
        "dominant_bottleneck_stage": (
            "primary_metrics_gate"
            if planner_context.get("dominant_bottleneck_stage") == "failure_diagnosis"
            else planner_context.get("dominant_bottleneck_stage")
        ),
        "avoid_structure_signatures": list(planner_context.get("avoid_structure_signatures") or [])[:12],
        "research_memory_positive": positive_memory,
        "research_memory_negative": list(planner_context.get("research_memory_negative") or [])[:1],
        "current_cycle_trial_evidence": list(planner_context.get("current_cycle_trial_evidence") or [])[:6],
        "exclude_expressions": list(
            planner_context.get("local_llm_exclude_expressions")
            or planner_context.get("local_poc_exclude_expressions")
            or []
        )[:30],
        "contract_feedback": list(planner_context.get("local_llm_contract_feedback") or [])[:8],
        "duplicate_pressure": planner_context.get("local_llm_duplicate_pressure") or {},
        "force_diversify": bool(planner_context.get("local_llm_force_diversify")),
        "recent_families": list(planner_context.get("local_llm_recent_families") or [])[:8],
        "recent_dataset_ids": list(planner_context.get("local_llm_recent_dataset_ids") or [])[:8],
        "repair_parent_expressions": list(planner_context.get("local_llm_repair_parent_expressions") or [])[:4],
        "repair_parent_counts": dict(planner_context.get("local_llm_repair_parent_counts") or {}),
        "max_repairs_per_batch": int(
            1
            if planner_context.get("local_llm_max_repairs_per_batch") is None
            else planner_context.get("local_llm_max_repairs_per_batch")
        ),
        "repair_slots_remaining": int(planner_context.get("local_llm_repair_slots_remaining") or 0),
        "exploration_slots_remaining": int(planner_context.get("local_llm_exploration_slots_remaining") or 0),
    }


def _build_messages(
    planner_context: dict[str, Any],
    *,
    batch_size: int,
    skill_context: str,
) -> list[dict[str, str]]:
    compact_context = _compact_planner_context(planner_context)
    context_json = json.dumps(compact_context, ensure_ascii=False, separators=(",", ":"), default=str)
    system = """You are the candidate-generation stage inside QuantGPT's WorldQuant research loop.
Follow the supplied local WQ Skill documents as authoritative workflow instructions.
This is research-only: never formally submit an Alpha and never claim an unobserved BRAIN result.
Use the provided planner/failure evidence to generate a semantically diverse next batch.
Prefer conservative core fields (open, high, low, close, volume, vwap, returns, cap, adv20/adv60/adv120) unless the planner context explicitly supplies another exact field id.
Prefer simple live-style FASTEXPR and avoid cosmetic window-only siblings.
When repairing an observed trial, preserve the exact parent expression and distinguish observed symptoms from plausible causes.
Return JSON only, with no markdown or prose outside the JSON object.
"""
    user = f"""Generate exactly {batch_size} research candidates for the next bounded BRAIN Simulation batch.

PLANNER_CONTEXT_JSON:
{context_json}

LOCAL_WQ_SKILLS:
{skill_context}

Return this shape:
{{
  "skill_candidates": [
    {{
      "route": "NEW_HYPOTHESIS | REPAIR | DIVERSIFY",
      "expression": "...",
      "hypothesis": "mechanism-first and falsifiable",
      "family": "short_family_name",
      "data_fields": ["exact_field_ids_used"],
      "knowledge_card_ids": [],
      "review_notes": "why this is worth one bounded test",
      "robustness_plan": {{
        "checks": [{{"universe": "TOP1000", "purpose": "what this check distinguishes"}}]
      }},
      "parent_expression": "required for REPAIR; exact expression from current_cycle_trial_evidence",
      "failure_signature": {{
        "observed_symptoms": ["required for REPAIR; facts only"],
        "plausible_causes": [{{"cause": "one plausible cause", "confidence": "low|medium|high"}}]
      }},
      "mutation_type": "required for REPAIR",
      "mutation_reason": "what one causal dimension is being tested",
      "diversity_case": {{
        "changed_dimensions": ["information_source"],
        "why_independent": "required for DIVERSIFY"
      }}
    }}
  ]
}}

Rules:
- Every expression must differ in mechanism/information source or structure, not just a lookback number.
- Every expression in exclude_expressions is HARD-FORBIDDEN; never return it again.
- If force_diversify=true, every candidate MUST use route=DIVERSIFY and diversity_case.changed_dimensions MUST include information_source or economic_mechanism. Prefer a family outside recent_families and change the data source/operator skeleton rather than only a window.
- selected_cells preserves the scheduler's exploitation/exploration allocation. If exploration_slots_remaining > 0, at least that many candidates MUST use route=DIVERSIFY and be genuinely different in information source/economic mechanism. If forced_exploration_fields is non-empty, an exploration candidate MUST use at least one exact field_id from that list; do not claim an exploration dataset without using one of its grounded fields. Only when forced_exploration_fields is empty may you fall back to an orthogonal core price/volume mechanism.
- data_fields must list every data field used by the expression and no operators.
- Use knowledge_card_ids only when an id is literally present in supplied context; otherwise [].
- Do not invent historical performance, Sharpe, Fitness, correlation, or platform checks.
- If current_cycle_trial_evidence exists, choose REPAIR only for a worthwhile parent and use its exact expression; otherwise choose DIVERSIFY or NEW_HYPOTHESIS.
- REPAIR is allowed only when parent_expression is listed in repair_parent_expressions and repair_slots_remaining > 0. Once repair slots are exhausted, use NEW_HYPOTHESIS or DIVERSIFY instead of another near-relative repair.
- REPAIR must include parent_expression, failure_signature.observed_symptoms, failure_signature.plausible_causes, mutation_type and mutation_reason. Change one causal dimension unless the supplied Skill explicitly justifies broader screening.
- DIVERSIFY must include diversity_case.changed_dimensions using only information_source, economic_mechanism, horizon_delay, structure, factor_exposure; include why_independent.
- NEW_HYPOTHESIS must not pretend to be a repair of a prior result.
- A structure_signature containing # is a pattern only. Never copy # into an executable expression or parent_expression.
- REPAIR is forbidden when current_cycle_trial_evidence is empty; parent_expression must exactly match one expression in that list.
- Keep robustness_plan to 1-4 targeted checks.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pi_thinking_level(thinking: str, reasoning_effort: str) -> str:
    requested = str(thinking or "").strip().lower()
    if requested == "enabled":
        return "max" if reasoning_effort == "max" else "high"
    if requested == "disabled":
        return "off"
    if requested in _PI_THINKING_LEVELS:
        return requested
    raise LocalCandidateGenerationError(
        "thinking must be enabled/disabled or one of off, minimal, low, medium, high, xhigh, max"
    )


def _request_candidate_content(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    thinking: str,
    reasoning_effort: str,
    request_attempts: int | None = None,
) -> str:
    pi_thinking = _pi_thinking_level(thinking, reasoning_effort)
    default_attempts = 1 if pi_thinking in {"high", "xhigh", "max"} else 2
    retries = max(
        1,
        min(
            3,
            int(request_attempts or os.environ.get("WQ_LOCAL_LLM_REQUEST_ATTEMPTS") or default_attempts),
        ),
    )
    prompt_parts = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "").strip()
        if content:
            prompt_parts.append(f"[{role}]\n{content}")
    prompt = "\n\n".join(prompt_parts).strip()
    if not prompt:
        raise LocalCandidateGenerationError("Pi Agent prompt is empty")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return run_pi_prompt(
                prompt,
                workspace=_REPO_ROOT,
                model=model,
                thinking=pi_thinking,
                timeout_seconds=timeout_seconds,
            )
        except (PiAgentError, OSError, ValueError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(0.5 * attempt)
    assert last_error is not None
    raise LocalCandidateGenerationError(
        f"Pi Agent request failed: {type(last_error).__name__}: {last_error}"
    ) from last_error


def generate_local_skill_batch(
    planner_context: dict[str, Any],
    *,
    batch_size: int = 8,
    model: str | None = None,
    timeout_seconds: float = 120.0,
    thinking: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generate one Skill-reviewed candidate batch through a local Pi Agent RPC process."""
    resolved_model = str(model or os.environ.get("WQ_PI_MODEL") or _DEFAULT_MODEL).strip()
    resolved_thinking = str(thinking or os.environ.get("WQ_PI_THINKING") or "max").strip().lower()
    resolved_effort = str(reasoning_effort or os.environ.get("WQ_LOCAL_LLM_REASONING_EFFORT") or "max").strip().lower()
    if resolved_effort not in {"high", "max"}:
        raise LocalCandidateGenerationError("reasoning_effort must be high or max")
    resolved_pi_thinking = _pi_thinking_level(resolved_thinking, resolved_effort)
    size = max(1, min(20, int(batch_size)))
    chunk_size = max(1, min(4, int(os.environ.get("WQ_LOCAL_LLM_CHUNK_SIZE") or 2)))
    effective_timeout = max(
        10.0,
        min(
            120.0,
            float(planner_context.get("local_llm_request_timeout_seconds") or timeout_seconds),
        ),
    )
    request_attempts = max(
        1,
        min(
            3,
            int(
                planner_context.get("local_llm_request_attempts")
                or os.environ.get("WQ_LOCAL_LLM_REQUEST_ATTEMPTS")
                or (1 if resolved_pi_thinking in {"high", "xhigh", "max"} else 2)
            ),
        ),
    )
    skill_names, skill_context = _load_skill_context(planner_context)

    candidates: list[dict[str, Any]] = []
    seen_expressions: set[str] = set()
    inherited_exclusions = {
        str(value).strip()
        for value in (
            planner_context.get("local_llm_exclude_expressions")
            or planner_context.get("local_poc_exclude_expressions")
            or []
        )
        if str(value).strip()
    }
    force_diversify = bool(planner_context.get("local_llm_force_diversify"))
    forced_exploration_cells = _forced_exploration_cells(planner_context.get("selected_cells"))
    exploration_slots_required = (
        1
        if forced_exploration_cells
        and (size >= 4 or bool(planner_context.get("local_llm_require_exploration_slot")))
        else 0
    )
    accepted_exploration_slots = 0
    recent_families = {
        str(value).strip()
        for value in (planner_context.get("local_llm_recent_families") or [])
        if str(value).strip()
    }
    recent_dataset_ids = {
        str(value).strip()
        for value in (planner_context.get("local_llm_recent_dataset_ids") or [])
        if str(value).strip()
    }
    recent_expression_blob = "\n".join(
        str(item.get("expression") or "")
        for item in (planner_context.get("current_cycle_trial_evidence") or [])
        if isinstance(item, dict)
    )
    repair_parent_expressions = {
        str(value).strip()
        for value in (planner_context.get("local_llm_repair_parent_expressions") or [])
        if str(value).strip()
    }
    enforce_repair_parent_whitelist = "local_llm_repair_parent_expressions" in planner_context
    configured_max_repairs = planner_context.get("local_llm_max_repairs_per_batch")
    max_repairs_per_batch = max(0, int(1 if configured_max_repairs is None else configured_max_repairs))
    accepted_repairs = 0
    planner_context_text = json.dumps(planner_context, ensure_ascii=False, default=str)
    request_count = 0
    chunk_attempt_limit = max(
        1,
        min(
            5,
            int(
                planner_context.get("local_llm_chunk_attempt_limit")
                or os.environ.get("WQ_LOCAL_LLM_CHUNK_ATTEMPTS")
                or 3
            ),
        ),
    )
    while len(candidates) < size:
        remaining = size - len(candidates)
        requested_chunk = min(chunk_size, remaining)
        added = 0
        rejection_feedback: list[dict[str, Any]] = []
        for _chunk_attempt in range(1, chunk_attempt_limit + 1):
            chunk_context = dict(planner_context)
            chunk_context["local_llm_exclude_expressions"] = [*inherited_exclusions, *seen_expressions]
            chunk_context["local_llm_repair_slots_remaining"] = max(0, max_repairs_per_batch - accepted_repairs)
            chunk_context["local_llm_exploration_slots_remaining"] = max(
                0,
                exploration_slots_required - accepted_exploration_slots,
            )
            if rejection_feedback:
                chunk_context["local_llm_contract_feedback"] = [
                    *list(planner_context.get("local_llm_contract_feedback") or []),
                    *rejection_feedback,
                ][-8:]
            messages = _build_messages(chunk_context, batch_size=requested_chunk, skill_context=skill_context)
            content = _request_candidate_content(
                model=resolved_model,
                messages=messages,
                timeout_seconds=effective_timeout,
                thinking=resolved_thinking,
                reasoning_effort=resolved_effort,
                request_attempts=request_attempts,
            )
            request_count += 1
            decoded = _extract_json_payload(content)
            raw_candidates = _candidate_items_from_payload(decoded)
            if not raw_candidates:
                rejection_feedback.append({"error": "LLM JSON did not contain candidate objects"})
                continue
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                normalized_item = dict(item)
                if force_diversify:
                    changed_dimensions = {
                        str(value)
                        for value in ((normalized_item.get("diversity_case") or {}).get("changed_dimensions") or [])
                    }
                    family = str(normalized_item.get("family") or "").strip()
                    dataset_id = str(normalized_item.get("dataset_id") or "").strip()
                    data_fields = [str(value).strip() for value in (normalized_item.get("data_fields") or []) if str(value).strip()]
                    if family and recent_families and family not in recent_families:
                        changed_dimensions.add("economic_mechanism")
                    if dataset_id and recent_dataset_ids and dataset_id not in recent_dataset_ids:
                        changed_dimensions.add("information_source")
                    if data_fields and recent_expression_blob and all(field not in recent_expression_blob for field in data_fields):
                        changed_dimensions.add("information_source")
                    if {"information_source", "economic_mechanism"} & changed_dimensions:
                        normalized_item["route"] = "DIVERSIFY"
                        diversity_case = dict(normalized_item.get("diversity_case") or {})
                        diversity_case["changed_dimensions"] = sorted(changed_dimensions)
                        diversity_case.setdefault(
                            "why_independent",
                            "forced diversification after repeated duplicate/zero-simulation rounds",
                        )
                        normalized_item["diversity_case"] = diversity_case
                candidate = _normalize_candidate(normalized_item, planner_context)
                expression = candidate["expression"]
                if candidate.get("local_llm_route") == "REPAIR":
                    parent_expression = str(candidate.get("parent_expression") or "").strip()
                    if enforce_repair_parent_whitelist and parent_expression not in repair_parent_expressions:
                        rejection_feedback.append(
                            {
                                "expression": expression,
                                "error": "repair_parent_not_eligible",
                                "parent_expression": parent_expression,
                            }
                        )
                        continue
                    if accepted_repairs >= max_repairs_per_batch:
                        rejection_feedback.append(
                            {
                                "expression": expression,
                                "error": "repair_batch_cap_reached",
                                "max_repairs_per_batch": max_repairs_per_batch,
                            }
                        )
                        continue
                if not expression or "#" in expression:
                    rejection_feedback.append({"expression": expression, "error": "invalid_expression"})
                    continue
                if expression in inherited_exclusions:
                    rejection_feedback.append({"expression": expression, "error": "duplicate_expression_history"})
                    continue
                if expression in seen_expressions:
                    rejection_feedback.append({"expression": expression, "error": "duplicate_expression_batch"})
                    continue
                if force_diversify:
                    changed_dimensions = {
                        str(value)
                        for value in ((candidate.get("diversity_case") or {}).get("changed_dimensions") or [])
                    }
                    if candidate.get("local_llm_route") != "DIVERSIFY" or not (
                        {"information_source", "economic_mechanism"} & changed_dimensions
                    ):
                        rejection_feedback.append(
                            {
                                "expression": expression,
                                "error": "forced_diversify_requires_new_information_source_or_mechanism",
                            }
                        )
                        continue
                if not candidate["hypothesis"] or not candidate["family"] or not candidate["data_fields"]:
                    rejection_feedback.append({"expression": expression, "error": "missing_candidate_metadata"})
                    continue
                fills_exploration_slot = _candidate_fills_exploration_slot(
                    candidate,
                    recent_expression_blob=recent_expression_blob,
                    forced_exploration_cells=forced_exploration_cells,
                    forced_exploration_fields=list(planner_context.get("forced_exploration_fields") or []),
                )
                remaining_capacity = size - len(candidates)
                remaining_exploration_slots = max(0, exploration_slots_required - accepted_exploration_slots)
                if remaining_exploration_slots >= remaining_capacity and not fills_exploration_slot:
                    rejection_feedback.append(
                        {
                            "expression": expression,
                            "error": "scheduler_exploration_slot_reserved",
                            "forced_exploration_cells": [
                                _compact_selected_cell(item) for item in forced_exploration_cells[:2]
                            ],
                        }
                    )
                    continue
                unresolved_fields = [
                    field
                    for field in candidate["data_fields"]
                    if not _data_field_is_grounded(str(field), planner_context_text)
                ]
                if unresolved_fields:
                    rejection_feedback.append(
                        {
                            "expression": expression,
                            "error": "unresolved_data_fields",
                            "fields": unresolved_fields,
                        }
                    )
                    continue
                seen_expressions.add(expression)
                candidates.append(candidate)
                if candidate.get("local_llm_route") == "REPAIR":
                    accepted_repairs += 1
                if fills_exploration_slot:
                    accepted_exploration_slots += 1
                added += 1
                if len(candidates) >= size:
                    break
            if added > 0:
                break
        if added == 0:
            if candidates:
                break
            raise LocalCandidateGenerationError(
                "LLM returned no new structurally usable candidates after hard duplicate/diversity filtering"
            )

    return {
        "provider": "pi-agent-rpc",
        "model": resolved_model,
        "thinking": resolved_pi_thinking,
        "reasoning_effort": resolved_effort if resolved_pi_thinking in {"high", "xhigh", "max"} else None,
        "requested": size,
        "generated": len(candidates[:size]),
        "requests": request_count,
        "skills_loaded": skill_names,
        "skill_candidates": candidates[:size],
    }
