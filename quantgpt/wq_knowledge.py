"""Multi-source Alpha research knowledge for WorldQuant autonomous research.

This module keeps literature/official evidence separate from empirical BRAIN
research memory.  Knowledge cards act as priors; real WQ trials remain the
source of empirical feedback and can calibrate card usefulness over time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import or_, select

from .db import _get_session_factory
from .models import WQKnowledgeCard, WQKnowledgeSource, WQResearchCandidate, WQResearchTrial

_ALLOWED_SOURCE_TYPES = {"textbook", "arxiv", "ssrn", "worldquant", "paper", "other"}
_ARXIV_API = "https://export.arxiv.org/api/query"
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value)}


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[: max(0, int(limit))]


def _string_list(value: Any, *, limit: int = 100) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _source_key(source_type: str, external_id: str | None, title: str) -> str:
    if external_id:
        return f"{source_type}:{external_id.strip()}"
    digest = hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()[:20]
    return f"{source_type}:sha256:{digest}"


def _serialize_source(row: WQKnowledgeSource, *, include_content: bool = False) -> dict[str, Any]:
    result = {
        "id": str(row.id),
        "source_key": row.source_key,
        "source_type": row.source_type,
        "title": row.title,
        "authors": list(row.authors or []),
        "published_year": row.published_year,
        "url": row.url,
        "external_id": row.external_id,
        "access_scope": row.access_scope,
        "metadata": dict(row.source_metadata or {}),
        "content_chars": len(row.content or ""),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if include_content:
        result["content"] = row.content
    return result


def _evidence_source_keys(evidence: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for item in evidence:
        key = str(item.get("source_key") or item.get("source_id") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _normalize_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in value[:50]:
        if not isinstance(raw, dict):
            continue
        source_key = str(raw.get("source_key") or raw.get("source_id") or "").strip()
        if not source_key:
            continue
        stance = str(raw.get("support") or raw.get("stance") or "mixed").lower().strip()
        if stance not in {"positive", "negative", "mixed", "neutral"}:
            stance = "mixed"
        out.append(
            {
                "source_key": source_key,
                "support": stance,
                "claim": _bounded_text(raw.get("claim"), 1200),
                "location": _bounded_text(raw.get("location"), 300),
                "content_scope": _bounded_text(raw.get("content_scope"), 80) or None,
            }
        )
    return out


def _card_key(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("card_key") or "").strip()
    if explicit:
        return explicit
    identity = "|".join(
        [
            str(payload.get("concept") or "").strip().lower(),
            str(payload.get("family") or "").strip().lower(),
            str(payload.get("hypothesis") or "").strip().lower(),
        ]
    )
    return "wqk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _card_payload(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _normalize_evidence(payload.get("evidence"))
    source_keys = _evidence_source_keys(evidence)
    requested_status = str(payload.get("status") or "active").lower().strip()
    # Single-source cards remain useful for audit/search, but cannot become an
    # active autonomous-research prior until corroborated by another source.
    status = "active" if requested_status == "active" and len(source_keys) >= 2 else "draft"
    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    return {
        "card_key": _card_key(payload),
        "concept": _bounded_text(payload.get("concept"), 120) or "unknown",
        "family": _bounded_text(payload.get("family"), 80) or "unknown",
        "hypothesis": _bounded_text(payload.get("hypothesis"), 4000),
        "mechanism": _string_list(payload.get("mechanism"), limit=30),
        "scope": dict(payload.get("scope") or {}) if isinstance(payload.get("scope"), dict) else {},
        "evidence": evidence,
        "source_keys": source_keys,
        "source_count": len(source_keys),
        "operators": _string_list(payload.get("operators") or (payload.get("wq_mapping") or {}).get("operators"), limit=50),
        "expression_templates": _string_list(
            payload.get("expression_templates") or (payload.get("wq_mapping") or {}).get("templates"), limit=50
        ),
        "failure_modes": _string_list(payload.get("failure_modes") or payload.get("expected_failure_modes"), limit=30),
        "mutation_strategies": _string_list(
            payload.get("mutation_strategies") or payload.get("robustness_actions"), limit=30
        ),
        "confidence": confidence,
        "status": status,
    }


def _serialize_card(row: WQKnowledgeCard, *, empirical: dict[str, Any] | None = None, score: float | None = None) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "card_key": row.card_key,
        "concept": row.concept,
        "family": row.family,
        "hypothesis": row.hypothesis,
        "mechanism": list(row.mechanism or []),
        "scope": dict(row.scope or {}),
        "evidence": list(row.evidence or []),
        "source_keys": list(row.source_keys or []),
        "source_count": int(row.source_count or 0),
        "operators": list(row.operators or []),
        "expression_templates": list(row.expression_templates or []),
        "failure_modes": list(row.failure_modes or []),
        "mutation_strategies": list(row.mutation_strategies or []),
        "confidence": float(row.confidence or 0.0),
        "status": row.status,
        "empirical": empirical or {"trials": 0},
        "score": round(float(score), 6) if score is not None else None,
    }


async def upsert_knowledge_source(
    *,
    source_type: str,
    title: str,
    content: str = "",
    source_key: str | None = None,
    authors: list[str] | None = None,
    published_year: int | None = None,
    url: str | None = None,
    external_id: str | None = None,
    access_scope: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_type = str(source_type or "other").lower().strip()
    if source_type not in _ALLOWED_SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {source_type}")
    title = str(title or "").strip()
    if not title:
        raise ValueError("title is required")
    source_key = str(source_key or _source_key(source_type, external_id, title)).strip()
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(WQKnowledgeSource).where(WQKnowledgeSource.source_key == source_key))
        row = result.scalar_one_or_none()
        values = {
            "source_type": source_type,
            "title": title,
            "authors": _string_list(authors, limit=50),
            "published_year": int(published_year) if published_year else None,
            "url": str(url or "").strip() or None,
            "external_id": str(external_id or "").strip() or None,
            "access_scope": str(access_scope or "").strip() or None,
            "content": str(content or ""),
            "source_metadata": dict(metadata or {}),
            "updated_at": _utcnow(),
        }
        if row is None:
            row = WQKnowledgeSource(source_key=source_key, **values)
            session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        await session.commit()
        await session.refresh(row)
        return _serialize_source(row)


async def upsert_knowledge_card(payload: dict[str, Any]) -> dict[str, Any]:
    values = _card_payload(dict(payload or {}))
    if not values["hypothesis"]:
        raise ValueError("hypothesis is required")
    factory = _get_session_factory()
    async with factory() as session:
        source_result = await session.execute(
            select(WQKnowledgeSource.source_key).where(WQKnowledgeSource.source_key.in_(values["source_keys"]))
        )
        known_source_keys = {str(value) for value in source_result.scalars().all()}
        missing_source_keys = [key for key in values["source_keys"] if key not in known_source_keys]
        if missing_source_keys:
            values["status"] = "draft"
        result = await session.execute(select(WQKnowledgeCard).where(WQKnowledgeCard.card_key == values["card_key"]))
        row = result.scalar_one_or_none()
        if row is None:
            row = WQKnowledgeCard(**values)
            session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
            row.updated_at = _utcnow()
        await session.commit()
        await session.refresh(row)
        serialized = _serialize_card(row)
        serialized["missing_source_keys"] = missing_source_keys
        return serialized


def _trial_empirical(rows: list[WQResearchTrial]) -> dict[str, dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "trials": 0,
            "candidates": 0,
            "sharpe": [],
            "fitness": [],
            "failures": Counter(),
        }
    )
    for row in rows:
        for card_id in _string_list(row.knowledge_card_ids, limit=20):
            item = accum[card_id]
            item["trials"] += 1
            if str(row.status or "").lower() == "candidate":
                item["candidates"] += 1
            if row.sharpe is not None:
                item["sharpe"].append(float(row.sharpe))
            if row.fitness is not None:
                item["fitness"].append(float(row.fitness))
            if row.failure_reason:
                item["failures"][str(row.failure_reason)] += 1
    out: dict[str, dict[str, Any]] = {}
    for card_id, item in accum.items():
        trials = int(item["trials"])
        out[card_id] = {
            "trials": trials,
            "candidates": int(item["candidates"]),
            "candidate_rate": round(int(item["candidates"]) / trials, 4) if trials else None,
            "mean_sharpe": round(sum(item["sharpe"]) / len(item["sharpe"]), 4) if item["sharpe"] else None,
            "mean_fitness": round(sum(item["fitness"]) / len(item["fitness"]), 4) if item["fitness"] else None,
            "top_failures": dict(item["failures"].most_common(5)),
        }
    return out


def _card_score(row: WQKnowledgeCard, *, query_tokens: set[str], families: set[str], empirical: dict[str, Any]) -> float:
    body_tokens = _tokens(
        {
            "concept": row.concept,
            "family": row.family,
            "hypothesis": row.hypothesis,
            "mechanism": row.mechanism,
            "operators": row.operators,
            "failure_modes": row.failure_modes,
            "mutation_strategies": row.mutation_strategies,
        }
    )
    lexical = len(query_tokens & body_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
    family = 1.0 if families and str(row.family or "").lower() in families else 0.0
    diversity = min(1.0, max(0.0, (int(row.source_count or 0) - 1) / 3.0))
    prior = float(row.confidence or 0.0) * 0.60 + diversity * 0.15 + lexical * 0.15 + family * 0.10
    trials = int(empirical.get("trials") or 0)
    if trials >= 3:
        candidate_rate = float(empirical.get("candidate_rate") or 0.0)
        mean_sharpe = float(empirical.get("mean_sharpe") or 0.0)
        mean_fitness = float(empirical.get("mean_fitness") or 0.0)
        empirical_score = min(1.0, candidate_rate * 1.5 + max(0.0, mean_sharpe) / 6.0 + max(0.0, mean_fitness) / 4.0)
        return prior * 0.7 + empirical_score * 0.3
    return prior


async def search_alpha_knowledge(
    *,
    query: str = "",
    families: list[str] | None = None,
    operators: list[str] | None = None,
    limit: int = 12,
    include_drafts: bool = False,
) -> dict[str, Any]:
    limit = max(1, min(50, int(limit)))
    query_tokens = _tokens(query)
    family_set = {str(value).lower().strip() for value in (families or []) if str(value).strip()}
    operator_set = {str(value).lower().strip() for value in (operators or []) if str(value).strip()}
    factory = _get_session_factory()
    async with factory() as session:
        stmt = select(WQKnowledgeCard)
        if not include_drafts:
            stmt = stmt.where(WQKnowledgeCard.status == "active")
        if family_set:
            stmt = stmt.where(or_(*[WQKnowledgeCard.family == family for family in family_set]))
        card_result = await session.execute(stmt)
        cards = list(card_result.scalars().all())
        trial_result = await session.execute(
            select(WQResearchTrial)
            .where(WQResearchTrial.knowledge_card_ids.is_not(None))
            .order_by(WQResearchTrial.created_at.desc())
            .limit(5000)
        )
        empirical = _trial_empirical(list(trial_result.scalars().all()))

    ranked: list[tuple[float, WQKnowledgeCard]] = []
    for row in cards:
        row_operators = {str(value).lower() for value in (row.operators or [])}
        if operator_set and not (operator_set & row_operators):
            continue
        score = _card_score(
            row,
            query_tokens=query_tokens,
            families=family_set,
            empirical=empirical.get(str(row.id), {"trials": 0}),
        )
        ranked.append((score, row))
    ranked.sort(key=lambda item: (item[0], float(item[1].confidence or 0.0), int(item[1].source_count or 0)), reverse=True)
    items = [
        _serialize_card(row, empirical=empirical.get(str(row.id), {"trials": 0}), score=score)
        for score, row in ranked[:limit]
    ]
    return {"query": query, "count": len(items), "cards": items}


async def load_knowledge_guidance(*, query: str = "", families: list[str] | None = None, limit: int = 24) -> dict[str, Any]:
    result = await search_alpha_knowledge(query=query, families=families, limit=limit, include_drafts=False)
    cards = result["cards"]
    family_confidence: dict[str, list[float]] = defaultdict(list)
    preferred_templates: list[dict[str, Any]] = []
    failure_mutations: list[dict[str, Any]] = []
    for card in cards:
        family_confidence[str(card.get("family") or "unknown")].append(float(card.get("score") or card.get("confidence") or 0.0))
        for expression in card.get("expression_templates") or []:
            preferred_templates.append(
                {
                    "expression": expression,
                    "card_id": card["id"],
                    "card_key": card["card_key"],
                    "family": card.get("family"),
                    "hypothesis": card.get("hypothesis"),
                    "source_keys": card.get("source_keys") or [],
                    "failure_modes": card.get("failure_modes") or [],
                    "mutation_strategies": card.get("mutation_strategies") or [],
                    "empirical": card.get("empirical") or {"trials": 0},
                    "confidence": card.get("confidence"),
                    "score": card.get("score"),
                }
            )
        if card.get("failure_modes") or card.get("mutation_strategies"):
            failure_mutations.append(
                {
                    "card_id": card["id"],
                    "family": card.get("family"),
                    "failure_modes": card.get("failure_modes") or [],
                    "mutation_strategies": card.get("mutation_strategies") or [],
                }
            )
    return {
        "policy": "multi_source_prior_empirically_calibrated",
        "cards": cards,
        "preferred_templates": preferred_templates,
        "failure_mutations": failure_mutations,
        "family_confidence": {
            family: round(sum(values) / len(values), 6) for family, values in family_confidence.items() if values
        },
    }


async def fetch_arxiv_sources(query: str, *, limit: int = 10, ingest: bool = False) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(50, int(limit)))
    # arXiv treats a whitespace-separated ``all:foo bar`` query more broadly
    # than users usually intend.  For plain natural-language queries require
    # each meaningful term; callers can still pass explicit arXiv syntax
    # (e.g. ``cat:q-fin.ST AND all:reversal``) unchanged.
    if ":" in query or re.search(r"\b(?:AND|OR|ANDNOT)\b", query, flags=re.IGNORECASE):
        search_query = query
    else:
        terms = re.findall(r'"[^"]+"|[^\s]+', query)[:12]
        search_query = " AND ".join(f"all:{term}" for term in terms)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "QuantGPT/2.8 knowledge-research"}) as client:
        response = await client.get(_ARXIV_API, params=params)
        response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        raw_id = _bounded_text(entry.findtext("atom:id", default="", namespaces=ns), 500)
        arxiv_id = raw_id.rstrip("/").split("/")[-1]
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
        published = entry.findtext("atom:published", default="", namespaces=ns)
        authors = [
            " ".join((author.findtext("atom:name", default="", namespaces=ns) or "").split())
            for author in entry.findall("atom:author", ns)
        ]
        categories = [item.attrib.get("term") for item in entry.findall("atom:category", ns) if item.attrib.get("term")]
        item = {
            "source_key": f"arxiv:{arxiv_id}",
            "source_type": "arxiv",
            "external_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published_year": int(published[:4]) if published[:4].isdigit() else None,
            "url": raw_id,
            "content": summary,
            "access_scope": "abstract",
            "metadata": {"published": published, "categories": categories, "query": query},
        }
        if ingest:
            item["stored"] = await upsert_knowledge_source(**item)
        items.append({key: value for key, value in item.items() if key != "content" or not ingest})
    return {"query": query, "count": len(items), "ingested": ingest, "sources": items}


def _extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(value[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM did not return a JSON object")


async def cross_distill_sources(
    source_keys: list[str],
    *,
    concept: str,
    family: str,
    research_goal: str = "WorldQuant BRAIN Alpha research",
) -> dict[str, Any]:
    keys = _string_list(source_keys, limit=12)
    if len(keys) < 2:
        raise ValueError("cross-distillation requires at least two source_keys")
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(WQKnowledgeSource).where(WQKnowledgeSource.source_key.in_(keys)))
        rows = list(result.scalars().all())
    if len(rows) < 2:
        raise ValueError("at least two stored sources are required")

    excerpts: list[str] = []
    for row in rows:
        content = _bounded_text(row.content, 12000)
        excerpts.append(
            f"SOURCE_KEY={row.source_key}\nTYPE={row.source_type}\nTITLE={row.title}\nACCESS={row.access_scope or 'unknown'}\nCONTENT:\n{content}"
        )
    source_blob = "\n\n--- SOURCE BREAK ---\n\n".join(excerpts)
    from .iteration import _call_llm

    system_prompt = (
        "You are an evidence-constrained quantitative research distiller. Produce one JSON knowledge card only. "
        "Use ONLY the supplied sources; do not add outside facts. Preserve disagreements instead of reconciling them by invention. "
        "Every evidence item must cite an exact SOURCE_KEY from the input. Prefer simple WorldQuant FASTEXPR mappings and mark uncertainty."
    )
    user_prompt = f"""Research goal: {research_goal}
Concept: {concept}
Family: {family}

Return JSON with exactly these keys:
concept, family, hypothesis, mechanism, scope, evidence, operators, expression_templates, failure_modes, mutation_strategies, confidence.
- evidence: list of source_key/support(positive|negative|mixed|neutral)/claim/location/content_scope
- operators: WorldQuant-style operator names only when justified by the evidence/mechanism
- expression_templates: simple generic FASTEXPR hypotheses, no invented data fields
- confidence: 0..1 and should decrease when sources conflict or only abstracts are available

SOURCES:
{source_blob}
"""
    raw = await asyncio.to_thread(
        _call_llm,
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=4096,
        clean_output=False,
    )
    card = _extract_json_object(raw)
    card["concept"] = card.get("concept") or concept
    card["family"] = card.get("family") or family
    stored = await upsert_knowledge_card(card)
    return {"stored": stored, "source_keys": [row.source_key for row in rows]}


async def explain_alpha_knowledge(*, account: str = "primary", alpha_id: str | None = None, expression: str | None = None) -> dict[str, Any]:
    alpha_id = str(alpha_id or "").strip()
    expression = str(expression or "").strip()
    if not alpha_id and not expression:
        raise ValueError("alpha_id or expression is required")
    factory = _get_session_factory()
    async with factory() as session:
        candidate = None
        if alpha_id:
            result = await session.execute(
                select(WQResearchCandidate).where(
                    WQResearchCandidate.account == account,
                    WQResearchCandidate.alpha_id == alpha_id,
                )
            )
            candidate = result.scalar_one_or_none()
        if candidate is None and expression:
            result = await session.execute(
                select(WQResearchCandidate)
                .where(WQResearchCandidate.account == account, WQResearchCandidate.expression == expression)
                .order_by(WQResearchCandidate.updated_at.desc())
                .limit(1)
            )
            candidate = result.scalar_one_or_none()

        trial_stmt = select(WQResearchTrial).where(WQResearchTrial.account == account)
        if alpha_id:
            trial_stmt = trial_stmt.where(WQResearchTrial.alpha_id == alpha_id)
        elif expression:
            trial_stmt = trial_stmt.where(WQResearchTrial.expression == expression)
        trial_result = await session.execute(trial_stmt.order_by(WQResearchTrial.created_at.desc()).limit(20))
        trials = list(trial_result.scalars().all())
        card_ids = set(_string_list(candidate.knowledge_card_ids if candidate else None, limit=50))
        for trial in trials:
            card_ids.update(_string_list(trial.knowledge_card_ids, limit=50))
        cards: list[WQKnowledgeCard] = []
        if card_ids:
            valid_card_ids = []
            for value in card_ids:
                try:
                    valid_card_ids.append(uuid.UUID(str(value)))
                except (TypeError, ValueError):
                    continue
            card_result = await session.execute(select(WQKnowledgeCard).where(WQKnowledgeCard.id.in_(valid_card_ids)))
            cards = list(card_result.scalars().all())

    return {
        "account": account,
        "alpha_id": alpha_id or (candidate.alpha_id if candidate else None),
        "expression": expression or (candidate.expression if candidate else (trials[0].expression if trials else None)),
        "knowledge_cards": [_serialize_card(row) for row in cards],
        "trials": [
            {
                "status": row.status,
                "sharpe": row.sharpe,
                "fitness": row.fitness,
                "turnover": row.turnover,
                "failure_reason": row.failure_reason,
                "knowledge_card_ids": list(row.knowledge_card_ids or []),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in trials
        ],
    }
