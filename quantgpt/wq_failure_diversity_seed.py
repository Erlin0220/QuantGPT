"""Repository-backed knowledge for WQ failure diagnosis and genuine diversification.

This seed reuses already-vetted WorldQuant official sources and research-methodology
sources. It adds decision cards rather than duplicating the underlying books/docs.
Facts and thresholds still come from live BRAIN; these cards guide interpretation.
"""

from __future__ import annotations

from typing import Any

from .wq_knowledge import upsert_knowledge_card


FAILURE_DIVERSITY_CARDS: list[dict[str, Any]] = [
    {
        "concept": "failure_signature_separates_observation_from_cause",
        "family": "failure_diagnosis",
        "hypothesis": (
            "A failed BRAIN check is an observation, not automatically the causal explanation. "
            "Diagnosis should preserve the exact platform evidence, distinguish symptoms from plausible causes, "
            "and only promote a cause when a targeted follow-up experiment can discriminate it from alternatives."
        ),
        "mechanism": [
            "Keep raw BRAIN metrics/checks, validation evidence and lineage context unchanged as the fact layer.",
            "Represent diagnosis as primary symptom, plausible causes, counter-evidence and discriminating next tests.",
            "Treat low Fitness, low Sharpe, turnover, concentration and sub-universe weakness as potentially interacting symptoms rather than mutually exclusive labels.",
            "Do not infer a causal repair merely because one metric crossed a threshold.",
        ],
        "scope": {"decision": "interpret a failed Alpha before repair/allocation"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-3", "support": "positive", "claim": "WorldQuant evaluates Alpha quality through multiple metrics/checks rather than one scalar outcome.", "location": "Lesson 3"},
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official examples separate hypothesis, implementation, simulation results and specific improvement ideas.", "location": "Alpha examples"},
            {"source_key": "textbook:santner-williams-notz-computer-experiments-2018", "support": "positive", "claim": "Computer-experiment methodology uses screening and sensitivity analysis to identify influential design dimensions before refinement.", "location": "screening and sensitivity chapters"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Calling every low-Fitness result a turnover problem creates false causal certainty.",
            "Changing several dimensions at once makes the diagnosis impossible to validate.",
            "A generic failure label can hide distinct causes such as weak signal, execution drag, exposure concentration or instability.",
        ],
        "mutation_strategies": [
            "Preserve raw failure facts, then state multiple plausible causes with confidence qualifiers.",
            "Choose one discriminating test per leading cause.",
            "Escalate from symptom label to causal diagnosis only after lineage or targeted-test evidence supports it.",
        ],
        "confidence": 0.95,
        "status": "active",
    },
    {
        "concept": "counterfactual_failure_diagnosis",
        "family": "failure_diagnosis",
        "hypothesis": (
            "When multiple causes fit the same BRAIN failure, prefer the smallest counterfactual experiment that would "
            "change the diagnosis. A useful repair test is diagnostic when its outcome would favor one explanation over another."
        ),
        "mechanism": [
            "Form competing explanations from the hypothesis, metrics, settings and lineage history.",
            "For each explanation, identify the smallest change that should improve the failing dimension while preserving unrelated dimensions.",
            "Prefer tests with high information value and clear attribution; avoid cosmetic variants that cannot falsify a cause.",
            "Feed the result back into Research Memory before deciding another repair.",
        ],
        "scope": {"workflow": "facts -> competing causes -> discriminating experiment -> updated diagnosis"},
        "evidence": [
            {"source_key": "textbook:powell-ryzhov-optimal-learning-2012", "support": "positive", "claim": "Optimal Learning frames expensive experiments by their value of information for later decisions.", "location": "knowledge-gradient / value-of-information treatment"},
            {"source_key": "textbook:santner-williams-notz-computer-experiments-2018", "support": "positive", "claim": "Sequential computer experiments use targeted designs to learn influential response dimensions efficiently.", "location": "sequential design"},
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official Alpha examples propose hypothesis-preserving improvements such as turnover control rather than unrelated random mutations.", "location": "potential improvements"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "A repair that changes field, sign, horizon and neutralization together cannot identify which explanation was correct.",
            "Repeated near-identical variants can consume Simulation budget without reducing diagnostic uncertainty.",
        ],
        "mutation_strategies": [
            "Change one causal dimension per diagnostic child.",
            "Use the outcome to update cause confidence, not merely to rank the child.",
            "Route to diversification/new hypothesis when no plausible local counterfactual remains informative.",
        ],
        "confidence": 0.93,
        "status": "active",
    },
    {
        "concept": "fitness_failure_requires_decomposition",
        "family": "failure_diagnosis",
        "hypothesis": (
            "Low Fitness should be decomposed using Sharpe, returns, turnover, horizon, concentration, coverage and "
            "cross-setting evidence before choosing a repair. The same low-Fitness label can arise from very different mechanisms."
        ),
        "mechanism": [
            "High/acceptable Sharpe with elevated turnover supports execution-drag or horizon-mismatch hypotheses more than weak-signal hypotheses.",
            "Weak Sharpe together with low Fitness supports a signal-quality hypothesis before parameter refinement.",
            "Concentration/sub-universe failures point to exposure/coverage instability rather than generic smoothing.",
            "Use live BRAIN checks as the authoritative symptom evidence and Research Memory for context-specific cause support.",
        ],
        "scope": {"failure": "low_fitness and related metric failures"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-3", "support": "positive", "claim": "WorldQuant Alpha evaluation uses Sharpe, Fitness, turnover and related checks jointly.", "location": "Lesson 3"},
            {"source_key": "worldquant:learn2quant-lesson-7", "support": "positive", "claim": "Holding frequency changes risk, return and transaction-cost behavior, making horizon relevant to turnover/efficiency diagnosis.", "location": "Lesson 7"},
            {"source_key": "worldquant:learn2quant-lesson-9", "support": "positive", "claim": "Common factor exposures and neutralization are distinct risk considerations that can affect apparent Alpha quality.", "location": "Lesson 9"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Blindly smoothing every low-Fitness Alpha can destroy a weak but potentially useful fast signal.",
            "Blindly reseeding every low-Fitness Alpha can discard a strong signal suffering implementation drag.",
        ],
        "mutation_strategies": [
            "Decompose the symptom before selecting repair classes.",
            "Name the evidence that supports and contradicts each proposed cause.",
            "If evidence is ambiguous, request a small diagnostic experiment rather than asserting a cause.",
        ],
        "confidence": 0.94,
        "status": "active",
    },
    {
        "concept": "diversity_requires_independent_information_dimensions",
        "family": "diversification",
        "hypothesis": (
            "A diversified Alpha pool should differ in the source of predictive information and economic mechanism, "
            "not only syntax. Meaningful diversity can come from dataset/information source, mechanism, horizon, factor exposure, "
            "or empirically different PnL behavior."
        ),
        "mechanism": [
            "Track diversity dimensions explicitly: dataset/information source, economic mechanism, horizon/delay, structural transform, common-factor exposure and PnL correlation.",
            "Require a diversification candidate to state which dimensions changed and why those changes should create independent information.",
            "Prefer changes in information source/mechanism before cosmetic parameter or operator changes.",
            "Use current BRAIN SC and observed PnL behavior as empirical validation of diversity.",
        ],
        "scope": {"objective": "genuinely low-correlation candidate and ACTIVE inventory"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-8", "support": "positive", "claim": "WorldQuant explicitly teaches diversity across data, models and combination approaches.", "location": "Lesson 8"},
            {"source_key": "worldquant:learn2quant-lesson-9", "support": "positive", "claim": "Common factor exposures can create shared risk even when expressions look different.", "location": "Lesson 9"},
            {"source_key": "worldquant:finding-alphas-official", "support": "positive", "claim": "WorldQuant frames Alpha research as a broad discipline spanning multiple signal sources and methods.", "location": "official introduction"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Different lookbacks on the same field/mechanism can be syntactically different but economically redundant.",
            "Different datasets can still collapse onto the same common factor exposure.",
            "A low expression-similarity score is not proof of low return correlation.",
        ],
        "mutation_strategies": [
            "Change information source or economic mechanism first.",
            "Use horizon/exposure changes only when they represent a coherent different source of edge.",
            "Confirm diversity with BRAIN SC/PnL evidence when available.",
        ],
        "confidence": 0.97,
        "status": "active",
    },
    {
        "concept": "self_correlation_failure_is_research_direction_feedback",
        "family": "diversification",
        "hypothesis": (
            "Self-Correlation failure should be treated as evidence that the new Alpha adds insufficiently independent information. "
            "The default response is an orthogonal research direction, not noise injection or adjacent parameter edits."
        ),
        "mechanism": [
            "Use the failed Alpha's family, dataset, mechanism, horizon and exposure profile to identify what is already saturated.",
            "Search underrepresented Knowledge/Data Explorer cells that differ on at least one information-level dimension.",
            "Keep a bounded exploitation slice for productive families but move exploration budget toward orthogonal sources.",
            "Reject random noise or irrelevant transforms intended only to change measured correlation.",
        ],
        "scope": {"trigger": "official SELF_CORRELATION / repeated local correlation evidence"},
        "evidence": [
            {"source_key": "worldquant:iqc-guidelines-2026", "support": "positive", "claim": "Official IQC rules include pool correlation constraints and anti-gaming/noise integrity requirements.", "location": "IQC guidelines"},
            {"source_key": "worldquant:learn2quant-lesson-8", "support": "positive", "claim": "WorldQuant teaches diversification as a core Alpha research objective.", "location": "Lesson 8"},
            {"source_key": "worldquant:brain-overview", "support": "positive", "claim": "BRAIN provides empirical feedback used to evaluate Alpha behavior.", "location": "BRAIN overview"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Noise injection can violate integrity rules and does not create a new economic edge.",
            "Window-only edits often preserve the same underlying information source.",
        ],
        "mutation_strategies": [
            "Switch dataset/information source or mechanism before local parameter edits.",
            "Use factor-exposure evidence to avoid recreating the same common risk through another syntax.",
            "Record the changed diversity dimensions for later empirical verification.",
        ],
        "confidence": 0.98,
        "status": "active",
    },
]


async def seed_failure_diversity_knowledge() -> dict[str, int]:
    for card in FAILURE_DIVERSITY_CARDS:
        await upsert_knowledge_card(card)
    return {"cards": len(FAILURE_DIVERSITY_CARDS)}
