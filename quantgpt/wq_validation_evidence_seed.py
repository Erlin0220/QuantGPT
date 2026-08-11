"""Repository-backed robustness and candidate-evidence methodology for WQ research.

These priors are distilled from primary papers, author-maintained textbooks, and
WorldQuant training. They define how to collect and interpret local evidence;
BRAIN remains authoritative for live eligibility and submission outcomes.
"""

from __future__ import annotations

from typing import Any

from .wq_knowledge import upsert_knowledge_card, upsert_knowledge_source


VALIDATION_EVIDENCE_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "ssrn",
        "source_key": "ssrn:bailey-borwein-lopez-zhu-pbo-2015",
        "title": "The Probability of Backtest Overfitting",
        "authors": ["David H. Bailey", "Jonathan M. Borwein", "Marcos Lopez de Prado", "Qiji Jim Zhu"],
        "published_year": 2015,
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
        "external_id": "SSRN:2326253",
        "access_scope": "public_abstract_and_ssrn_paper",
        "content": (
            "The paper develops combinatorially symmetric cross-validation (CSCV) to estimate the probability of "
            "backtest overfitting in investment simulations. It argues that standard hold-out logic can be unreliable "
            "for investment backtests and that the full set of related trials matters when judging whether a selected "
            "strategy is likely to degrade out of sample. QuantGPT should therefore treat lineage/variant history as "
            "robustness evidence instead of validating one Alpha in isolation."
        ),
        "metadata": {"kind": "primary_paper", "retrieved_date": "2026-08-11", "distillation_scope": ["PBO", "CSCV", "multiple_testing", "strategy_selection"]},
    },
    {
        "source_type": "ssrn",
        "source_key": "ssrn:bailey-lopez-deflated-sharpe-2014",
        "title": "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
        "authors": ["David H. Bailey", "Marcos Lopez de Prado"],
        "published_year": 2014,
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        "external_id": "SSRN:2460551",
        "access_scope": "public_abstract_and_ssrn_paper",
        "content": (
            "The Deflated Sharpe Ratio adjusts reported Sharpe evidence for multiple testing/selection bias and "
            "non-Normal returns. The transferable rule for QuantGPT is that repeated Alpha variants inflate apparent "
            "performance; local evidence should preserve the number and relationship of trials rather than rewarding "
            "the best observed Sharpe as if it were an isolated discovery."
        ),
        "metadata": {"kind": "primary_paper", "retrieved_date": "2026-08-11", "distillation_scope": ["DSR", "selection_bias", "multiple_testing", "non_normal_returns"]},
    },
    {
        "source_type": "ssrn",
        "source_key": "ssrn:bailey-lopez-probabilistic-sharpe-2012",
        "title": "The Sharpe Ratio Efficient Frontier",
        "authors": ["David H. Bailey", "Marcos Lopez de Prado"],
        "published_year": 2012,
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643",
        "external_id": "SSRN:1821643",
        "access_scope": "public_abstract_and_ssrn_paper",
        "content": (
            "The paper develops the Probabilistic Sharpe Ratio and explicitly accounts for finite track-record length, "
            "skewness and kurtosis. For QuantGPT, a Sharpe estimate is evidence with uncertainty, not a deterministic "
            "quality label; sample length and return-distribution shape should be preserved whenever available."
        ),
        "metadata": {"kind": "primary_paper", "retrieved_date": "2026-08-11", "distillation_scope": ["PSR", "Sharpe_uncertainty", "track_record_length", "non_normality"]},
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:murphy-probabilistic-machine-learning-2022",
        "title": "Probabilistic Machine Learning: An Introduction",
        "authors": ["Kevin P. Murphy"],
        "published_year": 2022,
        "url": "https://probml.github.io/pml-book/book1.html",
        "access_scope": "author_public_draft_cc_by_nc_nd",
        "content": (
            "Author-maintained MIT Press textbook presenting machine learning through probability, statistics, "
            "information theory, decision theory and optimization. For candidate ranking, the transferable principle "
            "is to represent uncertainty explicitly and avoid presenting an arbitrary deterministic score as a "
            "calibrated probability."
        ),
        "metadata": {"publisher": "MIT Press", "kind": "textbook", "retrieved_date": "2026-08-11", "distillation_scope": ["probabilistic_prediction", "uncertainty", "Bayesian_inference", "decision_theory"]},
    },
    {
        "source_type": "paper",
        "source_key": "paper:arrieta-ibarra-calibration-metrics-2022",
        "title": "Metrics of Calibration for Probabilistic Predictions",
        "authors": ["Imanol Arrieta-Ibarra", "Paman Gujral", "Jonathan Tannen", "Mark Tygert", "Cherie Xu"],
        "published_year": 2022,
        "url": "https://www.jmlr.org/papers/v23/22-0658.html",
        "access_scope": "open_jmlr_fulltext",
        "content": (
            "JMLR paper on assessing whether probabilistic predictions match observed frequencies. It emphasizes "
            "reliability/calibration diagnostics and the finite-sample difficulty of estimating miscalibration. For "
            "QuantGPT, an ACTIVE probability should only be treated as decision-grade after enough resolved outcomes "
            "exist to assess reliability; otherwise report uncertainty rather than fabricate precision."
        ),
        "metadata": {"kind": "primary_paper", "retrieved_date": "2026-08-11", "distillation_scope": ["calibration", "reliability_diagram", "finite_samples", "probabilistic_prediction"]},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-validation-evidence",
        "title": "Learn2Quant — Alpha Quality, Diversification and Risk Management",
        "authors": ["WorldQuant"],
        "published_year": 2026,
        "url": "https://www.worldquant.com/learn2quant/",
        "access_scope": "public",
        "content": (
            "WorldQuant's public Learn2Quant curriculum explicitly includes evaluating Alpha quality, building across "
            "data categories/idea types/horizons, diversifying an Alpha pool, managing risks and applying research "
            "techniques inside BRAIN. QuantGPT should keep current BRAIN Simulation/check responses authoritative while "
            "using local robustness and ranking methods as research evidence rather than hidden platform rules."
        ),
        "metadata": {"publisher": "WorldQuant", "kind": "official_training_series", "retrieved_date": "2026-08-11", "distillation_scope": ["alpha_quality", "diversification", "risk_management", "brain_feedback"]},
    },
]


VALIDATION_EVIDENCE_CARDS: list[dict[str, Any]] = [
    {
        "concept": "hypothesis_targeted_robustness_design",
        "family": "robustness_validation",
        "hypothesis": (
            "Robustness validation should perturb settings that can falsify the stated Alpha mechanism or expose an "
            "identified implementation risk. A fixed one-size-fits-all universe/neutralization grid wastes Simulation "
            "budget and can confuse irrelevant sensitivity with genuine fragility."
        ),
        "mechanism": [
            "Start from the Alpha hypothesis, horizon, field semantics and known failure risks.",
            "Choose a small set of cross-setting checks whose outcomes have distinct interpretations.",
            "Preserve raw BRAIN metrics/checks for each perturbation instead of collapsing them immediately into a magic pass ratio.",
            "Treat missing local robustness evidence as unknown/advisory, not an automatic failure."
        ],
        "scope": {"workflow": "hypothesis -> targeted stress plan -> BRAIN simulations -> evidence interpretation"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-validation-evidence", "support": "positive", "claim": "WorldQuant teaches Alpha quality, diversification and risk management through BRAIN feedback.", "location": "Learn2Quant curriculum"},
            {"source_key": "ssrn:bailey-borwein-lopez-zhu-pbo-2015", "support": "positive", "claim": "Investment strategy robustness should be evaluated in a multiple-testing-aware simulation framework rather than with a generic hold-out assumption.", "location": "abstract/framework"},
            {"source_key": "ssrn:bailey-lopez-probabilistic-sharpe-2012", "support": "positive", "claim": "Sharpe evidence depends on sampling uncertainty and non-normality.", "location": "abstract"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Fixed stress grids may test irrelevant dimensions.", "A pass-count threshold can hide which perturbation actually broke the hypothesis.", "Sparse evidence can be mistaken for evidence of failure."],
        "mutation_strategies": ["Select checks tied to hypothesis/failure risks.", "Keep per-check metrics and platform checks.", "Route structural collapse to diagnosis/repair; otherwise keep robustness evidence advisory."],
        "confidence": 0.94,
        "status": "active",
    },
    {
        "concept": "lineage_aware_multiple_testing_evidence",
        "family": "robustness_validation",
        "hypothesis": (
            "A candidate selected from many related variants should carry the history of those trials into robustness "
            "assessment. Selection among many tested siblings increases the chance that the apparent winner is a "
            "statistical fluke, so DSR/PBO-style evidence is more meaningful than judging the winning Sharpe alone."
        ),
        "mechanism": ["Track coherent parent/child variant families.", "Preserve the count of related trials used to discover the winner.", "Use DSR/PSR/PBO only when the required return history and comparable variants are available.", "Do not mix unrelated Alpha families into one PBO calculation."],
        "scope": {"decision": "how much confidence to place in a selected near-miss/winner"},
        "evidence": [
            {"source_key": "ssrn:bailey-lopez-deflated-sharpe-2014", "support": "positive", "claim": "DSR corrects for selection bias/multiple testing and non-normality.", "location": "abstract"},
            {"source_key": "ssrn:bailey-borwein-lopez-zhu-pbo-2015", "support": "positive", "claim": "PBO/CSCV estimates the probability a selected backtest is overfit.", "location": "abstract/framework"},
            {"source_key": "ssrn:bailey-lopez-probabilistic-sharpe-2012", "support": "positive", "claim": "PSR models uncertainty in Sharpe estimates using sample length and distribution shape.", "location": "abstract"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Treating the selected winner as one independent trial understates selection bias.", "Applying PBO across unrelated Alphas destroys the coherent-selection interpretation.", "Short/noisy PnL history can make local statistical evidence unavailable."],
        "mutation_strategies": ["Use lineage as the unit of related trials.", "Report evidence unavailable when sample requirements are not met.", "Keep statistical robustness as a soft ranking signal unless BRAIN itself returns a blocking check."],
        "confidence": 0.97,
        "status": "active",
    },
    {
        "concept": "calibrated_active_probability_requires_resolved_outcomes",
        "family": "candidate_evidence",
        "hypothesis": (
            "Do not label a hand-weighted metric score as P(ACTIVE). ACTIVE probability should come from resolved "
            "submission outcomes with explicit support/provenance, and its reliability should be evaluated with proper "
            "probability-calibration diagnostics. Before enough outcomes exist, report the probability as unavailable."
        ),
        "mechanism": [
            "Separate eligibility evidence from probabilistic outcome prediction.",
            "Use smoothed historical ACTIVE vs terminal-failure rates at the most specific context with adequate support.",
            "Fall back to broader family/global evidence when context-specific support is sparse.",
            "Evaluate resolved probability forecasts with Brier/reliability diagnostics rather than tuning arbitrary metric weights."
        ],
        "scope": {"output": "calibrated P(ACTIVE) only after a learning-maturity gate"},
        "evidence": [
            {"source_key": "textbook:murphy-probabilistic-machine-learning-2022", "support": "positive", "claim": "Probabilistic prediction should represent uncertainty explicitly within a statistical decision framework.", "location": "book framework"},
            {"source_key": "paper:arrieta-ibarra-calibration-metrics-2022", "support": "positive", "claim": "Calibration compares predicted probabilities with observed frequencies and is difficult to assess reliably in small samples.", "location": "abstract"},
            {"source_key": "worldquant:learn2quant-validation-evidence", "support": "positive", "claim": "BRAIN feedback remains the empirical environment for Alpha quality evaluation.", "location": "Learn2Quant curriculum"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["A weighted score can look precise while having no probability calibration.", "Small terminal-outcome samples can produce unstable context rates.", "Mixing local advisory evidence into an alleged probability hides uncertainty."],
        "mutation_strategies": ["Return probability=None during cold start.", "Use resolved ACTIVE/terminal outcomes for calibration.", "Report support and provenance with every probability."],
        "confidence": 0.98,
        "status": "active",
    },
    {
        "concept": "candidate_evidence_hierarchy_separates_rules_from_preferences",
        "family": "candidate_evidence",
        "hypothesis": (
            "Candidate selection should use a transparent evidence hierarchy: current BRAIN eligibility/official SC are "
            "hard facts; calibrated ACTIVE outcome evidence is the strongest local preference signal when mature; raw "
            "Fitness, Sharpe and returns are descriptive tie-breakers; local robustness/correlation/overfit evidence is "
            "advisory unless an explicit policy intentionally promotes it to a gate."
        ),
        "mechanism": ["Keep official platform blockers separate from local preferences.", "Prefer calibrated empirical outcome evidence when available.", "During cold start, rank transparently by primary BRAIN metrics instead of a hidden weighted sum.", "Preserve local robustness/correlation/overfit evidence as named fields so later decisions remain auditable."],
        "scope": {"decision": "which eligible candidate should be considered first"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-validation-evidence", "support": "positive", "claim": "WorldQuant treats BRAIN as the environment for applying Alpha quality/risk research.", "location": "Learn2Quant curriculum"},
            {"source_key": "paper:arrieta-ibarra-calibration-metrics-2022", "support": "positive", "claim": "Probabilistic confidence requires empirical calibration against outcomes.", "location": "abstract"},
            {"source_key": "ssrn:bailey-lopez-deflated-sharpe-2014", "support": "positive", "claim": "Selection and multiple testing can inflate apparent performance, so raw high metrics should not be interpreted as certainty.", "location": "abstract"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["One opaque score can mix hard rules, soft evidence and preferences.", "A high in-sample metric can dominate despite multiple-testing risk.", "Local proxies can accidentally become stricter than the platform."],
        "mutation_strategies": ["Use a lexicographic evidence hierarchy rather than arbitrary weights.", "Keep official checks as code-enforced blockers.", "Use local evidence for ranking/diagnosis and let BRAIN remain final authority."],
        "confidence": 0.96,
        "status": "active",
    },
]


async def seed_validation_evidence_knowledge() -> dict[str, int]:
    for source in VALIDATION_EVIDENCE_SOURCES:
        await upsert_knowledge_source(**source)
    for card in VALIDATION_EVIDENCE_CARDS:
        await upsert_knowledge_card(card)
    return {"sources": len(VALIDATION_EVIDENCE_SOURCES), "cards": len(VALIDATION_EVIDENCE_CARDS)}
