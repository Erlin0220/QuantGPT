"""Repository-backed WorldQuant official knowledge seed.

This module makes the distilled WorldQuant priors reproducible across fresh
QuantGPT databases.  Static source summaries are intentionally conservative:
claims come from the referenced WorldQuant material, while account-visible
operator/data-field snapshots are explicitly date-stamped and must never
replace live BRAIN catalog validation.
"""

from __future__ import annotations

from typing import Any

from .wq_knowledge import upsert_knowledge_card, upsert_knowledge_source


OFFICIAL_WORLDQUANT_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant",
        "title": "Learn2Quant",
        "content": (
            "Official WorldQuant Learn2Quant series. The curriculum centers Alpha research on generating original "
            "ideas; building Alphas across data categories, idea types, holding frequencies and delays; diversifying "
            "an Alpha pool; managing risk; and applying advanced research techniques. The series is designed to be "
            "followed inside BRAIN so ideas are implemented and tested rather than treated as abstract formulas."
        ),
        "url": "https://www.worldquant.com/learn2quant/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_training_series", "retrieved_date": "2026-08-10"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-1",
        "title": "Learn2Quant Lesson 1 — A Beginner's Guide to Quantitative Finance",
        "content": (
            "Lesson 1 introduces the stock market, hedge funds, quantitative research and model creation as the "
            "foundation for the rest of Learn2Quant, with BRAIN as the environment for applying the concepts."
        ),
        "url": "https://www.linkedin.com/posts/learn2quant-a-beginners-guide-to-quantitative-activity-7308971245465665536-oqDV",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 1},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-2",
        "title": "Learn2Quant Lesson 2 — Creating a Quant Alpha",
        "content": (
            "Lesson 2 moves from an Alpha idea to implementation and introduces long-short market neutralization. "
            "It prepares the researcher to test implementations through backtesting while respecting timing."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-creating-a-quant-alpha-lesson-activity-7310813207567233024-9Ura",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 2},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-3",
        "title": "Learn2Quant Lesson 3 — How Good is Your Alpha? A Metrics-Based Approach",
        "content": (
            "Lesson 3 focuses on evaluating Alpha quality with multiple metrics, the importance of backtesting and "
            "the need to avoid look-ahead bias."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-how-good-is-your-alpha-a-metrics-based-activity-7311536798206230530-iJUK",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 3},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-4",
        "title": "Learn2Quant Lesson 4 — Alpha Examples by Data Category: Part 1",
        "content": (
            "Lesson 4 moves into concrete Alpha examples by dataset category, starting with Price Volume and "
            "Fundamental data, with examples intended to be simulated on BRAIN."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-alpha-examples-by-data-category-activity-7315133377613889536-O0Mx",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 4},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-5",
        "title": "Learn2Quant Lesson 5 — Alpha Examples by Data Category: Part 2",
        "content": (
            "Lesson 5 continues dataset-category examples with Sentiment and Options data and tests the examples "
            "on BRAIN, expanding research breadth beyond Price Volume and Fundamental."
        ),
        "url": "https://www.linkedin.com/posts/learn2quant-alpha-examples-by-data-category-activity-7318716295858851840-h6mh",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 5},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-6",
        "title": "Learn2Quant Lesson 6 — Alpha Examples by Idea Type",
        "content": (
            "Lesson 6 organizes Alpha research by idea type and highlights reversion, momentum and seasonality as "
            "distinct predictive patterns to implement on BRAIN."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-alpha-examples-by-idea-type-activity-7323147723199688705-oaoc",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 6},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-7",
        "title": "Learn2Quant Lesson 7 — Alphas by Holding Frequencies and Delays",
        "content": (
            "Lesson 7 treats holding frequency and data delay as first-class research dimensions. Holding period "
            "affects expected risk, returns and transaction costs; data delays also have practical implications."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-alphas-by-holding-frequencies-activity-7327842662248554496-z5V6",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 7},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-8",
        "title": "Learn2Quant Lesson 8 — The Power of Diversity",
        "content": (
            "Lesson 8 explains how combining Alphas with sound hypotheses can produce a more robust research "
            "strategy and emphasizes diversity across models, data categories and combination techniques."
        ),
        "url": "https://www.linkedin.com/posts/nitish-maini-2a257318_learn2quant-the-power-of-diversity-lesson-activity-7334097408827531264-z4n_",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "presenter": "Nitish Maini", "kind": "official_training_lesson", "lesson": 8},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-9",
        "title": "Learn2Quant Lesson 9 — Risk Management",
        "content": (
            "Lesson 9 focuses on common factor risk versus idiosyncratic Alpha. It discusses market beta, momentum, "
            "size, value, growth, residual volatility, leverage and liquidity exposures; recommends a novel "
            "idiosyncratic Alpha pool and neutralizing unwanted common factor exposures; and shows a BRAIN example "
            "where risk neutralization can improve Sharpe and reduce drawdown with a possible marginal turnover cost."
        ),
        "url": "https://ytscribe.com/pl/v/eTq8iPhL1Ys",
        "access_scope": "public_transcript_mirror",
        "metadata": {
            "publisher": "WorldQuant",
            "presenter": "Nitish Maini",
            "kind": "official_training_lesson",
            "lesson": 9,
            "content_origin": "WorldQuant-published Learn2Quant video",
            "retrieval_note": "indexed third-party transcript mirror; official curriculum independently confirms risk management",
        },
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-10",
        "title": "Learn2Quant Lesson 10 — Advanced Research Ideas",
        "content": (
            "Lesson 10 discusses advanced statistical, machine-learning and deep-learning methods for improving data "
            "coverage, interpolating lower-frequency data, predicting future data such as returns or earnings, "
            "searching text/data fields, and compressing high-dimensional inputs. It explicitly notes overfitting "
            "risk when data fields outnumber observations and keeps BRAIN simulation as the feedback loop."
        ),
        "url": "https://ytscribe.com/ar/v/nNOFUVfDg3Y",
        "access_scope": "public_transcript_mirror",
        "metadata": {
            "publisher": "WorldQuant",
            "presenter": "Nitish Maini",
            "kind": "official_training_lesson",
            "lesson": 10,
            "content_origin": "WorldQuant-published Learn2Quant video",
            "retrieval_note": "indexed third-party transcript mirror; official curriculum independently confirms advanced research techniques",
        },
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:learn2quant-lesson-11-coda",
        "title": "Learn2Quant and BRAIN | Lesson 11 — official coda locator",
        "content": (
            "WorldQuant's official Ideas page exposes an embedded item titled 'Learn2Quant: Learn2Quant and BRAIN | "
            "Lesson 11'. The article describes BRAIN as a platform for testing theories, receiving immediate feedback "
            "and refining models. This entry is stored as a coda/locator and does not infer hidden video details."
        ),
        "url": "https://www.worldquant.com/ideas/bridging-the-talent-opportunity-gap-in-quantitative-finance/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_training_coda_locator", "lesson": 11},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:alpha-examples-104",
        "title": "104 – Alpha Examples by Data Category: Part 1",
        "content": (
            "Official BRAIN example using hypothesis -> implementation -> simulation -> improvement. Price/Volume: "
            "close-open short-horizon reversion, group_rank within subindustry, Delay-1 to avoid look-ahead, decay to "
            "smooth/reduce turnover, subindustry neutralization and a single-stock capital cap; suggested improvements "
            "include turnover and volatility-regime conditioning. Fundamental: operating cash flow relative to market "
            "capitalization, time-series z-score of improvement, market neutralization, and analyst cash-flow estimates "
            "as a forward-looking extension."
        ),
        "url": "https://worldquantbrain.com/alpha-examples",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant BRAIN", "kind": "official_alpha_example", "retrieved_date": "2026-08-10"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:finding-alphas-official",
        "title": "Finding Alphas — official WorldQuant introduction",
        "content": (
            "Official WorldQuant introduction to Finding Alphas. It frames Alpha research as practical predictive-signal "
            "design spanning information research, fundamental analysis, statistical arbitrage, Alpha diversity, "
            "technical design and advanced/complex designs, with hands-on experimentation rather than formula copying."
        ),
        "authors": ["Igor Tulchinsky"],
        "published_year": 2015,
        "url": "https://www.worldquant.com/ideas/how-to-identify-potential-profit-opportunities-from-market-behaviors-others-have-been-unable-to-quantify/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_book_introduction", "book": "Finding Alphas"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:consultant-program",
        "title": "Consultant Program for Quant Researchers | WorldQuant BRAIN",
        "content": (
            "Official consultant-program guidance tells users to learn quantitative-finance research/modeling in BRAIN, "
            "submit Alphas meeting BRAIN criteria and accumulate points; higher status expands research access such as "
            "data fields, regions, simulation history and advanced tooling."
        ),
        "url": "https://worldquantbrain.com/consultant",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant BRAIN", "kind": "official_consultant_program"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:brain-overview",
        "title": "WorldQuant BRAIN: Crowdsourcing Quantitative Research",
        "content": (
            "Official BRAIN overview describes datasets, research tools, performance dashboards and value-add measures "
            "for building/testing Alpha ideas in real time, with research quality and differentiated ideas central to "
            "the consultant ecosystem."
        ),
        "url": "https://www.worldquant.com/brain/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_brain_overview"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:iqc-2026",
        "title": "International Quant Championship 2026",
        "content": (
            "Official IQC page describes building Alphas from historical market data and predefined operators and "
            "evaluating both quality and quantity. 2026 training topics span Fundamental & Model, Price Volume, tips "
            "for submittable Alphas, D0, Options/Relationship/Vector and newly released datasets."
        ),
        "published_year": 2026,
        "url": "https://www.worldquant.com/brain/iqc/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_competition_and_training", "year": 2026},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:iqc-guidelines-2026",
        "title": "International Quant Championship 2026 Guidelines",
        "content": (
            "Official guidelines state that an Alpha is considered for scoring only if it qualifies for out-of-sample "
            "testing; combined Alpha pools are subject to a correlation test; and WorldQuant may disqualify deliberate "
            "noise/gaming. Research should optimize genuine predictive performance, robustness and differentiation."
        ),
        "published_year": 2026,
        "url": "https://www.worldquant.com/brain/iqc-guidelines/",
        "access_scope": "public",
        "metadata": {"publisher": "WorldQuant", "kind": "official_rules", "year": 2026},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:brain-learn-documentation",
        "title": "WorldQuant BRAIN Learn Documentation",
        "content": (
            "Official authenticated BRAIN documentation portal. The public page is retained only as the authoritative "
            "documentation locator; no hidden submission thresholds or inaccessible documentation body are inferred."
        ),
        "url": "https://platform.worldquantbrain.com/learn/documentation",
        "access_scope": "authenticated_locator",
        "metadata": {"publisher": "WorldQuant BRAIN", "kind": "official_documentation_portal", "retrieved_date": "2026-08-10"},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:brain-operators-snapshot-20260810",
        "title": "WorldQuant BRAIN FASTEXPR operator snapshot — 2026-08-10",
        "content": (
            "Account-visible /operators snapshot captured 2026-08-10 with 66 operators. Relevant operators include "
            "rank, zscore, winsorize, ts_mean, ts_zscore, ts_std_dev, ts_delay, ts_delta, ts_backfill, "
            "ts_decay_linear, trade_when, group_rank, group_neutralize, group_zscore and group_backfill. This is a "
            "dated reproducibility snapshot only; expression generation must validate against the current live catalog."
        ),
        "access_scope": "authenticated_snapshot",
        "metadata": {"publisher": "WorldQuant BRAIN", "kind": "operator_catalog_snapshot", "captured_date": "2026-08-10", "count": 66},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:brain-data-fields-cashflow-snapshot-20260810",
        "title": "WorldQuant BRAIN cash-flow Data Explorer snapshot — 2026-08-10",
        "content": (
            "USA/TOP3000/Delay1 Data Explorer snapshot for cash-flow concepts captured 2026-08-10. Account-visible "
            "MATRIX fields included operating_cashflow_reported_value, "
            "anl4_fs_detail_estimates_advanced_af_nd_cfo_mean, "
            "anl4_fs_detail_estimates_advanced_af_nd_cfo_median, anl4_cfo_mean, standardized_unexpected_cash_flow, "
            "est_fcf and op_cash_flow_stddev. This is a dated snapshot; live field availability must be checked before use."
        ),
        "access_scope": "authenticated_snapshot",
        "metadata": {"publisher": "WorldQuant BRAIN", "kind": "data_explorer_snapshot", "captured_date": "2026-08-10", "region": "USA", "universe": "TOP3000", "delay": 1},
    },
    {
        "source_type": "worldquant",
        "source_key": "worldquant:brain-submission-checks-snapshot-20260810",
        "title": "WorldQuant BRAIN submission-check snapshot — 2026-08-10",
        "content": (
            "Authenticated USA/TOP3000/Delay1 BRAIN simulation snapshot captured 2026-08-10 exposed checks including "
            "LOW_SHARPE, LOW_FITNESS, LOW_TURNOVER, HIGH_TURNOVER, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE, "
            "SELF_CORRELATION and MATCHES_COMPETITION. Observed numeric limits included Sharpe 1.25, Fitness 1.0, "
            "turnover 0.01 to 0.70; the sub-universe Sharpe limit varied by Alpha. UNITS can appear as a warning. "
            "These are dated account/platform observations, not universal or permanent official constants; current "
            "BRAIN responses remain authoritative at execution time."
        ),
        "access_scope": "authenticated_snapshot",
        "metadata": {
            "publisher": "WorldQuant BRAIN",
            "kind": "submission_check_snapshot",
            "captured_date": "2026-08-10",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "observed_limits": {"LOW_SHARPE": 1.25, "LOW_FITNESS": 1.0, "LOW_TURNOVER": 0.01, "HIGH_TURNOVER": 0.7},
        },
    },
]


OFFICIAL_WORLDQUANT_CARDS: list[dict[str, Any]] = [
    {
        "concept": "official_price_volume_reversion",
        "family": "price_volume",
        "hypothesis": (
            "Within comparable subindustries, a stock that closes below its open may exhibit short-horizon reversion "
            "and outperform, while a stock that closes above its open may revert downward. Implement the hypothesis "
            "with Delay-1 to avoid look-ahead, subindustry-relative ranking/neutralization, and turnover-aware smoothing "
            "rather than by adding unrelated complexity."
        ),
        "mechanism": [
            "Compare close-open behavior cross-sectionally within subindustry.",
            "Use Delay-1 for completed daily prices to avoid look-ahead.",
            "Use decay or conditional trading to reduce turnover without replacing the hypothesis.",
            "Treat volatility regime as a possible execution condition, as suggested by the official example.",
        ],
        "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "SUBINDUSTRY", "decay": 10},
        "evidence": [
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official Price/Volume example uses close-open reversion with group_rank, Delay-1, subindustry neutralization and decay, then suggests turnover/volatility conditioning.", "location": "Price Volume example"},
            {"source_key": "worldquant:learn2quant-lesson-4", "support": "positive", "claim": "Lesson 4 presents Price Volume examples intended to be simulated on BRAIN.", "location": "Lesson 4"},
            {"source_key": "worldquant:learn2quant", "support": "positive", "claim": "Learn2Quant teaches hypothesis generation, data categories, delays, diversification and risk.", "location": "What you'll learn"},
        ],
        "operators": ["group_rank", "trade_when", "ts_std_dev", "ts_decay_linear"],
        "expression_templates": ["-group_rank(close-open, subindustry)"],
        "failure_modes": ["High turnover can erase a raw daily reversion edge.", "The effect can be regime-dependent.", "Unrelated complexity breaks hypothesis-preserving repair."],
        "mutation_strategies": ["Change decay/execution settings before changing the economic signal.", "If turnover remains high, test a defensible volatility/regime condition.", "Preserve subindustry-relative comparison unless BRAIN feedback points elsewhere."],
        "confidence": 0.92,
        "status": "active",
    },
    {
        "concept": "official_cashflow_value_and_forward_estimates",
        "family": "fundamental_quality",
        "hypothesis": (
            "A firm's operating cash flow relative to market capitalization can act as a valuation signal; improvement "
            "in that ratio over time may be associated with future outperformance. A source-grounded extension is to "
            "replace stale reported cash flow with account-visible analyst forecasts of operating cash flow, while "
            "preserving the same cash-generation-to-market-value hypothesis."
        ),
        "mechanism": [
            "Normalize operating cash flow by market capitalization.",
            "Use time-series z-score to emphasize improvement/deterioration versus the firm's recent history.",
            "Test an analyst-estimate sibling as the official forward-looking extension.",
            "Handle sparse estimates conservatively; missingness treatment is implementation, not the hypothesis.",
        ],
        "scope": {"region": "USA", "universe": "TOP3000", "lookback_days": 60, "official_neutralization": "MARKET"},
        "evidence": [
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official Fundamental example uses operating cash flow relative to market cap, a 60-day time-series z-score and suggests analyst cash-flow predictions.", "location": "Fundamental example"},
            {"source_key": "worldquant:learn2quant-lesson-4", "support": "positive", "claim": "Lesson 4 presents Fundamental examples intended for BRAIN simulation.", "location": "Lesson 4"},
            {"source_key": "worldquant:brain-data-fields-cashflow-snapshot-20260810", "support": "neutral", "claim": "The dated account snapshot contains reported operating cash flow and analyst CFO estimate fields; live availability must be rechecked.", "location": "2026-08-10 Data Explorer snapshot"},
        ],
        "operators": ["ts_zscore", "ts_backfill", "rank", "divide"],
        "expression_templates": [
            "rank(ts_zscore(divide(ts_backfill(operating_cashflow_reported_value,60),cap),60))",
            "rank(ts_zscore(divide(ts_backfill(anl4_fs_detail_estimates_advanced_af_nd_cfo_mean,60),cap),60))",
        ],
        "failure_modes": ["Sparse/stale observations can reduce coverage.", "Raw levels may capture persistent structure rather than improvement.", "Analyst forecasts can be noisy.", "Unrelated ratio changes break the original hypothesis."],
        "mutation_strategies": ["Compare reported and analyst-forecast siblings separately.", "Adjust backfill conservatively if coverage is weak.", "Test mean/median forecast or a revision form while preserving cash-flow-to-cap logic."],
        "confidence": 0.90,
        "status": "active",
    },
    {
        "concept": "official_hypothesis_first_research_loop",
        "family": "unknown",
        "hypothesis": (
            "Generate Alpha candidates from an explicit economic or market-behavior hypothesis, map that hypothesis to "
            "an appropriate data category and live-supported operators, simulate it in BRAIN, then use observed failures "
            "to make small hypothesis-preserving repairs. Do not begin from arbitrary operator combinations and retrofit "
            "a story afterward."
        ),
        "mechanism": ["Start from an Alpha idea/market behavior.", "Choose data category, idea type, horizon and delay as part of the hypothesis.", "Use the smallest expression that tests it.", "Treat BRAIN failures as evidence and change one causal dimension at a time."],
        "scope": {"workflow": "idea -> data -> expression -> settings -> BRAIN simulation -> diagnose -> targeted repair"},
        "evidence": [
            {"source_key": "worldquant:learn2quant", "support": "positive", "claim": "Official curriculum teaches own Alpha ideas, data/idea categories, horizons/delays, diversification, risk and advanced research.", "location": "What you'll learn"},
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official examples are structured as hypothesis, implementation, simulation and potential improvements.", "location": "Price Volume/Fundamental examples"},
            {"source_key": "worldquant:finding-alphas-official", "support": "positive", "claim": "WorldQuant frames Alpha design as a practical research discipline spanning multiple methods.", "location": "official introduction"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Random operator search can lack stable interpretation.", "Changing many dimensions at once makes feedback unattributable.", "Post-hoc storytelling encourages overfit."],
        "mutation_strategies": ["State the hypothesis before variants.", "Change one field/transform/lookback/neutralization/execution dimension at a time.", "Switch hypotheses after repeated structural failure."],
        "confidence": 0.94,
        "status": "active",
    },
    {
        "concept": "official_alpha_pool_diversification",
        "family": "unknown",
        "hypothesis": (
            "Treat Alpha diversification as a first-class research objective: spread research across independent data "
            "categories, economic hypotheses and structural motifs, and respond to correlation risk by changing the "
            "source of predictive information rather than making cosmetic parameter edits."
        ),
        "mechanism": ["Build across multiple data categories and idea types.", "Use correlation feedback as evidence of insufficient differentiation.", "Prefer a new field family/mechanism over window-only mutations.", "Balance quantity with quality."],
        "scope": {"objective": "diverse low-correlation Alpha pool"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-8", "support": "positive", "claim": "Lesson 8 emphasizes diversity across models, data categories and combination techniques.", "location": "Lesson 8"},
            {"source_key": "worldquant:iqc-guidelines-2026", "support": "positive", "claim": "Official IQC rules apply a correlation test when pools are combined.", "location": "guidelines"},
            {"source_key": "worldquant:finding-alphas-official", "support": "positive", "claim": "WorldQuant's Finding Alphas material emphasizes Alpha diversity.", "location": "official introduction"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Window/decay tweaks can produce cosmetic clones.", "Overexploiting one family can saturate self-correlation.", "Quantity without quality wastes budget."],
        "mutation_strategies": ["On correlation risk, switch data category/mechanism before tuning another window.", "Track family/dataset/structure coverage.", "Keep bounded exploitation plus explicit exploration."],
        "confidence": 0.95,
        "status": "active",
    },
    {
        "concept": "official_oos_integrity_and_anti_gaming",
        "family": "unknown",
        "hypothesis": (
            "Optimize for genuine out-of-sample eligibility, performance and differentiation. Treat submission checks as "
            "evidence about robustness, not targets to game; never add meaningless noise solely to evade correlation or "
            "scoring rules."
        ),
        "mechanism": ["OOS qualification precedes IQC scoring.", "Correlation is an explicit pool-level constraint.", "Noise/gaming can be disqualified.", "Current BRAIN feedback should remain the final empirical authority."],
        "scope": {"policy": "research_integrity_and_oos"},
        "evidence": [
            {"source_key": "worldquant:iqc-guidelines-2026", "support": "positive", "claim": "Official rules cover OOS qualification, pool correlation testing and anti-gaming/noise enforcement.", "location": "guidelines"},
            {"source_key": "worldquant:brain-overview", "support": "positive", "claim": "BRAIN is designed for real-time empirical testing and performance feedback.", "location": "BRAIN overview"},
            {"source_key": "worldquant:consultant-program", "support": "positive", "claim": "Consultant progression depends on Alphas meeting BRAIN criteria and continued research quality/progress.", "location": "consultant path"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Noise injection can violate official rules and destroy interpretability.", "Sample-only optimization can fail OOS.", "Guessed local thresholds can become stale."],
        "mutation_strategies": ["For correlation, change data/hypothesis rather than add noise.", "Keep local gates advisory and current BRAIN checks authoritative.", "Use robustness evidence without over-tightening local filters."],
        "confidence": 0.98,
        "status": "active",
    },
    {
        "concept": "factor-risk neutralization for robust low-correlation BRAIN Alphas",
        "family": "risk_neutralization",
        "hypothesis": (
            "An Alpha pool can become more robust and less mutually correlated when unwanted common factor exposures "
            "are neutralized while the idiosyncratic signal is preserved; this should be tested as a risk-management "
            "implementation change rather than used to replace the original hypothesis."
        ),
        "mechanism": ["Common factor exposures can create correlation/shared drawdowns.", "Neutralization aims to preserve the idiosyncratic component.", "The official lesson shows a potential Sharpe/drawdown benefit with possible turnover cost.", "Neutralization complements rather than replaces true diversity."],
        "scope": {"workflow": "preserve hypothesis -> neutralize justified common factor risk -> re-simulate -> compare Sharpe/drawdown/turnover/correlation"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-9", "support": "positive", "claim": "Lesson 9 distinguishes common factor risks from idiosyncratic Alpha and demonstrates risk neutralization on BRAIN.", "location": "Lesson 9"},
            {"source_key": "worldquant:learn2quant-lesson-8", "support": "positive", "claim": "Lesson 8 emphasizes genuine diversity for robustness.", "location": "Lesson 8"},
            {"source_key": "worldquant:learn2quant", "support": "positive", "claim": "Risk management and diversification are core curriculum topics.", "location": "What you'll learn"},
        ],
        "operators": ["group_neutralize", "group_rank", "group_zscore", "rank", "zscore"],
        "expression_templates": [],
        "failure_modes": ["Over-neutralization can remove the intended signal.", "Improved Sharpe/drawdown can come with higher turnover.", "Shared underlying information can remain correlated after neutralization."],
        "mutation_strategies": ["Test neutralization while keeping the core field/transform unchanged.", "Compare the metrics that motivated the change.", "If correlation persists, switch data/mechanism rather than stack neutralizers."],
        "confidence": 0.93,
        "status": "active",
    },
    {
        "concept": "holding-frequency and delay aligned Alpha design",
        "family": "research_horizon",
        "hypothesis": (
            "Alpha construction should explicitly match the signal's intended holding frequency and observable data "
            "delay; this reduces timing mistakes and makes risk, return and transaction-cost trade-offs interpretable "
            "before BRAIN simulation."
        ),
        "mechanism": ["Holding frequency changes risk/return/transaction-cost profile.", "Delay determines what information is observable.", "Backtesting must avoid look-ahead.", "Decay can smooth a short-horizon signal without changing the hypothesis."],
        "scope": {"workflow": "state horizon/timing -> choose delay/settings -> implement -> BRAIN backtest -> compare turnover/risk/returns"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-7", "support": "positive", "claim": "Holding frequency affects expected risk, returns and transaction costs, and data delays have practical implications.", "location": "Lesson 7"},
            {"source_key": "worldquant:learn2quant-lesson-3", "support": "positive", "claim": "Backtesting and look-ahead avoidance are central to Alpha evaluation.", "location": "Lesson 3"},
            {"source_key": "worldquant:alpha-examples-104", "support": "positive", "claim": "Official Price/Volume example uses Delay-1 and decay to manage timing/turnover.", "location": "Price Volume example"},
        ],
        "operators": ["ts_delay", "ts_decay_linear", "ts_mean"],
        "expression_templates": [],
        "failure_modes": ["Using unavailable future data creates look-ahead.", "Horizon/signal-half-life mismatch can erase usefulness.", "Reactive implementations can create excessive turnover."],
        "mutation_strategies": ["Use justified decay/smoothing for turnover before changing the hypothesis.", "Fix timing with appropriate delay.", "Test meaningfully different horizons rather than adjacent-window sweeps."],
        "confidence": 0.95,
        "status": "active",
    },
    {
        "concept": "official idea-type and data-category research matrix",
        "family": "research_diversity",
        "hypothesis": (
            "Research breadth should be organized on two independent axes—data category and idea type—so candidate "
            "generation explores genuinely different information sources and mechanisms rather than many variants of "
            "one formula family."
        ),
        "mechanism": ["Data-category lessons span Price Volume, Fundamental, Sentiment and Options.", "Idea-type lesson spans reversion, momentum and seasonality.", "Diversity also spans models and combination techniques.", "Use the two axes as a coverage matrix."],
        "scope": {"data_categories": ["price_volume", "fundamental", "sentiment", "options"], "idea_types": ["reversion", "momentum", "seasonality"]},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-4", "support": "positive", "claim": "Lesson 4 covers Price Volume and Fundamental examples.", "location": "Lesson 4"},
            {"source_key": "worldquant:learn2quant-lesson-5", "support": "positive", "claim": "Lesson 5 extends examples to Sentiment and Options.", "location": "Lesson 5"},
            {"source_key": "worldquant:learn2quant-lesson-6", "support": "positive", "claim": "Lesson 6 covers reversion, momentum and seasonality idea types.", "location": "Lesson 6"},
            {"source_key": "worldquant:learn2quant-lesson-8", "support": "positive", "claim": "Lesson 8 emphasizes diversity across models/data/combination techniques.", "location": "Lesson 8"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["Syntactic diversity can still occupy one research cell.", "Window-only variants are not true diversity.", "New categories can fail if live fields are not checked."],
        "mutation_strategies": ["Move to underrepresented data-category × idea-type cells.", "Track both axes in lineage.", "Validate live fields/operators before simulation."],
        "confidence": 0.96,
        "status": "active",
    },
    {
        "concept": "official advanced data enhancement with overfit control",
        "family": "advanced_research",
        "hypothesis": (
            "Advanced statistical/ML methods are most useful when they improve data coverage, frequency, prediction "
            "targets, field search or feature compression while remaining constrained by BRAIN backtesting and explicit "
            "overfitting control; they should not be used merely to make an Alpha more complex."
        ),
        "mechanism": ["Advanced methods can fill missing history or enhance low-frequency data.", "They can predict future data or compress high-dimensional fields.", "High dimensionality can create overfit risk.", "BRAIN simulation remains the empirical feedback loop."],
        "scope": {"workflow": "identify concrete data limitation -> prefer existing BRAIN enhanced/model data -> use advanced method if needed -> compare to simple baseline"},
        "evidence": [
            {"source_key": "worldquant:learn2quant-lesson-10", "support": "positive", "claim": "Lesson 10 covers data enhancement, future-data prediction, feature compression, ML/DL/NLP/LLM ideas and high-dimensional overfit risk.", "location": "Lesson 10"},
            {"source_key": "worldquant:learn2quant", "support": "positive", "claim": "Advanced research techniques are part of the official curriculum and should be applied on BRAIN.", "location": "What you'll learn"},
            {"source_key": "worldquant:learn2quant-lesson-3", "support": "positive", "claim": "Backtesting and look-ahead control remain required when evaluating Alpha quality.", "location": "Lesson 3"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["High-dimensional models can overfit.", "Predicted/imputed values can introduce timing errors.", "Building a new ML stack when BRAIN already provides suitable data is unnecessary complexity."],
        "mutation_strategies": ["Search live BRAIN for existing model/enhanced fields first.", "Use advanced methods only for a named data limitation.", "Compare against a simpler baseline and revert if robustness/interpretability worsens."],
        "confidence": 0.91,
        "status": "active",
    },
    {
        "concept": "dated BRAIN check calibration",
        "family": "submission_quality",
        "hypothesis": (
            "Use current BRAIN simulation checks as dated calibration evidence for candidate review and diagnosis. "
            "Numeric limits observed from one account scope should be refreshed before reuse and should not be "
            "described as universal official constants."
        ),
        "mechanism": [
            "Authenticated BRAIN simulations expose Sharpe, Fitness, turnover, concentration, sub-universe and self-correlation checks.",
            "The 2026-08-10 USA/TOP3000/Delay1 snapshot observed Sharpe 1.25, Fitness 1.0 and turnover 0.01 to 0.70 numeric limits, while sub-universe limits varied by Alpha.",
            "Official public rules describe broader qualification, out-of-sample and correlation constraints rather than a universal fixed numeric table for every BRAIN scope.",
            "Candidate diagnosis should reference the current BRAIN response and label historical limits with their date and scope.",
        ],
        "scope": {"snapshot_date": "2026-08-10", "snapshot_scope": "USA/TOP3000/Delay1", "rule": "refresh current BRAIN checks before relying on numeric limits"},
        "evidence": [
            {"source_key": "worldquant:brain-submission-checks-snapshot-20260810", "support": "positive", "claim": "Authenticated BRAIN responses captured on 2026-08-10 expose current check names and observed numeric limits for USA/TOP3000/Delay1 simulations.", "location": "dated simulation snapshot"},
            {"source_key": "worldquant:iqc-guidelines-2026", "support": "positive", "claim": "Official rules establish out-of-sample qualification and correlation testing as evaluation constraints.", "location": "IQC guidelines"},
            {"source_key": "worldquant:consultant-program", "support": "positive", "claim": "Official consultant guidance refers researchers to BRAIN submission criteria.", "location": "consultant program"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": ["A dated numeric limit can become stale when the platform or scope changes.", "A generic local threshold can misdiagnose a candidate when live BRAIN shows another check.", "Repairing an unrelated metric wastes research effort when BRAIN identifies a different bottleneck."],
        "mutation_strategies": ["Read the latest BRAIN checks before choosing a repair direction.", "Annotate stored numeric limits with date/region/universe/delay.", "Use specific per-Alpha feedback for sub-universe, concentration and self-correlation issues.", "Re-run BRAIN after each targeted repair and retain the result in Research Memory."],
        "confidence": 0.97,
        "status": "active",
    },
]


async def seed_official_worldquant_knowledge() -> dict[str, int]:
    """Idempotently seed official WorldQuant source summaries and distilled cards."""
    for source in OFFICIAL_WORLDQUANT_SOURCES:
        await upsert_knowledge_source(**source)
    for card in OFFICIAL_WORLDQUANT_CARDS:
        await upsert_knowledge_card(card)
    return {"sources": len(OFFICIAL_WORLDQUANT_SOURCES), "cards": len(OFFICIAL_WORLDQUANT_CARDS)}
