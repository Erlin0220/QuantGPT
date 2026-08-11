"""Repository-backed economic-hypothesis priors for WorldQuant Alpha research.

These cards distill publisher/author materials from Finding Alphas (2e) and
Expected Returns.  They improve the quality of the hypothesis that precedes a
FASTEXPR; they do not define WorldQuant submission rules or guarantee that a
mechanism will survive BRAIN Simulation.
"""

from __future__ import annotations

from typing import Any

from .wq_knowledge import upsert_knowledge_card, upsert_knowledge_source


ECONOMIC_HYPOTHESIS_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "textbook",
        "source_key": "textbook:tulchinsky-finding-alphas-2e-2019",
        "title": "Finding Alphas: A Quantitative Approach to Building Trading Strategies, Second Edition",
        "authors": ["Igor Tulchinsky", "WorldQuant Virtual Research Center contributors"],
        "published_year": 2019,
        "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9781119571278",
        "access_scope": "publisher_metadata_toc_and_chapter_summaries",
        "content": (
            "Wiley/WorldQuant practitioner volume on predictive-signal design. The second edition organizes research "
            "around alpha design and evaluation, data and alpha design, turnover, correlation, overfitting/bias control, "
            "and structured exploration of alpha space. For QuantGPT the transferable rule is to begin with an interpretable "
            "research idea and data meaning, then use BRAIN evidence to evaluate and refine it rather than generating "
            "operator combinations first."
        ),
        "metadata": {
            "publisher": "Wiley",
            "kind": "textbook",
            "edition": 2,
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["alpha_design", "data_semantics", "structured_exploration", "robustness"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "chapter:finding-alphas-data-alpha-design-2019",
        "title": "Data and Alpha Design",
        "authors": ["Weijia Li"],
        "published_year": 2019,
        "url": "https://onlinelibrary.wiley.com/doi/10.1002/9781119571278.ch6",
        "access_scope": "publisher_chapter_summary",
        "content": (
            "The chapter summary states that data is central to alpha design: data can inspire ideas, should be sanity-checked, "
            "and must be understood semantically before robust alpha ideas can be formed. It also notes that an information-rich "
            "dataset can support many distinct alpha ideas. QuantGPT transfers this into a data-semantics gate before expression "
            "construction and encourages multiple economic questions from one useful dataset instead of shallow window variants."
        ),
        "metadata": {
            "publisher": "Wiley",
            "kind": "book_chapter",
            "book": "Finding Alphas, Second Edition",
            "chapter": 6,
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["data_understanding", "data_validation", "idea_generation"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "chapter:finding-alphas-triple-axis-plan-2019",
        "title": "The Triple-Axis Plan",
        "authors": ["Nitish Maini"],
        "published_year": 2019,
        "url": "https://onlinelibrary.wiley.com/doi/10.1002/9781119571278.ch11",
        "access_scope": "publisher_chapter_summary",
        "content": (
            "The chapter presents a structured way to explore alpha space using three axes: Ideas & Datasets, Performance "
            "Parameters, and Regions & Universes. QuantGPT uses the distinction to avoid spending an entire research batch on "
            "parameter-only siblings: semantic idea/data changes are a different research move from performance-parameter "
            "refinement, and region/universe changes are a separate axis again."
        ),
        "metadata": {
            "publisher": "Wiley",
            "kind": "book_chapter",
            "book": "Finding Alphas, Second Edition",
            "chapter": 11,
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["alpha_space", "ideas_datasets", "performance_parameters", "regions_universes"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:ilmanen-expected-returns-2011",
        "title": "Expected Returns: An Investor's Guide to Harvesting Market Rewards",
        "authors": ["Antti Ilmanen"],
        "published_year": 2011,
        "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118467190",
        "access_scope": "publisher_metadata_toc_and_summaries",
        "content": (
            "Wiley reference on expected-return determination. It balances historical evidence, economic/financial theory, "
            "and forward-looking/current market conditions; surveys rational drivers such as risk and liquidity premia and "
            "behavioral explanations such as extrapolation and overconfidence; and treats expected returns as time-varying. "
            "QuantGPT transfers these ideas as hypothesis-construction priors, not as direct short-horizon Alpha guarantees."
        ),
        "metadata": {
            "publisher": "Wiley",
            "kind": "textbook",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["risk_premia", "liquidity", "behavioral_finance", "forward_looking_indicators", "time_variation"],
        },
    },
    {
        "source_type": "other",
        "source_key": "author:aqr-ilmanen-expected-returns-framework-2011",
        "title": "Expected Returns: An Investors Guide to Harvesting Market Rewards",
        "authors": ["Antti Ilmanen"],
        "published_year": 2011,
        "url": "https://www.aqr.com/insights/research/book/expected-returns-an-investors-guide-to-harvesting-market-rewards",
        "access_scope": "author_employer_public_summary",
        "content": (
            "AQR's author page summarizes the framework: expected returns may reflect rational risk/liquidity premia or "
            "behavioral biases, may vary through time, and should be assessed using historical performance, theories, and "
            "forward-looking indicators rather than historical averages alone. It also highlights style sources such as value, "
            "carry and momentum. QuantGPT uses this as corroboration for mechanism-first, falsifiable Alpha hypotheses."
        ),
        "metadata": {
            "publisher": "AQR",
            "kind": "author_summary",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["mechanism_classes", "three_inputs", "time_varying_expected_returns"],
        },
    },
]


ECONOMIC_HYPOTHESIS_CARDS: list[dict[str, Any]] = [
    {
        "concept": "economic_mechanism_before_expression",
        "family": "economic_hypothesis",
        "hypothesis": (
            "Before constructing a FASTEXPR, state what information the data contains and why that information could predict "
            "future relative returns. Distinguish the economic or behavioral mechanism from the mathematical transform used to "
            "test it; an operator chain is an implementation, not the hypothesis itself."
        ),
        "mechanism": [
            "Describe the data in domain terms before choosing transforms.",
            "Classify the leading return driver, for example risk compensation, liquidity, behavioral underreaction/overreaction, information diffusion, valuation/carry, positioning/crowding, or another explicit mechanism.",
            "State expected direction and horizon before observing BRAIN metrics.",
            "Translate the mechanism into the smallest expression that can falsify it; add complexity only when it represents another explicit piece of the mechanism.",
        ],
        "scope": {"stage": "pre_FASTEXPR_hypothesis_design"},
        "evidence": [
            {"source_key": "chapter:finding-alphas-data-alpha-design-2019", "support": "positive", "claim": "The publisher summary emphasizes understanding and validating data so it can inspire robust alpha ideas.", "location": "chapter summary"},
            {"source_key": "textbook:ilmanen-expected-returns-2011", "support": "methodological_transfer", "claim": "Expected-return analysis combines theory about return drivers with empirical and current-condition evidence; QuantGPT transfers this as a hypothesis prior rather than a direct WQ rule.", "location": "book overview and Parts I-II"},
            {"source_key": "textbook:tulchinsky-finding-alphas-2e-2019", "support": "positive", "claim": "Finding Alphas organizes research around alpha design, data, evaluation and structured experimentation.", "location": "Design and Evaluation"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Starting from rank/ts_delta/ts_mean and inventing a story afterward encourages weak, repetitive hypotheses.",
            "A field can be statistically convenient but economically meaningless or misunderstood.",
            "Changing many transforms at once can hide which mechanism was actually tested.",
        ],
        "mutation_strategies": [
            "Change the information source or economic mechanism when the base hypothesis is weak.",
            "Change transforms only when the new transform tests a different aspect of the same stated mechanism.",
            "Keep the first implementation minimal so failure is interpretable.",
        ],
        "confidence": 0.94,
        "status": "active",
    },
    {
        "concept": "three_input_expected_return_triangulation",
        "family": "economic_hypothesis",
        "hypothesis": (
            "A stronger Alpha hypothesis separates three evidence inputs: historical/empirical prior, theoretical or economic "
            "mechanism, and forward-looking/current-condition evidence. Not every hypothesis will have strong evidence in all "
            "three buckets, but missing evidence must be marked as unknown rather than silently invented."
        ),
        "mechanism": [
            "Historical input asks whether the effect has a plausible empirical prior without treating the past average as permanent.",
            "Theory input explains why the effect could persist or why another market participant would leave it unarbitraged.",
            "Forward-looking input identifies an observable state, revision, valuation, liquidity or other current condition that should strengthen/weaken the effect when relevant.",
            "Record which input is absent; uncertainty is part of the hypothesis and informs experiment allocation.",
        ],
        "scope": {"evidence_buckets": ["historical", "theory", "forward_looking_current_condition"]},
        "evidence": [
            {"source_key": "textbook:ilmanen-expected-returns-2011", "support": "positive", "claim": "The publisher overview says expected-return judgments balance historical returns, theoretical considerations and current market conditions.", "location": "book overview"},
            {"source_key": "author:aqr-ilmanen-expected-returns-framework-2011", "support": "positive", "claim": "The author's AQR summary explicitly identifies historical performance, theories and forward-looking indicators as the main inputs.", "location": "author summary"},
            {"source_key": "chapter:finding-alphas-data-alpha-design-2019", "support": "positive", "claim": "Data understanding and sanity checks are prerequisites for robust alpha ideas.", "location": "chapter summary"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Historical average alone can become naive extrapolation.",
            "A plausible story without observable evidence is difficult to falsify.",
            "Post-hoc regime variables can turn current conditions into data mining rather than forward-looking evidence.",
        ],
        "mutation_strategies": [
            "If theory is strong but empirical evidence is weak, run a small discriminating test rather than a large family sweep.",
            "If a current-condition variable is part of the mechanism, predeclare its expected interaction before Simulation.",
            "If only a historical pattern exists, downgrade confidence and preserve exploration elsewhere.",
        ],
        "confidence": 0.92,
        "status": "active",
    },
    {
        "concept": "risk_behavior_liquidity_competing_explanations",
        "family": "economic_hypothesis",
        "hypothesis": (
            "For a predictive pattern, write at least one credible competing explanation when possible: rational compensation "
            "for risk/liquidity versus behavioral mispricing/information processing, or another explicit alternative. The two "
            "stories imply different factor exposures, horizons and falsifiers, which makes BRAIN experiments more informative."
        ),
        "mechanism": [
            "Risk/liquidity stories predict compensation for bearing an undesirable exposure or trading in costly states.",
            "Behavioral stories predict underreaction, overreaction, extrapolation, overconfidence or delayed information incorporation.",
            "The leading and competing stories should imply different observable consequences whenever possible.",
            "Use neutralization/conditioning only when it helps distinguish the competing explanations; do not add it cosmetically.",
        ],
        "scope": {"goal": "make_hypotheses_falsifiable_and_diagnostic"},
        "evidence": [
            {"source_key": "textbook:ilmanen-expected-returns-2011", "support": "positive", "claim": "Expected Returns separately surveys rational expected-return theories and behavioral finance and discusses liquidity as a return driver.", "location": "Chapters 5-7 and case studies"},
            {"source_key": "author:aqr-ilmanen-expected-returns-framework-2011", "support": "positive", "claim": "The author summary distinguishes rational risk/liquidity premia from psychological biases such as extrapolation and overconfidence.", "location": "author summary"},
            {"source_key": "textbook:tulchinsky-finding-alphas-2e-2019", "support": "positive", "claim": "Finding Alphas treats risk factors, bias control, data and alpha evaluation as explicit research concerns.", "location": "Design and Evaluation"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "A single unfalsifiable story makes every backtest outcome easy to rationalize.",
            "Neutralizing a true compensated risk premium may erase the signal; failing to neutralize a common factor may make a false Alpha look strong.",
            "Behavioral labels without a concrete information-processing mechanism add narrative but no testable content.",
        ],
        "mutation_strategies": [
            "Use factor-neutralization changes as discriminating tests only when the competing explanation predicts them.",
            "Change horizon when competing mechanisms imply different speeds of information incorporation.",
            "Prefer a new information source when neither leading nor competing explanation survives the first test.",
        ],
        "confidence": 0.90,
        "status": "active",
    },
    {
        "concept": "predeclared_state_dependent_alpha",
        "family": "economic_hypothesis",
        "hypothesis": (
            "Expected-return effects can vary with market state, but regime conditioning should be predeclared from the mechanism "
            "rather than added after a weak backtest. A state variable is justified only when the hypothesis predicts why the "
            "signal should strengthen, weaken or reverse in that state."
        ),
        "mechanism": [
            "State the baseline mechanism first.",
            "Name the state variable and expected interaction before testing it.",
            "Use the regime experiment as a falsifier/discriminating test, not as unrestricted parameter search.",
            "If conditioning reduces turnover but destroys the predicted effect, update the mechanism instead of trying many more thresholds.",
        ],
        "scope": {"stage": "hypothesis_or_failure_diagnosis", "rule": "predeclare_state_interaction"},
        "evidence": [
            {"source_key": "textbook:ilmanen-expected-returns-2011", "support": "methodological_transfer", "claim": "The book treats expected returns as time-varying and dependent on underlying conditions; QuantGPT transfers this as a disciplined regime-hypothesis prior.", "location": "book overview and broader themes"},
            {"source_key": "author:aqr-ilmanen-expected-returns-framework-2011", "support": "positive", "claim": "The author summary states that expected returns on return sources may vary over time.", "location": "author summary"},
            {"source_key": "textbook:tulchinsky-finding-alphas-2e-2019", "support": "positive", "claim": "Finding Alphas includes turnover, overfitting/bias control and alpha evaluation, which constrain post-hoc conditioning.", "location": "Design and Evaluation"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Trying many regime thresholds after seeing metrics is parameter chasing.",
            "Conditioning can mechanically improve turnover while removing the economic effect.",
            "A state variable with no mechanism increases complexity without information value.",
        ],
        "mutation_strategies": [
            "Test at most a small predeclared state interaction when the mechanism warrants it.",
            "If the predicted interaction fails, lower belief in the mechanism instead of scanning thresholds.",
            "Route to a new hypothesis when the baseline and its one informative state test both fail.",
        ],
        "confidence": 0.91,
        "status": "active",
    },
    {
        "concept": "structured_semantic_expansion_before_parameter_sweep",
        "family": "economic_hypothesis",
        "hypothesis": (
            "When inventory is weak, expand research first across Ideas & Datasets and genuinely different return mechanisms, "
            "then use performance-parameter refinement only for hypotheses that already show signal. Region/universe changes are "
            "a separate research axis and should not be confused with semantic diversification."
        ),
        "mechanism": [
            "Create a slate of distinct information sources and mechanisms before generating window siblings.",
            "Treat lookback/decay/truncation as performance-parameter refinement after semantic promise exists.",
            "Treat region/universe as another axis whose change can test portability but does not create a new economic mechanism by itself.",
            "Use Research Memory to avoid overused idea/data/structure cells while preserving bounded exploration.",
        ],
        "scope": {"inventory_mode": "replenishment", "priority": "semantic_breadth_before_numeric_depth"},
        "evidence": [
            {"source_key": "chapter:finding-alphas-triple-axis-plan-2019", "support": "positive", "claim": "The Triple-Axis Plan separates Ideas & Datasets, Performance Parameters, and Regions & Universes as distinct directions for alpha-space exploration.", "location": "chapter summary"},
            {"source_key": "chapter:finding-alphas-data-alpha-design-2019", "support": "positive", "claim": "An information-rich dataset can inspire many alpha ideas when the data is properly understood.", "location": "chapter summary"},
            {"source_key": "author:aqr-ilmanen-expected-returns-framework-2011", "support": "methodological_transfer", "claim": "The expected-return framework distinguishes multiple return drivers/styles, supporting semantic diversity rather than a single repeated motif.", "location": "author summary"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Large batches of window-only variants increase trial count without increasing hypothesis coverage.",
            "Changing universe can hide a weak mechanism rather than fix it.",
            "Parameter sweeps on a zero-signal hypothesis spend Simulation budget without learning much.",
        ],
        "mutation_strategies": [
            "When a whole slate has low Fitness, switch mechanism/information source before expanding numeric search.",
            "Reserve numeric refinement for near-misses with a credible mechanism.",
            "Track semantic coverage separately from parameter coverage.",
        ],
        "confidence": 0.93,
        "status": "active",
    },
]


async def seed_economic_hypothesis_knowledge() -> dict[str, int]:
    for source in ECONOMIC_HYPOTHESIS_SOURCES:
        await upsert_knowledge_source(**source)
    for card in ECONOMIC_HYPOTHESIS_CARDS:
        await upsert_knowledge_card(card)
    return {"sources": len(ECONOMIC_HYPOTHESIS_SOURCES), "cards": len(ECONOMIC_HYPOTHESIS_CARDS)}
