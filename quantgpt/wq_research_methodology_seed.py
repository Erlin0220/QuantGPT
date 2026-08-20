"""Repository-backed research-methodology knowledge for WQ experiment allocation.

These priors are distilled from authoritative textbooks and author/publisher
materials on expensive sequential experimentation, Bayesian optimization,
optimal learning, multi-armed bandits, and computer experiments. They guide
which BRAIN experiment to run next; they do not replace BRAIN metrics or invent
WorldQuant submission rules.
"""

from __future__ import annotations

from typing import Any

from .wq_knowledge import upsert_knowledge_card, upsert_knowledge_source


RESEARCH_METHODOLOGY_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "textbook",
        "source_key": "textbook:garnett-bayesian-optimization-2023",
        "title": "Bayesian Optimization",
        "authors": ["Roman Garnett"],
        "published_year": 2023,
        "url": "https://bayesoptbook.com/",
        "access_scope": "author_public_fulltext_personal_use",
        "content": (
            "Cambridge University Press monograph on optimizing expensive objective functions. The book develops "
            "Gaussian-process modeling, Bayesian sequential decision making, utility functions, practical acquisition "
            "policies, implementation, convergence, and extensions. For QuantGPT, the transferable principle is to "
            "treat a BRAIN Simulation as an expensive black-box evaluation and choose the next bounded refinement by "
            "expected utility under uncertainty rather than by point estimates alone."
        ),
        "metadata": {
            "publisher": "Cambridge University Press",
            "kind": "textbook",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["expensive_black_box", "sequential_decision", "acquisition_policy", "batch_evaluation"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:lattimore-szepesvari-bandit-algorithms-2020",
        "title": "Bandit Algorithms",
        "authors": ["Tor Lattimore", "Csaba Szepesvari"],
        "published_year": 2020,
        "url": "https://tor-lattimore.com/downloads/book/book.pdf",
        "access_scope": "author_public_fulltext",
        "content": (
            "Cambridge University Press textbook, with an author-provided free online version, on sequential decision "
            "making under uncertainty. It covers stochastic and adversarial bandits, UCB, contextual and linear "
            "bandits, pure exploration, Bayesian methods, Thompson sampling, non-stationarity, and related settings. "
            "For QuantGPT, it provides the exploration-versus-exploitation and context-dependent allocation framework "
            "for deciding which research cell or repair class deserves the next limited Simulation."
        ),
        "metadata": {
            "publisher": "Cambridge University Press",
            "kind": "textbook",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["exploration_exploitation", "contextual_bandits", "pure_exploration", "thompson_sampling"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:slivkins-introduction-multi-armed-bandits-2019",
        "title": "Introduction to Multi-Armed Bandits",
        "authors": ["Aleksandrs Slivkins"],
        "published_year": 2019,
        "url": "https://arxiv.org/abs/1904.07272",
        "external_id": "arXiv:1904.07272",
        "access_scope": "open_arxiv_fulltext",
        "content": (
            "Textbook-style introduction to multi-armed bandits covering stochastic and Bayesian bandits, adaptive "
            "exploration, contextual bandits, structured actions, and bandits with knapsacks. The budget-constrained "
            "view is directly relevant to QuantGPT: BRAIN Simulations are a limited resource, so new hypotheses, "
            "repairs, and diversification experiments should compete for budget instead of receiving equal effort."
        ),
        "metadata": {
            "publisher": "Foundations and Trends in Machine Learning",
            "kind": "open_textbook",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["adaptive_exploration", "contextual_bandits", "budget_constraints", "bandits_with_knapsacks"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:powell-ryzhov-optimal-learning-2012",
        "title": "Optimal Learning",
        "authors": ["Warren B. Powell", "Ilya O. Ryzhov"],
        "published_year": 2012,
        "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118309858",
        "access_scope": "publisher_metadata_and_author_materials",
        "content": (
            "Wiley textbook on collecting information to make effective decisions when information is expensive or "
            "time-consuming. It develops learning policies and gives special attention to the knowledge-gradient "
            "policy across belief models and online/offline settings. For QuantGPT, the key transfer is value of "
            "information: a Simulation is useful not only when it may immediately produce a Candidate, but when its "
            "result can materially improve the next research decision."
        ),
        "metadata": {
            "publisher": "Wiley",
            "kind": "textbook",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["optimal_learning", "knowledge_gradient", "value_of_information", "policy_comparison"],
        },
    },
    {
        "source_type": "textbook",
        "source_key": "textbook:santner-williams-notz-computer-experiments-2018",
        "title": "The Design and Analysis of Computer Experiments",
        "authors": ["Thomas J. Santner", "Brian J. Williams", "William I. Notz"],
        "published_year": 2018,
        "url": "https://link.springer.com/book/10.1007/978-1-4939-8847-1",
        "access_scope": "publisher_preview",
        "content": (
            "Springer textbook on experiments conducted through computer simulators. It covers stochastic-process "
            "models, Bayesian inference, space-filling and criterion-based designs, sensitivity analysis, variable "
            "screening, calibration, and optimization-oriented sequential design. For QuantGPT, BRAIN can be treated "
            "as an expensive simulator: first identify which factors materially affect the response, then spend "
            "additional runs on the dimensions that matter rather than changing many parameters simultaneously."
        ),
        "metadata": {
            "publisher": "Springer",
            "kind": "textbook",
            "edition": 2,
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["computer_experiments", "screening", "sensitivity_analysis", "sequential_design"],
        },
    },
    {
        "source_type": "paper",
        "source_key": "paper:reyes-powell-optimal-learning-lab-2020",
        "title": "Optimal Learning for Sequential Decisions in Laboratory Experimentation",
        "authors": ["Kristopher Reyes", "Warren B. Powell"],
        "published_year": 2020,
        "url": "https://arxiv.org/abs/2004.05417",
        "external_id": "arXiv:2004.05417",
        "access_scope": "open_arxiv_fulltext",
        "content": (
            "Open tutorial by Powell and Reyes on sequential experimentation when experiments are costly and most "
            "experiments may fail. It emphasizes belief models, learning policies, uncertainty reduction, and the "
            "knowledge gradient as a way to maximize the value of information from each experiment. This supplements "
            "Optimal Learning with an openly accessible, experiment-focused treatment."
        ),
        "metadata": {
            "kind": "author_tutorial",
            "retrieved_date": "2026-08-11",
            "distillation_scope": ["sequential_experimentation", "belief_models", "knowledge_gradient", "value_of_information"],
        },
    },
]


RESEARCH_METHODOLOGY_CARDS: list[dict[str, Any]] = [
    {
        "concept": "budgeted_contextual_simulation_allocation",
        "family": "research_allocation",
        "hypothesis": (
            "Treat new hypotheses, repair classes, and diversification routes as competing experiment choices under a "
            "limited BRAIN Simulation budget. Allocate more runs to choices with stronger context-specific evidence "
            "while preserving explicit exploration for uncertain or under-tested choices."
        ),
        "mechanism": [
            "Represent context with family, dataset, operator pattern, failure signature, parent metrics, and prior lineage outcomes.",
            "Use historical outcomes to update confidence in each experiment class rather than treating all repairs equally.",
            "Keep a bounded exploration share so a historically weak family is not permanently excluded by small-sample noise.",
            "Count Simulation as a scarce resource; a repair run competes directly with a new-hypothesis or diversification run.",
        ],
        "scope": {"decision": "which BRAIN experiment should consume the next Simulation budget"},
        "evidence": [
            {"source_key": "textbook:lattimore-szepesvari-bandit-algorithms-2020", "support": "positive", "claim": "Bandit algorithms formalize sequential exploration/exploitation, contextual decisions, pure exploration and Bayesian allocation.", "location": "Parts II, V, VII"},
            {"source_key": "textbook:slivkins-introduction-multi-armed-bandits-2019", "support": "positive", "claim": "The textbook covers adaptive exploration, contextual bandits and budget-constrained bandits with knapsacks.", "location": "Chapters 1, 8, 10"},
            {"source_key": "textbook:powell-ryzhov-optimal-learning-2012", "support": "positive", "claim": "Optimal Learning focuses on gathering expensive information with policies that improve later decisions.", "location": "book overview and knowledge-gradient treatment"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Equal allocation wastes runs on repeatedly unproductive repair classes.",
            "Pure exploitation can lock onto noisy early winners.",
            "A fixed family-level score can miss context-specific repair behavior.",
        ],
        "mutation_strategies": [
            "Rank experiment classes with uncertainty-aware evidence, not only mean candidate rate.",
            "Reserve budget for under-tested contexts.",
            "Compare repair, diversification and new-hypothesis experiments in the same allocation decision.",
        ],
        "confidence": 0.90,
        "status": "active",
    },
    {
        "concept": "value_of_information_before_repair",
        "family": "research_allocation",
        "hypothesis": (
            "A failed Alpha should receive another repair Simulation only when the proposed experiment has meaningful "
            "expected decision value: either a plausible chance of producing a better Candidate or enough information "
            "to change which research route should be chosen next. Repeated low-information tweaks should yield budget "
            "to a more informative experiment."
        ),
        "mechanism": [
            "Separate immediate expected performance from the value of learning something useful.",
            "Prefer experiments whose outcomes discriminate between competing explanations of the failure.",
            "Stop spending on a lineage when additional variants are dominated by alternative experiments in both promise and information value.",
            "Do not invent a universal numeric stopping threshold; calibrate confidence and costs from QuantGPT's own trial history.",
        ],
        "scope": {"decision": "repair vs stop vs switch hypothesis"},
        "evidence": [
            {"source_key": "textbook:powell-ryzhov-optimal-learning-2012", "support": "positive", "claim": "The book develops policies for collecting expensive information and gives special attention to the knowledge gradient.", "location": "book overview"},
            {"source_key": "paper:reyes-powell-optimal-learning-lab-2020", "support": "positive", "claim": "The tutorial frames laboratory experimentation around belief models, learning policies and maximizing value of information from each experiment.", "location": "tutorial overview"},
            {"source_key": "textbook:garnett-bayesian-optimization-2023", "support": "positive", "claim": "Bayesian optimization formulates expensive black-box optimization as sequential decision making with utility-driven policies.", "location": "Chapters 5-9"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Window-only variants can consume budget without resolving the root cause.",
            "A near-miss metric can look attractive even when the next repair is unlikely to change the decision.",
            "Fixed retry counts ignore differences in uncertainty and experiment cost/value.",
        ],
        "mutation_strategies": [
            "Choose repairs that isolate one plausible failure cause.",
            "Prefer a different research route when its expected decision value dominates another sibling tweak.",
            "Use Research Memory lineage outcomes as empirical evidence for repair value.",
        ],
        "confidence": 0.92,
        "status": "active",
    },
    {
        "concept": "bayesian_optimization_for_bounded_alpha_refinement",
        "family": "research_allocation",
        "hypothesis": (
            "Use Bayesian-optimization ideas only for bounded numeric or low-dimensional refinement after the economic "
            "hypothesis and expression structure are fixed. Do not use a numeric optimizer to substitute for semantic "
            "choices such as which dataset, information source, or market mechanism to research."
        ),
        "mechanism": [
            "Treat BRAIN as an expensive black-box evaluator for a small numeric refinement domain.",
            "Model both predicted response and uncertainty, then use an acquisition policy to choose the next setting.",
            "Suitable dimensions include a small number of lookbacks, decay, truncation or another explicitly bounded setting.",
            "Batch policies can be considered when several evaluations can run concurrently, while avoiding redundant near-identical points.",
        ],
        "scope": {"appropriate_for": "bounded low-dimensional near-miss refinement", "not_for": "economic hypothesis generation"},
        "evidence": [
            {"source_key": "textbook:garnett-bayesian-optimization-2023", "support": "positive", "claim": "The book is dedicated to expensive objective optimization using Bayesian modeling, utility functions and practical policies.", "location": "Chapters 1, 5-9, 11"},
            {"source_key": "textbook:santner-williams-notz-computer-experiments-2018", "support": "positive", "claim": "Computer-experiment methodology models simulator output and uses criterion-based and sequential designs for optimization.", "location": "Chapters 3-8"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Applying BO across arbitrary symbolic Alpha expressions creates a poorly defined search geometry.",
            "Too many mixed categorical/semantic choices make a small-budget surrogate unreliable.",
            "Over-refining one in-sample near-miss can create parameter chasing.",
        ],
        "mutation_strategies": [
            "Freeze the hypothesis and expression structure before numeric refinement.",
            "Keep the numeric domain small and pre-declared.",
            "Return to hypothesis/diversification research if the local refinement region remains weak.",
        ],
        "confidence": 0.91,
        "status": "active",
    },
    {
        "concept": "screening_and_sensitivity_before_expensive_refinement",
        "family": "research_allocation",
        "hypothesis": (
            "Before spending many BRAIN runs on a failed lineage, identify which design dimensions plausibly control "
            "the observed failure. Screen or vary a small number of factors to learn sensitivity, then focus sequential "
            "experiments on the influential dimensions instead of simultaneously changing field, structure, window, "
            "neutralization and execution settings."
        ),
        "mechanism": [
            "Use screening to distinguish influential dimensions from decorative or weak ones.",
            "Design experiments so outcomes remain attributable to specific changes.",
            "Use broader/space-filling exploration when the local response is poorly understood, then criterion-based refinement once structure is learned.",
            "Treat sensitivity evidence as guidance, not as a replacement for BRAIN's authoritative result.",
        ],
        "scope": {"workflow": "diagnose -> screen influential dimensions -> targeted sequential refinement"},
        "evidence": [
            {"source_key": "textbook:santner-williams-notz-computer-experiments-2018", "support": "positive", "claim": "The book covers space-filling and criterion-based designs, sensitivity analysis, variable screening and calibration for computer simulators.", "location": "Chapters 6-9"},
            {"source_key": "textbook:garnett-bayesian-optimization-2023", "support": "positive", "claim": "Model assessment and sequential utility-driven optimization require explicit treatment of uncertainty and model fit.", "location": "Chapters 4-9"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Changing many variables at once destroys causal attribution.",
            "Repeated local tweaks can miss a better region of the design space.",
            "A surrogate or local optimizer can be misleading before influential dimensions are identified.",
        ],
        "mutation_strategies": [
            "Change one causal dimension at a time when testing a specific diagnosis.",
            "Use a deliberately diverse small screening batch when the diagnosis is uncertain.",
            "Only refine parameters that have shown material response sensitivity.",
        ],
        "confidence": 0.90,
        "status": "active",
    },
    {
        "concept": "utility_defined_before_acquisition_policy",
        "family": "research_allocation",
        "hypothesis": (
            "For an expensive sequential experiment, define what makes the next observation useful before choosing a "
            "policy that ranks experiments. In QuantGPT, useful can mean Candidate upside, discrimination between failure "
            "causes, uncertainty reduction, or preserving scarce Simulation budget; raw Fitness alone is not the utility."
        ),
        "mechanism": [
            "State the downstream decision that the next Simulation could change.",
            "Define the relevant utility components before using an acquisition-like ranking.",
            "Account for uncertainty and experiment cost as part of the sequential decision.",
            "For batches, prefer complementary experiments whose joint outcomes answer different questions over redundant near-identical evaluations.",
        ],
        "scope": {"workflow": "define decision utility -> compare experiment choices -> select sequential/batch evaluation"},
        "evidence": [
            {"source_key": "textbook:garnett-bayesian-optimization-2023", "support": "positive", "claim": "Bayesian Optimization separates decision theory, utility functions, common policies and policy computation for expensive objectives.", "location": "Chapters 5-9 and 11"},
            {"source_key": "textbook:powell-ryzhov-optimal-learning-2012", "support": "positive", "claim": "Optimal Learning evaluates expensive information collection by how it improves later decisions, including knowledge-gradient/value-of-information thinking.", "location": "book overview and knowledge-gradient treatment"},
        ],
        "operators": [],
        "expression_templates": [],
        "failure_modes": [
            "Choosing a policy before defining utility can optimize the wrong objective.",
            "Using raw Fitness as the only utility ignores information value and Candidate/ACTIVE consequences.",
            "Parallel near-duplicates can waste a batch because their outcomes are jointly redundant.",
        ],
        "mutation_strategies": [
            "Write the decision consequence of each proposed experiment before ranking it.",
            "Prefer tests that separate plausible failure causes when immediate Candidate upside is uncertain.",
            "Diversify batch questions when several Simulations can run concurrently.",
        ],
        "confidence": 0.92,
        "status": "active",
    },
]


async def seed_research_methodology_knowledge() -> dict[str, int]:
    for source in RESEARCH_METHODOLOGY_SOURCES:
        await upsert_knowledge_source(**source)
    for card in RESEARCH_METHODOLOGY_CARDS:
        await upsert_knowledge_card(card)
    return {"sources": len(RESEARCH_METHODOLOGY_SOURCES), "cards": len(RESEARCH_METHODOLOGY_CARDS)}
