---
name: wq-alpha-diversify
description: Diversify a WorldQuant BRAIN Alpha research pool when self-correlation, family saturation, repeated near-clones, or poor inventory breadth becomes the bottleneck. Prove independent information across source, mechanism, horizon, exposure or PnL behavior instead of relying on syntactic novelty.
---

# WorldQuant Alpha Diversify

## R — Reading

Distilled from `worldquant:learn2quant`, especially Lessons 7–9, `worldquant:finding-alphas-official`, `worldquant:iqc-2026`, `worldquant:iqc-guidelines-2026`, plus QuantGPT's empirical Research Memory and the `diversification` Knowledge Cards.

WorldQuant explicitly teaches diversity across data categories, idea types, holding frequencies/delays, model/combination techniques, and risk exposures. Lesson 9 adds the critical distinction between superficially different expressions and genuinely different common-factor exposure. Official IQC material applies pool-level correlation constraints and prohibits anti-correlation gaming/noise.

## I — Interpretation

Diversification is an **information-independence problem**, not an expression-editing problem.

Two expressions can look different but still use the same information source, economic mechanism, horizon and factor exposure. Conversely, two structurally similar expressions can be meaningful siblings if they test genuinely different information with independent realized behavior.

The Skill therefore reasons on explicit diversity dimensions:

1. **information_source** — dataset/category/field provenance;
2. **economic_mechanism** — why the signal should predict returns;
3. **horizon_delay** — signal half-life, holding frequency and observable delay;
4. **structure** — transformation/operator motif;
5. **factor_exposure** — market/industry/style/common-risk dependence;
6. **empirical_behavior** — BRAIN SC and PnL/return correlation when available.

## A1 — Past application

A 20-day and 30-day revision rank built from the same analyst field and same mechanism is usually weak diversification even when its FASTEXPR differs. A cash-flow valuation signal and an options-skew signal differ at the information-source and economic-mechanism levels; if their common-factor exposure and realized PnL are also different, that is much stronger evidence of breadth.

## A2 — Future trigger

Invoke when:

- official SELF_CORRELATION fails;
- repeated local portfolio-correlation evidence indicates saturation;
- Research Memory shows one family/dataset/operator pattern dominating inventory;
- the candidate queue has low breadth;
- `wq-failure-diagnosis` identifies `correlation_saturation`;
- `wq-experiment-allocation` chooses `DIVERSIFY`.

For ordinary new-candidate generation without a saturation problem, use `wq-alpha-hypothesis`.

## E — Execution

1. Read Research Memory: ACTIVE/candidate family counts, dataset/category usage, operator-pattern usage, official SC failures, local PnL correlation evidence, factor/neutralization settings, and recent lineage outcomes.
2. Query `kb_search_alpha_knowledge` for `diversity independent information mechanism horizon factor exposure self correlation` and preserve relevant diversification card IDs.
3. Read live BRAIN Data Explorer. Prefer account-visible underused datasets/categories with a defensible economic hypothesis; never diversify into unavailable data.
4. Define the **saturation reference**: which existing Alpha/family/cell is the new candidate intended to be independent from, and what dimension is currently overrepresented?
5. Build a small slate of genuinely different research cells. When budget permits, cover at least 3 cells that differ in meaningful information-level dimensions, not merely lookbacks.
6. For every diversified candidate emit a structured `diversity_case`:
   - `reference`: saturated family/candidate/cell;
   - `changed_dimensions`: one or more of `information_source`, `economic_mechanism`, `horizon_delay`, `structure`, `factor_exposure`;
   - `why_independent`: concise economic reason the changed dimensions should add different information;
   - `unchanged_dimensions`: useful for showing what was intentionally held constant;
   - `empirical_evidence`: official SC/PnL/local-correlation evidence when available, otherwise `pending`;
   - `knowledge_card_ids`.
7. Prefer diversification moves in this order when official Self-Correlation/correlation saturation is the trigger:
   1. different information source / dataset category;
   2. different economic mechanism / idea type;
   3. meaningfully different horizon/delay when the mechanism supports another half-life;
   4. different factor exposure or justified neutralization when shared common-risk exposure is specifically evidenced;
   5. different structural transform that changes the information extraction, not merely syntax;
   6. fine parameter changes only as a last resort and never call them strong diversification by themselves.
8. A candidate is **not proven diversified** merely because `changed_dimensions` is non-empty. Mark evidence state:
   - `designed_diverse` — source/mechanism rationale is different but no realized correlation evidence yet;
   - `empirically_supported` — official SC / BRAIN PnL evidence supports independence;
   - `not_diverse` — official SC or strong realized correlation contradicts the design thesis.
9. Keep bounded exploitation for productive families, but reserve explicit exploration for underrepresented information sources. Use `wq-experiment-allocation` to decide the budget; this Skill defines what constitutes a meaningful diversity move.
10. Review each candidate with `wq-alpha-review`, then complete `wq-robustness-validation` and `wq-candidate-evidence`. For `RUN` candidates emit structured `skill_candidates` with:
    - `skill_chain: ["wq-alpha-hypothesis", "wq-failure-diagnosis", "wq-experiment-allocation", "wq-alpha-diversify", "wq-alpha-review", "wq-robustness-validation", "wq-candidate-evidence"]` when diversification follows a failure;
    - `diversity_case`;
    - new hypothesis/family/data fields;
    - review notes, `robustness_plan`, `candidate_evidence_policy`, and `knowledge_card_ids`.
11. Return through `wq_brain_autonomous_research` with deterministic fallback disabled. Persist family/dataset/structure/diversity lineage and update the diversity case after BRAIN SC/PnL evidence arrives.

## B — Boundary

- Do not add random noise, constants or irrelevant terms to lower measured correlation.
- Do not claim syntactic novelty is portfolio diversification.
- Do not force every candidate to change every diversity dimension; one strong information-level change can be meaningful.
- Do not abandon productive families solely for novelty; keep bounded exploitation.
- Do not use unavailable datasets or fields.
- Do not interpret local correlation proxies as official SC results.
- Do not neutralize mechanically. Exposure changes must be tied to an identified shared risk.

## Related skills

- `wq-alpha-hypothesis`
- `wq-failure-diagnosis`
- `wq-experiment-allocation`
- `wq-alpha-review`
- `wq-alpha-repair`
- `wq-robustness-validation`
- `wq-candidate-evidence`
