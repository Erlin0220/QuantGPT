---
name: wq-alpha-hypothesis
description: Generate WorldQuant BRAIN Alpha hypotheses and candidate FASTEXPR from official WorldQuant guidance plus the account's live BRAIN operators/data fields. Use when asked to find, design, generate, or research new WQ Alphas. Do not use for formal submission; stop at research candidates unless the submission gate is explicitly invoked.
---

# WorldQuant Alpha Hypothesis

## R — Reading

This skill is distilled from the official WorldQuant sources stored in QuantGPT Knowledge Sources, plus the mechanism-first priors produced by `wq-economic-hypothesis`, especially:

- `wq-economic-hypothesis`: combines *Finding Alphas, Second Edition* with Antti Ilmanen's *Expected Returns* framework so the research question states data semantics, expected-return driver, competing explanation, horizon and falsifiers before FASTEXPR construction.
- `worldquant:learn2quant`: start from your own Alpha ideas; vary data category, idea type, holding frequency and delay; diversify the pool and manage risk.
- `worldquant:alpha-examples-104`: the official workflow is hypothesis → implementation → simulation → potential improvement. Its Price/Volume example uses close-open reversion with `group_rank`, Delay-1, neutralization and turnover-aware decay. Its Fundamental example uses operating cash flow / market cap with a time-series z-score and suggests analyst forecasts as a forward-looking improvement.
- Learn2Quant Lessons 4–8: deliberately cross data categories (Price Volume, Fundamental, Sentiment, Options), idea types (reversion, momentum, seasonality), holding frequencies/delays and model/data diversity.
- Learn2Quant Lesson 9: separate common factor risk from idiosyncratic Alpha and use justified risk neutralization rather than allowing shared factor exposure to masquerade as diversity.
- Learn2Quant Lesson 10: advanced data enhancement/ML is justified only by a concrete data problem and must remain constrained by backtesting and overfit control.
- `worldquant:brain-operators-live` and live Data Explorer snapshots: field/operator availability must be checked against the authenticated account instead of assumed.

## I — Interpretation

A strong WQ candidate starts with a falsifiable market hypothesis, then uses the smallest live-supported expression that tests it. BRAIN simulation is the empirical judge. Do not start with random operators and invent a story after seeing metrics.

For new Alpha generation, the hypothesis must first pass through `wq-economic-hypothesis`. Treat its `economic_case` as the semantic contract: information source and expected-return mechanism come before the mathematical transform. This is deliberately stricter when recent research is dominated by `low_fitness`, because compiling more weak stories is not useful inventory replenishment.

## A1 — Past application

Official example 1: close below open may revert upward and close above open may revert downward; compare within subindustry and use Delay-1 to avoid look-ahead.

Official example 2: improving operating cash flow relative to market cap may predict outperformance; use time-series standardization, and test an analyst-estimate variant instead of only stale reported cash flow.

## A2 — Future trigger

Invoke when the user asks for new WorldQuant Alpha ideas, a batch of candidates, better candidate inventory, or a new research direction. If the task is instead to repair a failed simulation, use `wq-alpha-repair`. If the main problem is correlation/saturation, use `wq-alpha-diversify`.

## E — Execution

1. Read `kb_search_alpha_knowledge` for the target goal/family. Prefer active cards with multiple authoritative sources and useful empirical BRAIN feedback.
2. Invoke `wq-economic-hypothesis` and obtain 3–8 ordered `economic_case` objects. Reject cases whose mechanism, direction, horizon or falsifier is still vague; do not translate them into expressions yet.
3. Call `list_wq_operators`; never emit an operator that is absent from the live catalog unless explicitly treating it as an unvalidated proposal.
4. Call `wq_brain_data_catalog` for each case's relevant data concept. Use exact account-visible field IDs; do not invent field names. If live data semantics contradict the proposed case, update or discard the case rather than forcing the field into the story.
5. Convert each surviving `economic_case` into a concise hypothesis that preserves: information source, expected direction, economic/market mechanism, intended horizon/delay, competing explanation when relevant, likely common-factor exposure, state dependency if predeclared, and falsifiers.
6. Prefer semantic diversity across data categories, information sources and expected-return drivers before parameter diversity. Do not spend the batch on window-only variants of one motif. When a proposed advanced/ML route is useful, first check whether BRAIN already exposes a model/enhanced field before building anything locally.
7. Translate each hypothesis into the smallest plausible FASTEXPR. Keep the economic signal separate from simulation settings such as Delay, Decay, Neutralization and Truncation. A transform must have a stated role in the economic case; otherwise remove it.
8. For official-example descendants, preserve the source logic first. Example: test `-group_rank(close-open, subindustry)` as the close-open reversion seed; test cash-flow-to-cap with a time-series z-score and a live analyst cash-flow field as a separate forward-looking sibling.
9. Run `wq-alpha-review` before spending BRAIN budget. For each provisional `RUN`, let the review chain invoke `wq-robustness-validation` and `wq-candidate-evidence`. Package only fully reviewed candidates for `wq_brain_autonomous_research` as structured `skill_candidates`. Each new-hypothesis candidate should carry at least: `expression`, `hypothesis`, `family`, `data_fields`, `skill_chain: ["wq-economic-hypothesis", "wq-alpha-hypothesis", "wq-alpha-review", "wq-robustness-validation", "wq-candidate-evidence"]`, `review_decision: "RUN"`, concise `review_notes`, `robustness_plan`, `candidate_evidence_policy`, and the relevant `knowledge_card_ids`. Preserve the `economic_case` in the generation workspace/research notes even if the deterministic execution layer only persists the normalized hypothesis and card IDs.
10. Send those structured candidates to `wq_brain_autonomous_research` with deterministic fallback disabled. Research may simulate, diagnose and validate, but must not generate a hidden second generation or formally submit from this skill.
11. Feed the real result back into Knowledge/Research Memory. For actionable failures, invoke `wq-failure-diagnosis` then `wq-experiment-allocation`; use `wq-alpha-repair` only after `REPAIR`, `wq-alpha-diversify` after `DIVERSIFY`, and return to `wq-economic-hypothesis` after `NEW_HYPOTHESIS`.

## B — Boundary

- Never guess hidden BRAIN submission rules. Current BRAIN checks are the final authority.
- Never add meaningless noise to evade correlation or scoring checks.
- Never use a field/operator merely because it appeared in an old tutorial; live account availability wins.
- Do not optimize one in-sample result through many untracked parameter changes.
- Do not formal-submit an Alpha from this skill.

## Related skills

- `wq-economic-hypothesis`: mechanism-first semantic design before FASTEXPR.
- `wq-alpha-review`: pre-simulation quality review.
- `wq-alpha-repair`: failure-directed mutation after real BRAIN feedback.
- `wq-alpha-diversify`: pool-level diversification and correlation recovery.
