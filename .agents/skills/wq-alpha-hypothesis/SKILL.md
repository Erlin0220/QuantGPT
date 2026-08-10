---
name: wq-alpha-hypothesis
description: Generate WorldQuant BRAIN Alpha hypotheses and candidate FASTEXPR from official WorldQuant guidance plus the account's live BRAIN operators/data fields. Use when asked to find, design, generate, or research new WQ Alphas. Do not use for formal submission; stop at research candidates unless the submission gate is explicitly invoked.
---

# WorldQuant Alpha Hypothesis

## R — Reading

This skill is distilled from the official WorldQuant sources stored in QuantGPT Knowledge Sources, especially:

- `worldquant:learn2quant`: start from your own Alpha ideas; vary data category, idea type, holding frequency and delay; diversify the pool and manage risk.
- `worldquant:alpha-examples-104`: the official workflow is hypothesis → implementation → simulation → potential improvement. Its Price/Volume example uses close-open reversion with `group_rank`, Delay-1, neutralization and turnover-aware decay. Its Fundamental example uses operating cash flow / market cap with a time-series z-score and suggests analyst forecasts as a forward-looking improvement.
- Learn2Quant Lessons 4–8: deliberately cross data categories (Price Volume, Fundamental, Sentiment, Options), idea types (reversion, momentum, seasonality), holding frequencies/delays and model/data diversity.
- Learn2Quant Lesson 9: separate common factor risk from idiosyncratic Alpha and use justified risk neutralization rather than allowing shared factor exposure to masquerade as diversity.
- Learn2Quant Lesson 10: advanced data enhancement/ML is justified only by a concrete data problem and must remain constrained by backtesting and overfit control.
- `worldquant:brain-operators-live` and live Data Explorer snapshots: field/operator availability must be checked against the authenticated account instead of assumed.

## I — Interpretation

A strong WQ candidate starts with a falsifiable market hypothesis, then uses the smallest live-supported expression that tests it. BRAIN simulation is the empirical judge. Do not start with random operators and invent a story after seeing metrics.

## A1 — Past application

Official example 1: close below open may revert upward and close above open may revert downward; compare within subindustry and use Delay-1 to avoid look-ahead.

Official example 2: improving operating cash flow relative to market cap may predict outperformance; use time-series standardization, and test an analyst-estimate variant instead of only stale reported cash flow.

## A2 — Future trigger

Invoke when the user asks for new WorldQuant Alpha ideas, a batch of candidates, better candidate inventory, or a new research direction. If the task is instead to repair a failed simulation, use `wq-alpha-repair`. If the main problem is correlation/saturation, use `wq-alpha-diversify`.

## E — Execution

1. Read `kb_search_alpha_knowledge` for the target goal/family. Prefer active cards with multiple WorldQuant sources and useful empirical BRAIN feedback.
2. Call `list_wq_operators`; never emit an operator that is absent from the live catalog unless explicitly treating it as an unvalidated proposal.
3. Call `wq_brain_data_catalog` for the relevant data concept. Use exact account-visible field IDs; do not invent field names.
4. Write 3–8 hypotheses before expressions. Each hypothesis must state: expected direction, economic/market mechanism, intended horizon/delay, likely common-factor exposure, and what would falsify it.
5. Prefer diversity across the official data-category × idea-type matrix and structures. Do not spend the batch on window-only variants of one motif. When a proposed advanced/ML route is useful, first check whether BRAIN already exposes a model/enhanced field before building anything locally.
6. Translate each hypothesis into the smallest plausible FASTEXPR. Keep the economic signal separate from simulation settings such as Delay, Decay, Neutralization and Truncation.
7. For official-example descendants, preserve the source logic first. Example: test `-group_rank(close-open, subindustry)` as the close-open reversion seed; test cash-flow-to-cap with a time-series z-score and a live analyst cash-flow field as a separate forward-looking sibling.
8. Run `wq-alpha-review` before spending BRAIN budget. Package only `RUN` candidates for `wq_brain_autonomous_research` as structured `skill_candidates`. Each candidate must carry at least: `expression`, `hypothesis`, `family`, `data_fields`, `skill_chain: ["wq-alpha-hypothesis", "wq-alpha-review"]`, `review_decision: "RUN"`, and concise `review_notes`. Add `knowledge_card_ids` when applicable.
9. Send those structured candidates to `wq_brain_autonomous_research` with deterministic fallback disabled. Research may simulate, diagnose and validate, but must not generate a hidden second generation or formally submit from this skill.
10. Feed the real result back into Knowledge/Research Memory. For actionable failures, invoke `wq-alpha-repair`; for self-correlation/family saturation, invoke `wq-alpha-diversify`; review every new child again before the next BRAIN research call.

## B — Boundary

- Never guess hidden BRAIN submission rules. Current BRAIN checks are the final authority.
- Never add meaningless noise to evade correlation or scoring checks.
- Never use a field/operator merely because it appeared in an old tutorial; live account availability wins.
- Do not optimize one in-sample result through many untracked parameter changes.
- Do not formal-submit an Alpha from this skill.

## Related skills

- `wq-alpha-review`: pre-simulation quality review.
- `wq-alpha-repair`: failure-directed mutation after real BRAIN feedback.
- `wq-alpha-diversify`: pool-level diversification and correlation recovery.
